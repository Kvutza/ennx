//! Native FLAME forward evaluation over borrowed, contiguous CUDA BF16 weights.

use crate::CudaResult;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlameConfig {
    pub layers: u32,
    pub width: u32,
    pub heads: u32,
    pub vocab: u32,
    pub dense_width: u32,
    pub expert_width: u32,
    pub shared_width: u32,
    pub experts: u32,
    pub top_k: u32,
    pub context: u32,
    pub epsilon: f32,
    pub rope_base: f32,
}

impl FlameConfig {
    pub fn validate(&self, max_tokens: u32) -> CudaResult<()> {
        if [
            self.layers,
            self.width,
            self.heads,
            self.vocab,
            self.dense_width,
            self.expert_width,
            self.shared_width,
            self.experts,
            self.top_k,
            self.context,
        ]
        .iter()
        .any(|&x| x == 0 || x > i32::MAX as u32)
            || self.width % self.heads != 0
            || (self.width / self.heads) % 2 != 0
            || self.top_k > self.experts
            || self.width > i32::MAX as u32 / 3
            || [self.dense_width, self.expert_width, self.shared_width]
                .iter()
                .any(|&x| x > i32::MAX as u32 / 2)
            || max_tokens == 0
            || max_tokens > self.context
            || u64::from(max_tokens) * u64::from(self.top_k) > i32::MAX as u64
            || !self.epsilon.is_finite()
            || !(0.0..1.0).contains(&self.epsilon)
            || self.epsilon == 0.0
            || !self.rope_base.is_finite()
            || self.rope_base <= 1.0
        {
            return Err(
                "Invalid FLAME dimensions, attention heads, routing, or normalization".into(),
            );
        }
        Ok(())
    }
}

fn check_tokens(config: &FlameConfig, max_tokens: u32, tokens: &[i32]) -> CudaResult<()> {
    if tokens.is_empty()
        || tokens.len() > max_tokens as usize
        || tokens.iter().any(|&x| x < 0 || x as u32 >= config.vocab)
    {
        return Err("FLAME tokens must be nonempty, in vocabulary, and within max_tokens".into());
    }
    Ok(())
}

fn check_batch(
    config: &FlameConfig,
    max_tokens: u32,
    tokens: &[Vec<i32>],
    masks: &[Vec<bool>],
) -> CudaResult<()> {
    if tokens.is_empty() || tokens.len() != masks.len() {
        return Err("FLAME requires equally sized nonempty token and mask batches".into());
    }
    for (row, mask) in tokens.iter().zip(masks) {
        check_tokens(config, max_tokens, row)?;
        if row.len() < 2 || row.len() != mask.len() || mask[0] || !mask[1..].iter().any(|&x| x) {
            return Err("Each FLAME loss mask must match its tokens, leave token zero unscored, and score a target".into());
        }
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::*;
    use std::ffi::{CStr, c_char, c_void};
    use std::marker::PhantomData;
    use std::ptr::NonNull;
    use std::rc::Rc;

    unsafe extern "C" {
        #[link_name = "ennx_flame_create"]
        fn flame_create(
            config: *const FlameConfig,
            max_tokens: u32,
            error: *mut c_char,
            capacity: usize,
        ) -> *mut c_void;
        #[link_name = "ennx_flame_destroy"]
        fn flame_destroy(handle: *mut c_void);
        #[link_name = "ennx_flame_workspace"]
        fn flame_workspace(handle: *const c_void) -> u64;
        #[link_name = "ennx_flame_weights_len"]
        fn flame_weightslen(handle: *const c_void) -> u64;
        #[link_name = "ennx_flame_losses"]
        fn flame_losses(
            handle: *mut c_void,
            weights: *const u16,
            weights_len: usize,
            tokens: *const i32,
            masks: *const u8,
            lengths: *const u32,
            batch: usize,
            output: *mut f32,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        #[link_name = "ennx_flame_logits"]
        fn flame_logits(
            handle: *mut c_void,
            weights: *const u16,
            weights_len: usize,
            tokens: *const i32,
            length: u32,
            output: *mut f32,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
    }

    fn native_error(error: &[c_char; 1024]) -> String {
        // Every buffer starts zeroed; the C API always writes a terminated message.
        let text = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        if text.is_empty() {
            "Native FLAME evaluation failed".into()
        } else {
            text.into_owned()
        }
    }

    pub struct FlameEvaluator {
        handle: NonNull<c_void>,
        config: FlameConfig,
        max_tokens: u32,
        weights_len: usize,
        _thread: PhantomData<Rc<()>>,
    }

    impl FlameEvaluator {
        pub fn new(config: FlameConfig, max_tokens: u32) -> CudaResult<Self> {
            config.validate(max_tokens)?;
            let mut error = [0; 1024];
            let handle = NonNull::new(unsafe {
                flame_create(&config, max_tokens, error.as_mut_ptr(), error.len())
            })
            .ok_or_else(|| native_error(&error))?;
            let weights_len = unsafe { flame_weightslen(handle.as_ptr()) } as usize;
            Ok(Self {
                handle,
                config,
                max_tokens,
                weights_len,
                _thread: PhantomData,
            })
        }

        pub fn weights_len(&self) -> usize {
            self.weights_len
        }

        pub fn workspace_bytes(&self) -> u64 {
            unsafe { flame_workspace(self.handle.as_ptr()) }
        }

        pub fn vocab(&self) -> usize {
            self.config.vocab as usize
        }

        pub fn check_batch(&self, tokens: &[Vec<i32>], masks: &[Vec<bool>]) -> CudaResult<()> {
            check_batch(&self.config, self.max_tokens, tokens, masks)
        }

        pub fn check_tokens(&self, tokens: &[i32]) -> CudaResult<()> {
            check_tokens(&self.config, self.max_tokens, tokens)
        }

        fn check_weights(&self, pointer: u64, len: usize) -> CudaResult<()> {
            if pointer == 0 || pointer % 2 != 0 || len != self.weights_len {
                return Err(format!(
                    "FLAME requires {} contiguous CUDA BF16 weights",
                    self.weights_len
                ));
            }
            Ok(())
        }

        /// # Safety
        /// `pointer` must contain `len` readable BF16 values on CUDA device zero,
        /// ready on the legacy stream, and remain alive until this call returns.
        pub unsafe fn losses(
            &mut self,
            pointer: u64,
            len: usize,
            tokens: &[Vec<i32>],
            masks: &[Vec<bool>],
        ) -> CudaResult<Vec<f32>> {
            self.check_weights(pointer, len)?;
            self.check_batch(tokens, masks)?;
            let lengths = tokens
                .iter()
                .map(|row| row.len() as u32)
                .collect::<Vec<_>>();
            let tokens = tokens.iter().flatten().copied().collect::<Vec<_>>();
            let masks = masks
                .iter()
                .flatten()
                .map(|&x| u8::from(x))
                .collect::<Vec<_>>();
            let mut output = vec![0.0; lengths.len()];
            let mut error = [0; 1024];
            let status = unsafe {
                flame_losses(
                    self.handle.as_ptr(),
                    pointer as *const u16,
                    len,
                    tokens.as_ptr(),
                    masks.as_ptr(),
                    lengths.as_ptr(),
                    lengths.len(),
                    output.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(&error));
            }
            if output.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err("Native FLAME returned an invalid loss".into());
            }
            Ok(output)
        }

        /// # Safety
        /// Same borrowed CUDA BF16 pointer contract as [`Self::losses`].
        pub unsafe fn logits(
            &mut self,
            pointer: u64,
            len: usize,
            tokens: &[i32],
        ) -> CudaResult<Vec<f32>> {
            self.check_weights(pointer, len)?;
            self.check_tokens(tokens)?;
            let size = tokens
                .len()
                .checked_mul(self.vocab())
                .ok_or("FLAME logits size overflow")?;
            let mut output = vec![0.0; size];
            let mut error = [0; 1024];
            let status = unsafe {
                flame_logits(
                    self.handle.as_ptr(),
                    pointer as *const u16,
                    len,
                    tokens.as_ptr(),
                    tokens.len() as u32,
                    output.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(&error));
            }
            if output.iter().any(|x| !x.is_finite()) {
                return Err("Native FLAME returned nonfinite logits".into());
            }
            Ok(output)
        }
    }

    impl Drop for FlameEvaluator {
        fn drop(&mut self) {
            unsafe { flame_destroy(self.handle.as_ptr()) };
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use native::FlameEvaluator;

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FlameConfig {
        FlameConfig {
            layers: 2,
            width: 8,
            heads: 2,
            vocab: 13,
            dense_width: 12,
            expert_width: 5,
            shared_width: 7,
            experts: 4,
            top_k: 2,
            context: 16,
            epsilon: 1e-6,
            rope_base: 10000.0,
        }
    }

    #[test]
    fn dimensions_precuda() {
        let base = config();
        assert!(base.validate(8).is_ok());
        assert!(base.validate(0).is_err());
        assert!(base.validate(17).is_err());
        for heads in [0, 3, 8] {
            assert!(FlameConfig { heads, ..base }.validate(8).is_err());
        }
        for epsilon in [0.0, -1.0, 1.0, f32::NAN, f32::INFINITY] {
            assert!(FlameConfig { epsilon, ..base }.validate(8).is_err());
        }
        assert!(FlameConfig { top_k: 5, ..base }.validate(8).is_err());
        assert!(
            FlameConfig {
                dense_width: i32::MAX as u32,
                ..base
            }
            .validate(8)
            .is_err()
        );
        assert!(
            FlameConfig {
                context: i32::MAX as u32,
                ..base
            }
            .validate(i32::MAX as u32)
            .is_err()
        );
        assert!(
            FlameConfig {
                rope_base: f32::NAN,
                ..base
            }
            .validate(8)
            .is_err()
        );
    }

    #[test]
    fn loss_exportguard() {
        let config = config();
        assert!(check_batch(&config, 8, &[vec![1, 2]], &[vec![false, true]]).is_ok());
        for tokens in [vec![], vec![-1, 2], vec![1, 13], vec![1; 9]] {
            assert!(check_tokens(&config, 8, &tokens).is_err());
        }
        for masks in [
            vec![],
            vec![vec![true, true]],
            vec![vec![false, false]],
            vec![vec![false]],
        ] {
            assert!(check_batch(&config, 8, &[vec![1, 2]], &masks).is_err());
        }
        assert!(check_batch(&config, 8, &[], &[]).is_err());
    }
}
