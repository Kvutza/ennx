use crate::weights::ComputeDevice;

pub const METAL_OPS: &str = include_str!("dense/ops.metal");
pub const OPENCL_OPS: &str = include_str!("dense/ops.cl");

#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "opencl"))]
const TILE_ELEMENTS: usize = 65_536;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod cuda;

mod bf16;
mod linear;

pub use bf16::ParamBuffer;
pub use linear::{linear, DenseLinear, DenseView};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseLeaf {
    pub key: u64,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
}

impl DenseLeaf {
    pub fn new(key: u64, offset: usize, len: usize, scale: f32) -> Result<Self, String> {
        if len == 0 {
            return Err("dense leaf length must be positive".into());
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err("dense leaf scale must be finite and positive".into());
        }
        Ok(Self {
            key,
            offset,
            len,
            scale,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseTerm {
    pub seed: u64,
    pub coefficient: f32,
}

/// Stable leaf key for a named tensor shared with external model runtimes.
pub fn tensor_key(name: &str) -> u64 {
    name.as_bytes()
        .iter()
        .fold(1_469_598_103_934_665_603, |key, byte| {
            (key ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211)
        })
}

impl DenseTerm {
    pub fn new(seed: u64, coefficient: f32) -> Result<Self, String> {
        if !coefficient.is_finite() {
            return Err("dense coefficient must be finite".into());
        }
        Ok(Self { seed, coefficient })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseResult {
    pub values: Vec<f32>,
    pub changed: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "opencl"))]
pub(super) struct DenseTile {
    leaf: u32,
    start: u32,
    len: u32,
    pad: u32,
}

pub fn apply(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
    device: ComputeDevice,
) -> Result<DenseResult, String> {
    dense_validate(base, leaves, terms)?;

    let values = match device {
        ComputeDevice::Cpu => dense_cpu(base, leaves, terms)?,
        ComputeDevice::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::apply(base, leaves, terms, false)?
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("Metal dense directions are not available in this build".into());
            }
        }
        ComputeDevice::Agx => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::apply(base, leaves, terms, true)?
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("AGX dense directions are not available in this build".into());
            }
        }
        ComputeDevice::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                opencl::apply(base, leaves, terms)?
            }
            #[cfg(not(feature = "opencl"))]
            {
                return Err("OpenCL dense directions are not available in this build".into());
            }
        }
        ComputeDevice::Cuda => {
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            {
                cuda::apply(base, leaves, terms)?
            }
            #[cfg(not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64")))]
            {
                return Err("CUDA dense directions are not available in this build".into());
            }
        }
        ComputeDevice::Auto => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::apply(base, leaves, terms, true)
                    .or_else(|_| metal::apply(base, leaves, terms, false))?
            }
            #[cfg(all(
                feature = "cuda",
                target_os = "linux",
                target_arch = "x86_64",
                not(all(target_os = "macos", feature = "metal"))
            ))]
            {
                cuda::apply(base, leaves, terms)?
            }
            #[cfg(all(
                feature = "opencl",
                not(all(target_os = "macos", feature = "metal")),
                not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))
            ))]
            {
                opencl::apply(base, leaves, terms)?
            }
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                all(feature = "cuda", target_os = "linux", target_arch = "x86_64"),
                feature = "opencl"
            )))]
            {
                dense_cpu(base, leaves, terms)?
            }
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err("dense perturbation overflowed FP32".into());
    }
    let changed = base
        .iter()
        .zip(&values)
        .filter(|(before, after)| before != after)
        .count();
    if changed != base.len() {
        return Err(format!(
            "dense direction changed {changed} of {} parameters",
            base.len()
        ));
    }
    Ok(DenseResult { values, changed })
}

pub fn dist2(leaves: &[DenseLeaf], left: &[DenseTerm], right: &[DenseTerm]) -> Result<f64, String> {
    validate_leaves(leaves, None)?;
    validate_terms(left)?;
    validate_terms(right)?;
    zig_dist2(leaves, left, right)
}

fn dense_validate(base: &[f32], leaves: &[DenseLeaf], terms: &[DenseTerm]) -> Result<(), String> {
    if base.is_empty() {
        return Err("dense base cannot be empty".into());
    }
    if base.iter().any(|value| !value.is_finite()) {
        return Err("dense base values must be finite".into());
    }
    validate_leaves(leaves, Some(base.len()))?;
    validate_terms(terms)?;
    if !has_direction(terms) {
        return Err("dense terms cancel to zero".into());
    }
    Ok(())
}

pub(super) fn validate_leaves(leaves: &[DenseLeaf], expected: Option<usize>) -> Result<(), String> {
    if leaves.is_empty() {
        return Err("at least one dense leaf is required".into());
    }
    let mut end = 0usize;
    for leaf in leaves {
        if leaf.offset != end {
            return Err(format!(
                "dense leaf offset {} does not continue parameter offset {end}",
                leaf.offset
            ));
        }
        if leaf.len == 0 || !leaf.scale.is_finite() || leaf.scale <= 0.0 {
            return Err("dense leaves require positive lengths and scales".into());
        }
        end = leaf
            .offset
            .checked_add(leaf.len)
            .ok_or("dense parameter count overflow")?;
    }
    if expected.is_some_and(|count| end != count) {
        return Err(format!(
            "dense leaves cover {end} parameters, expected {}",
            expected.unwrap()
        ));
    }
    Ok(())
}

pub(super) fn validate_terms(terms: &[DenseTerm]) -> Result<(), String> {
    if terms.iter().any(|term| !term.coefficient.is_finite()) {
        return Err("dense coefficients must be finite".into());
    }
    Ok(())
}

#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "opencl"))]
pub(super) fn tiles(leaves: &[DenseLeaf]) -> Result<Vec<DenseTile>, String> {
    let mut tiles = Vec::new();
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let leaf_index = u32::try_from(leaf_index).map_err(|_| "dense leaf count exceeds u32")?;
        let mut start = 0usize;
        while start < leaf.len {
            let len = (leaf.len - start).min(TILE_ELEMENTS);
            tiles.push(DenseTile {
                leaf: leaf_index,
                start: u32::try_from(start).map_err(|_| "dense leaf length exceeds u32")?,
                len: u32::try_from(len).expect("dense tile length fits u32"),
                pad: 0,
            });
            start += len;
        }
    }
    Ok(tiles)
}

pub(super) fn has_direction(terms: &[DenseTerm]) -> bool {
    terms.iter().enumerate().any(|(index, term)| {
        !terms[..index]
            .iter()
            .any(|previous| previous.seed == term.seed)
            && coefficient(terms, term.seed) != 0.0
    })
}

fn coefficient(terms: &[DenseTerm], seed: u64) -> f64 {
    terms
        .iter()
        .filter(|term| term.seed == seed)
        .map(|term| f64::from(term.coefficient))
        .sum()
}

#[cfg(feature = "zig-dense")]
fn dense_cpu(base: &[f32], leaves: &[DenseLeaf], terms: &[DenseTerm]) -> Result<Vec<f32>, String> {
    let mut values = vec![0.0; base.len()];
    let mut changed = 0usize;
    let status = unsafe {
        ennx_dense_apply_f32(
            base.as_ptr(),
            base.len(),
            leaves.as_ptr(),
            leaves.len(),
            terms.as_ptr(),
            terms.len(),
            values.as_mut_ptr(),
            &mut changed,
        )
    };
    if status != 0 {
        return Err("Zig dense application rejected its inputs".into());
    }
    if changed != base.len() {
        return Err(format!(
            "Zig dense application changed {changed} of {} parameters",
            base.len()
        ));
    }
    Ok(values)
}

#[cfg(not(feature = "zig-dense"))]
fn dense_cpu(base: &[f32], leaves: &[DenseLeaf], terms: &[DenseTerm]) -> Result<Vec<f32>, String> {
    let mut values = vec![0.0; base.len()];
    reference(base, leaves, terms, &mut values)?;
    Ok(values)
}

#[cfg(feature = "zig-dense")]
fn zig_dist2(leaves: &[DenseLeaf], left: &[DenseTerm], right: &[DenseTerm]) -> Result<f64, String> {
    let mut distance = 0.0;
    let status = unsafe {
        ennx_dense_dist2(
            leaves.as_ptr(),
            leaves.len(),
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            &mut distance,
        )
    };
    if status == 0 {
        Ok(distance)
    } else {
        Err("Zig dense distance rejected its inputs".into())
    }
}

#[cfg(not(feature = "zig-dense"))]
fn zig_dist2(leaves: &[DenseLeaf], left: &[DenseTerm], right: &[DenseTerm]) -> Result<f64, String> {
    let energy: f64 = leaves
        .iter()
        .map(|leaf| leaf.len as f64 * f64::from(leaf.scale).powi(2))
        .sum();
    let mut seeds = Vec::with_capacity(left.len() + right.len());
    for term in left.iter().chain(right) {
        if !seeds.contains(&term.seed) {
            seeds.push(term.seed);
        }
    }
    let coefficient_distance: f64 = seeds
        .into_iter()
        .map(|seed| (coefficient(left, seed) - coefficient(right, seed)).powi(2))
        .sum();
    Ok(energy * coefficient_distance)
}

fn reference(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
    out: &mut [f32],
) -> Result<(), String> {
    for leaf in leaves {
        for local in 0..leaf.len {
            let index = leaf.offset + local;
            let mut sum = 0.0f32;
            let mut strongest = 0.0f32;
            let mut positive = true;
            for term in terms {
                if term.coefficient == 0.0 {
                    continue;
                }
                let direction = sign(term.seed, leaf.key, local as u64);
                sum += term.coefficient * direction;
                if term.coefficient.abs() > strongest {
                    strongest = term.coefficient.abs();
                    positive = (term.coefficient > 0.0) == (direction > 0.0);
                }
            }
            let candidate = base[index] + leaf.scale * sum;
            out[index] = if sum == 0.0 || candidate == base[index] {
                dense_next(base[index], positive)
            } else if candidate.is_finite() {
                candidate
            } else {
                return Err("dense perturbation overflowed FP32".into());
            };
        }
    }
    Ok(())
}

pub(crate) fn sign(seed: u64, leaf: u64, element: u64) -> f32 {
    let leaf = mix64(leaf ^ 0xd6e8_feb8_6659_fd93);
    let element = mix64(element ^ 0xa076_1d64_78bd_642f);
    if mix64(seed ^ leaf ^ element) & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

pub(super) fn dense_next(value: f32, positive: bool) -> f32 {
    let mut bits = value.to_bits();
    if value == 0.0 {
        return f32::from_bits(if positive { 1 } else { 0x8000_0001 });
    }
    if (value > 0.0) == positive {
        bits = bits.wrapping_add(1);
    } else {
        bits = bits.wrapping_sub(1);
    }
    let candidate = f32::from_bits(bits);
    if candidate.is_finite() {
        return candidate;
    }
    if (value > 0.0) == positive {
        f32::from_bits(value.to_bits().wrapping_sub(1))
    } else {
        f32::from_bits(value.to_bits().wrapping_add(1))
    }
}

fn mix64(input: u64) -> u64 {
    let mut value = input.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(feature = "zig-dense")]
extern "C" {
    fn ennx_dense_apply_f32(
        base: *const f32,
        num_values: usize,
        leaves: *const DenseLeaf,
        num_leaves: usize,
        terms: *const DenseTerm,
        num_terms: usize,
        out: *mut f32,
        changed: *mut usize,
    ) -> i32;

    fn ennx_dense_dist2(
        leaves: *const DenseLeaf,
        num_leaves: usize,
        left: *const DenseTerm,
        num_left: usize,
        right: *const DenseTerm,
        num_right: usize,
        distance: *mut f64,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves() -> [DenseLeaf; 2] {
        [
            DenseLeaf::new(11, 0, 4, 0.5).unwrap(),
            DenseLeaf::new(29, 4, 4, 1.25).unwrap(),
        ]
    }

    #[test]
    fn signs_match_the_zig_contract() {
        let expected = [
            1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
        ];
        for (element, expected) in expected.into_iter().enumerate() {
            assert_eq!(sign(0x1234_5678_9abc_def0, 11, element as u64), expected);
        }
    }

    #[test]
    fn tensor_keys_are_stable() {
        assert_eq!(
            tensor_key("blk.27.attn_q.weight"),
            3_843_877_851_495_245_630
        );
    }

    #[test]
    fn cpu_changes_the_complete_pytree() {
        let base = [0.5, -1.0, 2.0, 0.25, 4.0, -2.0, 0.75, -0.125];
        let terms = [DenseTerm::new(0x1234_5678_9abc_def0, 0.01).unwrap()];
        let result = apply(&base, &leaves(), &terms, ComputeDevice::Cpu).unwrap();
        assert_eq!(result.changed, base.len());
        assert!(base
            .iter()
            .zip(result.values)
            .all(|(left, right)| *left != right));
    }

    #[test]
    fn coefficient_distance_is_independent_of_parameter_count_work() {
        let left = [DenseTerm::new(7, 0.5).unwrap()];
        let right = [DenseTerm::new(7, -0.5).unwrap()];
        assert_eq!(dist2(&leaves(), &left, &right).unwrap(), 7.25);
    }
}
