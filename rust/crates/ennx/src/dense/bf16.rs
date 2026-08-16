use crate::weights::ComputeBackend;

use super::{has_direction, sign, validate_leaves, validate_terms, DenseLeaf, DenseTerm};

#[cfg(all(target_os = "macos", feature = "metal"))]
extern crate metal as metal_crate;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal_crate::Buffer;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod cuda;

/// A BF16 parameter tree whose base weights stay resident across perturbations.
pub struct Bf16Tree {
    len: usize,
    engine: Resident,
}

enum Resident {
    Cpu {
        base: Vec<u16>,
        candidate: Vec<u16>,
        leaves: Vec<DenseLeaf>,
    },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Resident),
    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    Cuda(cuda::Resident),
}

impl Bf16Tree {
    pub fn new(
        base: Vec<u16>,
        leaves: Vec<DenseLeaf>,
        backend: ComputeBackend,
    ) -> Result<Self, String> {
        validate(&base, &leaves)?;
        let len = base.len();
        let engine = match backend {
            ComputeBackend::Cpu => Resident::Cpu {
                candidate: base.clone(),
                base,
                leaves,
            },
            ComputeBackend::Metal | ComputeBackend::Agx | ComputeBackend::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Resident::Metal(metal::Resident::new(
                        &base,
                        &leaves,
                        backend != ComputeBackend::Metal,
                    )?)
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err("Metal BF16 pytree is not available in this build".into());
                }
            }
            ComputeBackend::OpenCl => return Err("OpenCL BF16 pytree is not available".into()),
            ComputeBackend::Cuda => {
                #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
                {
                    Resident::Cuda(cuda::Resident::new(&base, &leaves)?)
                }
                #[cfg(not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64")))]
                {
                    return Err("CUDA BF16 pytree is not available in this build".into());
                }
            }
        };
        Ok(Self { len, engine })
    }

    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    pub unsafe fn from_device(
        pointer: u64,
        len: usize,
        leaves: Vec<DenseLeaf>,
    ) -> Result<Self, String> {
        if pointer == 0 || len == 0 {
            return Err("CUDA BF16 pytree requires a non-empty device buffer".into());
        }
        validate_leaves(&leaves, Some(len))?;
        Ok(Self {
            len,
            engine: Resident::Cuda(unsafe { cuda::Resident::from_device(pointer, len, &leaves)? }),
        })
    }

    pub fn materialize(&mut self, terms: &[DenseTerm]) -> Result<(), String> {
        validate_terms(terms)?;
        if !has_direction(terms) {
            return Err("BF16 pytree terms cancel to zero".into());
        }
        match &mut self.engine {
            Resident::Cpu {
                base,
                candidate,
                leaves,
            } => {
                if let Err(error) = materialize(base, candidate, leaves, terms) {
                    candidate.clone_from(base);
                    return Err(error);
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Resident::Metal(engine) => engine.materialize(terms)?,
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            Resident::Cuda(engine) => engine.materialize(terms)?,
        }
        Ok(())
    }

    pub fn candidate(&self) -> Result<Vec<u16>, String> {
        match &self.engine {
            Resident::Cpu { candidate, .. } => Ok(candidate.clone()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Resident::Metal(engine) => Ok(engine.candidate()),
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            Resident::Cuda(engine) => engine.candidate(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn candidate_buffer(&self) -> Option<&Buffer> {
        match &self.engine {
            Resident::Metal(engine) => Some(engine.buffer()),
            Resident::Cpu { .. } => None,
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    pub fn device_ptr(&self, stream: Option<i64>) -> Result<(u64, usize, usize), String> {
        match &self.engine {
            Resident::Cuda(engine) => engine.device_ptr(stream),
            _ => Err("BF16 tree is not resident on CUDA".into()),
        }
    }
}

fn validate(base: &[u16], leaves: &[DenseLeaf]) -> Result<(), String> {
    if base.is_empty() {
        return Err("BF16 pytree base cannot be empty".into());
    }
    if base.iter().any(|&value| !decode(value).is_finite()) {
        return Err("BF16 pytree base values must be finite".into());
    }
    validate_leaves(leaves, Some(base.len()))
}

fn materialize(
    base: &[u16],
    out: &mut [u16],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
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
            let value = decode(base[index]) + leaf.scale * sum;
            if !value.is_finite() {
                return Err("BF16 pytree perturbation overflowed FP32".into());
            }
            let candidate = encode(value);
            out[index] = if sum == 0.0 || candidate == base[index] {
                next_finite(base[index], positive)
            } else {
                candidate
            };
        }
    }
    Ok(())
}

fn decode(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn encode(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn next_finite(value: u16, positive: bool) -> u16 {
    if value & 0x7fff == 0 {
        return if positive { 1 } else { 0x8001 };
    }
    let grows = (value & 0x8000 == 0) == positive;
    let candidate = if grows {
        value.wrapping_add(1)
    } else {
        value.wrapping_sub(1)
    };
    if decode(candidate).is_finite() {
        candidate
    } else if grows {
        value.wrapping_sub(1)
    } else {
        value.wrapping_add(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[test]
    fn auto_requires_metal() {
        let error = Bf16Tree::new(
            vec![encode(1.0)],
            vec![DenseLeaf::new(7, 0, 1, 1.0).unwrap()],
            ComputeBackend::Auto,
        )
        .err()
        .unwrap();
        assert_eq!(error, "Metal BF16 pytree is not available in this build");
    }

    #[test]
    fn candidate_starts_at_the_base() {
        let base = vec![encode(1.0), encode(-2.0)];
        let tree = Bf16Tree::new(
            base.clone(),
            vec![DenseLeaf::new(7, 0, base.len(), 1.0).unwrap()],
            ComputeBackend::Cpu,
        )
        .unwrap();
        assert_eq!(tree.candidate().unwrap(), base);
    }

    #[test]
    fn sub_ulp_directions_still_change_every_weight() {
        let base = vec![encode(1.0), encode(-2.0), encode(4.0), encode(-8.0)];
        let mut tree = Bf16Tree::new(
            base.clone(),
            vec![DenseLeaf::new(11, 0, base.len(), 1.0e-6).unwrap()],
            ComputeBackend::Cpu,
        )
        .unwrap();
        tree.materialize(&[DenseTerm::new(17, 1.0e-6).unwrap()])
            .unwrap();
        assert!(tree
            .candidate()
            .unwrap()
            .iter()
            .zip(base)
            .all(|(candidate, base)| *candidate != base));
    }

    #[test]
    fn next_value_stays_finite_at_the_bf16_limits() {
        assert!(decode(next_finite(0x7f7f, true)).is_finite());
        assert!(decode(next_finite(0xff7f, false)).is_finite());
    }

    #[test]
    fn rollback_overflow() {
        let base = vec![encode(1.0)];
        let mut tree = Bf16Tree::new(
            base.clone(),
            vec![DenseLeaf::new(7, 0, 1, f32::MAX).unwrap()],
            ComputeBackend::Cpu,
        )
        .unwrap();
        let error = tree
            .materialize(&[DenseTerm::new(11, f32::MAX).unwrap()])
            .unwrap_err();
        assert_eq!(error, "BF16 pytree perturbation overflowed FP32");
        assert_eq!(tree.candidate().unwrap(), base);
    }
}
