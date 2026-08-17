use crate::weights::ComputeDevice;

use super::{has_direction, sign, validate_leaves, validate_terms, DenseLeaf, DenseTerm};

#[cfg(all(target_os = "macos", feature = "metal"))]
extern crate metal as metal_crate;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal_crate::Buffer;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod cuda;

/// A BF16 parameter tree whose base weights stay resident across perturbations.
pub struct ParamBuffer {
    len: usize,
    device: ParamDevice,
    kind: ComputeDevice,
}

enum ParamDevice {
    Cpu {
        base: Vec<u16>,
        candidate: Vec<u16>,
        leaves: Vec<DenseLeaf>,
    },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Resident),
    #[cfg(feature = "opencl")]
    OpenCl(opencl::Resident),
    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    Cuda(cuda::Resident),
}

impl ParamBuffer {
    pub fn new(
        base: Vec<u16>,
        leaves: Vec<DenseLeaf>,
        device: ComputeDevice,
    ) -> Result<Self, String> {
        bf16_validate(&base, &leaves)?;
        let len = base.len();
        let (device, kind) = match device {
            ComputeDevice::Cpu => (
                ParamDevice::Cpu {
                    candidate: base.clone(),
                    base,
                    leaves,
                },
                ComputeDevice::Cpu,
            ),
            ComputeDevice::Metal | ComputeDevice::Agx => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    (
                        ParamDevice::Metal(metal::Resident::new(
                            &base,
                            &leaves,
                            device == ComputeDevice::Agx,
                        )?),
                        device,
                    )
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err("Metal BF16 pytree is not available in this build".into());
                }
            }
            ComputeDevice::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    (
                        ParamDevice::OpenCl(opencl::Resident::new(&base, &leaves)?),
                        ComputeDevice::OpenCl,
                    )
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(
                        "OpenCL BF16 parameter buffers are not available in this build".into(),
                    );
                }
            }
            ComputeDevice::Cuda => {
                #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
                {
                    (
                        ParamDevice::Cuda(cuda::Resident::new(&base, &leaves)?),
                        ComputeDevice::Cuda,
                    )
                }
                #[cfg(not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64")))]
                {
                    return Err("CUDA BF16 pytree is not available in this build".into());
                }
            }
            ComputeDevice::Auto => auto_device(&base, &leaves)?,
        };
        Ok(Self { len, device, kind })
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
            device: ParamDevice::Cuda(unsafe {
                cuda::Resident::from_device(pointer, len, &leaves)?
            }),
            kind: ComputeDevice::Cuda,
        })
    }

    pub fn materialize(&mut self, terms: &[DenseTerm]) -> Result<(), String> {
        validate_terms(terms)?;
        if !has_direction(terms) {
            return Err("BF16 pytree terms cancel to zero".into());
        }
        match &mut self.device {
            ParamDevice::Cpu {
                base,
                candidate,
                leaves,
            } => {
                if let Err(error) = bf16_apply(base, candidate, leaves, terms) {
                    candidate.clone_from(base);
                    return Err(error);
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            ParamDevice::Metal(device) => device.materialize(terms)?,
            #[cfg(feature = "opencl")]
            ParamDevice::OpenCl(device) => device.materialize(terms)?,
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            ParamDevice::Cuda(device) => device.materialize(terms)?,
        }
        Ok(())
    }

    pub fn candidate(&self) -> Result<Vec<u16>, String> {
        match &self.device {
            ParamDevice::Cpu { candidate, .. } => Ok(candidate.clone()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            ParamDevice::Metal(device) => Ok(device.candidate()),
            #[cfg(feature = "opencl")]
            ParamDevice::OpenCl(device) => device.candidate(),
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            ParamDevice::Cuda(device) => device.candidate(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> ComputeDevice {
        self.kind
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn candidate_buffer(&self) -> Option<&Buffer> {
        match &self.device {
            ParamDevice::Metal(device) => Some(device.buffer()),
            #[cfg(feature = "opencl")]
            ParamDevice::OpenCl(_) => None,
            ParamDevice::Cpu { .. } => None,
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    pub fn device_ptr(&self, stream: Option<i64>) -> Result<(u64, usize, usize), String> {
        match &self.device {
            ParamDevice::Cuda(device) => device.device_ptr(stream),
            _ => Err("BF16 tree is not resident on CUDA".into()),
        }
    }
}

fn auto_device(base: &[u16], leaves: &[DenseLeaf]) -> Result<(ParamDevice, ComputeDevice), String> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Ok(device) = metal::Resident::new(base, leaves, true) {
        return Ok((ParamDevice::Metal(device), ComputeDevice::Agx));
    }
    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    if let Ok(device) = cuda::Resident::new(base, leaves) {
        return Ok((ParamDevice::Cuda(device), ComputeDevice::Cuda));
    }
    #[cfg(feature = "opencl")]
    if let Ok(device) = opencl::Resident::new(base, leaves) {
        return Ok((ParamDevice::OpenCl(device), ComputeDevice::OpenCl));
    }
    Ok((
        ParamDevice::Cpu {
            base: base.to_vec(),
            candidate: base.to_vec(),
            leaves: leaves.to_vec(),
        },
        ComputeDevice::Cpu,
    ))
}

fn bf16_validate(base: &[u16], leaves: &[DenseLeaf]) -> Result<(), String> {
    if base.is_empty() {
        return Err("BF16 pytree base cannot be empty".into());
    }
    if base.iter().any(|&value| !bf16_decode(value).is_finite()) {
        return Err("BF16 pytree base values must be finite".into());
    }
    validate_leaves(leaves, Some(base.len()))
}

fn bf16_apply(
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
            let value = bf16_decode(base[index]) + leaf.scale * sum;
            if !value.is_finite() {
                return Err("BF16 pytree perturbation overflowed FP32".into());
            }
            let candidate = bf16_encode(value);
            out[index] = if sum == 0.0 || candidate == base[index] {
                bf16_next(base[index], positive)
            } else {
                candidate
            };
        }
    }
    Ok(())
}

fn bf16_decode(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn bf16_encode(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn bf16_next(value: u16, positive: bool) -> u16 {
    if value & 0x7fff == 0 {
        return if positive { 1 } else { 0x8001 };
    }
    let grows = (value & 0x8000 == 0) == positive;
    let candidate = if grows {
        value.wrapping_add(1)
    } else {
        value.wrapping_sub(1)
    };
    if bf16_decode(candidate).is_finite() {
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

    #[test]
    fn auto_selects_device() {
        let buffer = ParamBuffer::new(
            vec![bf16_encode(1.0)],
            vec![DenseLeaf::new(7, 0, 1, 1.0).unwrap()],
            ComputeDevice::Auto,
        )
        .unwrap();
        assert_ne!(buffer.device(), ComputeDevice::Auto);
    }

    #[test]
    fn candidate_starts_at_the_base() {
        let base = vec![bf16_encode(1.0), bf16_encode(-2.0)];
        let tree = ParamBuffer::new(
            base.clone(),
            vec![DenseLeaf::new(7, 0, base.len(), 1.0).unwrap()],
            ComputeDevice::Cpu,
        )
        .unwrap();
        assert_eq!(tree.candidate().unwrap(), base);
    }

    #[test]
    fn sub_ulp_directions_still_change_every_weight() {
        let base = vec![
            bf16_encode(1.0),
            bf16_encode(-2.0),
            bf16_encode(4.0),
            bf16_encode(-8.0),
        ];
        let mut tree = ParamBuffer::new(
            base.clone(),
            vec![DenseLeaf::new(11, 0, base.len(), 1.0e-6).unwrap()],
            ComputeDevice::Cpu,
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

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_matches_cpu() {
        let base = vec![
            bf16_encode(1.0),
            bf16_encode(-2.0),
            bf16_encode(4.0),
            bf16_encode(-8.0),
        ];
        let leaves = vec![DenseLeaf::new(7, 0, base.len(), 0.125).unwrap()];
        let terms = vec![
            DenseTerm::new(11, 0.75).unwrap(),
            DenseTerm::new(29, -0.25).unwrap(),
        ];
        let mut cpu = ParamBuffer::new(base.clone(), leaves.clone(), ComputeDevice::Cpu).unwrap();
        let mut opencl = match ParamBuffer::new(base, leaves, ComputeDevice::OpenCl) {
            Ok(buffer) => buffer,
            Err(error) if error.contains("no OpenCL") => return,
            Err(error) => panic!("OpenCL BF16 setup failed: {error}"),
        };
        cpu.materialize(&terms).unwrap();
        opencl.materialize(&terms).unwrap();
        assert_eq!(opencl.candidate().unwrap(), cpu.candidate().unwrap());
    }

    #[test]
    fn next_value_stays_finite_at_the_bf16_limits() {
        assert!(bf16_decode(bf16_next(0x7f7f, true)).is_finite());
        assert!(bf16_decode(bf16_next(0xff7f, false)).is_finite());
    }

    #[test]
    fn rollback_overflow() {
        let base = vec![bf16_encode(1.0)];
        let mut tree = ParamBuffer::new(
            base.clone(),
            vec![DenseLeaf::new(7, 0, 1, f32::MAX).unwrap()],
            ComputeDevice::Cpu,
        )
        .unwrap();
        let error = tree
            .materialize(&[DenseTerm::new(11, f32::MAX).unwrap()])
            .unwrap_err();
        assert_eq!(error, "BF16 pytree perturbation overflowed FP32");
        assert_eq!(tree.candidate().unwrap(), base);
    }
}
