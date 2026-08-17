use super::{make_steps, materialize, Ask, Cpu, Engine, Leaf, SparseEdit};
use crate::weights::ComputeDevice;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use super::cuda;
#[cfg(all(target_os = "macos", feature = "metal"))]
use super::metal;
#[cfg(feature = "opencl")]
use super::opencl;

impl Engine {
    #[allow(unused_variables)]
    pub(super) fn new(
        base: &[u8],
        leaves: &[Leaf],
        slots: usize,
        device: ComputeDevice,
    ) -> Result<Self, String> {
        match device {
            ComputeDevice::Cpu => Ok(Self::Cpu(Cpu::new(base, slots))),
            ComputeDevice::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Metal(metal::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err("Metal trial search is not available in this build".to_string())
                }
            }
            ComputeDevice::Agx => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Metal(metal::Engine::new_agx(base, leaves, slots)?))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err("AGX trial search is not available in this build".to_string())
                }
            }
            ComputeDevice::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Self::OpenCl(opencl::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err("OpenCL trial search is not available in this build".to_string())
                }
            }
            ComputeDevice::Cuda => {
                #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
                {
                    Ok(Self::Cuda(cuda::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")))]
                {
                    Err("CUDA trial search is not available in this build".to_string())
                }
            }
            ComputeDevice::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    return Ok(Self::Metal(
                        metal::Engine::new_agx(base, leaves, slots)
                            .or_else(|_| metal::Engine::new(base, leaves, slots))?,
                    ));
                }
                #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
                {
                    if let Ok(engine) = cuda::Engine::new(base, leaves, slots) {
                        return Ok(Self::Cuda(engine));
                    }
                }
                #[cfg(all(feature = "opencl", not(all(target_os = "macos", feature = "metal"))))]
                {
                    return Ok(Self::OpenCl(opencl::Engine::new(base, leaves, slots)?));
                }
                #[allow(unreachable_code)]
                Ok(Self::Cpu(Cpu::new(base, slots)))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        match self {
            Self::Cpu(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_sparse(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        seeds: &[u64],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        match self {
            Self::Cpu(engine) => {
                engine.ask_sparse(base, history, trial, seeds, edits, num_pert, leaves, config)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => {
                engine.ask_sparse(base, history, trial, seeds, edits, num_pert, leaves, config)
            }
            #[allow(unreachable_patterns)]
            _ => Err("sparse resident trials currently require CPU or CUDA".to_string()),
        }
    }

    #[allow(unused_variables)]
    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Result<Vec<u8>, String> {
        match self {
            Self::Cpu(engine) => Ok(engine.read(slot).to_vec()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => Ok(engine.read(slot, row_bytes)),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.read(slot, row_bytes),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => engine.read(slot),
        }
    }

    #[allow(unused_variables)]
    pub(super) fn write(&mut self, slot: usize, row: &[u8]) -> Result<(), String> {
        match self {
            Self::Cpu(engine) => {
                engine.read_mut(slot).copy_from_slice(row);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => {
                engine.write(slot, row);
                Ok(())
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.write(slot, row),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => engine.write(slot, row),
        }
    }

    #[allow(unused_variables)]
    pub(super) fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        leaves: &[Leaf],
        length: f32,
    ) -> Result<(), String> {
        let steps = make_steps(leaves, length);
        match self {
            Self::Cpu(engine) => {
                let base = engine.read(base_slot).to_vec();
                let row = materialize(&base, leaves, &steps, seed);
                engine.read_mut(trial_slot).copy_from_slice(&row);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => engine.materialize(base_slot, trial_slot, seed, &steps),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.materialize(base_slot, trial_slot, seed, &steps),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => engine.materialize(base_slot, trial_slot, seed, &steps),
        }
    }
}
