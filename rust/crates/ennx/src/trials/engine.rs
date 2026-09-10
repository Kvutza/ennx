use super::{make_steps, materialize, Ask, Center, Cpu, Engine, Parameter, SparseEdit};
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
        leaves: &[Parameter],
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
                    return Ok(Self::Metal(metal::Engine::new(base, leaves, slots)?));
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

    pub(super) fn device(&self) -> ComputeDevice {
        match self {
            Self::Cpu(_) => ComputeDevice::Cpu,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(_) => ComputeDevice::Metal,
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => ComputeDevice::OpenCl,
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(_) => ComputeDevice::Cuda,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        seeds: &[u64],
        leaves: &[Parameter],
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
    pub(super) fn ask_stream(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        base_seed: u64,
        count: usize,
        leaves: &[Parameter],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, u64, f32), String> {
        match self {
            Self::Cpu(engine) => engine.ask_stream(
                base,
                history,
                trial,
                base_seed,
                count,
                leaves,
                config,
                materialize_row,
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => engine.ask_stream(
                base,
                history,
                trial,
                base_seed,
                count,
                leaves,
                config,
                materialize_row,
            ),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.ask_stream(
                base,
                history,
                trial,
                base_seed,
                count,
                leaves,
                config,
                materialize_row,
            ),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(_) => Err("CUDA streamed trial seeds are not implemented".to_string()),
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
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        match self {
            Self::Cpu(engine) => {
                engine.ask_sparse(base, history, trial, seeds, edits, num_pert, leaves, config)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => {
                engine.ask_sparse(base, history, trial, seeds, edits, num_pert, leaves, config)
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => {
                engine.ask_sparse(base, history, trial, seeds, edits, num_pert, leaves, config)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => {
                engine.ask_sparse(base, history, trial, seeds, edits, num_pert, leaves, config)
            }
            #[allow(unreachable_patterns)]
            _ => Err(
                "sparse resident trials currently require CPU, Metal, OpenCL, or CUDA".to_string(),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sparse_stream(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        base_seed: u64,
        count: usize,
        num_pert: usize,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<(usize, u64, f32), String> {
        match self {
            Self::Cpu(engine) => engine.sparse_stream(
                base, history, trial, base_seed, count, num_pert, leaves, config,
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => engine.sparse_stream(
                base, history, trial, base_seed, count, num_pert, leaves, config,
            ),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.sparse_stream(
                base, history, trial, base_seed, count, num_pert, leaves, config,
            ),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(_) => Err("CUDA streamed sparse trials are not implemented".to_string()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_multi(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        regions: usize,
        candidates: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(device) => {
                device.ask_regions(base, history, regions, candidates, seeds, leaves, config)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(device) => {
                device.ask_regions(base, history, regions, candidates, seeds, leaves, config)
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(device) => {
                device.ask_regions(base, history, regions, candidates, seeds, leaves, config)
            }
            _ => {
                let mut results = Vec::with_capacity(regions);
                for region in 0..regions {
                    let start = region * candidates;
                    let end = start + candidates;
                    let (index, score) =
                        self.ask(base, history, 0, &seeds[start..end], leaves, config, false)?;
                    results.push((start + index, score));
                }
                Ok(results)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn regions_stream(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        regions: usize,
        candidates: usize,
        base_seed: u64,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(device) => device.regions_stream(
                base, history, regions, candidates, base_seed, leaves, config,
            ),
            #[cfg(feature = "opencl")]
            Self::OpenCl(device) => device.regions_stream(
                base, history, regions, candidates, base_seed, leaves, config,
            ),
            _ => {
                let seeds = super::cpu::seed_stream(base_seed, regions * candidates);
                self.ask_multi(base, history, regions, candidates, &seeds, leaves, config)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_tree(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        candidates: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        match self {
            Self::Cpu(device) => device.ask_centers(
                base,
                history,
                candidates,
                centers,
                region_centers,
                seeds,
                leaves,
                config,
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(device) => device.ask_centers(
                base,
                history,
                candidates,
                centers,
                region_centers,
                seeds,
                leaves,
                config,
            ),
            #[cfg(feature = "opencl")]
            Self::OpenCl(device) => device.ask_centers(
                base,
                history,
                candidates,
                centers,
                region_centers,
                seeds,
                leaves,
                config,
            ),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(device) => device.ask_centers(
                base,
                history,
                candidates,
                centers,
                region_centers,
                seeds,
                leaves,
                config,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn centers_stream(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        candidates: usize,
        centers: &[Center],
        region_centers: &[usize],
        base_seed: u64,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(device) => device.centers_stream(
                base,
                history,
                candidates,
                centers,
                region_centers,
                base_seed,
                leaves,
                config,
            ),
            #[cfg(feature = "opencl")]
            Self::OpenCl(device) => device.centers_stream(
                base,
                history,
                candidates,
                centers,
                region_centers,
                base_seed,
                leaves,
                config,
            ),
            _ => {
                let seeds = super::cpu::seed_stream(base_seed, region_centers.len() * candidates);
                self.ask_tree(
                    base,
                    history,
                    candidates,
                    centers,
                    region_centers,
                    &seeds,
                    leaves,
                    config,
                )
            }
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
    pub(super) fn byte_sum(&self, slot: usize, row_bytes: usize) -> Result<u64, String> {
        match self {
            Self::Cpu(engine) => Ok(engine
                .read(slot)
                .iter()
                .take(row_bytes)
                .fold(0u64, |sum, &byte| sum + u64::from(byte))),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => engine.byte_sum(slot),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.byte_sum(slot),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(engine) => Ok(engine
                .read(slot)?
                .iter()
                .take(row_bytes)
                .fold(0u64, |sum, &byte| sum + u64::from(byte))),
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
        leaves: &[Parameter],
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
