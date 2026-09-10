//! Shared Apple GPU runtime used by ENNX Metal backends.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

static RUNTIME: OnceLock<Result<Arc<Runtime>, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    G13,
    G14,
    G15,
    G16,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub target: Target,
}

pub fn device_info() -> Result<DeviceInfo, String> {
    Runtime::shared().map(|runtime| runtime.info().clone())
}

pub(crate) fn thread_group(width: u64) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

pub(crate) struct Runtime {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    info: DeviceInfo,
    pipelines: Mutex<HashMap<(u64, String), ComputePipelineState>>,
    schedules: Mutex<HashMap<u64, usize>>,
}

impl Runtime {
    pub(crate) fn shared() -> Result<Arc<Self>, String> {
        RUNTIME
            .get_or_init(|| Self::new().map(Arc::new))
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no default Metal device found")?;
        let name = device.name().to_string();
        let info = DeviceInfo {
            target: target_name(&name),
            name,
        };
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            info,
            pipelines: Mutex::new(HashMap::new()),
            schedules: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub(crate) fn pipeline(
        &self,
        source: &str,
        label: &str,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        self.compile(source, label, name, true)
    }

    pub(crate) fn precise(
        &self,
        source: &str,
        label: &str,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        self.compile(source, label, name, false)
    }

    fn compile(
        &self,
        source: &str,
        label: &str,
        name: &str,
        fast: bool,
    ) -> Result<ComputePipelineState, String> {
        let key = (source_hash(source), format!("{name}:{fast}"));
        if let Some(pipeline) = self
            .pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .get(&key)
        {
            return Ok(pipeline.to_owned());
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(fast);
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(|error| format!("{label} Metal compile: {error}"))?;
        let function = library
            .get_function(name, None)
            .map_err(|error| format!("missing Metal kernel {name}: {error}"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("Metal pipeline {name}: {error}"))?;
        self.pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .insert(key, pipeline.to_owned());
        Ok(pipeline)
    }

    pub(crate) fn buffer<T>(&self, elements: usize) -> Buffer {
        self.device.new_buffer(
            (elements.max(1) * size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        )
    }

    pub(crate) fn buffer_with<T>(&self, values: &[T]) -> Buffer {
        if values.is_empty() {
            return self.buffer::<T>(1);
        }
        self.device.new_buffer_with_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        )
    }

    pub(crate) fn schedule<F>(
        &self,
        family: &str,
        shape: &[usize],
        candidates: usize,
        mut evaluate: F,
    ) -> Result<usize, String>
    where
        F: FnMut(usize) -> Result<Option<Duration>, String>,
    {
        let key = source_hash(&format!("{family}:{shape:?}"));
        if let Some(choice) = self
            .schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .get(&key)
        {
            return Ok(*choice);
        }
        let mut best = None;
        for candidate in 0..candidates {
            let Some(elapsed) = evaluate(candidate)? else {
                continue;
            };
            if best.is_none_or(|(_, current)| elapsed < current) {
                best = Some((candidate, elapsed));
            }
        }
        let choice = best
            .map(|(candidate, _)| candidate)
            .ok_or_else(|| format!("no valid Apple GPU schedule for {family} {shape:?}"))?;
        self.schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .insert(key, choice);
        Ok(choice)
    }

    pub(crate) fn race<F>(
        &self,
        family: &str,
        shape: &[usize],
        candidates: usize,
        rounds: usize,
        mut evaluate: F,
    ) -> Result<usize, String>
    where
        F: FnMut(usize) -> Result<Option<Duration>, String>,
    {
        let key = source_hash(&format!("{family}:{shape:?}"));
        if let Some(choice) = self
            .schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .get(&key)
        {
            return Ok(*choice);
        }
        if candidates == 0 || rounds == 0 {
            return Err("Apple GPU race must have candidates and rounds".to_string());
        }
        let mut times = vec![Vec::with_capacity(rounds); candidates];
        let mut valid = vec![true; candidates];
        for round in 0..rounds {
            for offset in 0..candidates {
                let candidate = (round + offset) % candidates;
                if !valid[candidate] {
                    continue;
                }
                if let Some(elapsed) = evaluate(candidate)? {
                    times[candidate].push(elapsed);
                } else {
                    valid[candidate] = false;
                    times[candidate].clear();
                }
            }
        }
        let mut best = None;
        for (candidate, elapsed) in times.iter_mut().enumerate() {
            if !valid[candidate] || elapsed.is_empty() {
                continue;
            }
            elapsed.sort_unstable();
            let median = elapsed[elapsed.len() / 2];
            if best.is_none_or(|(_, current)| median < current) {
                best = Some((candidate, median));
            }
        }
        let choice = best
            .map(|(candidate, _)| candidate)
            .ok_or_else(|| format!("no valid Apple GPU race for {family} {shape:?}"))?;
        self.schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .insert(key, choice);
        Ok(choice)
    }
}

fn source_hash(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

fn target_name(name: &str) -> Target {
    let compact: String = name
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact.starts_with("AppleM1") {
        Target::G13
    } else if compact.starts_with("AppleM2") {
        Target::G14
    } else if compact.starts_with("AppleM3") {
        Target::G15
    } else if compact.starts_with("AppleM4") {
        Target::G16
    } else {
        Target::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::{device_info, source_hash, target_name, Runtime, Target};

    fn metal_unavailable4(error: &str) -> bool {
        error.contains("no default Metal device found")
    }

    #[test]
    fn maps_generations() {
        assert_eq!(target_name("Apple M1 Max"), Target::G13);
        assert_eq!(target_name("Apple M2"), Target::G14);
        assert_eq!(target_name("Apple M3 Pro"), Target::G15);
        assert_eq!(target_name("Apple M4"), Target::G16);
        assert_eq!(target_name("Future GPU"), Target::Unknown);
    }

    #[test]
    fn shared_buffers() {
        let runtime = match Runtime::shared() {
            Ok(runtime) => runtime,
            Err(error) if metal_unavailable4(&error) => return,
            Err(error) => panic!("{error}"),
        };
        let info = match device_info() {
            Ok(info) => info,
            Err(error) if metal_unavailable4(&error) => return,
            Err(error) => panic!("{error}"),
        };
        assert_eq!(info, runtime.info().clone());
        let source = "kernel void copy_one(device uint *x [[buffer(0)]]) { x[0] = x[0]; }";
        let first = runtime.pipeline(source, "test", "copy_one").unwrap();
        let second = runtime.pipeline(source, "test", "copy_one").unwrap();
        assert_eq!(
            first.thread_execution_width(),
            second.thread_execution_width()
        );
        assert_eq!(source_hash(source), source_hash(source));
        assert!(runtime.buffer::<u32>(4).length() >= 16);
        assert!(runtime.buffer_with(&[1u32, 2, 3]).length() >= 12);
    }
}
