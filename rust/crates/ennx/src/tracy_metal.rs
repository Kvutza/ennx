//! Metal GPU zones backed by stage-boundary timestamp counters.

use std::ops::Deref;
use std::sync::atomic::{AtomicI64, AtomicU16, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePassDescriptor, CounterSampleBuffer,
    CounterSampleBufferDescriptor, Device, MTLCounterSamplingPoint, MTLResourceOptions, NSRange,
};
use tracy_client_sys as sys;

const CONTEXT: u8 = 254;
const METAL: u8 = 6;
const MAX_SAMPLES: usize = 4096;

static INIT: OnceLock<Result<(), String>> = OnceLock::new();
static IDS: AtomicU16 = AtomicU16::new(0);
static LAST: AtomicI64 = AtomicI64::new(0);

pub(crate) struct Pool {
    counter: CounterSampleBuffer,
    next: usize,
}

pub(crate) struct Batch {
    counter: CounterSampleBuffer,
    output: Buffer,
    spans: Vec<(u16, u16, &'static str)>,
    start: usize,
    samples: usize,
}

pub(crate) struct Encoder<'a> {
    inner: &'a ComputeCommandEncoderRef,
    end: u16,
}

impl Pool {
    pub(crate) fn new(device: &Device) -> Result<Self, String> {
        setup(device)?;
        if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err("Metal stage-boundary counters are unavailable".to_string());
        }
        Ok(Self {
            counter: counter(device)?,
            next: 0,
        })
    }

    pub(crate) fn batch(&mut self, device: &Device, passes: usize) -> Result<Batch, String> {
        let samples = passes
            .checked_mul(2)
            .filter(|&count| count > 0 && count <= MAX_SAMPLES)
            .ok_or_else(|| format!("Tracy Metal pass count exceeds {}", MAX_SAMPLES / 2))?;
        if self.next + samples > MAX_SAMPLES {
            self.counter = counter(device)?;
            self.next = 0;
        }
        let start = self.next;
        self.next += samples;
        let output = device.new_buffer(
            (samples * size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Batch {
            counter: self.counter.to_owned(),
            output,
            spans: Vec::with_capacity(passes),
            start,
            samples,
        })
    }
}

impl Batch {
    pub(crate) fn encoder<'a>(
        &mut self,
        command: &'a CommandBufferRef,
        name: &'static str,
    ) -> Result<Encoder<'a>, String> {
        let offset = self.spans.len() * 2;
        if offset + 1 >= self.samples {
            return Err("Tracy Metal pass count was underestimated".to_string());
        }
        let sample = self.start + offset;
        let desc = ComputePassDescriptor::new();
        let attachment = desc
            .sample_buffer_attachments()
            .object_at(0)
            .ok_or("Metal counter attachment is unavailable")?;
        attachment.set_sample_buffer(&self.counter);
        attachment.set_start_of_encoder_sample_index(sample as u64);
        attachment.set_end_of_encoder_sample_index((sample + 1) as u64);
        let (start, end) = begin(name);
        self.spans.push((start, end, name));
        Ok(Encoder {
            inner: command.compute_command_encoder_with_descriptor(desc),
            end,
        })
    }

    pub(crate) fn resolve(&self, command: &CommandBufferRef) {
        let encoder = command.new_blit_command_encoder();
        encoder.resolve_counters(
            &self.counter,
            NSRange::new(self.start as u64, self.samples as u64),
            &self.output,
            0,
        );
        encoder.end_encoding();
    }

    pub(crate) fn upload(&self) -> Result<(), String> {
        let values = unsafe {
            std::slice::from_raw_parts(self.output.contents().cast::<u64>(), self.samples)
        };
        let mut last = LAST.load(Ordering::Relaxed);
        for (sample, &(start, end, name)) in self.spans.iter().enumerate() {
            let (begin, finish) = times(values[sample * 2], values[sample * 2 + 1], last)
                .map_err(|error| format!("{name}: {error}"))?;
            last = finish;
            unsafe {
                sys::___tracy_emit_gpu_time_serial(sys::___tracy_gpu_time_data {
                    gpuTime: begin,
                    queryId: start,
                    context: CONTEXT,
                });
                sys::___tracy_emit_gpu_time_serial(sys::___tracy_gpu_time_data {
                    gpuTime: finish,
                    queryId: end,
                    context: CONTEXT,
                });
            }
        }
        LAST.fetch_max(last, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn duration(&self) -> Result<Duration, String> {
        let values = unsafe {
            std::slice::from_raw_parts(self.output.contents().cast::<u64>(), self.samples)
        };
        let mut first = None;
        let mut last = None;
        for sample in 0..self.spans.len() {
            let begin = values[sample * 2];
            let end = values[sample * 2 + 1];
            if begin == 0 || end == 0 {
                continue;
            }
            let (Ok(begin), Ok(end)) = (timestamp(begin), timestamp(end)) else {
                continue;
            };
            if end < begin {
                continue;
            }
            first.get_or_insert(begin);
            last = Some(end);
        }
        let Some(first) = first else {
            return Ok(Duration::ZERO);
        };
        let Some(last) = last else {
            return Ok(Duration::ZERO);
        };
        if last < first {
            return Ok(Duration::ZERO);
        }
        Ok(Duration::from_nanos((last - first) as u64))
    }

    pub(crate) fn stages(&self) -> Result<Vec<(&'static str, Duration)>, String> {
        let values = unsafe {
            std::slice::from_raw_parts(self.output.contents().cast::<u64>(), self.samples)
        };
        self.spans
            .iter()
            .enumerate()
            .map(|(sample, &(_, _, name))| {
                let begin = values[sample * 2];
                let end = values[sample * 2 + 1];
                if begin == 0 || end == 0 {
                    return Ok((name, Duration::ZERO));
                }
                let duration = match (timestamp(begin), timestamp(end)) {
                    (Ok(begin), Ok(end)) if end >= begin => {
                        Duration::from_nanos((end - begin) as u64)
                    }
                    _ => Duration::ZERO,
                };
                Ok((name, duration))
            })
            .collect()
    }
}

impl Deref for Encoder<'_> {
    type Target = ComputeCommandEncoderRef;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl Drop for Encoder<'_> {
    fn drop(&mut self) {
        self.inner.end_encoding();
        unsafe {
            sys::___tracy_emit_gpu_zone_end_serial(sys::___tracy_gpu_zone_end_data {
                queryId: self.end,
                context: CONTEXT,
            });
        }
    }
}

fn setup(device: &Device) -> Result<(), String> {
    INIT.get_or_init(|| {
        let _ = crate::tracy::client();
        let mut cpu = 0;
        let mut gpu = 0;
        device.sample_timestamps(&mut cpu, &mut gpu);
        let gpu = timestamp(gpu.max(1))?;
        LAST.store(gpu, Ordering::Relaxed);
        let name = b"ENNX Metal";
        unsafe {
            sys::___tracy_emit_gpu_new_context_serial(sys::___tracy_gpu_new_context_data {
                gpuTime: gpu,
                period: 1.0,
                context: CONTEXT,
                flags: 0,
                type_: METAL,
            });
            sys::___tracy_emit_gpu_context_name_serial(sys::___tracy_gpu_context_name_data {
                context: CONTEXT,
                name: name.as_ptr().cast(),
                len: name.len() as u16,
            });
        }
        Ok(())
    })
    .clone()
}

fn begin(name: &'static str) -> (u16, u16) {
    let start = IDS.fetch_add(2, Ordering::Relaxed);
    let end = start.wrapping_add(1);
    let function = b"metal";
    let file = file!().as_bytes();
    let srcloc = unsafe {
        sys::___tracy_alloc_srcloc_name(
            line!(),
            file.as_ptr().cast(),
            file.len(),
            function.as_ptr().cast(),
            function.len(),
            name.as_ptr().cast(),
            name.len(),
            0,
        )
    };
    unsafe {
        sys::___tracy_emit_gpu_zone_begin_alloc_serial(sys::___tracy_gpu_zone_begin_data {
            srcloc,
            queryId: start,
            context: CONTEXT,
        });
    }
    (start, end)
}

fn timestamp(value: u64) -> Result<i64, String> {
    if value == 0 || value > i64::MAX as u64 {
        return Err(format!("invalid Metal timestamp {value}"));
    }
    Ok(value as i64)
}

fn counter(device: &Device) -> Result<CounterSampleBuffer, String> {
    let desc = CounterSampleBufferDescriptor::new();
    desc.set_storage_mode(metal::MTLStorageMode::Shared);
    desc.set_sample_count(MAX_SAMPLES as u64);
    let counters = device.counter_sets();
    let timestamps = counters
        .iter()
        .find(|set| set.name() == "timestamp")
        .ok_or("Metal timestamp counter is unavailable")?;
    desc.set_counter_set(timestamps);
    device.new_counter_sample_buffer_with_descriptor(&desc)
}

fn times(begin: u64, end: u64, last: i64) -> Result<(i64, i64), String> {
    if let (Ok(begin), Ok(end)) = (timestamp(begin), timestamp(end)) {
        if end >= begin {
            return Ok((begin, end));
        }
    }
    if last == 0 {
        return Err("unresolved Metal timestamp".to_string());
    }
    Ok((last + 5, last + 10))
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpu() {
        let device = metal::Device::system_default().unwrap();
        super::setup(&device).unwrap();
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let mut pool = super::Pool::new(&device).unwrap();
        let mut batch: super::Batch = pool.batch(&device, 1).unwrap();
        let encoder: super::Encoder<'_> = batch.encoder(command, "tracy.test").unwrap();
        let _: &metal::ComputeCommandEncoderRef = std::ops::Deref::deref(&encoder);
        drop(encoder);
        batch.resolve(command);
        command.commit();
        command.wait_until_completed();
        assert_eq!(batch.stages().unwrap().len(), 1);
        batch.upload().unwrap();
    }

    #[test]
    fn time() {
        assert_eq!(super::timestamp(7).unwrap(), 7);
        assert!(super::timestamp(0).is_err());
        assert_eq!(super::times(0, 0, 10).unwrap(), (15, 20));
        assert_eq!(super::times(9, 7, 10).unwrap(), (15, 20));
        assert!(super::times(0, 0, 0).is_err());
    }
}
