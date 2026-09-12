//! Narrow FP32/FP16 MPS GEMM bridge. Buffers remain owned by the Metal graph.

use std::collections::HashMap;

use metal::foreign_types::ForeignTypeRef;
use metal::objc::{
    __send_message as send,
    rc::StrongPtr,
    runtime::{BOOL, Class, NO, Object, Sel, YES},
};
use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputePipelineState, DeviceRef, MTLCommandBufferStatus,
    MTLOrigin, MTLSize,
};

use crate::apple_gpu::{Runtime, gpu_seconds};
use crate::fbt_metal::check_command;

// MPSMatrix is not wrapped by metal-rs 0.33. These selectors/types follow the
// public MPSCore/MPSMatrix SDK headers, not private GPU/compiler interfaces.
fn class(name: &str) -> Result<&'static Class, String> {
    Class::get(name).ok_or_else(|| format!("Missing MPS class {name}"))
}

unsafe fn owned(ptr: *mut Object) -> Result<StrongPtr, String> {
    if ptr.is_null() {
        Err("MPS object allocation failed".into())
    } else {
        Ok(unsafe { StrongPtr::new(ptr) })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Matrix<'a> {
    pub buffer: &'a BufferRef,
    pub rows: u32,
    pub cols: u32,
    pub batch: u32,
    /// Distance between matrices, in elements.
    pub stride: u64,
    /// Offset in elements, including submatrix row offsets.
    pub offset: u64,
    origin_row: u64,
    origin_col: u64,
    row_stride_cols: u32,
    data_type: u32,
    element_bytes: u64,
}

impl<'a> Matrix<'a> {
    pub fn new(buffer: &'a BufferRef, rows: u32, cols: u32) -> Self {
        Self {
            buffer,
            rows,
            cols,
            batch: 1,
            stride: u64::from(rows) * u64::from(cols),
            offset: 0,
            origin_row: 0,
            origin_col: 0,
            row_stride_cols: cols,
            data_type: metal::mps::MPSDataType::Float32 as u32,
            element_bytes: 4,
        }
    }

    pub fn half(buffer: &'a BufferRef, rows: u32, cols: u32) -> Self {
        Self {
            data_type: metal::mps::MPSDataType::Float16 as u32,
            element_bytes: 2,
            ..Self::new(buffer, rows, cols)
        }
    }

    pub fn layout(mut self, batch: u32, stride: u64, offset: u64) -> Self {
        self.batch = batch;
        self.stride = stride;
        self.offset = offset;
        self.origin_row = offset / u64::from(self.row_stride_cols);
        self.origin_col = offset % u64::from(self.row_stride_cols);
        self
    }

    pub fn row_view(mut self, rows: u32, offset: u64) -> Self {
        self.rows = rows;
        self.offset = offset;
        self.origin_row = offset / u64::from(self.row_stride_cols);
        self.origin_col = offset % u64::from(self.row_stride_cols);
        self
    }

    pub fn column_view(mut self, cols: u32, column: u32) -> Self {
        self.cols = cols;
        self.offset = u64::from(column);
        self.origin_row = 0;
        self.origin_col = u64::from(column);
        self
    }

    fn object(self, device: &DeviceRef) -> Result<StrongPtr, String> {
        let elements = u64::from(self.rows) * u64::from(self.cols);
        let row_stride = u64::from(self.row_stride_cols);
        let last_row = u64::from(self.rows.saturating_sub(1));
        let logical_end = self
            .origin_row
            .checked_add(last_row)
            .and_then(|row| row.checked_mul(row_stride))
            .and_then(|n| n.checked_add(self.origin_col))
            .and_then(|n| n.checked_add(u64::from(self.cols)))
            .ok_or("MPS matrix range overflow")?;
        let end_elements = if self.stride == 0 {
            logical_end
        } else {
            self.stride
                .checked_mul(u64::from(self.batch.saturating_sub(1)))
                .and_then(|n| n.checked_add(logical_end))
                .ok_or("MPS matrix range overflow")?
        };
        let end = end_elements
            .checked_mul(self.element_bytes)
            .ok_or("MPS matrix range overflow")?;
        let stride_bytes = self
            .stride
            .checked_mul(self.element_bytes)
            .ok_or("MPS matrix stride overflow")?;
        if self.rows == 0
            || self.cols == 0
            || self.batch == 0
            || (self.stride != 0 && self.stride < elements)
            || self.row_stride_cols < self.cols
            || self.origin_col + u64::from(self.cols) > row_stride
            || (self.stride != 0 && self.stride % row_stride != 0)
            || (self.batch > 1 && self.stride != 0 && self.offset + elements > self.stride)
            || end > self.buffer.length()
            || self.buffer.device().registry_id() != device.registry_id()
        {
            return Err(format!(
                "Invalid MPS matrix: rows={} cols={} batch={} stride={} elements={} offset={} end={} buf_len={}",
                self.rows,
                self.cols,
                self.batch,
                self.stride,
                elements,
                self.offset,
                end,
                self.buffer.length()
            ));
        }
        unsafe {
            // Use matrix origins for row views. The target MPS batched GEMM
            // path did not preserve nonzero buffer offsets in the sentinel test.
            let batch_div = if self.stride == 0 {
                1
            } else {
                u64::from(self.batch)
            };
            // Spare buffer capacity is not padding between batch matrices.
            let full_rows = (if self.batch > 1 {
                self.stride / row_stride
            } else {
                0
            })
            .max(self.origin_row + u64::from(self.rows));
            let physical_bytes = full_rows
                .checked_mul(row_stride)
                .and_then(|n| n.checked_mul(self.element_bytes))
                .and_then(|n| n.checked_mul(batch_div))
                .ok_or("MPS physical extent overflow")?;
            if physical_bytes > self.buffer.length() {
                return Err("MPS physical extent exceeds allocation".into());
            }
            let matrix_bytes = if self.stride == 0 {
                0
            } else {
                stride_bytes.max(full_rows * row_stride * self.element_bytes)
            };
            let desc: *mut Object = send(
                class("MPSMatrixDescriptor")?,
                Sel::register(
                    "matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:",
                ),
                (
                    full_rows,
                    row_stride,
                    self.batch as u64,
                    row_stride * self.element_bytes,
                    matrix_bytes,
                    self.data_type,
                ),
            )
            .map_err(|e| e.to_string())?;
            if desc.is_null() {
                return Err("MPS matrix descriptor failed".into());
            }
            let object: *mut Object =
                send(class("MPSMatrix")?, Sel::register("alloc"), ()).map_err(|e| e.to_string())?;
            owned(
                send(
                    object,
                    Sel::register("initWithBuffer:descriptor:"),
                    (self.buffer, desc),
                )
                .map_err(|e| e.to_string())?,
            )
        }
    }
}

type MatrixKey = (usize, u32, u32, u32, u64, u64, u64, u64, u32, u32);

#[derive(Default)]
pub(crate) struct Matmul {
    kernels: HashMap<(u32, u32, u32, bool, u64, u32), StrongPtr>,
    matrices: HashMap<MatrixKey, StrongPtr>,
}

impl Matmul {
    pub fn encode(
        &mut self,
        device: &DeviceRef,
        command: &CommandBufferRef,
        a: Matrix<'_>,
        b: Matrix<'_>,
        c: Matrix<'_>,
        transpose_b: bool,
        alpha: f64,
    ) -> Result<(), String> {
        check_command(command)?;
        let (m, k, n) = (a.rows, a.cols, c.cols);
        if c.rows != m
            || a.batch != b.batch
            || a.batch != c.batch
            || (b.rows, b.cols) != if transpose_b { (n, k) } else { (k, n) }
            || !alpha.is_finite()
            || a.data_type != b.data_type
            || a.element_bytes != b.element_bytes
            || std::ptr::eq(a.buffer, c.buffer)
            || std::ptr::eq(b.buffer, c.buffer)
        {
            return Err(format!(
                "MPS GEMM shape mismatch: a({}x{}, {}B), b({}x{}, {}B), c({}x{}, {}B), m={}, n={}, k={}, trans_b={}, a_buf={:p}, b_buf={:p}, c_buf={:p}",
                a.rows,
                a.cols,
                a.element_bytes,
                b.rows,
                b.cols,
                b.element_bytes,
                c.rows,
                c.cols,
                c.element_bytes,
                m,
                n,
                k,
                transpose_b,
                a.buffer,
                b.buffer,
                c.buffer
            ));
        }
        let get_obj = |matrices: &mut HashMap<MatrixKey, StrongPtr>,
                       mat: Matrix<'_>|
         -> Result<StrongPtr, String> {
            let key = (
                mat.buffer.as_ptr() as usize,
                mat.rows,
                mat.cols,
                mat.batch,
                mat.stride,
                mat.offset,
                mat.origin_row,
                mat.origin_col,
                mat.row_stride_cols,
                mat.data_type,
            );
            if let Some(obj) = matrices.get(&key) {
                Ok(obj.clone())
            } else {
                let obj = mat.object(device)?;
                matrices.insert(key, obj.clone());
                Ok(obj)
            }
        };
        let a_obj = get_obj(&mut self.matrices, a)?;
        let b_obj = get_obj(&mut self.matrices, b)?;
        let c_obj = get_obj(&mut self.matrices, c)?;
        let key = (m, n, k, transpose_b, alpha.to_bits(), a.data_type);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.kernels.entry(key) {
            let object = unsafe {
                let allocated: *mut Object = send(
                    class("MPSMatrixMultiplication")?,
                    Sel::register("alloc"),
                    (),
                )
                .map_err(|e| e.to_string())?;
                owned(send(allocated,
                    Sel::register("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"),
                    (device, NO as BOOL, if transpose_b { YES } else { NO },
                     m as u64, n as u64, k as u64, alpha, 0.0f64))
                    .map_err(|e| e.to_string())?)?
            };
            entry.insert(object);
        }
        let kernel = &self.kernels[&key];
        unsafe {
            for (selector, matrix) in [
                ("setLeftMatrixOrigin:", a),
                ("setRightMatrixOrigin:", b),
                ("setResultMatrixOrigin:", c),
            ] {
                let origin = MTLOrigin {
                    x: matrix.origin_row,
                    y: matrix.origin_col,
                    z: 0,
                };
                send::<Object, _, ()>(**kernel, Sel::register(selector), (origin,))
                    .map_err(|e| e.to_string())?;
            }
            send::<Object, _, ()>(**kernel, Sel::register("setBatchSize:"), (a.batch as u64,))
                .map_err(|e| e.to_string())?;
            send::<Object, _, ()>(
                **kernel,
                Sel::register("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"),
                (command, *a_obj, *b_obj, *c_obj),
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GateUpProbe {
    pub production_first_seconds: f64,
    pub candidate_first_seconds: f64,
    pub production_second_seconds: f64,
    pub candidate_second_seconds: f64,
    pub qualified: bool,
}

const GATE_UP_M: u32 = 8192;
const GATE_UP_N: u32 = 6656;
const GATE_UP_K: u32 = 1536;

struct GateUpBuffers {
    input: Buffer,
    weights: Buffer,
    gate_up: Buffer,
    production: Buffer,
    candidate: Buffer,
    count: usize,
}

fn probe_value(index: usize, exponent: u16, seed: u32) -> u16 {
    let mut hash = index as u32 ^ seed;
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x846ca68b);
    hash ^= hash >> 16;
    exponent | (hash as u16 & 0x8000) | ((hash as u16 & 7) << 7)
}

fn gate_up_buffers(runtime: &Runtime) -> GateUpBuffers {
    let input_values = (0..(GATE_UP_M * GATE_UP_K) as usize)
        .map(|index| probe_value(index, 0x3800, 17))
        .collect::<Vec<_>>();
    let weight_values = (0..(GATE_UP_K * 2 * GATE_UP_N) as usize)
        .map(|index| probe_value(index, 0x1800, 123))
        .collect::<Vec<_>>();
    let count = (GATE_UP_M * GATE_UP_N) as usize;
    GateUpBuffers {
        input: runtime.buffer_with(&input_values),
        weights: runtime.buffer_with(&weight_values),
        gate_up: runtime.buffer_with(&vec![0x7e00u16; (GATE_UP_M * 2 * GATE_UP_N) as usize]),
        production: runtime.buffer_with(&vec![0x7e00u16; count + 64]),
        candidate: runtime.buffer_with(&vec![0x7e00u16; count + 64]),
        count,
    }
}

fn submit_gate_up(
    runtime: &Runtime,
    gemm: &mut Matmul,
    buffers: &GateUpBuffers,
    pipelines: (&ComputePipelineState, &ComputePipelineState),
    candidate: bool,
) -> Result<f64, String> {
    let command = runtime.queue.new_command_buffer();
    let output = if candidate {
        &buffers.candidate
    } else {
        &buffers.production
    };
    encode_gate_up_probe(
        runtime,
        gemm,
        command,
        candidate,
        &buffers.input,
        &buffers.weights,
        &buffers.gate_up,
        output,
        pipelines.0,
        pipelines.1,
    )?;
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "gate/up probe command failed: {:?}",
            command.status()
        ));
    }
    gpu_seconds(command).ok_or_else(|| "Metal did not report gate/up GPU timing".into())
}

fn validate_gate_up(buffers: &GateUpBuffers) -> Result<(), String> {
    let read = |buffer: &Buffer| unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u16>(), buffers.count + 64)
    };
    let expected = read(&buffers.production);
    let actual = read(&buffers.candidate);
    if !expected[buffers.count..].iter().all(|&bits| bits == 0x7e00)
        || !actual[buffers.count..].iter().all(|&bits| bits == 0x7e00)
    {
        return Err("gate/up probe output-tail canary was overwritten".into());
    }
    let mut nonzero = 0usize;
    for index in 0..buffers.count {
        let (left, right) = (expected[index], actual[index]);
        if left & 0x7c00 == 0x7c00 || right & 0x7c00 == 0x7c00 {
            return Err(format!(
                "gate/up probe produced non-finite output at {index}"
            ));
        }
        if left != right && !(left & 0x7fff == 0 && right & 0x7fff == 0) {
            return Err(format!(
                "gate/up probe FP16 mismatch at {index}: {left:#06x} != {right:#06x}"
            ));
        }
        nonzero += usize::from(left & 0x7fff != 0);
    }
    if nonzero <= buffers.count / 2 {
        return Err("gate/up probe output is unexpectedly sparse".into());
    }
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure_gate_up(
    order: [bool; 4],
    submit: &mut impl FnMut(bool) -> Result<f64, String>,
) -> Result<(f64, f64), String> {
    for _ in 0..2 {
        for candidate in order {
            submit(candidate)?;
        }
    }
    let mut production = Vec::new();
    let mut candidate = Vec::new();
    for _ in 0..7 {
        let mut production_pair = Vec::with_capacity(2);
        let mut candidate_pair = Vec::with_capacity(2);
        for is_candidate in order {
            let seconds = submit(is_candidate)?;
            if is_candidate {
                candidate_pair.push(seconds);
            } else {
                production_pair.push(seconds);
            }
        }
        production.push(production_pair.iter().sum::<f64>() / 2.0);
        candidate.push(candidate_pair.iter().sum::<f64>() / 2.0);
    }
    Ok((median(&mut production), median(&mut candidate)))
}

fn encode_gate_up_probe(
    runtime: &Runtime,
    gemm: &mut Matmul,
    command: &CommandBufferRef,
    candidate: bool,
    input: &BufferRef,
    weights: &BufferRef,
    gate_up: &BufferRef,
    output: &BufferRef,
    fused: &ComputePipelineState,
    glu: &ComputePipelineState,
) -> Result<(), String> {
    if candidate {
        let parameters = [GATE_UP_M, GATE_UP_N, GATE_UP_K];
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fused);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weights), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 12, parameters.as_ptr().cast());
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(GATE_UP_N).div_ceil(64),
                height: u64::from(GATE_UP_M).div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    } else {
        gemm.encode(
            &runtime.device,
            command,
            Matrix::half(input, GATE_UP_M, GATE_UP_K),
            Matrix::half(weights, GATE_UP_K, 2 * GATE_UP_N),
            Matrix::half(gate_up, GATE_UP_M, 2 * GATE_UP_N),
            false,
            1.0,
        )?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(glu);
        encoder.set_buffer(0, Some(gate_up), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, (&GATE_UP_N as *const u32).cast());
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(GATE_UP_M),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
    Ok(())
}

/// Compare the candidate fused gate/up operation with the complete production
/// MPS gate/up plus SwiGLU operation at the exact production shape.
pub fn run_gate_up_probe(minimum_speedup_percent: f64) -> Result<GateUpProbe, String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let source = include_str!("fbt_prefill.metal");
        let fused = runtime.precise(source, "full-space BO gate/up probe", "fbt_gemm_glu_half")?;
        let glu = runtime.precise(
            source,
            "full-space BO gate/up probe",
            "fbt_prefill_glu_half_fused",
        )?;
        let buffers = gate_up_buffers(&runtime);
        let mut gemm = Matmul::default();
        let mut submit = |is_candidate: bool| -> Result<f64, String> {
            submit_gate_up(&runtime, &mut gemm, &buffers, (&fused, &glu), is_candidate)
        };
        submit(false)?;
        submit(true)?;
        validate_gate_up(&buffers)?;
        let first = measure_gate_up([false, true, false, true], &mut submit)?;
        let second = measure_gate_up([true, false, true, false], &mut submit)?;
        let ratio = 1.0 - minimum_speedup_percent / 100.0;
        let qualified = first.1 <= first.0 * ratio && second.1 <= second.0 * ratio;
        eprintln!(
            "TURBO_ENN_GATE_UP_PROBE production_first_seconds={:.6} candidate_first_seconds={:.6} production_second_seconds={:.6} candidate_second_seconds={:.6} minimum_speedup_percent={minimum_speedup_percent:.3} qualified={qualified}",
            first.0, first.1, second.0, second.1
        );
        Ok(GateUpProbe {
            production_first_seconds: first.0,
            candidate_first_seconds: first.1,
            production_second_seconds: second.0,
            candidate_second_seconds: second.1,
            qualified,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apple_gpu::Runtime;

    #[test]
    #[ignore = "exact production-shape gate/up feasibility probe"]
    fn gate_up_feasibility() {
        let result = super::run_gate_up_probe(10.0).unwrap();
        assert!(result.production_first_seconds.is_finite());
        assert!(result.candidate_first_seconds.is_finite());
        assert!(result.production_second_seconds.is_finite());
        assert!(result.candidate_second_seconds.is_finite());
    }

    #[test]
    fn batch_parity() {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared().unwrap();
            let mut gemm = Matmul::default();
            for transpose in [false, true] {
                for m in [1, 256] {
                    let (n, k, batch) = (13, 17, 4);
                    let a_stride = 257 * k;
                    let a_offset = (257 - m) * k;
                    let c_stride = 257 * n;
                    let c_offset = (257 - m) * n;
                    let av: Vec<f32> = (0..a_stride * batch)
                        .map(|i| (i % 19) as f32 * 0.03 - 0.2)
                        .collect();
                    let bv: Vec<f32> = (0..n * k * batch)
                        .map(|i| (i % 23) as f32 * 0.02 - 0.1)
                        .collect();
                    let a = runtime.buffer_with(&av);
                    let b = runtime.buffer_with(&bv);
                    let c = runtime.buffer_with(&vec![-999.0f32; (c_stride * batch) as usize]);
                    let matrix_a =
                        Matrix::new(&a, m, k).layout(batch, a_stride as u64, a_offset as u64);
                    let matrix_b = Matrix {
                        batch,
                        ..Matrix::new(
                            &b,
                            if transpose { n } else { k },
                            if transpose { k } else { n },
                        )
                    };
                    let matrix_c =
                        Matrix::new(&c, m, n).layout(batch, c_stride as u64, c_offset as u64);
                    let command = runtime.queue.new_command_buffer();
                    gemm.encode(
                        &runtime.device,
                        command,
                        matrix_a,
                        matrix_b,
                        matrix_c,
                        transpose,
                        0.5,
                    )
                    .unwrap();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                    let actual = unsafe {
                        std::slice::from_raw_parts(
                            c.contents().cast::<f32>(),
                            (c_stride * batch) as usize,
                        )
                    };
                    for sample in 0..batch {
                        for row in 0..257 - m {
                            assert_eq!(actual[(sample * c_stride + row * n) as usize], -999.0);
                        }
                        for row in 0..m {
                            for col in 0..n {
                                let expected: f64 = (0..k)
                                    .map(|j| {
                                        let ai = sample * a_stride + a_offset + row * k + j;
                                        let bi = sample * n * k
                                            + if transpose { col * k + j } else { j * n + col };
                                        f64::from(av[ai as usize])
                                            * f64::from(bv[bi as usize])
                                            * 0.5
                                    })
                                    .sum();
                                assert!(
                                    (f64::from(
                                        actual[(sample * c_stride + c_offset + row * n + col)
                                            as usize]
                                    ) - expected)
                                        .abs()
                                        < 1e-5
                                );
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    #[ignore = "exact-shape diagnostic; exercised by tools/fbt-bo --gemm"]
    fn bench_gemm_layouts() {
        use crate::apple_gpu::gpu_seconds;
        use metal::MTLSize;
        use std::time::Instant;
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared().unwrap();
            let custom_gate_up = runtime
                .precise(
                    include_str!("fbt_prefill.metal"),
                    "FBT GEMM layouts",
                    "fbt_gemm_half",
                )
                .unwrap();
            let custom_gate_up_128 = runtime
                .precise(
                    include_str!("fbt_prefill.metal"),
                    "FBT GEMM layouts",
                    "fbt_gemm_128_half",
                )
                .unwrap();
            let value = |index: usize, exponent: u16, seed: u32| {
                let mut hash = index as u32 ^ seed;
                hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
                hash = (hash ^ (hash >> 15)).wrapping_mul(0x846ca68b);
                hash ^= hash >> 16;
                exponent | (hash as u16 & 0x8000) | ((hash as u16 & 7) << 7)
            };
            let half_value = |bits: u16| -> f64 {
                let exponent = (bits >> 10) & 31;
                let fraction = bits & 1023;
                let magnitude = if exponent == 0 {
                    f64::from(fraction) * 2.0f64.powi(-24)
                } else {
                    (1.0 + f64::from(fraction) / 1024.0) * 2.0f64.powi(i32::from(exponent) - 15)
                };
                if bits & 0x8000 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            };
            for (name, m, n, k) in [
                ("gate_up", 8192u32, 13312u32, 1536u32),
                ("down", 8192, 1536, 6656),
                ("qkvg", 8192, 3088, 1536),
                ("attention_out", 8192, 1536, 1536),
                ("readout", 2048, 100352, 1536),
            ] {
                let a_values: Vec<u16> = (0..(m * k) as usize)
                    .map(|i| value(i, 0x3800, 17))
                    .collect();
                let b_values: Vec<u16> = (0..(k * n) as usize)
                    .map(|i| value(i, 0x2400, 123))
                    .collect();
                let bt_values: Vec<u16> = (0..(n * k) as usize)
                    .map(|i| value(i % k as usize * n as usize + i / k as usize, 0x2400, 123))
                    .collect();
                let a = runtime.buffer_with(&a_values);
                let b = runtime.buffer_with(&b_values);
                let bt = runtime.buffer_with(&bt_values);
                let count = (m * n) as usize;
                let variants = if name == "gate_up" {
                    4
                } else if name == "readout" {
                    3
                } else {
                    2
                };
                let outputs: Vec<_> = (0..variants)
                    .map(|_| runtime.buffer_with(&vec![0x7e00u16; count + 64]))
                    .collect();
                let split_gate;
                let split_up;
                let split_gate_weights;
                let split_up_weights;
                let split_gate_output;
                let split_up_output;
                let split_width = if name == "gate_up" { n / 2 } else { 0 };
                if name == "gate_up" {
                    split_gate = b_values
                        .chunks_exact(n as usize)
                        .flat_map(|row| row[..split_width as usize].iter().copied())
                        .collect::<Vec<_>>();
                    split_up = b_values
                        .chunks_exact(n as usize)
                        .flat_map(|row| row[split_width as usize..].iter().copied())
                        .collect::<Vec<_>>();
                    split_gate_weights = Some(runtime.buffer_with(&split_gate));
                    split_up_weights = Some(runtime.buffer_with(&split_up));
                    split_gate_output =
                        Some(
                            runtime.buffer_with(&vec![0x7e00u16; (m * split_width) as usize + 64]),
                        );
                    split_up_output =
                        Some(
                            runtime.buffer_with(&vec![0x7e00u16; (m * split_width) as usize + 64]),
                        );
                } else {
                    split_gate = Vec::new();
                    split_up = Vec::new();
                    split_gate_weights = None;
                    split_up_weights = None;
                    split_gate_output = None;
                    split_up_output = None;
                }
                let split_variant = variants;
                let mut split_times = Vec::new();
                let mut times = vec![Vec::new(); variants];
                let mut gemm = Matmul::default();
                // Identical nonzero operands, equal warmups, rotated order. Checks are untimed.
                for iteration in 0..9 {
                    let schedule = variants + usize::from(name == "gate_up");
                    for offset in 0..schedule {
                        let variant = (iteration + offset) % schedule;
                        let started = Instant::now();
                        let command = runtime.queue.new_command_buffer();
                        if name == "gate_up" && variant == split_variant {
                            gemm.encode(
                                &runtime.device,
                                command,
                                Matrix::half(&a, m, k),
                                Matrix::half(split_gate_weights.as_ref().unwrap(), k, split_width),
                                Matrix::half(split_gate_output.as_ref().unwrap(), m, split_width),
                                false,
                                1.0,
                            )
                            .unwrap();
                            gemm.encode(
                                &runtime.device,
                                command,
                                Matrix::half(&a, m, k),
                                Matrix::half(split_up_weights.as_ref().unwrap(), k, split_width),
                                Matrix::half(split_up_output.as_ref().unwrap(), m, split_width),
                                false,
                                1.0,
                            )
                            .unwrap();
                        } else if variant < 2 {
                            gemm.encode(
                                &runtime.device,
                                command,
                                Matrix::half(&a, m, k),
                                if variant == 0 {
                                    Matrix::half(&b, k, n)
                                } else {
                                    Matrix::half(&bt, n, k)
                                },
                                Matrix::half(&outputs[variant], m, n),
                                variant == 1,
                                1.0,
                            )
                            .unwrap();
                        } else if name == "readout" {
                            for column in (0..n).step_by(8192) {
                                let width = 8192.min(n - column);
                                gemm.encode(
                                    &runtime.device,
                                    command,
                                    Matrix::half(&a, m, k),
                                    Matrix::half(&bt, width, k)
                                        .row_view(width, u64::from(column) * u64::from(k)),
                                    Matrix::half(&outputs[variant], m, width)
                                        .row_view(m, u64::from(column) * u64::from(m)),
                                    true,
                                    1.0,
                                )
                                .unwrap();
                            }
                        } else {
                            let parameters = [m, n, k];
                            let encoder = command.new_compute_command_encoder();
                            encoder.set_compute_pipeline_state(if variant == 2 {
                                &custom_gate_up
                            } else {
                                &custom_gate_up_128
                            });
                            encoder.set_buffer(0, Some(&a), 0);
                            encoder.set_buffer(1, Some(&b), 0);
                            encoder.set_buffer(2, Some(&outputs[variant]), 0);
                            encoder.set_bytes(3, 12, parameters.as_ptr().cast());
                            encoder.dispatch_thread_groups(
                                MTLSize {
                                    width: u64::from(n).div_ceil(64),
                                    height: u64::from(m).div_ceil(if variant == 2 {
                                        64
                                    } else {
                                        128
                                    }),
                                    depth: 1,
                                },
                                MTLSize {
                                    width: if variant == 2 { 128 } else { 256 },
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            encoder.end_encoding();
                        }
                        command.commit();
                        let encode = started.elapsed().as_secs_f64();
                        command.wait_until_completed();
                        let wall = started.elapsed().as_secs_f64();
                        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                        if iteration >= 2 {
                            let sample = (gpu_seconds(command).unwrap(), encode, wall);
                            if name == "gate_up" && variant == split_variant {
                                split_times.push(sample);
                            } else {
                                times[variant].push(sample);
                            }
                        }
                    }
                }
                let read = |buffer: &metal::Buffer| unsafe {
                    std::slice::from_raw_parts(buffer.contents().cast::<u16>(), count + 64)
                };
                let reference = read(&outputs[0]);
                // Eight independent CPU dot products include row/column and block boundaries.
                for row in [0, m - 1] {
                    for column in [0, 8191.min(n - 1), 8192.min(n - 1), n - 1] {
                        let expected: f64 = (0..k as usize)
                            .map(|inner| {
                                half_value(a_values[row as usize * k as usize + inner])
                                    * half_value(b_values[inner * n as usize + column as usize])
                            })
                            .sum();
                        let actual = half_value(reference[(row * n + column) as usize]);
                        assert!(
                            (expected - actual).abs() <= expected.abs() / 2048.0 + 1e-6,
                            "{name} CPU reference at ({row}, {column}): {expected} vs {actual}"
                        );
                    }
                }
                for variant in 0..variants {
                    let actual = read(&outputs[variant]);
                    assert!(actual[count..].iter().all(|&x| x == 0x7e00));
                    let mut nonzero = 0usize;
                    for row in 0..m as usize {
                        for column in 0..n as usize {
                            let index = row * n as usize + column;
                            let actual_index = if name == "readout" && variant == 2 {
                                let start = column / 8192 * 8192;
                                start * m as usize + row * 8192.min(n as usize - start) + column
                                    - start
                            } else {
                                index
                            };
                            let bits = actual[actual_index];
                            assert!(bits & 0x7c00 != 0x7c00, "{name}: non-finite output");
                            assert!(
                                bits == reference[index]
                                    || (bits & 0x7fff == 0 && reference[index] & 0x7fff == 0),
                                "{name} variant={variant} ({row},{column}) mismatch"
                            );
                            nonzero += usize::from(bits & 0x7fff != 0);
                        }
                    }
                    assert!(nonzero > count / 2);
                    let mut gpu: Vec<_> = times[variant].iter().map(|x| x.0).collect();
                    let mut encode: Vec<_> = times[variant].iter().map(|x| x.1).collect();
                    let mut wall: Vec<_> = times[variant].iter().map(|x| x.2).collect();
                    gpu.sort_by(f64::total_cmp);
                    encode.sort_by(f64::total_cmp);
                    wall.sort_by(f64::total_cmp);
                    let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
                    let layout = match (name, variant) {
                        (_, 0) => "NN",
                        (_, 1) => "NT",
                        ("gate_up", 2) => "custom-64x64",
                        ("gate_up", 3) => "custom-128x64",
                        ("readout", 2) => "NT-block8192",
                        _ => unreachable!(),
                    };
                    eprintln!(
                        "FBT_GEMM name={name} layout={layout} m={m} n={n} k={k} gpu_seconds={:.6} tflops={:.3} encode_seconds={:.6} wall_seconds={:.6} samples={gpu:?}",
                        gpu[3],
                        flops / gpu[3] / 1e12,
                        encode[3],
                        wall[3]
                    );
                }
                if name == "gate_up" {
                    let split_count = (m * split_width) as usize;
                    let read_split = |buffer: &metal::Buffer| unsafe {
                        std::slice::from_raw_parts(
                            buffer.contents().cast::<u16>(),
                            split_count + 64,
                        )
                    };
                    let gate = read_split(split_gate_output.as_ref().unwrap());
                    let up = read_split(split_up_output.as_ref().unwrap());
                    assert!(
                        gate[(m * split_width) as usize..]
                            .iter()
                            .all(|&x| x == 0x7e00)
                    );
                    assert!(
                        up[(m * split_width) as usize..]
                            .iter()
                            .all(|&x| x == 0x7e00)
                    );
                    for row in [0, m - 1] {
                        for column in [0, split_width - 1] {
                            let packed_gate = reference[(row * n + column) as usize];
                            let packed_up = reference[(row * n + split_width + column) as usize];
                            let split_gate = gate[(row * split_width + column) as usize];
                            let split_up = up[(row * split_width + column) as usize];
                            assert!(
                                packed_gate == split_gate
                                    || (packed_gate & 0x7fff == 0 && split_gate & 0x7fff == 0)
                            );
                            assert!(
                                packed_up == split_up
                                    || (packed_up & 0x7fff == 0 && split_up & 0x7fff == 0)
                            );
                        }
                    }
                    let mut gpu: Vec<_> = split_times.iter().map(|x| x.0).collect();
                    let mut encode: Vec<_> = split_times.iter().map(|x| x.1).collect();
                    let mut wall: Vec<_> = split_times.iter().map(|x| x.2).collect();
                    gpu.sort_by(f64::total_cmp);
                    encode.sort_by(f64::total_cmp);
                    wall.sort_by(f64::total_cmp);
                    let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
                    eprintln!(
                        "FBT_GEMM name={name} layout=NN-split2 m={m} n={n} k={k} gpu_seconds={:.6} tflops={:.3} encode_seconds={:.6} wall_seconds={:.6} samples={gpu:?}",
                        gpu[3],
                        flops / gpu[3] / 1e12,
                        encode[3],
                        wall[3]
                    );
                }
            }
        });
    }
}
