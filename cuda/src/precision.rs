//! Device FP16 conversion checked against the independent `half` implementation.
use crate::{CudaResult, cuda_error, read_prefix};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use ennx_cuda_kernels::fbt_model;
use half::f16;

pub fn check() -> CudaResult<usize> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    // SAFETY: generated binding loads its matching CUDA module.
    let module = unsafe { fbt_model::load(&context) }.map_err(cuda_error)?;
    let mut values = (0..=u16::MAX)
        .map(|bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    // Test every positive FP16 rounding boundary and its adjacent FP32 values,
    // then reflect through zero. This includes subnormals and ties to even.
    for bits in 0..0x7bff {
        let a = f16::from_bits(bits).to_f32();
        let b = f16::from_bits(bits + 1).to_f32();
        let midpoint = ((a + b) * 0.5).to_bits();
        for bits in [midpoint - 1, midpoint, midpoint + 1] {
            values.extend([f32::from_bits(bits), -f32::from_bits(bits)]);
        }
    }
    for bits in [
        0_u32,
        1,
        0x7f7f_ffff,
        65520.0_f32.to_bits() - 1,
        65520.0_f32.to_bits(),
        65520.0_f32.to_bits() + 1,
        0x7f80_0000,
        0x7f80_0001,
        0x7fff_ffff,
    ] {
        values.extend([f32::from_bits(bits), f32::from_bits(bits | 0x8000_0000)]);
    }
    let mut bits = 17_u32;
    for _ in 0..131072 {
        bits ^= bits << 13;
        bits ^= bits >> 17;
        bits ^= bits << 5;
        values.push(f32::from_bits(bits));
    }
    let input = DeviceBuffer::from_host(&stream, &values).map_err(cuda_error)?;
    let mut narrowed = DeviceBuffer::zeroed(&stream, values.len()).map_err(cuda_error)?;
    let mut widened = DeviceBuffer::zeroed(&stream, values.len()).map_err(cuda_error)?;
    let launch = module
        .prepare_convert_check(LaunchConfig1D::new(
            values.len().div_ceil(256) as u32,
            256,
            0,
        ))
        .map_err(cuda_error)?;
    module
        .convert_check(
            &stream,
            &launch,
            &input,
            &mut narrowed,
            &mut widened,
            values.len() as u32,
        )
        .map_err(cuda_error)?;
    let narrowed = read_prefix(&narrowed, &stream, values.len())?;
    let widened = read_prefix(&widened, &stream, values.len())?;
    for (i, &value) in values.iter().enumerate() {
        let expected = if value.is_nan() {
            let bits = value.to_bits();
            ((bits >> 16) & 0x8000 | 0x7e00 | (bits & 0x7f_ffff) >> 13) as u16
        } else {
            f16::from_f32(value).to_bits()
        };
        if narrowed[i] != expected {
            return Err(format!(
                "FP16 narrowing mismatch: input={:08x} got={:04x} expected={expected:04x}",
                value.to_bits(),
                narrowed[i]
            ));
        }
        let bits = i as u16;
        let expected = if bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0 {
            (u32::from(bits & 0x8000) << 16) | 0x7fc0_0000 | (u32::from(bits & 0x03ff) << 13)
        } else {
            f16::from_bits(bits).to_f32().to_bits()
        };
        if widened[i] != expected {
            return Err(format!(
                "FP16 widening mismatch: input={bits:04x} got={:08x} expected={expected:08x}",
                widened[i]
            ));
        }
    }
    context.check_err().map_err(cuda_error)?;
    println!(
        "FP16_PARITY ok=true widening_patterns=65536 narrowing_cases={} signed_zero=true subnormal=true ties_even=true nan_payload=true",
        values.len()
    );
    Ok(values.len())
}
