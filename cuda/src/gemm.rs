//! Runtime-assembled PTX recipes, checked against independent FP16 arithmetic.
use crate::{CudaResult, cuda_error, timing_event};
use cuda_core::{
    CudaContext, CudaFunction, CudaStream, DeviceBuffer, simt::launch_kernel_on_stream,
};
use ptx_synth::{TuringGemmConfig, synthesize_turing_fp16_gemm};
use std::{ffi::c_void, sync::Arc};

pub(crate) struct Gemm {
    function: CudaFunction,
}

impl Gemm {
    pub(crate) fn from_env(context: &Arc<CudaContext>) -> CudaResult<Option<Self>> {
        match std::env::var("ENNX_SYNTH_GEMM") {
            Ok(value) => {
                let k = value
                    .parse()
                    .map_err(|_| "ENNX_SYNTH_GEMM must be 8, 16, 32, or 64")?;
                let kernel = Self::new(context, k)?;
                kernel.check(&context.default_stream())?;
                Ok(Some(kernel))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn new(context: &Arc<CudaContext>, k: u32) -> CudaResult<Self> {
        if ![8, 16, 32, 64].contains(&k) {
            return Err("PTX GEMM staging depth must be 8, 16, 32, or 64".into());
        }
        let name = format!("ennx_gemm_k{k}");
        let ptx = synthesize_turing_fp16_gemm(&TuringGemmConfig {
            name: name.clone(),
            tile_k: k,
        });
        let source = format!("/tmp/{name}-{}.ptx", std::process::id());
        let binary = format!("/tmp/{name}-{}.cubin", std::process::id());
        std::fs::write(&source, &ptx).map_err(cuda_error)?;
        let assembled = std::process::Command::new("ptxas")
            .args([
                "-arch=sm_75",
                "-O3",
                "--warn-on-spills",
                &source,
                "-o",
                &binary,
            ])
            .output()
            .map_err(cuda_error)?;
        if !assembled.status.success() {
            return Err(format!(
                "PTX GEMM assembly failed: {}",
                String::from_utf8_lossy(&assembled.stderr)
            ));
        }
        let module = context.load_module_from_file(&binary).map_err(cuda_error)?;
        let function = module.load_function(&name).map_err(cuda_error)?;
        Ok(Self { function })
    }

    pub(crate) fn launch(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaResult<()> {
        self.range(stream, a, b, c, m, n, k, 0)
    }

    pub(crate) fn range(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
        first: usize,
    ) -> CudaResult<()> {
        if m == 0
            || n == 0
            || k == 0
            || first
                .checked_add(m)
                .and_then(|v| v.checked_mul(k))
                .is_none_or(|v| v > a.len())
            || k.checked_mul(n).is_none_or(|v| v > b.len())
            || m.checked_mul(n).is_none_or(|v| v > c.len())
        {
            return Err("PTX GEMM buffer extent mismatch".into());
        }
        let mut ap = a.cu_deviceptr() + (first * k * 2) as u64;
        let mut bp = b.cu_deviceptr();
        let mut cp = c.cu_deviceptr();
        let mut m = u32::try_from(m).map_err(cuda_error)?;
        let mut n = u32::try_from(n).map_err(cuda_error)?;
        let mut k = u32::try_from(k).map_err(cuda_error)?;
        let mut args = [
            (&mut ap as *mut u64).cast::<c_void>(),
            (&mut bp as *mut u64).cast::<c_void>(),
            (&mut cp as *mut u64).cast::<c_void>(),
            (&mut m as *mut u32).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut k as *mut u32).cast::<c_void>(),
        ];
        // SAFETY: checked contiguous FP16 matrices match the generated PTX ABI;
        // one block covers 64x16 outputs, including predicated boundary tiles.
        unsafe {
            launch_kernel_on_stream(
                &self.function,
                (n.div_ceil(16), m.div_ceil(64), 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
            .map_err(cuda_error)
        }
    }

    fn check(&self, stream: &CudaStream) -> CudaResult<()> {
        for (m, n, k) in [(1, 1, 1), (63, 625, 216), (65, 33, 9), (129, 81, 65)] {
            let a = data(m * k, 17);
            let b = data(k * n, 29);
            let ad = DeviceBuffer::from_host(stream, &a).map_err(cuda_error)?;
            let bd = DeviceBuffer::from_host(stream, &b).map_err(cuda_error)?;
            let cd = DeviceBuffer::zeroed(stream, m * n).map_err(cuda_error)?;
            self.launch(stream, &ad, &bd, &cd, m, n, k)?;
            let actual = cd.to_host_vec(stream).map_err(cuda_error)?;
            for row in 0..m {
                for col in 0..n {
                    let expected = (0..k)
                        .map(|i| {
                            half::f16::from_bits(a[row * k + i]).to_f32()
                                * half::f16::from_bits(b[i * n + col]).to_f32()
                        })
                        .sum::<f32>();
                    let expected = half::f16::from_f32(expected).to_bits();
                    let observed = actual[row * n + col];
                    if half::f16::from_bits(observed).to_f32()
                        != half::f16::from_bits(expected).to_f32()
                    {
                        return Err(format!(
                            "PTX GEMM parity failed at {m}x{n}x{k} [{row},{col}]: {observed:04x} != {expected:04x}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn data(len: usize, salt: usize) -> Vec<u16> {
    (0..len)
        .map(|i| half::f16::from_f32(((i * salt + i / 7) % 31) as f32 / 16.0 - 0.9375).to_bits())
        .collect()
}

pub fn bench() -> CudaResult<()> {
    println!("{}", measure()?);
    Ok(())
}

pub(crate) fn measure() -> CudaResult<String> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    // SAFETY: generated bindings load their matching embedded module.
    let baseline = unsafe { ennx_cuda_kernels::fbt_model::load(&context) }.map_err(cuda_error)?;
    let mut records = Vec::new();
    for depth in [8, 16, 32, 64] {
        let kernel = Gemm::new(&context, depth)?;
        kernel.check(&stream)?;
        for n in [512, 625, 640, 8192] {
            let (m, k) = (4096, 512);
            let a = DeviceBuffer::from_host(&stream, &data(m * k, 17)).map_err(cuda_error)?;
            let b = DeviceBuffer::from_host(&stream, &data(k * n, 29)).map_err(cuda_error)?;
            let c = DeviceBuffer::zeroed(&stream, m * n).map_err(cuda_error)?;
            let mut reference = DeviceBuffer::zeroed(&stream, m * n).map_err(cuda_error)?;
            kernel.launch(&stream, &a, &b, &c, m, n, k)?;
            stream.synchronize().map_err(cuda_error)?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = timing_event(&stream)?;
                kernel.launch(&stream, &a, &b, &c, m, n, k)?;
                let end = timing_event(&stream)?;
                stream.synchronize().map_err(cuda_error)?;
                samples.push(start.elapsed_ms(&end).map_err(cuda_error)?);
            }
            samples.sort_by(f32::total_cmp);
            crate::model::matmul(None, &baseline, &stream, &a, &b, &mut reference, m, n, k)?;
            stream.synchronize().map_err(cuda_error)?;
            let mut control = Vec::new();
            for _ in 0..5 {
                let start = timing_event(&stream)?;
                crate::model::matmul(None, &baseline, &stream, &a, &b, &mut reference, m, n, k)?;
                let end = timing_event(&stream)?;
                stream.synchronize().map_err(cuda_error)?;
                control.push(start.elapsed_ms(&end).map_err(cuda_error)?);
            }
            control.sort_by(f32::total_cmp);
            let actual = c.to_host_vec(&stream).map_err(cuda_error)?;
            let expected = reference.to_host_vec(&stream).map_err(cuda_error)?;
            let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            if mismatches != 0 {
                return Err(format!(
                    "PTX GEMM differs from CUDA-Oxide on {m}x{n}x{k}: {mismatches} elements"
                ));
            }
            records.push(ennx_wire::json::json!({"tile_k":depth,"m":m,"n":n,"k":k,"samples_ms":samples,"median_ms":samples[2],"baseline_ms":control[2],"speedup":control[2]/samples[2],"cpu_edge_parity":true,"baseline_mismatches":mismatches}));
        }
    }
    ennx_wire::json::pretty_string(&ennx_wire::json::json!({"schema":"ennx.ptx.gemm.v1","hardware":context.device_name().map_err(cuda_error)?,"kernels":records,"full_generation_loop":false})).map_err(cuda_error)
}
