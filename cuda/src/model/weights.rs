use crate::{CudaResult, cuda_error};
use cuda_core::{CudaStream, DeviceBuffer};
use deser::Deserialize;
use ennx_wire::json::Value;
use memmap2::MmapOptions;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

#[derive(Deserialize)]
struct Header {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(super) struct Site {
    pub predictor: DeviceBuffer<u16>,
    pub bias: DeviceBuffer<u16>,
    pub control: DeviceBuffer<u16>,
    pub norm: DeviceBuffer<u16>,
}
pub(super) struct Layer {
    pub qkv: DeviceBuffer<u16>,
    pub output: DeviceBuffer<u16>,
    pub router: DeviceBuffer<u16>,
    pub gate: DeviceBuffer<u16>,
    pub down: DeviceBuffer<u16>,
    pub attention: Site,
    pub moe: Site,
}
pub(crate) struct DiffusionWeights {
    pub mask: DeviceBuffer<u16>,
    pub index: DeviceBuffer<u16>,
}

pub(crate) struct Weights {
    pub diffusion: Option<DiffusionWeights>,
    pub embedding: DeviceBuffer<u16>,
    pub readout: DeviceBuffer<u16>,
    pub final_norm: DeviceBuffer<u16>,
    pub layers: Vec<Layer>,
}

struct Checkpoint {
    map: memmap2::Mmap,
    start: usize,
    headers: BTreeMap<String, Header>,
    diffusion: bool,
}
impl Checkpoint {
    fn open(path: &Path) -> CudaResult<Self> {
        let file = File::open(path).map_err(cuda_error)?;
        // SAFETY: read-only mapping retained for every tensor upload.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(cuda_error)?;
        if map.len() < 8 {
            return Err("truncated Safetensors prefix".into());
        }
        let len = usize::try_from(u64::from_le_bytes(map[..8].try_into().unwrap()))
            .map_err(cuda_error)?;
        let start = len.checked_add(8).ok_or("checkpoint header overflow")?;
        if start > map.len() {
            return Err("truncated Safetensors header".into());
        }
        let mut root: BTreeMap<String, Value> =
            ennx_wire::json::from_slice(&map[8..start]).map_err(cuda_error)?;
        let metadata: BTreeMap<String, String> = ennx_wire::json::from_value(
            root.remove("__metadata__")
                .ok_or("checkpoint metadata missing")?,
        )
        .map_err(cuda_error)?;
        let diffusion = match metadata.get("format").map(String::as_str) {
            Some("ennx.fbt-pisa1-looped-mhc4-rope.v1") => false,
            Some("ennx.fbt-pisa1-diffusion-mhc4-rope.v1") => true,
            _ => return Err("checkpoint must be a looped or diffusion mHC RoPE model".into()),
        };
        for (name, expected) in [
            ("position_encoding", "rope"),
            ("rope_base", "10000"),
            ("rope_dimensions", "64"),
        ] {
            if metadata.get(name).map(String::as_str) != Some(expected) {
                return Err(format!("checkpoint {name} must be {expected}"));
            }
        }
        let mut headers = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in root {
            let header: Header = ennx_wire::json::from_value(value).map_err(cuda_error)?;
            let count = header
                .shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .ok_or("tensor shape overflow")?;
            let [a, b] = header.data_offsets;
            if header.dtype != "F16"
                || a > b
                || count.checked_mul(2) != Some(b - a)
                || start.checked_add(b).is_none_or(|end| end > map.len())
            {
                return Err(format!("invalid F16 tensor {name}"));
            }
            ranges.push((a, b));
            headers.insert(name, header);
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err("overlapping checkpoint tensors".into());
        }
        Ok(Self {
            map,
            start,
            headers,
            diffusion,
        })
    }
    fn tensor(&self, name: &str, shape: &[usize], layer: Option<usize>) -> CudaResult<Vec<u16>> {
        let header = self
            .headers
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if header.shape != shape {
            return Err(format!(
                "tensor {name} shape {:?}, expected {shape:?}",
                header.shape
            ));
        }
        let [mut a, mut b] = header.data_offsets;
        if let Some(layer) = layer {
            let stride = (b - a) / 5;
            a += layer * stride;
            b = a + stride;
        }
        Ok(self.map[self.start + a..self.start + b]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }
}
impl Weights {
    pub fn open(path: &Path, stream: &CudaStream) -> CudaResult<Self> {
        let checkpoint = Checkpoint::open(path)?;
        let upload = |name: &str, shape: &[usize], layer| {
            DeviceBuffer::from_host(stream, &checkpoint.tensor(name, shape, layer)?)
                .map_err(cuda_error)
        };
        let readout = checkpoint.tensor("embedding_readout", &[512, 8192], None)?;
        let embedding =
            DeviceBuffer::from_host(stream, &crate::prefill::pack_embedding(&readout, 512, 8192))
                .map_err(cuda_error)?;
        let readout = DeviceBuffer::from_host(stream, &readout).map_err(cuda_error)?;
        let final_norm = upload("final_norm", &[512], None)?;
        let mut layers = Vec::with_capacity(5);
        for layer in 0..5 {
            let site = |attention| -> CudaResult<Site> {
                let prefix = if attention {
                    "mhc_attention"
                } else {
                    "mhc_moe"
                };
                Ok(Site {
                    predictor: upload(&format!("{prefix}_predictor"), &[5, 2048, 24], Some(layer))?,
                    bias: upload(&format!("{prefix}_bias"), &[5, 24], Some(layer))?,
                    control: upload(&format!("{prefix}_control"), &[5, 4], Some(layer))?,
                    norm: upload(
                        if attention {
                            "attention_norm"
                        } else {
                            "ffn_norm"
                        },
                        &[5, 512],
                        Some(layer),
                    )?,
                })
            };
            layers.push(Layer {
                qkv: upload("qkv", &[5, 512, 640], Some(layer))?,
                output: upload("attention_output", &[5, 512, 512], Some(layer))?,
                router: upload("router", &[5, 512, 625], Some(layer))?,
                gate: upload("expert_gate_up", &[5, 626, 512, 432], Some(layer))?,
                down: upload("expert_down", &[5, 626, 216, 512], Some(layer))?,
                attention: site(true)?,
                moe: site(false)?,
            });
        }
        let diffusion = if checkpoint.diffusion {
            Some(DiffusionWeights {
                mask: upload("mask_embed", &[512], None)?,
                index: upload("index_query", &[64, 64], None)?,
            })
        } else {
            None
        };
        Ok(Self {
            diffusion,
            embedding,
            readout,
            final_norm,
            layers,
        })
    }
}

impl Weights {
    pub(super) fn fixture(stream: &CudaStream) -> CudaResult<Self> {
        let upload =
            |value, len| DeviceBuffer::from_host(stream, &vec![value; len]).map_err(cuda_error);
        let mut layers = Vec::new();
        for _ in 0..5 {
            let site = || -> CudaResult<Site> {
                Ok(Site {
                    predictor: upload(0, 2048 * 24)?,
                    bias: upload(0, 24)?,
                    control: upload(0, 4)?,
                    norm: upload(0x3c00, 512)?,
                })
            };
            layers.push(Layer {
                qkv: upload(0x1800, 512 * 640)?,
                output: upload(0x1800, 512 * 512)?,
                router: upload(0, 512 * 625)?,
                gate: upload(0, 626 * 512 * 432)?,
                down: upload(0, 626 * 216 * 512)?,
                attention: site()?,
                moe: site()?,
            });
        }
        Ok(Self {
            diffusion: None,
            embedding: upload(0x3c00, 8192 * 512)?,
            readout: upload(0x3c00, 512 * 8192)?,
            final_norm: upload(0x3c00, 512)?,
            layers,
        })
    }
}
