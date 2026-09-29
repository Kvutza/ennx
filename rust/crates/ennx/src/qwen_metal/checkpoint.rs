use super::*;

#[derive(Clone, Copy)]
pub(super) struct QwenConfig {
    pub(super) layers: u32,
    pub(super) hidden: u32,
    pub(super) intermediate: u32,
    pub(super) heads: u32,
    pub(super) kv_heads: u32,
    pub(super) vocab: u32,
    pub(super) eos_token_id: u32,
    pub(super) context: u32,
    pub(super) epsilon: f32,
    pub(super) rope_theta: f32,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            layers: 28,
            hidden: 1536,
            intermediate: 8960,
            heads: 12,
            kv_heads: 2,
            vocab: 151_936,
            eos_token_id: 151_643,
            context: 32_768,
            epsilon: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

impl QwenConfig {
    pub(super) fn validate(self, max_tokens: u32) -> Result<()> {
        if self.layers == 0
            || self.hidden == 0
            || self.intermediate == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.vocab == 0
            || self.context == 0
            || self.eos_token_id >= self.vocab
            || self.hidden % self.heads != 0
            || self.heads % self.kv_heads != 0
            || (self.hidden / self.heads) % 2 != 0
            || max_tokens == 0
            || max_tokens > self.context
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.epsilon >= 1.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
        {
            return Err("Invalid Qwen dimensions, heads, or normalization".into());
        }
        Ok(())
    }

    pub(super) fn head_dim(self) -> u32 {
        self.hidden / self.heads
    }

    pub(super) fn kv_width(self) -> u32 {
        self.kv_heads * self.head_dim()
    }
}

#[derive(Default, Clone, Copy)]
pub(super) struct Layer {
    pub(super) input_norm: usize,
    pub(super) post_norm: usize,
    pub(super) q_weight: usize,
    pub(super) q_bias: usize,
    pub(super) k_weight: usize,
    pub(super) k_bias: usize,
    pub(super) v_weight: usize,
    pub(super) v_bias: usize,
    pub(super) o_weight: usize,
    pub(super) gate_weight: usize,
    pub(super) up_weight: usize,
    pub(super) down_weight: usize,
}

pub(super) struct Layout {
    pub(super) layers: Vec<Layer>,
    pub(super) embedding: usize,
    pub(super) final_norm: usize,
    pub(super) tensors: BTreeMap<String, (usize, usize)>,
    pub(super) len: usize,
}

impl Layout {
    pub(super) fn new(c: QwenConfig) -> Result<Self> {
        let mut tensors = BTreeMap::<String, (usize, usize)>::new();
        let mut add = |name: String, dims: &[usize]| -> Result<()> {
            let length = dims.iter().try_fold(1usize, |value, &dim| {
                value
                    .checked_mul(dim)
                    .ok_or_else(|| "Qwen layout shape overflow".to_string())
            })?;
            tensors.insert(name, (0, length));
            Ok(())
        };
        let h = c.hidden as usize;
        let kv = c.kv_width() as usize;
        let intermediate = c.intermediate as usize;
        add("model.embed_tokens.weight".into(), &[c.vocab as usize, h])?;
        add("model.norm.weight".into(), &[h])?;
        for layer in 0..c.layers {
            let p = format!("model.layers.{layer}.");
            add(p.clone() + "input_layernorm.weight", &[h])?;
            add(p.clone() + "post_attention_layernorm.weight", &[h])?;
            add(p.clone() + "self_attn.q_proj.weight", &[h, h])?;
            add(p.clone() + "self_attn.q_proj.bias", &[h])?;
            add(p.clone() + "self_attn.k_proj.weight", &[kv, h])?;
            add(p.clone() + "self_attn.k_proj.bias", &[kv])?;
            add(p.clone() + "self_attn.v_proj.weight", &[kv, h])?;
            add(p.clone() + "self_attn.v_proj.bias", &[kv])?;
            add(p.clone() + "self_attn.o_proj.weight", &[h, h])?;
            add(p.clone() + "mlp.gate_proj.weight", &[intermediate, h])?;
            add(p.clone() + "mlp.up_proj.weight", &[intermediate, h])?;
            add(p + "mlp.down_proj.weight", &[h, intermediate])?;
        }
        let mut len = 0usize;
        for (offset, length) in tensors.values_mut() {
            *offset = len;
            len = len
                .checked_add(*length)
                .ok_or("Qwen layout offset overflow")?;
        }
        len.checked_mul(2).ok_or("Qwen weight byte-size overflow")?;
        let mut layers = Vec::with_capacity(c.layers as usize);
        for layer in 0..c.layers {
            let p = format!("model.layers.{layer}.");
            let get = |name: &str| tensors[&(p.clone() + name)].0;
            layers.push(Layer {
                input_norm: get("input_layernorm.weight"),
                post_norm: get("post_attention_layernorm.weight"),
                q_weight: get("self_attn.q_proj.weight"),
                q_bias: get("self_attn.q_proj.bias"),
                k_weight: get("self_attn.k_proj.weight"),
                k_bias: get("self_attn.k_proj.bias"),
                v_weight: get("self_attn.v_proj.weight"),
                v_bias: get("self_attn.v_proj.bias"),
                o_weight: get("self_attn.o_proj.weight"),
                gate_weight: get("mlp.gate_proj.weight"),
                up_weight: get("mlp.up_proj.weight"),
                down_weight: get("mlp.down_proj.weight"),
            });
        }
        Ok(Self {
            layers,
            embedding: tensors["model.embed_tokens.weight"].0,
            final_norm: tensors["model.norm.weight"].0,
            tensors,
            len,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct QwenBlock {
    pub(super) key: u64,
    pub(super) offset: usize,
    pub(super) length: usize,
    pub(super) scale: f32,
    pub(super) weight: f32,
}

#[derive(Deserialize)]
pub(super) struct SafeTensorRecord {
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    pub(super) data_offsets: [usize; 2],
}

pub(super) fn bf16_stats(bytes: &[u8], name: &str) -> Result<(f32, bool)> {
    let mut peak = 0.0f64;
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7f80 == 0x7f80 {
            return Err(format!(
                "Qwen tensor {name} contains a non-finite BF16 value"
            ));
        }
        peak = peak.max(f64::from(f32::from_bits(u32::from(bits) << 16).abs()));
    }
    if peak == 0.0 {
        return Ok((0.0, false));
    }
    let mut normalized_sum = 0.0f64;
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let value = f64::from(f32::from_bits(u32::from(bits) << 16)) / peak;
        normalized_sum += value * value;
    }
    let rms = peak * (normalized_sum / (bytes.len() / 2) as f64).sqrt();
    if !rms.is_finite() || rms <= 0.0 || rms > f32::MAX as f64 {
        return Err(format!("Qwen tensor {name} has an invalid FP32 RMS"));
    }
    Ok((rms as f32, true))
}
