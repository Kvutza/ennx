use super::prefill::*;

impl Prefill {
    pub(super) fn new(model: &Model, batch: u32, length: u32) -> Result<Self, String> {
        let c = model.config;
        let Allocation {
            rows,
            d,
            kv,
            n,
            f,
            weights,
            attention,
            logits,
        } = Allocation::checked(model, batch, length)?;
        let alloc = |elements| allocate::<f32>(&model.runtime, elements);
        let pipe = |name| model.runtime.precise(SOURCE, "FBT full prefill", name);
        Ok(Self {
            batch,
            length,
            rows,
            matmul: RefCell::default(),
            feedback: RefCell::new(Feedback::new(
                FeedbackConfig {
                    width: c.width,
                    token_norm: c.feedback_token_norm,
                    fused_norm: c.feedback_fused_norm,
                },
                rows,
            )?),
            widen: pipe("fbt_widen_weights")?,
            prepare: pipe("fbt_prefill_prepare")?,
            softmax: pipe("fbt_prefill_softmax")?,
            unpack: pipe("fbt_prefill_unpack")?,
            flash_attention: pipe("fbt_prefill_flash_attention")?,
            shift: pipe("fbt_prefill_shift")?,
            glu: pipe("fbt_prefill_glu")?,
            weights: alloc(weights)?,
            tokens: allocate::<u32>(&model.runtime, n)?,
            targets: allocate::<u32>(&model.runtime, n)?,
            mask: allocate::<u32>(&model.runtime, n)?,
            losses: alloc(n)?,
            x: alloc(n * d)?,
            embed: alloc(n * d)?,
            normalized: alloc(n * d)?,
            branch: alloc(n * d)?,
            previous: alloc(n * d)?,
            history: alloc(n * d)?,
            qkv: float_qkv(model, n, d, kv)?,
            head_major: head_major(model, n * d)?,
            gates: if model.optimized {
                alloc(1)?
            } else {
                alloc(n * u64::from(c.heads))?
            },
            ff_gate: if model.optimized {
                alloc(1)?
            } else {
                alloc(n * f)?
            },
            ff_up: if model.optimized {
                alloc(1)?
            } else {
                alloc(n * f)?
            },
            scores: alloc(attention)?,
            logits: alloc(logits)?,
            rms_half: pipe("fbt_prefill_rms_half")?,
            residual_half: pipe("fbt_prefill_residual_half")?,
            glu_half: pipe("fbt_prefill_glu_half")?,
            prepare_half: pipe("fbt_prefill_prepare_half")?,
            flash_attention_half: pipe("fbt_prefill_flash_attention_half")?,
            flash_attention_half_96: pipe("fbt_prefill_flash_attention_half_96")?,
            normalized_half: allocate::<u16>(&model.runtime, n * d)?,
            embed_half: allocate::<u16>(&model.runtime, n * d)?,
            branch_half: allocate::<u16>(&model.runtime, n * d)?,
            qkv_half: half_qkv(&model.runtime, n * d, n * kv)?,
            gates_half: allocate::<u16>(&model.runtime, n * u64::from(c.heads))?,
            ff_gate_half: allocate::<u16>(&model.runtime, n * f)?,
            ff_up_half: allocate::<u16>(&model.runtime, n * f)?,
            cross_entropy_blocked_half: pipe("fbt_cross_entropy_blocked_half")?,
            logits_half: allocate::<u16>(
                &model.runtime,
                u64::from(2048u32.min(rows)) * u64::from(c.vocab),
            )?,
            unit_rms_half: pipe("fbt_prefill_unit_rms_half")?,
            feedback_combine_norm: pipe("fbt_feedback_combine_norm")?,
            qkvg_half: allocate::<u16>(&model.runtime, n * 3088)?,
            prepare_half_fused: pipe("fbt_prefill_prepare_half_fused")?,
            ff_gate_up_half: allocate::<u16>(&model.runtime, n * 2 * f)?,
            glu_half_fused: pipe("fbt_prefill_glu_half_fused")?,
            history_half: allocate::<u16>(&model.runtime, n * d)?,
            shift_half: pipe("fbt_prefill_shift_half")?,
            glu_gemm: pipe("fbt_gemm_glu_half")?,
        })
    }
}

struct Allocation {
    rows: u32,
    d: u64,
    kv: u64,
    n: u64,
    f: u64,
    weights: u64,
    attention: u64,
    logits: u64,
}
impl Allocation {
    fn checked(model: &Model, batch: u32, length: u32) -> Result<Self, String> {
        let c = model.config;
        let supported_half_attention =
            c.width / c.heads == 128 || (c.width == 1536 && c.heads == 16 && c.kv_heads == 8);
        if model.optimized && !supported_half_attention {
            return Err(
                "Optimized prefill requires 128-wide heads or the 1536-wide 16Q/8KV configuration"
                    .into(),
            );
        }
        if batch == 0 || length == 0 || length > c.capacity {
            return Err("Invalid FBT prefill batch/length".into());
        }
        let rows = batch
            .checked_mul(length)
            .ok_or("FBT prefill row overflow")?;
        let heads = batch
            .checked_mul(c.heads)
            .ok_or("FBT prefill head overflow")?;
        rows.checked_mul(c.width.max(c.intermediate))
            .ok_or("FBT prefill coordinate overflow")?;
        let d = u64::from(c.width);
        let kv = u64::from(c.kv_heads) * (d / u64::from(c.heads));
        let n = u64::from(rows);
        let f = u64::from(c.intermediate);
        let weights = if model.optimized {
            1
        } else {
            d * u64::from(c.vocab).max(f).max(d)
        };
        let attention = if model.optimized && (c.width / c.heads == 128 || c.width / c.heads == 96)
        {
            1
        } else {
            u64::from(heads) * u64::from(length.min(QUERY_BLOCK)) * u64::from(length)
        };
        let partials_elements = u64::from(rows) * u64::from(c.vocab).div_ceil(32) * 4;
        let logits = if model.optimized {
            partials_elements
        } else {
            u64::from(c.chunk.min(rows)) * u64::from(c.vocab)
        };
        // Six graph states, raw QKV, four head-major states, two FFN states;
        // Feedback additionally owns a row scale and one full-width scratch.
        let total = if model.optimized {
            4 * (n * (11 * d + 5) + logits)
        } else {
            4 * (n * (12 * d + 2 * kv + u64::from(c.heads) + 2 * f + 5)
                + weights
                + attention
                + logits)
        };
        let largest = [weights, attention, logits, n * d, n * f, n * kv]
            .into_iter()
            .max()
            .unwrap()
            * 4;
        check_memory(&model.runtime, total, largest)?;
        if !metal::mps::mps_supports_device(&model.runtime.device) {
            return Err("MPS does not support this Metal device".into());
        }
        Ok(Self {
            rows,
            d,
            kv,
            n,
            f,
            weights,
            attention,
            logits,
        })
    }
}

fn float_qkv(model: &Model, n: u64, d: u64, kv: u64) -> Result<[Buffer; 3], String> {
    let sizes = if model.optimized {
        [1; 3]
    } else {
        [n * d, n * kv, n * kv]
    };
    Ok([
        allocate::<f32>(&model.runtime, sizes[0])?,
        allocate::<f32>(&model.runtime, sizes[1])?,
        allocate::<f32>(&model.runtime, sizes[2])?,
    ])
}
fn head_major(model: &Model, elements: u64) -> Result<[Buffer; 4], String> {
    Ok([
        allocate::<u16>(&model.runtime, elements)?,
        allocate::<u16>(&model.runtime, elements)?,
        allocate::<u16>(&model.runtime, elements)?,
        allocate::<f32>(&model.runtime, if model.optimized { 1 } else { elements })?,
    ])
}
fn half_qkv(runtime: &Runtime, query: u64, kv: u64) -> Result<[Buffer; 3], String> {
    Ok([
        allocate::<u16>(runtime, query)?,
        allocate::<u16>(runtime, kv)?,
        allocate::<u16>(runtime, kv)?,
    ])
}
