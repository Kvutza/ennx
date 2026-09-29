use crate::CudaResult;
use crate::model::{FbtModel, ModelOutput};
use std::fs;
use std::path::Path;

pub fn check() -> CudaResult<()> {
    crate::precision::check()?;
    super::routing::check()?;
    let mut model = FbtModel::fixture()?;
    let tokens = vec![0; 4096];
    for (visits, value) in [(1, 0x4600), (2, 0x4800), (4, 0x4a00)] {
        let result = model.run(&tokens, visits, 0.0, 17)?;
        if result.tokens.iter().any(|&token| token != 0)
            || result.hidden.iter().any(|&v| v != 0x3c00)
            || result.stream_probe.iter().any(|&v| v != value)
        {
            return Err(format!(
                "full model fixture mismatch at {visits} recurrent visits; probe={:?}",
                &result.stream_probe[..8]
            ));
        }
        report(&result);
    }
    let sampled = model.run(&tokens, 2, 0.8, 17)?;
    let expected = model.generate(&[0; 128], 2, 0.8, 17, 1)?;
    let unrolled = model.generate(&[0; 128], 2, 0.8, 17, 2)?;
    if expected.tokens[128..] != sampled.tokens[127..4095]
        || expected.tokens != unrolled.tokens
        || expected.tokens[128..]
            .iter()
            .all(|&token| token == expected.tokens[128])
    {
        return Err(
            "accepted-prefix generation changed with unrolling or stochastic sampling collapsed"
                .into(),
        );
    }
    println!(
        "GENERATION_PARITY ok=true prompt=128 generated=3968 temperature=0.8 unroll=1,2 passes={}/{}",
        expected.passes, unrolled.passes
    );
    println!(
        "MODEL_PARITY ok=true layers=5 experts=625 shared=1 streams=4 recurrence=1,2,4 rope=true readout=true"
    );
    Ok(())
}

pub fn run(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 7 {
        return Err("usage: ./ennx cuda model CHECKPOINT TOKENS_JSON OUTPUT_DIR [CONTEXT=4096] [VISITS=2] [TEMPERATURE=0.8] [SEED=17]".into());
    }
    let tokens: Vec<u32> =
        ennx_wire::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let context = arg(args, 3, 4096_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let mut model = FbtModel::open(Path::new(&args[0]), tokens.len(), context)?;
    let result = model.run(&tokens, visits, temperature, seed)?;
    fs::write(
        output.join("tokens.json"),
        ennx_wire::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let hidden = result
        .hidden
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(output.join("hidden.f16"), hidden).map_err(|e| e.to_string())?;
    let record = ennx_wire::json::json!({ "stage": "cuda-full-model-forward", "checkpoint": args[0], "rows": tokens.len(), "context": context, "physical-layers": 5, "layer-visits": result.visits, "temperature": temperature, "seed": seed, "device-ms": result.device_ms, "wall-ms": result.wall_ms });
    fs::write(
        output.join("result.json"),
        ennx_wire::json::pretty_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    report(&result);
    Ok(())
}
pub fn generate(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 9 {
        return Err("usage: ./ennx cuda generate CHECKPOINT PROMPT_JSON OUTPUT_DIR [CONTEXT=4096] [VISITS=2] [TEMPERATURE=0.8] [SEED=17] [UNROLL=2] [CHUNK=0]".into());
    }
    let prompt: Vec<u32> =
        ennx_wire::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let context = arg(args, 3, 4096_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let unroll = arg(args, 7, 2_usize)?;
    let chunk = arg(args, 8, 0_usize)?;
    let output = Path::new(&args[2]);
    let mut model = if chunk == 0 {
        FbtModel::open(Path::new(&args[0]), context, context)?
    } else {
        FbtModel::open_streamed(Path::new(&args[0]), context, chunk)?
    };
    let (result, evaluated) = if chunk == 0 {
        let result = model.generate(&prompt, visits, temperature, seed, unroll)?;
        let evaluated = context * result.passes;
        (result, evaluated)
    } else {
        model.generate_streamed(&prompt, visits, temperature, seed, unroll, chunk)?
    };
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    fs::write(
        output.join("tokens.json"),
        ennx_wire::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let record = ennx_wire::json::json!({ "stage": "cuda-accepted-prefix-generation", "checkpoint": args[0], "context": context, "prompt-tokens": result.prompt_tokens, "drafted-tokens": 0, "evaluated-positions": evaluated, "repaired-tokens": context - result.prompt_tokens, "generated-tokens": context - result.prompt_tokens, "committed-tokens": context - result.prompt_tokens, "physical-layers": 5, "layer-visits": result.visits, "temperature": temperature, "seed": seed, "unroll": unroll, "chunk-rows": chunk, "passes": result.passes, "device-ms": result.device_ms, "wall-ms": result.wall_ms, "full-bo-loop": false });
    fs::write(
        output.join("result.json"),
        ennx_wire::json::pretty_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "GENERATE prompt={} generated={} passes={} executions={} device_ms={:.3} wall_ms={:.3}",
        result.prompt_tokens,
        context - result.prompt_tokens,
        result.passes,
        result.visits,
        result.device_ms,
        result.wall_ms
    );
    Ok(())
}

pub fn bench(args: &[String]) -> CudaResult<()> {
    if args.len() < 2 || args.len() > 10 {
        return Err("usage: generation-bench CHECKPOINT OUTPUT_DIR [CONTEXT=4096] [PROMPT=128] [VISITS=2] [TEMPERATURE=0.8] [SEED=17] [UNROLL=2] [CHUNK=0] [PROMPT_JSON]".into());
    }
    let context = arg(args, 2, 4096_usize)?;
    let prompt_tokens = arg(args, 3, 128_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let unroll = arg(args, 7, 2_usize)?;
    let chunk = arg(args, 8, 0_usize)?;
    if prompt_tokens == 0 || prompt_tokens >= context {
        return Err("generation benchmark prompt must be nonempty and shorter than context".into());
    }
    let prompt: Vec<u32> = if let Some(path) = args.get(9) {
        ennx_wire::json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        vec![0; prompt_tokens]
    };
    if prompt.len() != prompt_tokens || prompt.iter().any(|&t| t >= 8192) {
        return Err("benchmark prompt file must match the declared count and vocabulary".into());
    }
    let output = Path::new(&args[1]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    if std::env::var_os("ENNX_SYNTH_GEMM").is_some() {
        fs::write(output.join("gemm.json"), crate::gemm::measure()?).map_err(|e| e.to_string())?;
    }
    let precision_cases = crate::precision::check()?;
    let loaded_at = std::time::Instant::now();
    let mut model = if chunk == 0 {
        FbtModel::open(Path::new(&args[0]), context, context)?
    } else {
        FbtModel::open_streamed(Path::new(&args[0]), context, chunk)?
    };
    let profile = if chunk == 0 {
        Some(model.profile_wave(&vec![0; context], visits, temperature, seed)?)
    } else {
        None
    };
    let (result, evaluated) = if chunk == 0 {
        let result = model.generate(&prompt, visits, temperature, seed, unroll)?;
        let evaluated = context * result.passes;
        (result, evaluated)
    } else {
        model.generate_streamed(&prompt, visits, temperature, seed, unroll, chunk)?
    };
    let mut comparisons = Vec::new();
    let load_and_generation_ms = loaded_at.elapsed().as_secs_f64() * 1000.0;
    if context > 65536 && chunk != 0 {
        let other_chunk = if chunk == 512 { 1024 } else { 512 };
        let mut reference = FbtModel::open_streamed(Path::new(&args[0]), context, other_chunk)?;
        let (other, other_evaluated) =
            reference.generate_streamed(&prompt, visits, temperature, seed, unroll, other_chunk)?;
        if other.tokens != result.tokens || other.frontiers != result.frontiers {
            return Err("million-context generation changed with chunk size".into());
        }
        comparisons.push(ennx_wire::json::json!({
            "case": "chunk-invariance", "reference_chunk_rows": other_chunk,
            "reference_device_ms": other.device_ms,
            "reference_evaluated_positions": other_evaluated,
            "token_parity": true, "frontier_parity": true,
            "packed_reference": false
        }));
    }
    if chunk != 0 && context <= 65536 {
        let mut reference = FbtModel::open(Path::new(&args[0]), context, context)?;
        let packed = reference.generate(&prompt, visits, temperature, seed, unroll)?;
        if packed.tokens != result.tokens || packed.frontiers != result.frontiers {
            return Err("streamed generation differs from packed reference".into());
        }
        comparisons.push(ennx_wire::json::json!({
            "case": "selected-chunk", "chunk_rows": chunk,
            "packed_device_ms": packed.device_ms,
            "streamed_device_ms": result.device_ms,
            "packed_evaluated_positions": context * packed.passes,
            "streamed_evaluated_positions": evaluated,
            "token_parity": true, "frontier_parity": true
        }));
    }
    if context == 4096 && chunk == 0 {
        let prompt = vec![0; prompt_tokens];
        let varied = (0..context)
            .map(|i| ((i * 7919 + 17) % 8192) as u32)
            .collect::<Vec<_>>();
        let mut streamed = FbtModel::open(Path::new(&args[0]), context, context)?;
        for (case, input, chunk, case_seed, case_unroll) in [
            ("zero-512", prompt.as_slice(), 512, seed, unroll),
            ("zero-2048", prompt.as_slice(), 2048, seed, unroll),
            (
                "varied-512",
                &varied[..prompt_tokens],
                512,
                seed.wrapping_add(1),
                2,
            ),
        ] {
            let packed = model.generate(input, visits, temperature, case_seed, case_unroll)?;
            let (cached, evaluated) = streamed.generate_streamed(
                input,
                visits,
                temperature,
                case_seed,
                case_unroll,
                chunk,
            )?;
            if packed.tokens != cached.tokens || packed.frontiers != cached.frontiers {
                let mismatch = packed
                    .tokens
                    .iter()
                    .zip(&cached.tokens)
                    .position(|(a, b)| a != b);
                return Err(format!(
                    "streamed parity failed: case={case} first_token={mismatch:?} packed_frontiers={:?} cached_frontiers={:?}",
                    packed.frontiers, cached.frontiers
                ));
            }
            comparisons.push(ennx_wire::json::json!({
                "case": case,
                "chunk_rows": chunk,
                "seed": case_seed,
                "unroll": case_unroll,
                "packed_device_ms": packed.device_ms,
                "streamed_device_ms": cached.device_ms,
                "streamed_wall_ms": cached.wall_ms,
                "packed_evaluated_positions": context * packed.passes,
                "streamed_evaluated_positions": evaluated,
                "committed_tokens": context - input.len(),
                "token_parity": true,
                "frontier_parity": true
            }));
            println!(
                "STREAMED_PARITY case={case} chunk={chunk} evaluated={evaluated} packed_ms={:.3} streamed_ms={:.3}",
                packed.device_ms, cached.device_ms
            );
        }
    }
    let generated = context - prompt_tokens;
    let record = ennx_wire::json::json!({
        "schema": "ennx.cuda.generation.v1",
        "fp16_narrowing_cases": precision_cases,
        "fp16_widening_patterns": 65536,
        "synth_gemm_tile_k": std::env::var("ENNX_SYNTH_GEMM").ok(),
        "synth_scope": std::env::var_os("ENNX_SYNTH_GEMM").map(|_| "readout_only"),
        "streamed_comparison": comparisons,
        "chunk_rows": chunk,
        "repair_chunk_rows": if chunk == 0 { 0 } else { super::cached::repair_chunk(chunk)? },
        "logit_scratch_bytes": model.scratch.logits.len() * 2,
        "readout_chunk_rows": model.scratch.logits.len() / 8192,
        "target": "sm_75",
        "hardware": model.context.device_name().map_err(crate::cuda_error)?,
        "prompt_source": if args.get(9).is_some() { "token_file" } else { "zero_token_fixture" },
        "checkpoint": args[0],
        "context_tokens": context,
        "prompt_tokens": prompt_tokens,
        "drafted_tokens": 0,
        "evaluated_positions": evaluated,
        "repaired_tokens": generated,
        "generated_tokens": generated,
        "committed_tokens": generated,
        "repair_waves": result.passes,
        "frontiers": result.frontiers,
        "layer_visits_per_wave": result.visits,
        "temperature": temperature,
        "seed": seed,
        "unroll": unroll,
        "device_ms": result.device_ms,
        "wall_ms": result.wall_ms,
        "first_wave_ms": result.first_wave_ms,
        "checkpoint_load_and_generation_ms": load_and_generation_ms,
        "first_wave_chunk_profiles": result.chunk_profiles.iter().map(|(first, p)| ennx_wire::json::json!({
            "first_position": first, "rows": chunk,
            "embed_ms": p.embed_ms, "mhc_ms": p.mhc_ms, "qkv_ms": p.qkv_ms,
            "index_ms": p.index_ms, "pisa_ms": p.pisa_ms, "output_ms": p.output_ms,
            "moe_ms": p.moe_ms, "readout_ms": p.readout_ms, "total_ms": p.total_ms
        })).collect::<Vec<_>>(),
        "wave_profile": profile.map(|profile| ennx_wire::json::json!({
            "embed_ms": profile.embed_ms,
            "mhc_ms": profile.mhc_ms,
            "qkv_ms": profile.qkv_ms,
            "index_ms": profile.index_ms,
            "pisa_ms": profile.pisa_ms,
            "output_ms": profile.output_ms,
            "attention_ms": profile.attention_ms,
            "moe_ms": profile.moe_ms,
            "readout_ms": profile.readout_ms,
            "total_ms": profile.total_ms
        })),
        "full_generation_loop": true,
        "full_bo_loop": false
    });
    let json = ennx_wire::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    fs::write(
        output.join("tokens.json"),
        ennx_wire::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

fn arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> CudaResult<T> {
    args.get(index).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| format!("invalid argument {index}: {value}"))
    })
}
fn report(result: &ModelOutput) {
    println!(
        "MODEL rows={} executions={} device_ms={:.3} wall_ms={:.3}",
        result.tokens.len(),
        result.visits,
        result.device_ms,
        result.wall_ms
    );
}
