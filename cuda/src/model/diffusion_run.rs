use super::*;
use std::fs;

pub fn run(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 10 {
        return Err("usage: ./ennx cuda diffusion CHECKPOINT PROMPT_JSON OUTPUT_DIR [CONTEXT=4096] [VISITS=2] [TEMPERATURE=0.8] [SEED=17] [STEPS=2] [BLOCK=128] [SOFT=true]".into());
    }
    let arg = |index: usize, default: &str| {
        args.get(index)
            .map(String::as_str)
            .unwrap_or(default)
            .to_owned()
    };
    let prompt: Vec<u32> = ennx_wire::json::from_slice(&fs::read(&args[1]).map_err(cuda_error)?)
        .map_err(cuda_error)?;
    let context = arg(3, "4096").parse::<usize>().map_err(cuda_error)?;
    let visits = arg(4, "2").parse().map_err(cuda_error)?;
    let temperature = arg(5, "0.8").parse().map_err(cuda_error)?;
    let seed = arg(6, "17").parse().map_err(cuda_error)?;
    let steps = arg(7, "2").parse().map_err(cuda_error)?;
    let block = arg(8, "128").parse().map_err(cuda_error)?;
    let soft = arg(9, "true").parse().map_err(cuda_error)?;
    let mut model = FbtModel::open(Path::new(&args[0]), context, context)?;
    let result = model.denoise(&prompt, visits, temperature, seed, steps, block, soft)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(cuda_error)?;
    fs::write(
        output.join("tokens.json"),
        ennx_wire::json::to_string(&result.tokens).map_err(cuda_error)?,
    )
    .map_err(cuda_error)?;
    let record = ennx_wire::json::json!({
        "stage": "cuda-learned-block-diffusion", "checkpoint": args[0],
        "context": context, "prompt-tokens": prompt.len(), "generated-tokens": context - prompt.len(),
        "steps": steps, "block": block, "soft": soft, "index": "independent",
        "layer-visits": result.visits, "temperature": temperature, "seed": seed,
        "evaluated-positions": steps * context, "repair-waves": 0,
        "device-ms": result.device_ms, "wall-ms": result.wall_ms,
    });
    fs::write(
        output.join("result.json"),
        ennx_wire::json::pretty_string(&record).map_err(cuda_error)?,
    )
    .map_err(cuda_error)?;
    println!(
        "DIFFUSION generated={} steps={} executions={} device_ms={:.3} wall_ms={:.3}",
        context - prompt.len(),
        steps,
        result.visits,
        result.device_ms,
        result.wall_ms
    );
    Ok(())
}

pub fn check() -> CudaResult<()> {
    let mut model = FbtModel::fixture()?;
    let prompt = vec![0; 128];
    let tokens = vec![0; 4096];
    let last_seed = super::diffusion::mix(17 ^ 2 ^ 0x6469_6666);
    let expected = model.run(&tokens, 2, 0.8, last_seed)?;
    let mut index = vec![0u16; 4096];
    for dim in 0..64 {
        index[dim * 64 + dim] = 0x3c00;
    }
    model.weights.diffusion = Some(weights::DiffusionWeights {
        mask: DeviceBuffer::from_host(&model.stream, &vec![0x3c00; 512]).map_err(cuda_error)?,
        index: DeviceBuffer::from_host(&model.stream, &index).map_err(cuda_error)?,
    });
    let result = model.denoise(&prompt, 2, 0.8, 17, 3, 128, true)?;
    if result.tokens[..128] != prompt || result.tokens[128..] != expected.tokens[128..] {
        return Err(
            "diffusion fixture changed the prompt or disagrees with uniform vocabulary sampling"
                .into(),
        );
    }
    println!(
        "DIFFUSION_PARITY ok=true generated=3968 steps=3 block=128 soft=true learned-mask=true learned-index=true repair-waves=0 device_ms={:.3}",
        result.device_ms
    );
    Ok(())
}
