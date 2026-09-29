use super::*;

pub(super) fn validate(runtime: &Runtime, pisa: &Pisa1, qkv: &BufferRef) -> Result<f64, String> {
    let qkv_values = half_values(qkv, (ROWS * QKV_WIDTH) as usize);
    let rows = [0, 63, 64, 511, 2048, 4095, 4096, 8191];
    let mut maximum = 0.0f64;
    for row in rows {
        let command = runtime.queue.new_command_buffer();
        pisa.exact_range(command, qkv, row, 1);
        complete(command)?;
        let actual = half_values(&pisa.exact_output, (ROWS * QUERY_WIDTH) as usize);
        let sequence = row / CONTEXT;
        let position = row % CONTEXT;
        for head in 0..QUERY_HEADS {
            let mut logits = Vec::with_capacity(position as usize + 1);
            for token in 0..=position {
                let source = sequence * CONTEXT + token;
                let mut score = 0.0f32;
                for dim in 0..HEAD_DIM {
                    score +=
                        decode_half(qkv_values[(row * QKV_WIDTH + head * HEAD_DIM + dim) as usize])
                            * decode_half(
                                qkv_values[(source * QKV_WIDTH + QUERY_WIDTH + dim) as usize],
                            );
                }
                logits.push(score * 0.125);
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = logits
                .iter()
                .map(|&score| (score - max).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            for dim in 0..HEAD_DIM {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(token, &weight)| {
                        let source = sequence * CONTEXT + token as u32;
                        weight
                            * decode_half(
                                qkv_values
                                    [(source * QKV_WIDTH + QUERY_WIDTH + HEAD_DIM + dim) as usize],
                            )
                    })
                    .sum::<f32>()
                    / total;
                let index = (row * QUERY_WIDTH + head * HEAD_DIM + dim) as usize;
                let error = f64::from((decode_half(actual[index]) - expected).abs());
                if !error.is_finite() {
                    return Err(format!(
                        "PISA exact fallback produced non-finite output at {index}"
                    ));
                }
                maximum = maximum.max(error);
            }
        }
    }
    if maximum > 0.002 {
        return Err(format!(
            "PISA exact fallback max sampled absolute error {maximum:.9} exceeds 0.002"
        ));
    }
    eprintln!("TURBO_ENN_PISA1_EXACT_FALLBACK rows={rows:?} max_abs_error={maximum:.9}");
    Ok(maximum)
}
