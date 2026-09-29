use super::*;

pub(super) fn decode_half(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    let fraction = f64::from(bits & 1023);
    let magnitude = match exponent {
        0 => fraction * 2.0f64.powi(-24),
        31 => f64::NAN,
        _ => (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25),
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

pub(super) fn compare_half(
    reference: &BufferRef,
    candidate: &BufferRef,
    elements: usize,
    label: &str,
) -> Result<f64, String> {
    let read = |buffer: &BufferRef| unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u16>(), elements + 64)
    };
    let reference = read(reference);
    let candidate = read(candidate);
    if !reference[elements..].iter().all(|&bits| bits == 0x7e00)
        || !candidate[elements..].iter().all(|&bits| bits == 0x7e00)
    {
        return Err(format!("{label} overwrote its output canary"));
    }
    let mut maximum = 0.0f64;
    let mut maximum_index = 0usize;
    let mut maximum_reference = 0.0f64;
    let mut maximum_candidate = 0.0f64;
    for index in 0..elements {
        let left = decode_half(reference[index]);
        let right = decode_half(candidate[index]);
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("{label} produced a non-finite output at {index}"));
        }
        let error = (left - right).abs();
        if error > maximum {
            maximum = error;
            maximum_index = index;
            maximum_reference = left;
            maximum_candidate = right;
        }
    }
    eprintln!(
        "TURBO_ENN_MOE_PARITY operation={label} max_abs_error={maximum:.9} index={maximum_index} reference={maximum_reference:.9} candidate={maximum_candidate:.9}"
    );
    Ok(maximum)
}

pub(super) fn matmul_element(
    input: &BufferRef,
    weights: &BufferRef,
    row: u32,
    column: u32,
    input_width: u32,
    output_width: u32,
) -> f64 {
    let expert = row / EXPERT_ROWS;
    let input = unsafe {
        std::slice::from_raw_parts(
            input.contents().cast::<u16>(),
            (ROWS * input_width) as usize,
        )
    };
    let weights = unsafe {
        std::slice::from_raw_parts(
            weights.contents().cast::<u16>(),
            (EXPERTS * input_width * output_width) as usize,
        )
    };
    let mut sum = 0.0f32;
    for k in 0..input_width {
        let left = decode_half(input[(row * input_width + k) as usize]) as f32;
        let weight = ((expert * input_width + k) * output_width + column) as usize;
        sum += left * decode_half(weights[weight]) as f32;
    }
    f64::from(sum)
}

pub(super) fn matmul_dense(
    input: &BufferRef,
    weights: &BufferRef,
    row: u32,
    column: u32,
    input_width: u32,
    output_width: u32,
) -> f64 {
    let input = unsafe {
        std::slice::from_raw_parts(
            input.contents().cast::<u16>(),
            (ROWS * input_width) as usize,
        )
    };
    let weights = unsafe {
        std::slice::from_raw_parts(
            weights.contents().cast::<u16>(),
            (input_width * output_width) as usize,
        )
    };
    let mut sum = 0.0f32;
    for k in 0..input_width {
        let left = decode_half(input[(row * input_width + k) as usize]) as f32;
        sum += left * decode_half(weights[(k * output_width + column) as usize]) as f32;
    }
    f64::from(sum)
}

pub(super) fn buffer_element(buffer: &BufferRef, row: u32, column: u32, width: u32) -> f64 {
    let values = unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u16>(), (ROWS * width) as usize)
    };
    decode_half(values[(row * width + column) as usize])
}

pub(super) fn validate_tail(buffers: &Buffers) -> Result<f64, String> {
    let state = buffer_element(&buffers.feedback_state, 0, 0, WIDTH);
    let gate = buffer_element(&buffers.feedback_gate, 0, 0, WIDTH);
    let feedback = buffer_element(&buffers.feedback, 0, 0, WIDTH);
    let expected_feedback = state / (1.0 + (-gate).exp());
    let feedback_error = (feedback - expected_feedback).abs();

    let logits = unsafe {
        std::slice::from_raw_parts(buffers.logits()?.contents().cast::<u16>(), VOCAB as usize)
    };
    let maximum = logits
        .iter()
        .map(|&bits| decode_half(bits))
        .fold(f64::NEG_INFINITY, f64::max);
    let total = logits
        .iter()
        .map(|&bits| (decode_half(bits) - maximum).exp())
        .sum::<f64>();
    let labels = unsafe {
        std::slice::from_raw_parts(buffers.labels.contents().cast::<u32>(), ROWS as usize)
    };
    let expected_loss = total.ln() + maximum - decode_half(logits[labels[0] as usize]);
    let losses = unsafe {
        std::slice::from_raw_parts(buffers.losses.contents().cast::<f32>(), ROWS as usize + 64)
    };
    if losses[..ROWS as usize]
        .iter()
        .any(|value| !value.is_finite())
        || losses[ROWS as usize..].iter().any(|value| !value.is_nan())
    {
        return Err(
            "feedback/readout tail produced non-finite loss or overwrote its canary".into(),
        );
    }
    let loss_error = (f64::from(losses[0]) - expected_loss).abs();
    let maximum_error = feedback_error.max(loss_error);
    if feedback_error > 0.002 || loss_error > 0.002 {
        return Err(format!(
            "feedback/readout parity failed: feedback={feedback_error:.9}, loss={loss_error:.9}"
        ));
    }
    eprintln!(
        "TURBO_ENN_TAIL_PARITY feedback_max_abs_error={feedback_error:.9} loss_max_abs_error={loss_error:.9}"
    );
    Ok(maximum_error)
}

pub(super) fn materialized_layer(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<f64, String> {
    let command = runtime.queue.new_command_buffer();
    encode(pipelines, buffers, command, Some(Stage::Gate));
    encode(pipelines, buffers, command, Some(Stage::Group));
    encode_tensorops(
        command,
        &tensorops.gate_up,
        &[
            &buffers.grouped,
            &buffers.materialized_gate_up,
            &buffers.gate_up,
        ],
        GATE_UP,
    )?;
    encode(pipelines, buffers, command, Some(Stage::Swiglu));
    encode_tensorops(
        command,
        &tensorops.down,
        &[
            &buffers.activation,
            &buffers.materialized_down,
            &buffers.down,
        ],
        WIDTH,
    )?;
    encode(pipelines, buffers, command, Some(Stage::Ungroup));
    complete(command)
}
