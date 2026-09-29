use super::*;

impl QwenEvaluator {
    pub(super) fn prefill_linear(
        &self,
        command: &CommandBufferRef,
        input: W,
        weights: &Buffer,
        offset: usize,
        output: W,
        rows: u32,
        inside: u32,
        outside: u32,
    ) -> Result<()> {
        let Some(mps) = self.mps.as_ref().filter(|_| rows >= MPS_MINROWS) else {
            self.linear(
                command, input, weights, offset, output, rows, inside, outside,
            );
            return Ok(());
        };
        let elements = u64::from(inside)
            .checked_mul(u64::from(outside))
            .ok_or("Qwen MPS matrix size overflow")?;
        let element_bytes = if mps.mode == MpsMode::F16 { 2 } else { 4 };
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or("Qwen MPS matrix byte-size overflow")?;
        if bytes > mps.weights.length() {
            return Err("Qwen MPS weight scratch is too small".into());
        }
        if mps.mode == MpsMode::F32 {
            self.encode(
                command,
                "qwen_widen",
                &[(weights, offset as u64 * 2), (&mps.weights, 0)],
                &elements,
                thread_group(elements.div_ceil(256)),
            );
            return mps.matmul.borrow_mut().encode(
                &self.runtime.device,
                command,
                MpsMatrix::new(self.buffer(input), rows, inside),
                MpsMatrix::new(&mps.weights, outside, inside),
                MpsMatrix::new(self.buffer(output), rows, outside),
                true,
                1.0,
            );
        }

        let input_elements = u64::from(rows)
            .checked_mul(u64::from(inside))
            .ok_or("Qwen MPS input size overflow")?;
        let output_elements = u64::from(rows)
            .checked_mul(u64::from(outside))
            .ok_or("Qwen MPS output size overflow")?;
        let input_scratch = mps.input.as_ref().expect("MPS half input missing");
        let output_scratch = mps.output.as_ref().expect("MPS half output missing");
        if input_elements
            .checked_mul(2)
            .is_none_or(|bytes| bytes > input_scratch.length())
            || output_elements
                .checked_mul(2)
                .is_none_or(|bytes| bytes > output_scratch.length())
        {
            return Err("Qwen MPS half scratch is too small".into());
        }
        self.encode(
            command,
            "qwen_f32_to_f16",
            &[(self.buffer(input), 0), (input_scratch, 0)],
            &input_elements,
            thread_group(input_elements.div_ceil(256)),
        );
        self.encode(
            command,
            "qwen_bf16_f16",
            &[(weights, offset as u64 * 2), (&mps.weights, 0)],
            &elements,
            thread_group(elements.div_ceil(256)),
        );
        mps.matmul.borrow_mut().encode(
            &self.runtime.device,
            command,
            MpsMatrix::half(input_scratch, rows, inside),
            MpsMatrix::half(&mps.weights, outside, inside),
            MpsMatrix::half(output_scratch, rows, outside),
            true,
            1.0,
        )?;
        self.encode(
            command,
            "qwen_f16_to_f32",
            &[(output_scratch, 0), (self.buffer(output), 0)],
            &output_elements,
            thread_group(output_elements.div_ceil(256)),
        );
        Ok(())
    }

    pub(super) fn half_gemm(
        &self,
        command: &CommandBufferRef,
        input: &Buffer,
        weights: &Buffer,
        offset: usize,
        output: &Buffer,
        rows: u32,
        inside: u32,
        outside: u32,
    ) -> Result<()> {
        let mps = self
            .mps
            .as_ref()
            .filter(|mps| mps.mode == MpsMode::F16)
            .ok_or("Qwen half GEMM requires the FP16 MPS backend")?;
        let elements = u64::from(inside)
            .checked_mul(u64::from(outside))
            .ok_or("Qwen half GEMM weight size overflow")?;
        let input_bytes = u64::from(rows)
            .checked_mul(u64::from(inside))
            .and_then(|value| value.checked_mul(2))
            .ok_or("Qwen half GEMM input size overflow")?;
        let output_bytes = u64::from(rows)
            .checked_mul(u64::from(outside))
            .and_then(|value| value.checked_mul(2))
            .ok_or("Qwen half GEMM output size overflow")?;
        if elements
            .checked_mul(2)
            .is_none_or(|bytes| bytes > mps.weights.length())
            || input_bytes > input.length()
            || output_bytes > output.length()
        {
            return Err("Qwen half GEMM scratch is too small".into());
        }
        self.encode(
            command,
            "qwen_bf16_f16",
            &[(weights, offset as u64 * 2), (&mps.weights, 0)],
            &elements,
            thread_group(elements.div_ceil(256)),
        );
        mps.matmul.borrow_mut().encode(
            &self.runtime.device,
            command,
            MpsMatrix::half(input, rows, inside),
            MpsMatrix::half(&mps.weights, outside, inside),
            MpsMatrix::half(output, rows, outside),
            true,
            1.0,
        )
    }

    pub(super) fn mlp_expand(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        rows: u32,
        layer: &Layer,
    ) -> Result<bool> {
        let Some(mps) = self
            .mps
            .as_ref()
            .filter(|mps| mps.mode == MpsMode::F16 && rows >= MPS_MINROWS)
        else {
            return Ok(false);
        };
        let c = self.config;
        let input = mps.input.as_ref().expect("MPS half input missing");
        let input_elements = u64::from(rows)
            .checked_mul(u64::from(c.hidden))
            .ok_or("Qwen MLP input size overflow")?;
        let output_elements = u64::from(rows)
            .checked_mul(u64::from(c.intermediate))
            .ok_or("Qwen MLP output size overflow")?;
        let weight_elements = usize::try_from(c.hidden)
            .ok()
            .and_then(|hidden| {
                usize::try_from(c.intermediate)
                    .ok()
                    .and_then(|width| hidden.checked_mul(width))
            })
            .ok_or("Qwen MLP weight size overflow")?;
        if layer
            .gate_weight
            .checked_add(weight_elements)
            .is_none_or(|offset| offset != layer.up_weight)
        {
            return Ok(false);
        }
        self.encode(
            command,
            "qwen_f32_to_f16",
            &[(self.buffer(W::Norm), 0), (input, 0)],
            &input_elements,
            thread_group(input_elements.div_ceil(256)),
        );
        self.half_gemm(
            command,
            input,
            weights,
            layer.gate_weight,
            self.buffer(W::Gate),
            rows,
            c.hidden,
            c.intermediate
                .checked_mul(2)
                .ok_or("Qwen MLP fused width overflow")?,
        )?;
        let shape = [output_elements, u64::from(c.intermediate)];
        self.encode(
            command,
            "qwen_silu16",
            &[(self.buffer(W::Gate), 0), (self.buffer(W::Activation), 0)],
            &shape,
            thread_group(output_elements.div_ceil(256)),
        );
        Ok(true)
    }

    pub(super) fn mlp_reduce(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        rows: u32,
        layer: &Layer,
    ) -> Result<()> {
        let mps = self
            .mps
            .as_ref()
            .filter(|mps| mps.mode == MpsMode::F16)
            .ok_or("Qwen half MLP reduction requires the FP16 MPS backend")?;
        let c = self.config;
        let output = mps.output.as_ref().expect("MPS half output missing");
        self.half_gemm(
            command,
            self.buffer(W::Activation),
            weights,
            layer.down_weight,
            output,
            rows,
            c.intermediate,
            c.hidden,
        )?;
        let elements = u64::from(rows)
            .checked_mul(u64::from(c.hidden))
            .ok_or("Qwen MLP reduction size overflow")?;
        self.encode(
            command,
            "qwen_f16_to_f32",
            &[(output, 0), (self.buffer(W::Attended), 0)],
            &elements,
            thread_group(elements.div_ceil(256)),
        );
        Ok(())
    }
}
