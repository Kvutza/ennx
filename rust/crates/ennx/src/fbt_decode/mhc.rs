use super::*;

impl Decoder {
    fn mhc_site(execution: u32, attention: bool) -> u32 {
        execution * 2 + u32::from(!attention)
    }

    pub(super) fn mhc_prepare(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        layer: u32,
        execution: u32,
        attention: bool,
    ) {
        let site = Self::mhc_site(execution, attention);
        let stream = (site % 2) as usize;
        let predictor_stride = u64::from(MHC_INPUT * MHC_COEFFICIENTS);
        let (predictor, bias, control, norm) = if attention {
            (
                weights.mhc_attention_predictor,
                weights.mhc_attention_bias,
                weights.mhc_attention_control,
                weights.attention_norm,
            )
        } else {
            (
                weights.mhc_moe_predictor,
                weights.mhc_moe_bias,
                weights.mhc_moe_control,
                weights.ffn_norm,
            )
        };
        let shape = [1u32, WIDTH, weights.architecture.kernel_code(), 0];
        bytes(encoder, 5, &shape);
        launch(
            encoder,
            &self.kernels.mhc_predict,
            &[
                (&self.mhc_streams[stream], 0),
                (
                    weights.buffer,
                    predictor + half_bytes(u64::from(layer) * predictor_stride),
                ),
                (
                    weights.buffer,
                    bias + half_bytes(u64::from(layer * MHC_COEFFICIENTS)),
                ),
                (weights.buffer, control + half_bytes(u64::from(layer) * 4)),
                (&self.mhc_coefficients, 0),
            ],
            thread_group(1),
            256,
        );
        bytes(encoder, 4, &shape);
        launch(
            encoder,
            &self.kernels.mhc_mix_rms,
            &[
                (&self.mhc_streams[stream], 0),
                (&self.mhc_coefficients, 0),
                (weights.buffer, norm + half_bytes(u64::from(layer * WIDTH))),
                (&self.normalized, 0),
            ],
            thread_group(1),
            128,
        );
    }

    fn mhc_update(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        layer: u32,
        execution: u32,
        attention: bool,
        branch: &BufferRef,
    ) {
        let site = Self::mhc_site(execution, attention);
        let source = (site % 2) as usize;
        let destination = 1 - source;
        let shape = [1u32, WIDTH, weights.architecture.kernel_code(), 0];
        bytes(encoder, 4, &shape);
        launch(
            encoder,
            &self.kernels.mhc_update,
            &[
                (&self.mhc_streams[source], 0),
                (branch, 0),
                (&self.mhc_coefficients, 0),
                (&self.mhc_streams[destination], 0),
            ],
            thread_group(8),
            256,
        );
    }

    pub(super) fn mhc_finalize(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
    ) {
        let shape = [1u32, WIDTH, weights.architecture.kernel_code(), 0];
        bytes(encoder, 3, &shape);
        launch(
            encoder,
            &self.kernels.mhc_mean_rms,
            &[
                (&self.mhc_streams[0], 0),
                (weights.buffer, weights.final_norm),
                (&self.normalized, 0),
            ],
            thread_group(1),
            128,
        );
    }

    pub(super) fn finish_mhc(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        layer: u32,
        execution: u32,
    ) {
        self.mhc_update(encoder, weights, layer, execution, true, &self.projected);
        self.mhc_prepare(encoder, weights, layer, execution, false);
        let (_, gate, down) = ffn_offsets(layer);
        self.gemv(
            encoder,
            (&self.normalized, 0),
            (
                &self.router,
                half_bytes(u64::from(layer * WIDTH * PADDED_EXPERTS)),
            ),
            (&self.route_scores, 0),
            WIDTH,
            PADDED_EXPERTS,
            0,
        );
        bytes(
            encoder,
            4,
            &[1u32, WIDTH, routing::ROUTED_EXPERTS, 3, 64, 1],
        );
        launch(
            encoder,
            &self.kernels.route,
            &[
                (&self.route_scores, 0),
                (&self.experts, 0),
                (&self.route_weights, 0),
                (&self.margin, 0),
            ],
            thread_group(1),
            32,
        );
        self.gemv(
            encoder,
            (&self.normalized, 0),
            (weights.buffer, weights.gate_up + gate),
            (&self.gate, 0),
            WIDTH,
            432,
            1,
        );
        bytes(
            encoder,
            2,
            &MoeShape {
                rows: 4,
                width: WIDTH,
                experts: 4,
                rows_per_expert: 1,
                expert_width: 216,
            },
        );
        launch(
            encoder,
            &self.kernels.activation,
            &[(&self.gate, 0), (&self.activation, 0)],
            thread_group(7),
            128,
        );
        self.gemv(
            encoder,
            (&self.activation, 0),
            (weights.buffer, weights.down + down),
            (&self.down, 0),
            216,
            WIDTH,
            2,
        );
        launch(
            encoder,
            &self.kernels.combine_branch,
            &[
                (&self.down, 0),
                (&self.route_weights, 0),
                (&self.residual, 0),
            ],
            thread_group(4),
            128,
        );
        self.mhc_update(encoder, weights, layer, execution, false, &self.residual);
    }
}
