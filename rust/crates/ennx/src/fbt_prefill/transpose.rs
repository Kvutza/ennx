use super::prefill::*;

impl Model {
    pub fn ensure_transposed(&mut self) -> Result<(), String> {
        if self.transposed_revision == self.revision && !self.transposed_weights.is_empty() {
            return Ok(());
        }
        self.allocate_transposed()?;
        let command = self.runtime.queue.new_command_buffer().to_owned();
        let pipe = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_transpose_half")?;
        for (idx, p) in self.parameters.iter().enumerate() {
            if let Some(out) = &self.transposed_weights[idx] {
                let (n, k) = (p.shape[0], p.shape[1]);
                let shape = [n, k];
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipe);
                encoder.set_buffer(0, Some(&p.buffer), 0);
                encoder.set_buffer(1, Some(out), 0);
                encoder.set_bytes(2, 8, shape.as_ptr().cast());
                let grid = MTLSize {
                    width: u64::from(k).div_ceil(16),
                    height: u64::from(n).div_ceil(16),
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, tg);
                encoder.end_encoding();
            }
        }
        if let Some([b0, b1]) = &self.transposed_feedback {
            let p = &self.parameters[self.feedback_weights];
            let width = self.config.width;
            let shape = [width, width];
            let matrix_bytes = u64::from(width) * u64::from(width) * 2;
            for (out_buf, offset) in [(b0, 0), (b1, matrix_bytes)] {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipe);
                encoder.set_buffer(0, Some(&p.buffer), offset);
                encoder.set_buffer(1, Some(out_buf), 0);
                encoder.set_bytes(2, 8, shape.as_ptr().cast());
                let grid = MTLSize {
                    width: u64::from(width).div_ceil(16),
                    height: u64::from(width).div_ceil(16),
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, tg);
                encoder.end_encoding();
            }
        }
        self.enqueue_packed(&command)?;
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("Failed to transpose weights".into());
        }
        self.transposed_revision = self.revision;
        Ok(())
    }

    pub(super) fn trace_transposed(
        &self,
        trace: &mut TraceRecorder,
    ) -> Result<(usize, bool), String> {
        if self.transposed_revision == self.revision && !self.transposed_weights.is_empty() {
            trace.host(
                "weight_layout_reuse",
                format!("revision={}", self.revision),
                || {},
            );
            return Ok((1, false));
        }
        if self.transposed_weights.len() != self.parameters.len()
            || self.transposed_feedback.is_none()
            || self.transposed_qkvg.len() != self.config.layers as usize
            || self.transposed_gate_up.len() != self.config.layers as usize
        {
            return Err("FBT traced weight layouts were not preallocated".into());
        }

        let start = trace.operations.len();
        let transpose = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_transpose_half")?;
        for (index, parameter) in self.parameters.iter().enumerate() {
            if let Some(output) = &self.transposed_weights[index] {
                let (n, k) = (parameter.shape[0], parameter.shape[1]);
                let shape = [n, k];
                trace.gpu(
                    self,
                    "metal",
                    0,
                    None,
                    "weight_transpose",
                    format!("parameter={} n={n} k={k}", parameter.name),
                    |command| {
                        let encoder = command.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&transpose);
                        encoder.set_buffer(0, Some(&parameter.buffer), 0);
                        encoder.set_buffer(1, Some(output), 0);
                        encoder.set_bytes(2, 8, shape.as_ptr().cast());
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: u64::from(k).div_ceil(16),
                                height: u64::from(n).div_ceil(16),
                                depth: 1,
                            },
                            MTLSize {
                                width: 16,
                                height: 16,
                                depth: 1,
                            },
                        );
                        encoder.end_encoding();
                        Ok(())
                    },
                )?;
            }
        }

        self.trace_feedback(trace, &transpose)?;
        self.trace_qkvg(trace)?;
        self.trace_gateup(trace)?;
        Ok((trace.operations.len() - start, true))
    }

    fn allocate_transposed(&mut self) -> Result<(), String> {
        if self.transposed_weights.is_empty() {
            self.transposed_weights = (0..self.parameters.len()).map(|_| None).collect();
            for (idx, p) in self.parameters.iter().enumerate() {
                if p.shape.len() == 2 {
                    let is_qkvg = self
                        .layers
                        .iter()
                        .any(|l| l.q == idx || l.k == idx || l.v == idx || l.head_gate == idx);
                    let is_gate_up = self.layers.iter().any(|l| l.gate == idx || l.up == idx);
                    if is_qkvg && self.config.width == 1536 {
                        continue;
                    }
                    if is_gate_up && (self.config.width == 1536 && self.config.intermediate == 6656)
                    {
                        continue;
                    }
                    let n = p.shape[0];
                    let k = p.shape[1];
                    let elements = u64::from(n) * u64::from(k);
                    let buf = allocate::<u16>(&self.runtime, elements)?;
                    self.transposed_weights[idx] = Some(buf);
                }
            }
        }
        if self.transposed_feedback.is_none() {
            let width = self.config.width;
            let elements = u64::from(width) * u64::from(width);
            let b0 = allocate::<u16>(&self.runtime, elements)?;
            let b1 = allocate::<u16>(&self.runtime, elements)?;
            self.transposed_feedback = Some([b0, b1]);
        }
        Ok(())
    }

    fn enqueue_packed(&mut self, command: &CommandBufferRef) -> Result<(), String> {
        if self.config.width == 1536 {
            if self.transposed_qkvg.is_empty() {
                let width = self.config.width;
                let elements = u64::from(width) * 3088;
                self.transposed_qkvg = (0..self.config.layers)
                    .map(|_| allocate::<u16>(&self.runtime, elements).ok())
                    .collect();
            }
            let pack_pipe = self
                .runtime
                .precise(SOURCE, "FBT full prefill", "fbt_pack_qkvg")?;
            for (idx, l) in self.layers.iter().enumerate() {
                if let Some(out) = &self.transposed_qkvg[idx] {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_pipe);
                    encoder.set_buffer(0, Some(&self.parameters[l.q].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[l.k].buffer), 0);
                    encoder.set_buffer(2, Some(&self.parameters[l.v].buffer), 0);
                    encoder.set_buffer(3, Some(&self.parameters[l.head_gate].buffer), 0);
                    encoder.set_buffer(4, Some(out), 0);
                    encoder.set_bytes(5, 4, (&self.config.width as *const u32).cast());
                    let grid = MTLSize {
                        width: 3088,
                        height: u64::from(self.config.width),
                        depth: 1,
                    };
                    let tg = MTLSize {
                        width: 32,
                        height: 8,
                        depth: 1,
                    };
                    encoder.dispatch_threads(grid, tg);
                    encoder.end_encoding();
                }
            }
        }
        if self.config.width == 1536 && self.config.intermediate == 6656 {
            if self.transposed_gate_up.is_empty() {
                let width = self.config.width;
                let elements = u64::from(width) * 13312;
                self.transposed_gate_up = (0..self.config.layers)
                    .map(|_| allocate::<u16>(&self.runtime, elements).ok())
                    .collect();
            }
            let pack_pipe = self
                .runtime
                .precise(SOURCE, "FBT full prefill", "fbt_pack_gate_up")?;
            for (idx, l) in self.layers.iter().enumerate() {
                if let Some(out) = &self.transposed_gate_up[idx] {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_pipe);
                    encoder.set_buffer(0, Some(&self.parameters[l.gate].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[l.up].buffer), 0);
                    encoder.set_buffer(2, Some(out), 0);
                    encoder.set_bytes(3, 4, (&self.config.width as *const u32).cast());
                    let grid = MTLSize {
                        width: 13312,
                        height: u64::from(self.config.width),
                        depth: 1,
                    };
                    let tg = MTLSize {
                        width: 32,
                        height: 8,
                        depth: 1,
                    };
                    encoder.dispatch_threads(grid, tg);
                    encoder.end_encoding();
                }
            }
        }
        Ok(())
    }

    fn trace_feedback(
        &self,
        trace: &mut TraceRecorder,
        transpose: &metal::ComputePipelineStateRef,
    ) -> Result<(), String> {
        let feedback = self.transposed_feedback.as_ref().unwrap();
        let parameter = &self.parameters[self.feedback_weights];
        let width = self.config.width;
        let shape = [width, width];
        let matrix_bytes = u64::from(width) * u64::from(width) * 2;
        for (matrix, (output, offset)) in [
            (feedback[0].as_ref(), 0),
            (feedback[1].as_ref(), matrix_bytes),
        ]
        .into_iter()
        .enumerate()
        {
            trace.gpu(
                self,
                "metal",
                0,
                None,
                "feedback_weight_transpose",
                format!("matrix={matrix} n={width} k={width}"),
                |command| {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&transpose);
                    encoder.set_buffer(0, Some(&parameter.buffer), offset);
                    encoder.set_buffer(1, Some(output), 0);
                    encoder.set_bytes(2, 8, shape.as_ptr().cast());
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: u64::from(width).div_ceil(16),
                            height: u64::from(width).div_ceil(16),
                            depth: 1,
                        },
                        MTLSize {
                            width: 16,
                            height: 16,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    Ok(())
                },
            )?;
        }

        Ok(())
    }

    fn trace_qkvg(&self, trace: &mut TraceRecorder) -> Result<(), String> {
        let pack_qkvg = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_pack_qkvg")?;
        for (index, layer) in self.layers.iter().enumerate() {
            let output = self.transposed_qkvg[index]
                .as_ref()
                .ok_or_else(|| format!("Missing traced QKVG layout for layer {index}"))?;
            trace.gpu(
                self,
                "metal",
                0,
                Some(index as u32),
                "qkvg_weight_pack",
                format!("width={} packed=3088", self.config.width),
                |command| {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_qkvg);
                    encoder.set_buffer(0, Some(&self.parameters[layer.q].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[layer.k].buffer), 0);
                    encoder.set_buffer(2, Some(&self.parameters[layer.v].buffer), 0);
                    encoder.set_buffer(3, Some(&self.parameters[layer.head_gate].buffer), 0);
                    encoder.set_buffer(4, Some(output), 0);
                    encoder.set_bytes(5, 4, (&self.config.width as *const u32).cast());
                    encoder.dispatch_threads(
                        MTLSize {
                            width: 3088,
                            height: u64::from(self.config.width),
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 8,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    Ok(())
                },
            )?;
        }

        Ok(())
    }

    fn trace_gateup(&self, trace: &mut TraceRecorder) -> Result<(), String> {
        let pack_gate_up = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_pack_gate_up")?;
        for (index, layer) in self.layers.iter().enumerate() {
            let output = self.transposed_gate_up[index]
                .as_ref()
                .ok_or_else(|| format!("Missing traced gate/up layout for layer {index}"))?;
            trace.gpu(
                self,
                "metal",
                0,
                Some(index as u32),
                "gate_up_weight_pack",
                format!("width={} packed=13312", self.config.width),
                |command| {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_gate_up);
                    encoder.set_buffer(0, Some(&self.parameters[layer.gate].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[layer.up].buffer), 0);
                    encoder.set_buffer(2, Some(output), 0);
                    encoder.set_bytes(3, 4, (&self.config.width as *const u32).cast());
                    encoder.dispatch_threads(
                        MTLSize {
                            width: 13312,
                            height: u64::from(self.config.width),
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 8,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    Ok(())
                },
            )?;
        }
        Ok(())
    }
}
