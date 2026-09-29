use super::*;

impl FlameEvaluator {
    pub(super) fn forward(&self, weights: &Buffer, rows: u32) -> Result<()> {
        let c = self.config;
        let p = self.shape(rows);
        let nh = u64::from(rows) * u64::from(c.width);
        let mut command = self.runtime.queue.new_command_buffer().to_owned();
        self.embedding(&command, weights, &p, nh);
        for (i, layer) in self.layout.layers.iter().enumerate() {
            self.attention(&command, weights, layer, rows);
            self.normalized(&command, weights, layer.mlp_norm, rows);
            if i == 0 {
                self.mlp(
                    &command,
                    weights,
                    W::Norm,
                    W::Update,
                    layer.first,
                    layer.second,
                    rows,
                    c.dense_width,
                );
                self.elementwise(
                    &command,
                    "flame_residual",
                    &[(self.b(W::X), 0), (self.b(W::Update), 0)],
                    &p,
                    nh,
                );
                continue;
            }
            self.linear(
                &command,
                self.b(W::Norm),
                0,
                weights,
                layer.router,
                self.b(W::RouteLogits),
                rows,
                c.width,
                c.experts,
            );
            self.encode(
                &command,
                "flame_router",
                &[
                    (self.b(W::RouteLogits), 0),
                    (self.b(W::Probs), 0),
                    (self.b(W::Indices), 0),
                ],
                &p,
                thread_group(u64::from(rows)),
            );
            let grouped = Shape {
                sequence: self.max_tokens,
                ..p
            };
            self.elementwise(
                &command,
                "flame_group",
                &[
                    (self.b(W::Indices), 0),
                    (self.b(W::Slots), 0),
                    (self.b(W::Counts), 0),
                ],
                &grouped,
                u64::from(c.experts),
            );
            self.mlp(
                &command,
                weights,
                W::Norm,
                W::Update,
                layer.first,
                layer.second,
                rows,
                c.shared_width,
            );
            finish(&command)?;
            let counts = self.read::<u32>(W::Counts, c.experts as usize);
            if counts.iter().any(|&count| count > rows)
                || counts.iter().map(|&x| u64::from(x)).sum::<u64>()
                    != u64::from(rows) * u64::from(c.top_k)
            {
                return Err(
                    "FLAME invalid expert route counts (possibly nonfinite router logits)".into(),
                );
            }
            command = self.runtime.queue.new_command_buffer().to_owned();
            for (expert, &count) in counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let slots = (
                    self.b(W::Slots),
                    expert as u64 * u64::from(self.max_tokens) * 4,
                );
                let ep = self.shape(count);
                let elements = u64::from(count) * u64::from(c.width);
                self.elementwise(
                    &command,
                    "flame_gather",
                    &[(self.b(W::Norm), 0), slots, (self.b(W::Gathered), 0)],
                    &ep,
                    elements,
                );
                self.mlp(
                    &command,
                    weights,
                    W::Gathered,
                    W::ExpertOutput,
                    layer.expert_first + expert * 2 * c.expert_width as usize * c.width as usize,
                    layer.expert_second + expert * c.width as usize * c.expert_width as usize,
                    count,
                    c.expert_width,
                );
                self.elementwise(
                    &command,
                    "flame_scatter",
                    &[(self.b(W::ExpertOutput), 0), slots, (self.b(W::Routed), 0)],
                    &ep,
                    elements,
                );
            }
            self.elementwise(
                &command,
                "flame_combine",
                &[
                    (self.b(W::X), 0),
                    (self.b(W::Update), 0),
                    (self.b(W::Routed), 0),
                    (self.b(W::Probs), 0),
                ],
                &p,
                nh,
            );
        }
        self.normalized(&command, weights, self.layout.final_norm, rows);
        finish(&command)
    }

    fn embedding(&self, command: &CommandBufferRef, weights: &Buffer, p: &Shape, nh: u64) {
        self.elementwise(
            command,
            "flame_embedding",
            &[
                (weights, self.layout.embedding as u64 * 2),
                (self.b(W::Tokens), 0),
                (self.b(W::X), 0),
            ],
            p,
            nh,
        );
    }

    fn attention(&self, command: &CommandBufferRef, weights: &Buffer, layer: &Layer, rows: u32) {
        let c = self.config;
        let p = self.shape(rows);
        let nh = u64::from(rows) * u64::from(c.width);
        self.normalized(command, weights, layer.attention_norm, rows);
        self.linear(
            command,
            self.b(W::Norm),
            0,
            weights,
            layer.qkv,
            self.b(W::Qkv),
            rows,
            c.width,
            3 * c.width,
        );
        self.elementwise(
            command,
            "flame_rotary",
            &[
                (self.b(W::Qkv), 0),
                (self.b(W::Q), 0),
                (self.b(W::K), 0),
                (self.b(W::V), 0),
            ],
            &p,
            nh,
        );
        let d = c.width / c.heads;
        let qk = Matmul {
            m: rows,
            n: rows,
            k: d,
            transpose_b: 1,
            stride_a: u64::from(rows) * u64::from(d),
            stride_b: u64::from(rows) * u64::from(d),
            stride_c: u64::from(rows) * u64::from(rows),
        };
        self.encode(
            command,
            "flame_matmul",
            &[(self.b(W::Q), 0), (self.b(W::K), 0), (self.b(W::Scores), 0)],
            &qk,
            MTLSize {
                width: u64::from(rows).div_ceil(32),
                height: u64::from(rows).div_ceil(32),
                depth: u64::from(c.heads),
            },
        );
        self.encode(
            command,
            "flame_softmax",
            &[(self.b(W::Scores), 0)],
            &p,
            thread_group(u64::from(rows) * u64::from(c.heads)),
        );
        let av = Matmul {
            m: rows,
            n: d,
            k: rows,
            transpose_b: 0,
            stride_a: qk.stride_c,
            stride_b: qk.stride_b,
            stride_c: qk.stride_a,
        };
        self.encode(
            command,
            "flame_matmul",
            &[
                (self.b(W::Scores), 0),
                (self.b(W::V), 0),
                (self.b(W::Qkv), 0),
            ],
            &av,
            MTLSize {
                width: u64::from(d).div_ceil(32),
                height: u64::from(rows).div_ceil(32),
                depth: u64::from(c.heads),
            },
        );
        self.elementwise(
            command,
            "flame_unpack",
            &[(self.b(W::Qkv), 0), (self.b(W::Attended), 0)],
            &p,
            nh,
        );
        self.linear(
            command,
            self.b(W::Attended),
            0,
            weights,
            layer.projection,
            self.b(W::Update),
            rows,
            c.width,
            c.width,
        );
        self.elementwise(
            command,
            "flame_residual",
            &[(self.b(W::X), 0), (self.b(W::Update), 0)],
            &p,
            nh,
        );
    }
}
