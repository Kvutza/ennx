use super::*;

impl SearchState {
    pub(super) fn realized_history(&self) -> bool {
        self.exact_history && !self.latent_history
    }

    /// Latent independent proposals are ranked from their procedural norm and
    /// the incumbent-to-history distances. They do not need four realized
    /// model-sized rows before selection; only the selected row must be
    /// materialized and measured exactly.
    pub(super) fn analytic_pool(&self) -> bool {
        self.latent_history && self.threshold.is_none()
    }

    pub(super) fn latent_norm(&self, candidate: usize) -> f32 {
        let radius = f64::from(self.radius(candidate));
        self.blocks
            .iter()
            .map(|block| {
                let step = f64::from(block.scale) * radius;
                block.len as f64 * f64::from(block.weight) * step * step
            })
            .sum::<f64>() as f32
    }

    pub(super) fn latent_components(&self, candidate: usize) -> [f32; FAMILIES] {
        let mut norms = [0.0f64; FAMILIES];
        let Some(family) = &self.family else {
            return [0.0; FAMILIES];
        };
        let radius = f64::from(self.radius(candidate));
        for (index, block) in self.blocks.iter().enumerate() {
            let step = f64::from(block.scale) * radius;
            norms[family.groups[index]] +=
                block.len as f64 * f64::from(family.original[index].weight) * step * step;
        }
        norms.map(|norm| norm as f32)
    }

    #[cfg(test)]
    pub(super) fn params(&self, root: u64, candidate: usize) -> Params {
        let radius = self.radius(candidate);
        Params {
            seed: self.procedural_seed(root, candidate),
            radius,
            alternate_radius: radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            ..Params::default()
        }
    }
    pub(super) fn pool_params(&self, root: u64) -> Params {
        Params {
            seed: self.procedural_seed(root, 0),
            stream_seed: if self.pool_layout == crate::procedural_pool::ProceduralPool::legacy() {
                candidate_seed(root, 2)
            } else {
                root
            },
            basis_seed: self
                .threshold
                .as_ref()
                .map_or(0, threshold::ThresholdSearch::basis),
            radius: self.radius(0),
            alternate_radius: self.radius(1),
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            program: u32::from(self.threshold.is_some()),
            blocks: self.blocks.len() as u32,
            base_slot: self.resident_identities[..self.resident_history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .and_then(|slot| u32::try_from(slot).ok())
                .unwrap_or(u32::MAX),
            ..Params::default()
        }
    }
    pub(super) fn select_params(
        &self,
        root: u64,
        mut config: Ask,
        forced_candidate: Option<usize>,
    ) -> SelectionParams {
        if let Some((neighbors, epistemic, aleatoric, y_scale)) = self.fitted_enn {
            config.neighbors = neighbors.min(self.history);
            config.epistemic_scale = epistemic;
            config.aleatoric_scale = aleatoric;
            config.y_scale = y_scale;
        }
        let base_index = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id)
            .unwrap_or(0);
        let base_distances = std::array::from_fn(|i| {
            if self.implicit_history && i < self.history {
                self.pairwise_distances[base_index * MAX_HISTORY + i]
            } else {
                0.0
            }
        });
        let mut local_scales = [1.0; MAX_HISTORY];
        if self.distance_scaling == crate::config::DistanceScaling::SelfTuning && self.history > 1 {
            let mut values = Vec::with_capacity(self.history - 1);
            for (row, scale) in local_scales[..self.history].iter_mut().enumerate() {
                values.clear();
                values.extend(
                    (0..self.history)
                        .filter(|&column| column != row)
                        .map(|column| self.total_distance(row, column)),
                );
                values.sort_by(f32::total_cmp);
                *scale = values[self.local_scale_neighbors.min(values.len()) - 1].max(1.0e-12);
            }
        }
        let resident_indices = std::array::from_fn(|i| {
            self.identities[..self.history]
                .iter()
                .position(|identity| {
                    i < self.resident_history && *identity == self.resident_identities[i]
                })
                .unwrap_or(0) as u32
        });
        SelectionParams {
            root_seed: root,
            basis_seed: self
                .threshold
                .as_ref()
                .map_or(0, threshold::ThresholdSearch::basis),
            outcomes: std::array::from_fn(|i| self.outcomes.get(i).copied().unwrap_or(0.0)),
            variances: std::array::from_fn(|i| self.variances.get(i).copied().unwrap_or(0.0)),
            draws: std::array::from_fn(|i| {
                let identity = if self.relative {
                    i as i64 + 1
                } else {
                    self.identities.get(i).copied().unwrap_or(0)
                };
                crate::hash::normal_metric(config.seed, identity, 0) as f32
            }),
            base_distances,
            local_scales,
            latent_norms: std::array::from_fn(|candidate| self.latent_norm(candidate)),
            axis_history: std::array::from_fn(|i| {
                self.axis.as_ref().map_or(0.0, |axis| axis.history[i])
            }),
            axis_candidates: std::array::from_fn(|candidate| {
                self.axis_candidate(root, candidate).unwrap_or(0.0)
            }),
            axis_base: self.axis.as_ref().map_or(0.0, |axis| axis.base),
            axis_weight: self.axis.as_ref().map_or(0.0, |axis| axis.weight),
            axis_enabled: u32::from(self.axis.is_some()),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
            radius: self.radius(0),
            alternate_radius: self.radius(1),
            neighbors: config.neighbors as u32,
            history: self.history as u32,
            acquisition: match config.acquisition {
                AcquisitionKind::Ucb => 0,
                AcquisitionKind::Thompson => 1,
                AcquisitionKind::Pareto => 2,
            },
            tiles: self.tiles.len() as u32,
            mode: self.perturbation.shader(),
            program: u32::from(self.threshold.is_some()),
            resident_history: self.physical_history() as u32,
            resident_indices,
            implicit_history: u32::from(self.implicit_history),
            exact_history: u32::from(
                self.realized_history() && self.history > self.resident_history,
            ),
            latent_history: u32::from(self.latent_history),
            forced_candidate: forced_candidate.unwrap_or(4) as u32,
            distance_scaling: u32::from(
                self.distance_scaling == crate::config::DistanceScaling::SelfTuning,
            ),
            local_scale_neighbors: self.local_scale_neighbors as u32,
            incumbent_index: base_index as u32,
            candidate_floor: self
                .reliability
                .as_ref()
                .is_some_and(|controller| self.pool_layout.slots() != 1 && controller.force_fresh())
                .then_some(2)
                .unwrap_or(0),
        }
    }

    pub(super) fn physical_history(&self) -> usize {
        if self.implicit_history {
            self.resident_history
        } else {
            self.history
        }
    }
    pub(super) fn radius(&self, candidate: usize) -> f32 {
        (self.length
            * if self.pool_layout.slots() == 1 {
                1.0
            } else if candidate & 1 == 0 {
                0.5
            } else {
                2.0
            })
        .clamp(self.length_config.length_min, self.length_config.length_max) as f32
    }
    pub(super) fn ensure_ref(&mut self) -> Result<(), String> {
        autoreleasepool(|| self.init_reference())
    }

    pub(super) fn init_reference(&mut self) -> Result<(), String> {
        if self.independent_fp16 {
            return Ok(());
        }
        let seed = self
            .reference_seed
            .ok_or("Enable correlated Metal sampling first")?;
        if self.reference.is_none() {
            preflight(&self.runtime, &[self.row_bytes()], self.row_bytes())?;
            self.reference = Some(self.runtime.buffer::<u16>(self.dimensions));
            let command = self.runtime.queue.new_command_buffer();
            self.encode_ref(
                command,
                Params {
                    seed,
                    initialize: 1,
                    ..Params::default()
                },
            );
            if let Err(error) = finish(command).and_then(|()| self.check_reference()) {
                self.reference = None;
                return Err(error);
            }
        }
        Ok(())
    }
    pub(super) fn check_reference(&self) -> Result<(), String> {
        if read::<f32>(&self.reference_scales, self.blocks.len())
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            Err("Invalid Metal BF16 reference RMS".into())
        } else {
            Ok(())
        }
    }
}
