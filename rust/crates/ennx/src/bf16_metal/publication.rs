use super::*;

impl SearchState {
    pub(super) fn publish_ask(&mut self) -> Result<Proposals, String> {
        self.publish_start(Instant::now())
    }

    pub(super) fn publish_start(&mut self, start: Instant) -> Result<Proposals, String> {
        let total_ms = start.elapsed().as_secs_f32() * 1000.0;
        let decision = read::<Decision>(&self.decision, 1)[0];
        if decision.valid == 0
            || decision.index >= 4
            || !decision.score.is_finite()
            || !decision.predicted_mean.is_finite()
            || !decision.predicted_standard_error.is_finite()
            || decision.predicted_standard_error < 0.0
            || !decision.incumbent_mean.is_finite()
            || !decision.incumbent_standard_error.is_finite()
            || decision.incumbent_standard_error < 0.0
        {
            return Err("No finite Metal BF16 acquisition candidate".into());
        }
        if decision.mode != self.perturbation.shader() {
            return Err("Metal BF16 selected perturbation mode changed".into());
        }
        if decision.program != u32::from(self.threshold.is_some()) {
            return Err("Metal BF16 selected proposal program changed".into());
        }
        let index = decision.index as usize;
        let distances = read::<f32>(&self.pool_distances, 4 * self.history);
        if distances.iter().any(|distance| !distance.is_finite()) {
            return Err("Metal BF16 candidate-pool history distance is nonfinite".into());
        }
        let pool = distances
            .chunks_exact(self.history)
            .enumerate()
            .map(|(candidate, row)| {
                (
                    candidate,
                    self.procedural_seed(decision.root_seed, candidate),
                    self.radius(candidate),
                    if !self.independent_fp16 && self.pool_layout.slots() != 1 && candidate < 2 {
                        0.75
                    } else {
                        0.0
                    },
                    self.identities[..self.history]
                        .iter()
                        .copied()
                        .zip(row.iter().copied())
                        .collect(),
                )
            })
            .collect::<Vec<PoolDescription>>();
        if pool[index].1 != decision.seed || pool[index].2 != decision.radius {
            return Err("Metal BF16 selected descriptor disagrees with its pool".into());
        }
        let (pool_radii, pool_cosines, reference_cosines) = self.read_geometry()?;
        let history_distances = pool[index].4.clone();
        let axis_coordinate = self.axis_candidate(decision.root_seed, index);
        let axis_value = self
            .axis
            .as_ref()
            .zip(axis_coordinate)
            .map(|(axis, coordinate)| axis.value(coordinate));
        let partials = read::<Partial>(&self.partials, self.tiles.len() * 4);
        if self.analytic_pool() {
            let selected = &partials[index * self.tiles.len()..(index + 1) * self.tiles.len()];
            if selected
                .iter()
                .any(|partial| partial.invalid != 0 || !partial.squared.is_finite())
            {
                return Err("Metal BF16 selected proposal is nonfinite".into());
            }
            if selected.iter().all(|partial| partial.changed == 0) {
                return Err("Metal BF16 selected proposal did not change".into());
            }
        }
        let mut changes = vec![(0u64, 0.0f64); self.blocks.len()];
        for (tile_index, tile) in self.tiles.iter().enumerate() {
            let partial = partials[index * self.tiles.len() + tile_index];
            changes[tile.leaf as usize].0 += u64::from(partial.changed);
            changes[tile.leaf as usize].1 += f64::from(partial.squared);
        }
        let mut bounds = None;
        let family_distances = if self.realized_history()
            && self.history > self.resident_history
            && self.family.is_some()
        {
            let values = read::<f32>(&self.pool_family_distances, 4 * self.history * FAMILIES);
            bounds = Some(self.inspect_bounds(&distances, &values, decision.root_seed));
            Some(
                values[index * self.history * FAMILIES..(index + 1) * self.history * FAMILIES]
                    .chunks_exact(FAMILIES)
                    .map(|row| row.try_into().unwrap())
                    .collect(),
            )
        } else if let Some(family) = &self.family {
            let base = self.identities[..self.history]
                .iter()
                .position(|&id| id == self.base_id)
                .ok_or("Family history lost its incumbent")?;
            if self.latent_history {
                let norms = self.latent_components(index);
                Some(
                    (0..self.history)
                        .map(|row| {
                            std::array::from_fn(|group| {
                                family.components[base * MAX_HISTORY + row][group] + norms[group]
                            })
                        })
                        .collect(),
                )
            } else {
                let mut norms = [0.0; FAMILIES];
                let mut resident = [[0.0; FAMILIES]; 2];
                for (t, tile) in self.tiles.iter().enumerate() {
                    let leaf = tile.leaf as usize;
                    let g = family.groups[leaf];
                    let p = partials[index * self.tiles.len() + t];
                    norms[g] += p.squared * family.original[leaf].weight;
                    resident[0][g] += p.anchor / family.weights[g];
                    resident[1][g] += p.rejected / family.weights[g];
                }
                let mut rows = (0..self.history)
                    .map(|i| {
                        std::array::from_fn(|g| {
                            family.components[base * MAX_HISTORY + i][g] + norms[g]
                        })
                    })
                    .collect::<Vec<_>>();
                for (slot, values) in resident.iter().enumerate().take(self.resident_history) {
                    let row = self.identities[..self.history]
                        .iter()
                        .position(|&id| id == self.resident_identities[slot])
                        .ok_or("Family history lost resident identity")?;
                    rows[row] = *values;
                }
                if rows.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err("Invalid family distance component".into());
                }
                Some(rows)
            }
        } else {
            None
        };
        let round = Proposals {
            pool_layout: self.pool_layout,
            owner: self.owner,
            id: self.next_id,
            base_id: self.base_id,
            index,
            seed: decision.seed,
            score: decision.score,
            length: decision.radius,
            predicted_mean: decision.predicted_mean,
            predicted_standard_error: decision.predicted_standard_error,
            incumbent_mean: decision.incumbent_mean,
            incumbent_standard_error: decision.incumbent_standard_error,
            changes,
            history_distances,
            family_distances,
            block_scales: self.blocks.iter().map(|b| b.scale).collect(),
            pool,
            pool_radii,
            pool_cosines,
            reference_cosines,
            bounds,
            axis_coordinate,
            axis_value,
            basis_seed: decision.basis_seed,
            direction_norm: self.latent_norm(index),
            threshold_table: self.threshold_table(index)?,
        };
        self.pending = Some(round.clone());
        self.next_id = self.next_id.wrapping_add(1);
        self.started = true;
        if self.profiling && self.last_profile.is_none() {
            self.last_profile = Some(AskProfile {
                score_ms: total_ms,
                pick_ms: 0.0,
                materialize_ms: 0.0,
                total_ms,
            });
        }
        Ok(round)
    }

    fn read_geometry(&self) -> Result<([f32; 4], [Option<f32>; 6], [Option<f32>; 4]), String> {
        if self.analytic_pool() {
            return Ok((
                std::array::from_fn(|candidate| self.latent_norm(candidate).sqrt()),
                [None; 6],
                [None; 4],
            ));
        }
        let geometry = read::<f32>(&self.pool_geometry, POOL_METRICS * self.tiles.len());
        if geometry.iter().any(|value| !value.is_finite()) {
            return Err("Metal BF16 candidate-pool geometry is nonfinite".into());
        }
        let mut totals = [0.0f64; POOL_METRICS];
        for (metric, total) in totals.iter_mut().enumerate() {
            *total = geometry[metric * self.tiles.len()..(metric + 1) * self.tiles.len()]
                .iter()
                .map(|&value| f64::from(value))
                .sum();
        }
        if totals[..4].iter().any(|&norm| norm < 0.0) {
            return Err("Metal BF16 candidate pool has negative squared radius".into());
        }
        let pool_radii = std::array::from_fn(|candidate| totals[candidate].sqrt() as f32);
        #[cfg(not(test))]
        let (pool_cosines, reference_cosines) = ([None; 6], [None; 4]);
        #[cfg(test)]
        let (pool_cosines, reference_cosines) = {
            let mut pool_cosines = [None; 6];
            for (pair, &(left, right)) in POOL_PAIRS.iter().enumerate() {
                let denominator = f64::from(pool_radii[left]) * f64::from(pool_radii[right]);
                if denominator == 0.0 {
                    continue;
                }
                let cosine = totals[pair + 4] / denominator;
                if !cosine.is_finite() || cosine.abs() > 1.001 {
                    return Err("Metal BF16 candidate-pool cosine is invalid".into());
                }
                pool_cosines[pair] = Some(cosine.clamp(-1.0, 1.0) as f32);
            }
            if totals[10] < 0.0 {
                return Err("Metal BF16 reference has negative squared radius".into());
            }
            let reference_radius = totals[10].sqrt();
            let mut reference_cosines = [None; 4];
            for candidate in 0..4 {
                let denominator = f64::from(pool_radii[candidate]) * reference_radius;
                if denominator == 0.0 {
                    continue;
                }
                let cosine = totals[candidate + 11] / denominator;
                if !cosine.is_finite() || cosine.abs() > 1.001 {
                    return Err("Metal BF16 candidate-reference cosine is invalid".into());
                }
                reference_cosines[candidate] = Some(cosine.clamp(-1.0, 1.0) as f32);
            }
            (pool_cosines, reference_cosines)
        };
        Ok((pool_radii, pool_cosines, reference_cosines))
    }
}
