use super::{decode_code, make_steps, Ask, Center, LeafStep, Parameter, SparseEdit};
use super::{sparse, tree};
use crate::util::insert_neighbor;
use crate::weights::AcquisitionKind;

pub(super) struct Cpu {
    rows: Vec<u8>,
    row_bytes: usize,
}

impl Cpu {
    pub(super) fn new(base: &[u8], slots: usize) -> Self {
        let mut rows = vec![0; slots * base.len()];
        rows[..base.len()].copy_from_slice(base);
        Self {
            rows,
            row_bytes: base.len(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        let steps = make_steps(leaves, config.length);
        let base = self.read(base_slot).to_vec();
        let draws = if config.acquisition == AcquisitionKind::Thompson {
            crate::weights::thompson_history_draws(history, config.seed)
        } else {
            Vec::new()
        };
        let mut best_index = 0;
        let mut best_score = f32::NEG_INFINITY;
        let mut nearest = vec![(f32::INFINITY, 0usize); config.neighbors];
        for (index, &seed) in seeds.iter().enumerate() {
            nearest.fill((f32::INFINITY, 0usize));
            for (observation_index, &(slot, _)) in history.iter().enumerate() {
                let distance = trial_distance(&base, self.read(slot), leaves, &steps, seed);
                insert_neighbor(&mut nearest, distance, observation_index);
            }
            let score = score(&nearest, history, &draws, config);
            if score > best_score || (score == best_score && index < best_index) {
                best_index = index;
                best_score = score;
            }
        }

        if materialize_row {
            let row = materialize(&base, leaves, &steps, seeds[best_index]);
            self.read_mut(trial_slot).copy_from_slice(&row);
        }
        Ok((best_index, best_score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        base_seed: u64,
        count: usize,
        leaves: &[Parameter],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, u64, f32), String> {
        let seeds = seed_stream(base_seed, count);
        let (index, score) = self.ask(
            base_slot,
            history,
            trial_slot,
            &seeds,
            leaves,
            config,
            materialize_row,
        )?;
        Ok((index, seeds[index], score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_sparse(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        let base = self.read(base_slot).to_vec();
        let rows = history
            .iter()
            .map(|&(slot, _)| self.read(slot))
            .collect::<Vec<_>>();
        let (index, score) = sparse::sparse_select(
            &base, &rows, history, seeds, edits, num_pert, leaves, config,
        );
        let row = sparse::sparse_materialize(
            &base,
            seeds[index],
            &edits[index * num_pert..(index + 1) * num_pert],
            leaves,
            config.length,
        );
        self.read_mut(trial_slot).copy_from_slice(&row);
        Ok((index, score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sparse_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        base_seed: u64,
        count: usize,
        num_pert: usize,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<(usize, u64, f32), String> {
        let seeds = seed_stream(base_seed, count);
        let edits = super::sparse::make_edits(&seeds, leaves, num_pert)?;
        let (index, score) = self.ask_sparse(
            base_slot, history, trial_slot, &seeds, &edits, num_pert, leaves, config,
        )?;
        Ok((index, seeds[index], score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_centers(
        &self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        tree::cpu_ask(
            self.read(base_slot),
            &self.rows,
            self.row_bytes,
            history,
            seeds_per_region,
            centers,
            region_centers,
            seeds,
            leaves,
            config,
        )
    }

    pub(super) fn read(&self, slot: usize) -> &[u8] {
        &self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }

    pub(super) fn read_mut(&mut self, slot: usize) -> &mut [u8] {
        &mut self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }
}

pub(super) fn check_count(count: usize, observations: usize, config: Ask) -> Result<(), String> {
    if count == 0 {
        return Err("ask requires at least one seed".to_string());
    }
    if config.neighbors == 0 || config.neighbors > observations {
        return Err(format!(
            "neighbor count must be between one and {observations}"
        ));
    }
    for (name, value) in [
        ("length", config.length),
        ("epistemic_scale", config.epistemic_scale),
        ("aleatoric_scale", config.aleatoric_scale),
        ("y_scale", config.y_scale),
        ("beta", config.beta),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and nonnegative"));
        }
    }
    Ok(())
}

pub(super) fn check_ask(seeds: &[u64], observations: usize, config: Ask) -> Result<(), String> {
    check_count(seeds.len(), observations, config)
}

pub(crate) fn perturb(code: u32, seed: u64, element: u32, step: LeafStep) -> u32 {
    let random = hash(seed, element);
    let sign = random & 1;
    let extra = u32::from((random >> 1) < (step.threshold >> 1));
    let amount = step.whole + extra;
    if amount == 0 {
        return code;
    }
    let max_code = (1u32 << step.bits) - 1;
    if sign == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

pub(super) fn hash(seed: u64, element: u32) -> u32 {
    let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= (seed >> 32) as u32;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

pub(super) fn seed_at(base_seed: u64, index: u32) -> u64 {
    let low = hash(base_seed, index) as u64;
    let high = hash(base_seed, index ^ 0xa511_e9b3) as u64;
    low | (high << 32)
}

pub(super) fn seed_stream(base_seed: u64, count: usize) -> Vec<u64> {
    (0..count)
        .map(|index| seed_at(base_seed, index as u32))
        .collect()
}

pub(super) fn materialize(
    base: &[u8],
    leaves: &[Parameter],
    steps: &[LeafStep],
    seed: u64,
) -> Vec<u8> {
    let mut row = vec![0u8; base.len()];
    for (&leaf, &step) in leaves.iter().zip(steps) {
        match leaf.bits {
            4 => {
                for element in 0..leaf.length {
                    let byte = step.byte_offset as usize + element / 2;
                    let shift = (element & 1) * 4;
                    let code = u32::from((base[byte] >> shift) & 0x0f);
                    let value = perturb(code, seed, leaf.offset as u32 + element as u32, step);
                    row[byte] |= (value as u8) << shift;
                }
            }
            8 => {
                for element in 0..leaf.length {
                    let byte = step.byte_offset as usize + element;
                    let code = u32::from(base[byte]);
                    row[byte] =
                        perturb(code, seed, leaf.offset as u32 + element as u32, step) as u8;
                }
            }
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
    row
}

pub(super) fn trial_distance(
    base: &[u8],
    observation: &[u8],
    leaves: &[Parameter],
    steps: &[LeafStep],
    seed: u64,
) -> f32 {
    let mut distance = 0.0f32;
    for (&leaf, &step) in leaves.iter().zip(steps) {
        let byte_offset = step.byte_offset as usize;
        let element_offset = leaf.offset as u32;
        if leaf.bits == 4 {
            for element in 0..leaf.length {
                let byte = byte_offset + element / 2;
                let shift = (element & 1) * 4;
                let code = u32::from((base[byte] >> shift) & 0x0f);
                let candidate_code = perturb(code, seed, element_offset + element as u32, step);
                let observed_code = u32::from((observation[byte] >> shift) & 0x0f);
                let candidate_val = decode_code(candidate_code, leaf.encoding, leaf.scale);
                let observed_val = decode_code(observed_code, leaf.encoding, leaf.scale);
                let delta = candidate_val - observed_val;
                distance = delta.mul_add(delta * leaf.weight, distance);
            }
        } else {
            for element in 0..leaf.length {
                let byte = byte_offset + element;
                let code = u32::from(base[byte]);
                let candidate_code = perturb(code, seed, element_offset + element as u32, step);
                let observed_code = u32::from(observation[byte]);
                let candidate_val = decode_code(candidate_code, leaf.encoding, leaf.scale);
                let observed_val = decode_code(observed_code, leaf.encoding, leaf.scale);
                let delta = candidate_val - observed_val;
                distance = delta.mul_add(delta * leaf.weight, distance);
            }
        }
    }
    distance
}

pub(super) fn score(
    nearest: &[(f32, usize)],
    history: &[(usize, f32)],
    draws: &[f32],
    config: Ask,
) -> f32 {
    let mut weight_sum = 0.0;
    let mut weighted_value = 0.0;
    let mut weighted_noise = 0.0;
    let mut weight_squared_sum = 0.0;
    let reference_weight = if config.acquisition == AcquisitionKind::Thompson {
        let variance = 1.0e-9 + config.epistemic_scale * nearest[0].0 + config.aleatoric_scale;
        (1.0 / variance.max(1.0e-12)).max(f32::MIN_POSITIVE)
    } else {
        1.0
    };
    for &(distance, index) in nearest {
        let variance = 1.0e-9 + config.epistemic_scale * distance + config.aleatoric_scale;
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * history[index].1;
        if config.acquisition == AcquisitionKind::Thompson {
            let draw_weight = weight / reference_weight;
            weighted_noise += draw_weight * draws[index];
            weight_squared_sum += draw_weight * draw_weight;
        }
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * config.y_scale;
    match config.acquisition {
        AcquisitionKind::Ucb => mean + config.beta * se,
        AcquisitionKind::Thompson => {
            mean + se * (weighted_noise / weight_squared_sum.sqrt().max(1.0e-12))
        }
        // Legacy scalar heuristic for the resident search path.
        // The multiobjective Pareto-front selector lives in `crate::acquisition`.
        AcquisitionKind::Pareto => mean + se,
    }
}
