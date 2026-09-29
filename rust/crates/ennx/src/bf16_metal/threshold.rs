use super::*;
use crate::threshold::{TABLE_WORDS, ThresholdModel, ThresholdTable};
use std::collections::BTreeMap;

#[derive(Clone)]
struct ProgramStep {
    table: ThresholdTable,
    norm: f32,
}

pub(super) struct ThresholdSearch {
    basis_seed: Option<u64>,
    adaptive: bool,
    steps: usize,
    model: ThresholdModel,
    tables: Vec<ThresholdTable>,
    history: BTreeMap<i64, ProgramStep>,
}

impl ThresholdSearch {
    pub(super) fn new(noise_floor: f64, adaptive: bool) -> Result<Self, String> {
        Ok(Self {
            basis_seed: None,
            adaptive,
            steps: 0,
            model: ThresholdModel::new(noise_floor)?,
            tables: Vec::with_capacity(4),
            history: BTreeMap::new(),
        })
    }

    pub(super) fn compact(&mut self, incumbent: i64) {
        self.history.retain(|identity, _| *identity == incumbent);
        self.tables.clear();
    }

    pub(super) fn basis(&self) -> u64 {
        self.basis_seed.unwrap_or(0)
    }

    pub(super) fn prepare(
        &mut self,
        root: u64,
        buffer: &Buffer,
        identities: &[i64],
        incumbent: i64,
        norms: [f32; 4],
    ) -> Result<(u64, Vec<f32>), String> {
        let basis = *self
            .basis_seed
            .get_or_insert_with(|| crate::hash::splitmix64(root ^ 0x7074_665f_6261_7369));
        self.tables.clear();
        let draws = if self.adaptive {
            self.model.draw_batch(root, &[0, 1])?
        } else {
            vec![ThresholdTable::basis(self.steps); 2]
        };
        for candidate in 0..4 {
            self.tables.push(draws[candidate / 2].clone());
        }
        unsafe {
            for (candidate, table) in self.tables.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    table.words().as_ptr(),
                    buffer.contents().cast::<u64>().add(candidate * TABLE_WORDS),
                    TABLE_WORDS,
                );
            }
        }
        let mut distances = Vec::with_capacity(5 * identities.len());
        for (candidate, table) in self.tables.iter().enumerate() {
            distances.extend(identities.iter().map(|identity| {
                self.history.get(identity).map_or(norms[candidate], |step| {
                    program_distance(table, norms[candidate], step)
                })
            }));
        }
        let incumbent_step = self.history.get(&incumbent);
        distances.extend(identities.iter().map(|identity| {
            match (incumbent_step, self.history.get(identity)) {
                (Some(left), Some(right)) => program_distance(&left.table, left.norm, right),
                (Some(left), None) => left.norm,
                (None, Some(right)) => right.norm,
                (None, None) => 0.0,
            }
        }));
        Ok((basis, distances))
    }

    pub(super) fn table(&self, candidate: usize) -> Result<ThresholdTable, String> {
        self.tables
            .get(candidate)
            .cloned()
            .ok_or_else(|| "threshold candidate table is unavailable".into())
    }

    pub(super) fn observe(
        &mut self,
        identity: i64,
        table: &ThresholdTable,
        radius: f32,
        norm: f32,
        improvement: f64,
        variance: f64,
    ) -> Result<(), String> {
        if !norm.is_finite() || norm < 0.0 {
            return Err("threshold direction norm must be finite and nonnegative".into());
        }
        if self.adaptive {
            self.model
                .observe(table, f64::from(radius), improvement, variance)?;
        }
        self.steps = self.steps.saturating_add(1);
        self.history.insert(
            identity,
            ProgramStep {
                table: table.clone(),
                norm,
            },
        );
        Ok(())
    }
}

fn program_distance(table: &ThresholdTable, norm: f32, step: &ProgramStep) -> f32 {
    let correlation = 1.0 - 0.5 * table.distance(&step.table);
    let cross = 2.0 * (f64::from(norm) * f64::from(step.norm)).sqrt() * correlation;
    (f64::from(norm) + f64::from(step.norm) - cross).max(0.0) as f32
}

impl SearchState {
    pub(super) fn prepare_threshold(&mut self, root: u64) -> Result<u64, String> {
        if self.threshold.is_none() {
            return Ok(0);
        }
        let identities = self.identities[..self.history].to_vec();
        let norms = std::array::from_fn(|candidate| self.latent_norm(candidate));
        let threshold = self.threshold.as_mut().unwrap();
        let (basis, distances) = threshold.prepare(
            root,
            &self.threshold_tables,
            &identities,
            self.base_id,
            norms,
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                distances.as_ptr(),
                self.pool_distances.contents().cast::<f32>(),
                distances.len(),
            );
        }
        Ok(basis)
    }

    pub(super) fn threshold_table(
        &self,
        candidate: usize,
    ) -> Result<Option<ThresholdTable>, String> {
        self.threshold
            .as_ref()
            .map(|threshold| threshold.table(candidate))
            .transpose()
    }
}
