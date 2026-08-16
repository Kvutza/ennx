use std::collections::VecDeque;

use ndarray::Array1;

use crate::trials::Ask;
use crate::trust_region::{TRLengthConfig, TurboTrustRegion};

const MAX_HISTORY: usize = 128;
const MAX_PENDING: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bf16Block {
    pub key: u64,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
    pub weight: f32,
}

impl Bf16Block {
    pub fn new(
        key: u64,
        offset: usize,
        len: usize,
        scale: f32,
        weight: f32,
    ) -> Result<Self, String> {
        if len == 0 {
            return Err("BF16 block length must be positive".to_string());
        }
        if !scale.is_finite() || scale <= 0.0 || !weight.is_finite() || weight <= 0.0 {
            return Err("BF16 block scale and weight must be positive".to_string());
        }
        Ok(Self {
            key,
            offset,
            len,
            scale,
            weight,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bf16Trial {
    id: u64,
    slot: usize,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
    pub length: f32,
}

#[derive(Debug, Clone, Copy)]
struct Record {
    slot: usize,
    value: f32,
    variance: f32,
}

/// Stateful TuRBO search over full-precision BF16 model weights on CUDA.
pub struct Bf16Search {
    engine: ennx_cuda::Bf16SearchEngine,
    capacity: usize,
    pending_capacity: usize,
    history: VecDeque<Record>,
    pending: Vec<Bf16Trial>,
    next_id: u64,
    trust: TurboTrustRegion,
    best: f32,
    best_variance: f32,
    restarts: usize,
}

impl Bf16Search {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u16],
        base_value: f32,
        base_variance: f32,
        blocks: Vec<Bf16Block>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        let len = check_search(
            base.len(),
            base_value,
            base_variance,
            &blocks,
            capacity,
            pending_capacity,
        )?;
        let slots = slot_count(capacity, pending_capacity)?;
        let leaves = cuda_blocks(&blocks);
        let engine = ennx_cuda::Bf16SearchEngine::new(base, &leaves, slots)?;
        Self::create(
            engine,
            len,
            base_value,
            base_variance,
            capacity,
            pending_capacity,
            length,
        )
    }

    /// Copy a contiguous device-0 BF16 allocation into resident search state.
    ///
    /// # Safety
    /// `pointer` must address at least `len * 2` readable bytes on CUDA device 0.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_device(
        pointer: u64,
        len: usize,
        base_value: f32,
        base_variance: f32,
        blocks: Vec<Bf16Block>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        let dimensions = check_search(
            len,
            base_value,
            base_variance,
            &blocks,
            capacity,
            pending_capacity,
        )?;
        let slots = slot_count(capacity, pending_capacity)?;
        let leaves = cuda_blocks(&blocks);
        let engine =
            unsafe { ennx_cuda::Bf16SearchEngine::from_device(pointer, len, &leaves, slots)? };
        Self::create(
            engine,
            dimensions,
            base_value,
            base_variance,
            capacity,
            pending_capacity,
            length,
        )
    }

    fn create(
        engine: ennx_cuda::Bf16SearchEngine,
        dimensions: usize,
        base_value: f32,
        base_variance: f32,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        engine.copy_row(0, 1)?;
        let mut trust = TurboTrustRegion::new(dimensions, length);
        trust.set_num_arms(1);
        let initial = Array1::from_vec(vec![f64::from(base_value)]);
        trust
            .update_with_incumbent_new_batch(&initial.view(), 1, f64::from(base_value))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            engine,
            capacity,
            pending_capacity,
            history: VecDeque::from([Record {
                slot: 1,
                value: base_value,
                variance: base_variance,
            }]),
            pending: Vec::with_capacity(pending_capacity),
            next_id: 0,
            trust,
            best: base_value,
            best_variance: base_variance,
            restarts: 0,
        })
    }

    pub fn ask(&mut self, seeds: &[u64], config: Ask) -> Result<Bf16Trial, String> {
        Ok(self.ask_batch(seeds, 1, config)?[0])
    }

    pub fn ask_batch(
        &mut self,
        seeds: &[u64],
        arms: usize,
        config: Ask,
    ) -> Result<Vec<Bf16Trial>, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish outstanding BF16 trials before ask".to_string());
        }
        if arms == 0 || arms > self.pending_capacity || seeds.is_empty() || seeds.len() % arms != 0
        {
            return Err("BF16 batch shape exceeds pending capacity".to_string());
        }
        let candidates = seeds.len() / arms;
        let slots = self.free_slots(arms)?;
        let history_slots = self
            .history
            .iter()
            .map(|record| record.slot as u32)
            .collect::<Vec<_>>();
        let outcomes = self
            .history
            .iter()
            .map(|record| record.value)
            .collect::<Vec<_>>();
        let variances = self
            .history
            .iter()
            .map(|record| record.variance)
            .collect::<Vec<_>>();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let length = self.trust.length() as f32;
        self.trust.set_num_arms(arms);
        let selections = self.engine.ask(
            0,
            &history_slots,
            &outcomes,
            &variances,
            &slots,
            seeds,
            &draws,
            candidates,
            length,
            ennx_cuda::Ask {
                neighbors: config.neighbors,
                acquisition: crate::weights::acquisition_code(config.acquisition),
                epistemic_scale: config.epistemic_scale,
                aleatoric_scale: config.aleatoric_scale,
                y_scale: config.y_scale,
                beta: config.beta,
            },
        )?;
        let mut trials = Vec::with_capacity(arms);
        for (&slot, selection) in slots.iter().zip(selections) {
            let index = selection.index as usize;
            let trial = Bf16Trial {
                id: self.next_id,
                slot: slot as usize,
                index,
                seed: seeds[index],
                score: selection.score,
                length,
            };
            self.next_id = self.next_id.wrapping_add(1);
            self.pending.push(trial);
            trials.push(trial);
        }
        Ok(trials)
    }

    pub fn tell(&mut self, trial: Bf16Trial, value: f32, variance: f32) -> Result<bool, String> {
        Ok(self.tell_batch(&[trial], &[value], &[variance])?[0])
    }

    pub fn tell_batch(
        &mut self,
        trials: &[Bf16Trial],
        values: &[f32],
        variances: &[f32],
    ) -> Result<Vec<bool>, String> {
        check_tell(trials, values, variances)?;
        self.check_trials(trials)?;
        let mut accepted = Vec::with_capacity(trials.len());
        for ((trial, &value), &variance) in trials.iter().zip(values).zip(variances) {
            let accept = value > self.best;
            if accept {
                self.engine.copy_row(trial.slot, 0)?;
                self.best = value;
                self.best_variance = variance;
            }
            if self.history.len() == self.capacity {
                self.history.pop_front();
            }
            self.history.push_back(Record {
                slot: trial.slot,
                value,
                variance,
            });
            self.pending.retain(|candidate| candidate.id != trial.id);
            accepted.push(accept);
        }
        let batch = Array1::from_iter(values.iter().map(|value| f64::from(*value)));
        let total = self.trust.prev_num_obs() + values.len();
        self.trust
            .update_with_incumbent_new_batch(&batch.view(), total, f64::from(self.best))
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() && self.pending.is_empty() {
            self.restart()?;
        }
        Ok(accepted)
    }

    pub fn device_row(
        &self,
        trial: Bf16Trial,
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        let pending = self.pending_for(trial)?;
        self.engine.device_row(pending.slot, stream)
    }

    pub fn read(&self, trial: Bf16Trial) -> Result<Vec<u16>, String> {
        let pending = self.pending_for(trial)?;
        self.engine.read(pending.slot)
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.engine.set_profiling(enabled);
    }

    pub fn last_profile(&self) -> Option<ennx_cuda::AskProfile> {
        self.engine.last_profile()
    }

    pub fn length(&self) -> f64 {
        self.trust.length()
    }

    pub fn best(&self) -> f32 {
        self.best
    }

    pub fn best_variance(&self) -> f32 {
        self.best_variance
    }

    pub fn restarts(&self) -> usize {
        self.restarts
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn len(&self) -> usize {
        self.engine.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engine.is_empty()
    }

    fn restart(&mut self) -> Result<(), String> {
        self.history.clear();
        self.engine.copy_row(0, 1)?;
        self.history.push_back(Record {
            slot: 1,
            value: self.best,
            variance: self.best_variance,
        });
        self.trust.restart();
        self.restarts += 1;
        Ok(())
    }

    fn check_trials(&self, trials: &[Bf16Trial]) -> Result<(), String> {
        for (index, trial) in trials.iter().enumerate() {
            if trials[..index].contains(trial) {
                return Err("BF16 tell batch contains a duplicate trial".to_string());
            }
            self.pending_for(*trial)?;
        }
        Ok(())
    }

    fn pending_for(&self, trial: Bf16Trial) -> Result<Bf16Trial, String> {
        self.pending
            .iter()
            .copied()
            .find(|pending| pending.id == trial.id && *pending == trial)
            .ok_or_else(|| "BF16 trial does not match an outstanding ask".to_string())
    }

    fn free_slots(&self, count: usize) -> Result<Vec<u32>, String> {
        let slots = (1..slot_count(self.capacity, self.pending_capacity)?)
            .filter(|slot| {
                self.history.iter().all(|record| record.slot != *slot)
                    && self.pending.iter().all(|trial| trial.slot != *slot)
            })
            .take(count)
            .map(|slot| slot as u32)
            .collect::<Vec<_>>();
        if slots.len() != count {
            Err("not enough free BF16 model slots".to_string())
        } else {
            Ok(slots)
        }
    }
}

fn check_search(
    len: usize,
    base_value: f32,
    base_variance: f32,
    blocks: &[Bf16Block],
    capacity: usize,
    pending_capacity: usize,
) -> Result<usize, String> {
    if !base_value.is_finite() || !base_variance.is_finite() || base_variance < 0.0 {
        return Err("BF16 base value and variance are invalid".to_string());
    }
    if capacity == 0 || capacity > MAX_HISTORY {
        return Err(format!(
            "BF16 history capacity must be in 1..={MAX_HISTORY}"
        ));
    }
    if pending_capacity == 0 || pending_capacity > MAX_PENDING {
        return Err(format!(
            "BF16 pending capacity must be in 1..={MAX_PENDING}"
        ));
    }
    let mut expected = 0usize;
    for block in blocks {
        if block.offset != expected {
            return Err("BF16 blocks must form a contiguous layout".to_string());
        }
        expected = expected
            .checked_add(block.len)
            .ok_or("BF16 block layout overflow")?;
    }
    if expected == 0 || expected != len {
        return Err(format!(
            "BF16 blocks cover {expected} weights, expected {len}"
        ));
    }
    Ok(expected)
}

fn check_tell(trials: &[Bf16Trial], values: &[f32], variances: &[f32]) -> Result<(), String> {
    if trials.is_empty() || trials.len() != values.len() || trials.len() != variances.len() {
        return Err(
            "BF16 trials, values, and variances must have equal non-zero length".to_string(),
        );
    }
    if values.iter().any(|value| !value.is_finite())
        || variances
            .iter()
            .any(|variance| !variance.is_finite() || *variance < 0.0)
    {
        return Err("BF16 values and variances must be finite".to_string());
    }
    Ok(())
}

fn slot_count(capacity: usize, pending: usize) -> Result<usize, String> {
    capacity
        .checked_add(pending)
        .and_then(|slots| slots.checked_add(1))
        .ok_or("BF16 resident slot count overflow".to_string())
}

fn cuda_blocks(blocks: &[Bf16Block]) -> Vec<ennx_cuda::Bf16Leaf> {
    blocks
        .iter()
        .map(|block| ennx_cuda::Bf16Leaf {
            key: block.key,
            offset: block.offset as u64,
            length: block.len as u64,
            scale: block.scale,
            weight: block.weight,
        })
        .collect()
}
