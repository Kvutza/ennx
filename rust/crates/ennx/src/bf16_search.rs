use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::trials::Ask;
use crate::trust_region::TRLengthConfig;

const MAX_HISTORY: usize = 128;
const MAX_PENDING: usize = 32;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TellMode {
    Legacy,
    Paired,
    PairedRelative,
}

fn check_tellmode(current: Option<TellMode>, requested: TellMode) -> Result<(), String> {
    if current.is_some_and(|mode| mode != requested) {
        return Err(
            "Cannot mix legacy tell, tell_paired, and tell_relative on the same BF16 search".into(),
        );
    }
    if current.is_none() && requested == TellMode::PairedRelative {
        return Err("Enable paired-relative BF16 search before tell_relative".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamBlock {
    pub key: u64,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
    pub weight: f32,
}

impl ParamBlock {
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
pub struct Proposal {
    owner: u64,
    id: u64,
    slot: usize,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
    pub length: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Proposals {
    trials: Vec<Proposal>,
}

impl Proposals {
    pub fn arms(&self) -> usize {
        self.trials.len()
    }
}

/// Stateful BF16 weight search on CUDA, with sign or Gaussian proposals.
pub struct SearchState {
    owner: u64,
    tell_mode: Option<TellMode>,
    engine: ennx_cuda::Bf16SearchEngine,
    dimensions: usize,
    capacity: usize,
    pending_capacity: usize,
    history: VecDeque<usize>,
    pending: Vec<Proposal>,
    next_id: u64,
    length: f64,
    length_config: TRLengthConfig,
    best: f32,
    best_variance: f32,
    restarts: usize,
    queued: Option<usize>,
    failure_limit: Option<usize>,
    correlated: bool,
    gaussian: bool,
    started: bool,
}

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u16],
        base_value: f32,
        base_variance: f32,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        check_length(length)?;
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
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        check_length(length)?;
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
        mut engine: ennx_cuda::Bf16SearchEngine,
        dimensions: usize,
        base_value: f32,
        base_variance: f32,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        engine.copy_row(0, 1)?;
        engine.init_search(
            base_value,
            base_variance,
            capacity,
            length.length_init,
            length.length_min,
            length.length_max,
        )?;
        Ok(Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            tell_mode: None,
            engine,
            dimensions,
            capacity,
            pending_capacity,
            history: VecDeque::from([1]),
            pending: Vec::with_capacity(pending_capacity),
            next_id: 0,
            length: length.length_init,
            length_config: length,
            best: base_value,
            best_variance: base_variance,
            restarts: 0,
            queued: None,
            failure_limit: None,
            correlated: false,
            gaussian: false,
            started: false,
        })
    }

    /// Enable correlated Gaussian proposals and the GPU accepted-radius controller once,
    /// before any asks. Requires one pending arm and no failure-tolerance override.
    pub fn enable_correlated(&mut self, reference_seed: u64) -> Result<(), String> {
        if self.gaussian || self.started || !self.pending.is_empty() || self.queued.is_some() {
            return Err("Enable correlated BF16 sampling once, before any rounds".into());
        }
        if self.pending_capacity != 1 || self.failure_limit.is_some() {
            return Err(
                "Correlated BF16 sampling requires max_pending=1 and no failure tolerance".into(),
            );
        }
        let length = correlated_length(self.length_config)?;
        self.engine.init_search(
            self.best,
            self.best_variance,
            self.capacity,
            length.length_init,
            length.length_min,
            length.length_max,
        )?;
        self.engine.enable_correlated(reference_seed)?;
        self.length_config = length;
        self.length = length.length_init;
        self.correlated = true;
        self.gaussian = true;
        Ok(())
    }

    /// Enable independent Gaussian proposals once, before any asks.
    /// This mode retains TuRBO and does not allocate a reference direction.
    pub fn enable_gaussian(&mut self) -> Result<(), String> {
        if self.gaussian || self.started || !self.pending.is_empty() || self.queued.is_some() {
            return Err("Enable Gaussian BF16 sampling once, before any rounds".into());
        }
        self.engine.enable_gaussian()?;
        self.gaussian = true;
        Ok(())
    }

    /// Pin a zero-valued incumbent anchor and use paired-relative observations.
    /// Enable once on a fresh correlated search, before any asks or tells.
    pub fn enable_relative(&mut self, failure_tolerance: usize) -> Result<(), String> {
        check_relativeinit(
            self.correlated,
            self.started || !self.pending.is_empty() || self.queued.is_some(),
            self.tell_mode,
            self.capacity,
            failure_tolerance,
        )?;
        self.engine.enable_relative(failure_tolerance)?;
        self.tell_mode = Some(TellMode::PairedRelative);
        Ok(())
    }

    pub fn ask(&mut self, seeds: &[u64], config: Ask) -> Result<Proposal, String> {
        Ok(self.ask_batch(seeds, 1, config)?[0])
    }

    pub fn ask_batch(
        &mut self,
        seeds: &[u64],
        arms: usize,
        config: Ask,
    ) -> Result<Vec<Proposal>, String> {
        if self.correlated {
            return Err("Correlated BF16 sampling requires ask_round, not explicit seeds".into());
        }
        self.sync()?;
        if !self.pending.is_empty() {
            return Err("tell must finish outstanding BF16 trials before ask".to_string());
        }
        if arms == 0 || arms > self.pending_capacity || seeds.is_empty() || seeds.len() % arms != 0
        {
            return Err("BF16 batch shape exceeds pending capacity".to_string());
        }
        let candidates = seeds.len() / arms;
        let slots = self.free_slots(arms)?;
        let length = self.length as f32;
        self.started = true;
        let selections = self.engine.ask(
            0,
            self.history.len(),
            &slots,
            seeds,
            candidates,
            length,
            config.seed,
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
            let trial = Proposal {
                owner: self.owner,
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

    pub fn ask_round(
        &mut self,
        arms: usize,
        candidates: usize,
        seed_root: u64,
        config: Ask,
    ) -> Result<Proposals, String> {
        if self.correlated && (arms != 1 || candidates != 4 || self.pending_capacity != 1) {
            return Err(
                "Correlated BF16 rounds require arms=1, candidates=4, max_pending=1".into(),
            );
        }
        if !self.pending.is_empty() {
            return Err("tell must finish the outstanding BF16 round".to_string());
        }
        if arms == 0
            || arms > self.pending_capacity
            || candidates == 0
            || arms.checked_mul(candidates).is_none()
        {
            return Err("BF16 round shape exceeds pending capacity".to_string());
        }
        let slots = self.free_slots(arms)?;
        self.started = true;
        self.engine.ask_seeded(
            0,
            self.capacity,
            &slots,
            candidates,
            seed_root,
            1.0,
            config.seed,
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
        for &slot in &slots {
            let trial = Proposal {
                owner: self.owner,
                id: self.next_id,
                slot: slot as usize,
                index: 0,
                seed: 0,
                score: 0.0,
                length: 0.0,
            };
            self.next_id = self.next_id.wrapping_add(1);
            self.pending.push(trial);
            trials.push(trial);
        }
        Ok(Proposals { trials })
    }

    pub fn tell(&mut self, trial: Proposal, value: f32, variance: f32) -> Result<bool, String> {
        Ok(self.tell_batch(&[trial], &[value], &[variance])?[0])
    }

    pub fn tell_batch(
        &mut self,
        trials: &[Proposal],
        values: &[f32],
        variances: &[f32],
    ) -> Result<Vec<bool>, String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        check_tell(trials, values, variances)?;
        self.check_trials(trials)?;
        let slots = trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, trials.len()));
        let output = self
            .engine
            .tell(&slots, values, variances, self.capacity, tolerance)?;
        self.tell_mode = Some(TellMode::Legacy);
        Ok(self.finish_tell(trials, output))
    }

    /// Consume contiguous device-0 FP32 rewards and variances.
    ///
    /// # Safety
    /// The pointers must address `trials.len() * 4` readable bytes on CUDA device 0.
    pub unsafe fn tell_device(
        &mut self,
        trials: &[Proposal],
        values: u64,
        variances: Option<u64>,
    ) -> Result<Vec<bool>, String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        if trials.is_empty() {
            return Err("BF16 tell batch cannot be empty".to_string());
        }
        self.check_trials(trials)?;
        let slots = trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, trials.len()));
        let output = unsafe {
            self.engine.tell_device(
                &slots,
                values,
                variances,
                trials.len(),
                self.capacity,
                tolerance,
            )?
        };
        self.tell_mode = Some(TellMode::Legacy);
        Ok(self.finish_tell(trials, output))
    }

    pub fn tell_round(
        &mut self,
        round: &Proposals,
        values: &[f32],
        variances: &[f32],
    ) -> Result<Vec<bool>, String> {
        self.tell_batch(&round.trials, values, variances)
    }

    pub fn queue_round(
        &mut self,
        round: &Proposals,
        values: &[f32],
        variances: &[f32],
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        check_tell(&round.trials, values, variances)?;
        self.check_trials(&round.trials)?;
        let slots = round
            .trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, round.arms()));
        self.engine
            .queue_values(&slots, values, variances, self.capacity, tolerance)?;
        self.tell_mode = Some(TellMode::Legacy);
        self.pending.clear();
        self.queued = Some(round.arms());
        Ok(())
    }

    /// Queue a paired comparison. The explicit decision is independent of cached rewards.
    /// Rejection refreshes only the incumbent measurement; FIFO stores the candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn tell_paired(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::Paired)?;
        check_paired(
            self.correlated,
            round.arms(),
            value,
            variance,
            incumbent_value,
            incumbent_variance,
        )?;
        self.check_trials(&round.trials)?;
        self.engine.queue_paired(
            round.trials[0].slot as u32,
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            accept,
            self.capacity,
        )?;
        self.tell_mode = Some(TellMode::Paired);
        self.pending.clear();
        self.queued = Some(1);
        Ok(())
    }

    /// Queue an explicit decision with improvement measured against this incumbent.
    /// Absolute measurements update reporting only; accepted steps reset relative history.
    /// Rejections count toward radius contraction only when `reject_is_failure` is true.
    #[allow(clippy::too_many_arguments)]
    pub fn tell_relative(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        reject_is_failure: bool,
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::PairedRelative)?;
        check_paired(
            self.correlated,
            round.arms(),
            value,
            variance,
            incumbent_value,
            incumbent_variance,
        )?;
        check_improvement(improvement, improvement_variance)?;
        self.check_trials(&round.trials)?;
        self.engine.queue_relative(
            round.trials[0].slot as u32,
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
            accept,
            self.capacity,
            reject_is_failure,
        )?;
        self.pending.clear();
        self.queued = Some(1);
        Ok(())
    }

    /// Copy row zero into an unused, already allocated pending slot.
    pub fn snapshot_incumbent(&self) -> Result<(), String> {
        self.check_idle()?;
        self.engine.snapshot_incumbent(self.capacity + 1)
    }

    pub fn device_incumbent(&self, stream: Option<i64>) -> Result<(u64, usize, usize), String> {
        self.check_idle()?;
        let (pointer, _, _) = self.engine.device_row(self.capacity + 1, stream)?;
        Ok((pointer, 1, self.len()))
    }

    pub fn sync_consumers(&self) -> Result<(), String> {
        self.engine.sync_consumers()
    }

    fn check_idle(&self) -> Result<(), String> {
        if !self.pending.is_empty() || self.queued.is_some() {
            return Err(
                "Finish outstanding BF16 proposals and sync before exporting incumbent".into(),
            );
        }
        Ok(())
    }

    /// Consume device rewards for an opaque resident round.
    ///
    /// # Safety
    /// The pointers must address `round.arms() * 4` readable CUDA bytes.
    pub unsafe fn finish_round(
        &mut self,
        round: &Proposals,
        values: u64,
        variances: Option<u64>,
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        self.check_trials(&round.trials)?;
        let slots = round
            .trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, round.arms()));
        unsafe {
            self.engine.queue_tell(
                &slots,
                values,
                variances,
                round.arms(),
                self.capacity,
                tolerance,
            )?;
        }
        self.tell_mode = Some(TellMode::Legacy);
        self.pending.clear();
        self.queued = Some(round.arms());
        Ok(())
    }

    pub fn sync(&mut self) -> Result<Vec<bool>, String> {
        let Some(count) = self.queued else {
            return Ok(Vec::new());
        };
        let output = self.engine.collect_tell(count)?;
        self.queued = None;
        let accepted = output.accepted.clone();
        self.update_state(output);
        Ok(accepted)
    }

    pub fn device_row(
        &self,
        trial: Proposal,
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        let pending = self.pending_for(trial)?;
        self.engine.device_row(pending.slot, stream)
    }

    pub fn device_batch(
        &mut self,
        trials: &[Proposal],
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        if trials.is_empty() {
            return Err("BF16 device batch cannot be empty".to_string());
        }
        self.check_trials(trials)?;
        let slots = trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        self.engine.device_batch(&slots, stream)
    }

    pub fn device_round(
        &mut self,
        round: &Proposals,
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        self.check_trials(&round.trials)?;
        self.engine.device_round(round.arms(), stream)
    }

    pub fn describe(
        &self,
        round: &Proposals,
    ) -> Result<Vec<ennx_cuda::ProposalDescription>, String> {
        self.check_trials(&round.trials)?;
        self.engine.describe(round.arms())
    }

    pub fn set_tolerance(&mut self, tolerance: usize) -> Result<(), String> {
        if self.correlated {
            return Err("Correlated BF16 sampling does not use failure tolerance".into());
        }
        if tolerance == 0
            || u32::try_from(tolerance).is_err()
            || !self.pending.is_empty()
            || self.queued.is_some()
        {
            return Err(
                "BF16 failure tolerance must be a positive u32 set between completed rounds".into(),
            );
        }
        self.failure_limit = Some(tolerance);
        Ok(())
    }

    /// Selected candidate index and Gaussian persistence for each pending arm.
    /// Correlated indices 0/1 report 0.75; fresh and independent proposals report 0.
    pub fn geometry(&self, round: &Proposals) -> Result<Vec<(usize, f32)>, String> {
        self.check_trials(&round.trials)?;
        self.engine.geometry(round.arms())
    }

    pub fn read_best(&mut self) -> Result<Vec<u16>, String> {
        self.sync()?;
        if !self.pending.is_empty() {
            return Err("Finish the pending BF16 round before exporting best weights".into());
        }
        self.engine.read(0)
    }

    /// Copy the stored correlated reference to host memory as raw BF16 words.
    /// This model-sized copy is for validation; other sampler modes return an error.
    pub fn read_reference(&mut self) -> Result<Vec<u16>, String> {
        self.sync()?;
        if !self.pending.is_empty() {
            return Err("Finish the pending BF16 round before exporting the reference".into());
        }
        self.engine.read_reference()
    }

    pub fn read(&self, trial: Proposal) -> Result<Vec<u16>, String> {
        let pending = self.pending_for(trial)?;
        self.engine.read(pending.slot)
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.engine.set_profiling(enabled);
    }

    pub fn last_profile(&self) -> Option<ennx_cuda::AskProfile> {
        self.engine.last_profile()
    }

    pub fn length(&mut self) -> Result<f64, String> {
        self.sync()?;
        Ok(self.length)
    }

    pub fn best(&mut self) -> Result<f32, String> {
        self.sync()?;
        Ok(self.best)
    }

    pub fn best_variance(&mut self) -> Result<f32, String> {
        self.sync()?;
        Ok(self.best_variance)
    }

    pub fn restarts(&mut self) -> Result<usize, String> {
        self.sync()?;
        Ok(self.restarts)
    }

    pub fn history_len(&mut self) -> Result<usize, String> {
        self.sync()?;
        Ok(self.history.len())
    }

    pub fn len(&self) -> usize {
        self.engine.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engine.is_empty()
    }

    fn finish_tell(&mut self, trials: &[Proposal], output: ennx_cuda::TellOutput) -> Vec<bool> {
        self.pending
            .retain(|candidate| !trials.iter().any(|trial| trial.id == candidate.id));
        let accepted = output.accepted.clone();
        self.update_state(output);
        accepted
    }

    fn update_state(&mut self, output: ennx_cuda::TellOutput) {
        self.history = (1..=output.history).collect();
        self.length = output.length;
        self.best = output.best;
        self.best_variance = output.best_variance;
        self.restarts = output.restarts;
    }

    fn check_trials(&self, trials: &[Proposal]) -> Result<(), String> {
        if self.queued.is_some() {
            return Err("Sync the queued BF16 tell before another update".into());
        }
        for (index, trial) in trials.iter().enumerate() {
            if trials[..index].contains(trial) {
                return Err("BF16 tell batch contains a duplicate trial".to_string());
            }
            self.pending_for(*trial)?;
        }
        Ok(())
    }

    fn pending_for(&self, trial: Proposal) -> Result<Proposal, String> {
        self.pending
            .iter()
            .copied()
            .find(|pending| pending.id == trial.id && *pending == trial)
            .ok_or_else(|| "BF16 trial does not match an outstanding ask".to_string())
    }

    fn free_slots(&self, count: usize) -> Result<Vec<u32>, String> {
        let slots = (self.capacity + 1..slot_count(self.capacity, self.pending_capacity)?)
            .filter(|slot| self.pending.iter().all(|trial| trial.slot != *slot))
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

fn failure_tolerance(dimensions: usize, arms: usize) -> usize {
    let arm_count = arms as f64;
    (4.0_f64 / arm_count)
        .max(dimensions as f64 / arm_count)
        .ceil()
        .max(1.0) as usize
}

fn check_length(length: TRLengthConfig) -> Result<(), String> {
    if [length.length_min, length.length_init, length.length_max]
        .iter()
        .any(|&value| !value.is_finite() || !(value as f32).is_finite() || value as f32 <= 0.0)
        || length.length_init < length.length_min
        || length.length_init > length.length_max
    {
        return Err(
            "BF16 radius bounds must be ordered, positive, and representable as FP32".into(),
        );
    }
    Ok(())
}

fn correlated_length(length: TRLengthConfig) -> Result<TRLengthConfig, String> {
    check_length(length)?;
    let mut min = length.length_min as f32;
    let mut max = length.length_max as f32;
    // Accepted FP32 radii become FP64 state. Round bounds inward so this round-trip
    // cannot put the next round's state outside the caller's requested interval.
    if f64::from(min) < length.length_min {
        min = min.next_up();
    }
    if f64::from(max) > length.length_max {
        max = max.next_down();
    }
    if !min.is_finite() || min >= max {
        return Err("Correlated BF16 radius bounds must contain two distinct FP32 radii".into());
    }
    let (min, max) = (f64::from(min), f64::from(max));
    Ok(TRLengthConfig::new(
        length.length_init.clamp(min, max),
        min,
        max,
    ))
}

fn check_search(
    len: usize,
    base_value: f32,
    base_variance: f32,
    blocks: &[ParamBlock],
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

fn check_tell(trials: &[Proposal], values: &[f32], variances: &[f32]) -> Result<(), String> {
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

fn check_paired(
    correlated: bool,
    arms: usize,
    value: f32,
    variance: f32,
    incumbent_value: f32,
    incumbent_variance: f32,
) -> Result<(), String> {
    if !correlated || arms != 1 {
        return Err("tell_paired requires a single-arm correlated BF16 round".into());
    }
    if !value.is_finite()
        || !incumbent_value.is_finite()
        || !variance.is_finite()
        || variance < 0.0
        || !incumbent_variance.is_finite()
        || incumbent_variance < 0.0
    {
        return Err(
            "Paired rewards must be finite and both variances finite and nonnegative".into(),
        );
    }
    Ok(())
}

fn check_relativeinit(
    correlated: bool,
    started: bool,
    mode: Option<TellMode>,
    capacity: usize,
    failure_tolerance: usize,
) -> Result<(), String> {
    if !correlated || started || mode.is_some() || capacity < 2 {
        return Err("Enable paired-relative mode once on a fresh correlated search with capacity >= 2, before any asks or tells".into());
    }
    if failure_tolerance == 0 || u32::try_from(failure_tolerance).is_err() {
        return Err("Paired-relative failure_tolerance must be a positive u32".into());
    }
    Ok(())
}

fn check_improvement(improvement: f32, variance: f32) -> Result<(), String> {
    if !improvement.is_finite() || !variance.is_finite() || variance < 0.0 {
        return Err(
            "Paired improvement must be finite and its variance finite and nonnegative".into(),
        );
    }
    Ok(())
}

fn slot_count(capacity: usize, pending: usize) -> Result<usize, String> {
    capacity
        .checked_add(pending)
        .and_then(|slots| slots.checked_add(1))
        .ok_or("BF16 resident slot count overflow".to_string())
}

fn cuda_blocks(blocks: &[ParamBlock]) -> Vec<ennx_cuda::Bf16Leaf> {
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

#[cfg(test)]
mod radius_tests {
    use super::*;

    #[test]
    fn collapsed_bounds() {
        let ulp = f64::from(1.0_f32.next_up()) - 1.0;
        for (min, max) in [
            (1.0, 1.0),
            (1.0 + f64::EPSILON, 1.0 + 2.0 * f64::EPSILON),
            (1.0 + ulp * 0.25, 1.0 + ulp * 0.75),
            (1.0 + ulp * 0.5, 1.0 + ulp * 1.5),
        ] {
            let config = TRLengthConfig::new((min + max) * 0.5, min, max);
            assert!(check_length(config).is_ok());
            let error = correlated_length(config).unwrap_err();
            assert!(error.contains("two distinct FP32 radii"), "{error}");
        }
    }

    #[test]
    fn rounded_bounds() {
        let config = TRLengthConfig::new(0.01, 0.0001, 0.08);
        let rounded = correlated_length(config).unwrap();
        assert_eq!(rounded.length_init, config.length_init);
        assert!(rounded.length_min >= config.length_min);
        assert!(rounded.length_max <= config.length_max);
        assert_eq!(rounded.length_min, f64::from((0.0001_f64 as f32).next_up()));
        assert_eq!(rounded.length_max, f64::from(0.08_f64 as f32));
        assert!(check_length(TRLengthConfig::new(0.01, 0.01, 0.01)).is_ok());
    }

    #[test]
    fn clamped_radius() {
        let min = 1.0 + f64::EPSILON;
        let max = 2.0 - f64::EPSILON;
        for initial in [min, max] {
            let rounded = correlated_length(TRLengthConfig::new(initial, min, max)).unwrap();
            let expected = if initial == min {
                f64::from(1.0_f32.next_up())
            } else {
                f64::from(2.0_f32.next_down())
            };
            assert_eq!(rounded.length_init, expected);
            assert!(rounded.length_init >= min && rounded.length_init <= max);
        }
    }

    #[test]
    fn adjacent_bounds() {
        let min = f64::from(1.0_f32);
        let max = f64::from(1.0_f32.next_up());
        let config = TRLengthConfig::new(min, min, max);
        assert_eq!(correlated_length(config).unwrap(), config);
    }

    #[test]
    fn accepted_bound() {
        for config in [
            TRLengthConfig::default(),
            TRLengthConfig::new(0.01, 0.0001, 0.08),
            TRLengthConfig::new(1.0, 1.0 - f64::EPSILON, 2.0 - f64::EPSILON),
            TRLengthConfig::new(1e-40, 1e-41, 1e-39),
            TRLengthConfig::new(1e37, 1e36, f64::from(f32::MAX)),
        ] {
            let rounded = correlated_length(config).unwrap();
            for factor in [0.5, 2.0] {
                let mut radius = rounded.length_init;
                for _ in 0..128 {
                    // Mirror candidate_radius and the accepted-radius state update.
                    radius = f64::from(
                        ((radius * factor)
                            .max(rounded.length_min)
                            .min(rounded.length_max)) as f32,
                    );
                    assert!(radius >= config.length_min && radius <= config.length_max);
                    assert!(radius >= rounded.length_min && radius <= rounded.length_max);
                    assert!(
                        check_length(TRLengthConfig::new(
                            radius,
                            rounded.length_min,
                            rounded.length_max,
                        ))
                        .is_ok()
                    );
                }
            }
        }
    }

    #[test]
    fn bad_radii() {
        for config in [
            TRLengthConfig::new(f64::NAN, 0.001, 0.1),
            TRLengthConfig::new(0.01, 0.0, 0.1),
            TRLengthConfig::new(0.01, 0.02, 0.1),
            TRLengthConfig::new(0.01, 0.001, 0.005),
            TRLengthConfig::new(0.01, f64::MIN_POSITIVE, 0.1),
            TRLengthConfig::new(0.01, 0.001, f64::MAX),
        ] {
            assert!(correlated_length(config).is_err());
        }
    }
}

#[cfg(test)]
mod paired_tests {
    use super::*;

    #[test]
    fn validate_measure() {
        assert!(check_paired(true, 1, -12.0, 0.25, -13.0, 0.0).is_ok());
        assert!(check_paired(false, 1, 0.0, 0.0, 0.0, 0.0).is_err());
        for arms in [0, 2, MAX_PENDING] {
            assert!(check_paired(true, arms, 0.0, 0.0, 0.0, 0.0).is_err());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_paired(true, 1, invalid, 0.0, 0.0, 0.0).is_err());
            assert!(check_paired(true, 1, 0.0, 0.0, invalid, 0.0).is_err());
        }
        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_paired(true, 1, 0.0, invalid, 0.0, 0.0).is_err());
            assert!(check_paired(true, 1, 0.0, 0.0, 0.0, invalid).is_err());
        }
    }

    #[test]
    fn tell_modes() {
        for mode in [TellMode::Legacy, TellMode::Paired] {
            assert!(check_tellmode(None, mode).is_ok());
            assert!(check_tellmode(Some(mode), mode).is_ok());
        }
        assert!(check_tellmode(Some(TellMode::Legacy), TellMode::Paired).is_err());
        assert!(check_tellmode(Some(TellMode::Paired), TellMode::Legacy).is_err());
        assert!(check_tellmode(None, TellMode::PairedRelative).is_err());
        for mode in [TellMode::Legacy, TellMode::Paired, TellMode::PairedRelative] {
            assert!(check_tellmode(Some(mode), mode).is_ok());
            for other in [TellMode::Legacy, TellMode::Paired, TellMode::PairedRelative] {
                assert_eq!(check_tellmode(Some(mode), other).is_ok(), mode == other);
            }
        }
    }

    #[test]
    fn mode_validation() {
        assert!(check_relativeinit(true, false, None, 2, 4).is_ok());
        assert!(check_relativeinit(true, false, None, MAX_HISTORY, u32::MAX as usize).is_ok());
        assert!(check_relativeinit(false, false, None, 2, 4).is_err());
        assert!(check_relativeinit(true, true, None, 2, 4).is_err());
        for mode in [TellMode::Legacy, TellMode::Paired, TellMode::PairedRelative] {
            assert!(check_relativeinit(true, false, Some(mode), 2, 4).is_err());
        }
        for capacity in [0, 1] {
            assert!(check_relativeinit(true, false, None, capacity, 4).is_err());
        }
        assert!(check_relativeinit(true, false, None, 2, 0).is_err());
        if usize::BITS > 32 {
            assert!(check_relativeinit(true, false, None, 2, usize::MAX).is_err());
        }
    }

    #[test]
    fn valid_improve() {
        for improvement in [-1.0, 0.0, 1.0, f32::MAX, -f32::MAX] {
            assert!(check_improvement(improvement, 0.0).is_ok());
            assert!(check_improvement(improvement, f32::MAX).is_ok());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_improvement(invalid, 0.0).is_err());
            assert!(check_improvement(0.0, invalid).is_err());
        }
        assert!(check_improvement(0.0, -f32::from_bits(1)).is_err());
    }

    #[test]
    fn identity_owner() {
        let proposal = Proposal {
            owner: 1,
            id: 0,
            slot: 3,
            index: 0,
            seed: 0,
            score: 0.0,
            length: 0.0,
        };
        let foreign = Proposal {
            owner: 2,
            ..proposal
        };
        assert_ne!(proposal, foreign);
        assert_ne!(proposal, Proposal { id: 1, ..proposal });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_order() {
        let block = ParamBlock::new(0, 0, 1, 1.0, 1.0).unwrap();
        for config in [
            TRLengthConfig::new(f64::NAN, 0.001, 0.1),
            TRLengthConfig::new(0.01, 0.0, 0.1),
            TRLengthConfig::new(0.01, 0.02, 0.1),
            TRLengthConfig::new(0.01, 0.001, 0.005),
            TRLengthConfig::new(0.01, f64::MIN_POSITIVE, 0.1),
            TRLengthConfig::new(0.01, 0.001, f64::MAX),
        ] {
            let error = SearchState::new(&[0x3f80], 0.0, 0.0, vec![block], 2, 1, config)
                .err()
                .expect("Invalid bounds must fail before accessing CUDA");
            assert!(error.contains("radius bounds"), "{error}");
        }
        assert!(check_length(TRLengthConfig::default()).is_ok());
        assert!(check_length(TRLengthConfig::new(0.01, 0.01, 0.01)).is_ok());
    }
}
