//! Candidate search and evaluation coordination.
//!
//! Region adaptation is separate from device computation. This API is experimental.
#[cfg(test)]
mod device_tests;
mod region;
#[cfg(test)]
mod region_tests;
#[cfg(test)]
mod view_tests;
pub use self::region::TrustRegion;
pub use crate::trials::{device_views, DeviceView, Parameter, Search, Trial};

use crate::trials::Ask;
use crate::trust_region::TRLengthConfig;
use crate::weights::ComputeDevice;

/// Coordinates candidate evaluation and region adaptation.
pub struct Optimizer {
    search: Search,
    region: TrustRegion,
}

impl Optimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Parameter>,
        capacity: usize,
        device: ComputeDevice,
        num_pert: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        Self::new_batch(
            base, base_value, leaves, capacity, device, num_pert, length, 1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_batch(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Parameter>,
        capacity: usize,
        device: ComputeDevice,
        num_pert: usize,
        length: TRLengthConfig,
        pending_capacity: usize,
    ) -> Result<Self, String> {
        let dimensions = leaves.iter().try_fold(0usize, |total, leaf| {
            total
                .checked_add(leaf.length)
                .ok_or("parameter count overflow")
        })?;
        let region = TrustRegion::new(dimensions, num_pert, base_value, length)?;
        Self::with_region(base, leaves, capacity, device, pending_capacity, region)
    }

    pub fn with_region(
        base: &[u8],
        leaves: Vec<Parameter>,
        capacity: usize,
        device: ComputeDevice,
        pending_capacity: usize,
        mut region: TrustRegion,
    ) -> Result<Self, String> {
        let dimensions = leaves.iter().try_fold(0usize, |total, leaf| {
            total
                .checked_add(leaf.length)
                .ok_or("parameter count overflow")
        })?;
        if dimensions != region.dimensions() {
            return Err("region dimensions must match search parameters".to_string());
        }
        let mut search = Search::new_batch(
            base,
            region.best()?,
            leaves,
            capacity,
            pending_capacity,
            device,
        )?;
        #[cfg(feature = "opencl")]
        if let Some(context) = search.opencl_context() {
            region.attach_context(context)?;
        } else {
            region.attach(search.device())?;
        }
        #[cfg(not(feature = "opencl"))]
        region.attach(search.device())?;
        bind_region(&mut search, &region);
        search.start_state(region.best()?)?;
        Ok(Self { search, region })
    }

    pub fn ask(&mut self, seeds: &[u64], mut config: Ask) -> Result<Trial, String> {
        if !matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            config.length = self.region.length()? as f32;
        }
        self.search
            .ask_sparse(seeds, self.region.num_pert(), config)
    }

    pub fn ask_stream(
        &mut self,
        base_seed: u64,
        count: usize,
        mut config: Ask,
    ) -> Result<Trial, String> {
        if !matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            config.length = self.region.length()? as f32;
        }
        self.search
            .sparse_stream(base_seed, count, self.region.num_pert(), config)
    }

    pub fn ask_batch(
        &mut self,
        seeds: &[u64],
        arms: usize,
        mut config: Ask,
    ) -> Result<Vec<Trial>, String> {
        if !matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            config.length = self.region.length()? as f32;
        }
        self.search
            .ask_batch(seeds, arms, self.region.num_pert(), config)
    }

    pub fn batch_stream(
        &mut self,
        base_seed: u64,
        arms: usize,
        candidates: usize,
        mut config: Ask,
    ) -> Result<Vec<Trial>, String> {
        if !matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            config.length = self.region.length()? as f32;
        }
        self.search
            .batch_stream(base_seed, arms, candidates, self.region.num_pert(), config)
    }

    pub fn row(&self, trial: Trial) -> Result<Vec<u8>, String> {
        self.search.row(trial)
    }

    /// Borrow the encoded row without downloading its contents.
    ///
    /// Complete evaluator work before releasing the borrow and updating the optimizer.
    pub fn device_view(&self, trial: Trial) -> Result<DeviceView<'_>, String> {
        self.search.device_view(trial)
    }

    pub fn device_views(&self, trials: &[Trial]) -> Result<Vec<DeviceView<'_>>, String> {
        device_views(&self.search, trials)
    }

    /// Evaluate a device row and advance state only after successful evaluation.
    /// An evaluator error leaves the trial pending, so it can be retried.
    pub fn evaluate<F>(&mut self, trial: Trial, evaluator: F) -> Result<f32, String>
    where
        F: FnOnce(DeviceView<'_>) -> Result<f32, String>,
    {
        let value = evaluator(self.device_view(trial)?)?;
        self.observe(trial, value)?;
        Ok(value)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_row(&self, trial: Trial) -> Result<(u64, usize, usize), String> {
        self.search.device_row(trial)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_batch(&self, trials: &[Trial]) -> Result<Vec<(u64, usize, usize)>, String> {
        self.search.device_batch(trials)
    }

    /// Advance search state without returning acceptance telemetry.
    pub fn observe(&mut self, trial: Trial, value: f32) -> Result<(), String> {
        if matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            self.search.observe(trial, value)
        } else {
            self.tell(trial, value).map(|_| ())
        }
    }

    pub fn tell(&mut self, trial: Trial, value: f32) -> Result<bool, String> {
        if matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            self.search.observe(trial, value)?;
            return self.region.accepted();
        }
        if !value.is_finite() {
            return Err("trial value must be finite".to_string());
        }
        self.search.check_pending(&[trial])?;
        let remaining = self.search.pending_len() - 1;
        let accepted = self.region.prepare(value, remaining)?;
        self.search.tell(trial, value, accepted)?;
        if self.region.finish(value, remaining)? {
            self.search.restart(self.region.best()?)?;
        }
        bind_region(&mut self.search, &self.region);
        Ok(accepted)
    }

    pub fn tell_batch(&mut self, trials: &[Trial], values: &[f32]) -> Result<Vec<bool>, String> {
        if trials.is_empty() || trials.len() != values.len() {
            return Err("batch trials and values must have the same non-zero length".to_string());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("batch values must be finite".to_string());
        }
        self.search.check_pending(trials)?;
        trials
            .iter()
            .zip(values)
            .map(|(trial, value)| self.tell(*trial, *value))
            .collect()
    }

    pub fn region(&self) -> &TrustRegion {
        &self.region
    }

    pub fn length(&self) -> Result<f64, String> {
        self.region.length()
    }
    pub fn probability(&self) -> f64 {
        self.region.probability()
    }
    pub fn best(&self) -> Result<f32, String> {
        self.region.best()
    }
    pub fn restarts(&self) -> Result<usize, String> {
        self.region.restarts()
    }
    pub fn history_len(&self) -> Result<usize, String> {
        if matches!(
            self.search.device(),
            ComputeDevice::Metal | ComputeDevice::OpenCl
        ) {
            self.search.state_word(1).map(|count| count as usize)
        } else {
            Ok(self.search.history_len())
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn leaves() -> Vec<Parameter> {
        vec![Parameter::new(0, 100, 8, 0.1, 1.0, 0.1).unwrap()]
    }

    fn hash_seed(seed: u64, element: u32) -> u32 {
        let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= (seed >> 32) as u32;
        value = value.wrapping_mul(0x846c_a68b);
        value ^ (value >> 15)
    }

    fn stream_seed(base: u64, index: u32) -> u64 {
        let low = hash_seed(base, index) as u64;
        let high = hash_seed(base, index ^ 0xa511_e9b3) as u64;
        low | (high << 32)
    }

    #[test]
    fn owns_state() {
        let mut search = Optimizer::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
        )
        .unwrap();
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let trial = search.ask(&[7, 11], ask).unwrap();
        assert_eq!(search.probability(), 0.2);
        assert!(search.tell(trial, 1.0).unwrap());
        assert_eq!(search.best().unwrap(), 1.0);
        let trial = search.ask(&[13, 17], ask).unwrap();
        assert!(!search.tell(trial, 0.5).unwrap());
        assert_eq!(search.history_len().unwrap(), 3);
    }

    #[test]
    fn trial_identity() {
        let create = || {
            Optimizer::new(
                &[8; 100],
                0.0,
                leaves(),
                8,
                ComputeDevice::Cpu,
                20,
                TRLengthConfig::default(),
            )
            .unwrap()
        };
        let (mut first, mut second) = (create(), create());
        let config = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let left = first.ask(&[7], config).unwrap();
        let right = second.ask(&[7], config).unwrap();
        assert_ne!(left, right);
        assert!(first.row(right).is_err());
        assert!(first.tell(right, 1.0).is_err());
        assert_eq!(first.history_len().unwrap(), 1);
        let mut diagnostic = left;
        diagnostic.score = f32::NAN;
        assert_eq!(left, diagnostic);
        assert!(first.tell_batch(&[left, diagnostic], &[1.0, 2.0]).is_err());
        assert_eq!(first.history_len().unwrap(), 1);
        first.tell(left, 1.0).unwrap();
        assert!(first.tell(left, 2.0).is_err());
        second.tell(right, 1.0).unwrap();
    }

    #[test]
    fn adapts_length() {
        let mut search = Optimizer::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
        )
        .unwrap();
        let initial = search.length().unwrap();
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        for (seed, reward) in [(7, 1.0), (11, 2.0), (13, 3.0), (17, 4.0)] {
            let trial = search.ask(&[seed], ask).unwrap();
            assert!(search.tell(trial, reward).unwrap());
        }
        assert!(search.length().unwrap() > initial);
    }

    #[test]
    fn batches_trials() {
        let mut search = Optimizer::new_batch(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
            2,
        )
        .unwrap();
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let trials = search.ask_batch(&[7, 11, 13, 17], 2, ask).unwrap();
        assert_eq!(trials.len(), 2);
        assert!(search.ask(&[19], ask).is_err());
        assert!(search.tell(trials[1], 1.0).unwrap());
        assert!(!search.tell(trials[0], 0.5).unwrap());
        assert_eq!(search.history_len().unwrap(), 3);
        assert_eq!(search.best().unwrap(), 1.0);
    }

    #[test]
    fn stream_sparse() {
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let seeds = (0..4)
            .map(|index| stream_seed(0x1234_5678_9abc_def0, index))
            .collect::<Vec<_>>();
        let mut explicit = Optimizer::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
        )
        .unwrap();
        let mut streamed = Optimizer::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
        )
        .unwrap();
        let left = explicit.ask(&seeds, ask).unwrap();
        let right = streamed
            .ask_stream(0x1234_5678_9abc_def0, seeds.len(), ask)
            .unwrap();
        assert_eq!(left.index, right.index);
        assert_eq!(left.seed, right.seed);
        assert_eq!(explicit.row(left).unwrap(), streamed.row(right).unwrap());
    }

    fn assert_trial(left: Trial, right: Trial) {
        assert_eq!(left.index, right.index);
        assert_eq!(left.seed, right.seed);
        assert_eq!(left.score.to_bits(), right.score.to_bits());
    }

    fn new_optimizer(device: ComputeDevice) -> Result<Optimizer, String> {
        Optimizer::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            device,
            20,
            TRLengthConfig::default(),
        )
    }

    fn trajectory(device: ComputeDevice, acquisition: crate::weights::AcquisitionKind) {
        let mut cpu = new_optimizer(ComputeDevice::Cpu).unwrap();
        let mut target = new_optimizer(device).unwrap();
        let config = Ask {
            acquisition,
            neighbors: 1,
            seed: 0xfeed_beef,
            ..Ask::default()
        };
        for round in 0..6 {
            let base_seed = 0x1234_5678_9abc_def0 ^ (round as u64).wrapping_mul(0x9e37_79b9);
            let left = cpu.ask_stream(base_seed, 5, config).unwrap();
            let right = target.ask_stream(base_seed, 5, config).unwrap();
            assert_trial(left, right);
            assert_eq!(cpu.row(left).unwrap(), target.row(right).unwrap());

            let value = [1.0, 0.5, 1.25, 0.25, 1.5, 0.75][round];
            assert_eq!(
                cpu.tell(left, value).unwrap(),
                target.tell(right, value).unwrap()
            );
            assert_eq!(
                cpu.best().unwrap().to_bits(),
                target.best().unwrap().to_bits()
            );
            assert_eq!(
                cpu.length().unwrap().to_bits(),
                target.length().unwrap().to_bits()
            );
            assert_eq!(cpu.restarts().unwrap(), target.restarts().unwrap());
            assert_eq!(cpu.history_len().unwrap(), target.history_len().unwrap());
        }
    }

    fn trajectory_all(device: ComputeDevice) {
        for acquisition in [
            crate::weights::AcquisitionKind::Ucb,
            crate::weights::AcquisitionKind::Thompson,
            crate::weights::AcquisitionKind::Pareto,
        ] {
            trajectory(device, acquisition);
        }
    }

    #[test]
    fn stream_batch() {
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let mut search = Optimizer::new_batch(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
            2,
        )
        .unwrap();
        let trials = search
            .batch_stream(0x1234_5678_9abc_def0, 2, 3, ask)
            .unwrap();
        assert_eq!(trials.len(), 2);
        assert_ne!(trials[0].seed, trials[1].seed);
        assert!(search.ask_stream(9, 2, ask).is_err());
    }

    #[test]
    fn tells_batch() {
        let mut search = Optimizer::new_batch(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeDevice::Cpu,
            20,
            TRLengthConfig::default(),
            2,
        )
        .unwrap();
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let trials = search.ask_batch(&[7, 11, 13, 17], 2, ask).unwrap();
        assert_eq!(
            search.tell_batch(&trials, &[0.5, 1.0]).unwrap(),
            [true, true]
        );
        assert_eq!(search.best().unwrap(), 1.0);
        assert_eq!(search.history_len().unwrap(), 3);
    }

    #[test]
    fn cpu_trajectory() {
        trajectory_all(ComputeDevice::Cpu);
    }

    #[test]
    fn metal_trajectory() {
        match new_optimizer(ComputeDevice::Metal) {
            Ok(_) => trajectory_all(ComputeDevice::Metal),
            Err(error) => eprintln!("skipping Metal trajectory test: {error}"),
        }
    }

    #[test]
    fn opencl_trajectory() {
        match new_optimizer(ComputeDevice::OpenCl) {
            Ok(_) => trajectory_all(ComputeDevice::OpenCl),
            Err(error) => eprintln!("skipping OpenCL trajectory test: {error}"),
        }
    }
}

#[allow(unused_variables)]
fn bind_region(search: &mut Search, region: &TrustRegion) {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Some(buffer) = region.metal_buffer() {
        search.metal_region(buffer);
    }
    #[cfg(feature = "opencl")]
    if let Some(buffer) = region.opencl_buffer() {
        search.opencl_region(buffer);
    }
}
