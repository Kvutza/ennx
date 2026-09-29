use super::*;

impl ConfigOverrides {
    pub fn model_seed(&self) -> u64 {
        self.model_seeded(0)
    }

    pub fn model_seeded(&self, rep: u32) -> u64 {
        self.model_seed
            .unwrap_or_else(|| self.derived_seed(rep, "model", 0))
    }

    pub fn sample_seeded(&self, rep: u32) -> u64 {
        self.generation
            .as_ref()
            .and_then(|config| config.seed)
            .unwrap_or_else(|| self.derived_seed(rep, "sampling", 0))
    }

    pub fn reference_seed(&self) -> u64 {
        self.reference_seed
            .unwrap_or_else(|| self.derived_seed(0, "reference", 0))
    }

    pub fn proposal_seed(&self) -> u64 {
        self.proposal_seeded(0)
    }

    pub fn proposal_seeded(&self, rep: u32) -> u64 {
        self.proposal_seed
            .unwrap_or_else(|| self.derived_seed(rep, "proposal", 0))
    }

    pub fn acquisition_seed(&self) -> u64 {
        self.acquisition_seeded(0)
    }

    pub fn acquisition_seeded(&self, rep: u32) -> u64 {
        self.acquisition_seed
            .unwrap_or_else(|| self.derived_seed(rep, "acquisition", 0))
    }

    pub(crate) fn derived_seed(&self, rep: u32, domain: &str, index: u64) -> u64 {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |bytes: &[u8]| {
            state ^= bytes.len() as u64;
            state = crate::hash::splitmix64(state);
            for byte in bytes {
                state ^= u64::from(*byte);
                state = state.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        mix(b"ennx-tune-seed-v2");
        mix(&[
            experiment_code(self.experiment),
            model_code(self.model),
            corpus_code(self.corpus),
        ]);
        mix(domain.as_bytes());
        mix(&rep.to_le_bytes());
        mix(&index.to_le_bytes());
        crate::hash::splitmix64(state) & 0x3fff_ffff_ffff_ffff
    }
}

fn experiment_code(experiment: Option<TurboEnnExperiment>) -> u8 {
    match experiment {
        None => 0,
        Some(TurboEnnExperiment::EndToEnd) => 1,
        Some(TurboEnnExperiment::MoeLayer) => 2,
        Some(TurboEnnExperiment::Pretrain) => 3,
        Some(TurboEnnExperiment::Generation) => 4,
    }
}

fn model_code(model: Option<PretrainModel>) -> u8 {
    match model {
        None => 0,
        Some(PretrainModel::FbtPisa1MoeV1) => 1,
        Some(PretrainModel::FbtPisa1Residual1V1) => 2,
        Some(PretrainModel::FbtPisa1ProjectedBoundaryV1) => 3,
        Some(PretrainModel::FbtPisa1Hc4V1) => 4,
        Some(PretrainModel::FbtPisa1Mhc4V1) => 5,
        Some(PretrainModel::FbtPisa1LoopedMhc4V1) => 6,
        Some(PretrainModel::FbtPisa1DiffusionMhc4V1) => 7,
    }
}

fn corpus_code(corpus: Option<PretrainCorpus>) -> u8 {
    match corpus {
        None => 0,
        Some(PretrainCorpus::StackV3PythonPilotV1) => 1,
        Some(PretrainCorpus::Fineweb10btPilotV1) => 2,
        Some(PretrainCorpus::StackV3Python800kV1) => 3,
    }
}
