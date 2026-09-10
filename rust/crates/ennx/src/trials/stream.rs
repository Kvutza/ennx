use super::{check_count, tree, trial_id, Center, Pending, Search, Trial};
use crate::trials::{cpu, Ask};

impl Search {
    pub fn ask_stream(
        &mut self,
        base_seed: u64,
        count: usize,
        config: Ask,
    ) -> Result<Trial, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        check_count(count, self.history_bound(), config)?;
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history: Vec<(usize, f32)> = self.history_view();
        let (index, seed, score) = self.engine.ask_stream(
            self.base,
            &history,
            slot,
            base_seed,
            count,
            &self.leaves,
            config,
            true,
        )?;
        let id = trial_id()?;
        self.pending.push(Pending {
            id,
            slot,
            seed,
            length: config.length,
            materialized: true,
        });
        Ok(Trial {
            id,
            index,
            seed,
            score,
        })
    }

    pub fn sparse_stream(
        &mut self,
        base_seed: u64,
        count: usize,
        num_pert: usize,
        config: Ask,
    ) -> Result<Trial, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        check_count(count, self.history_bound(), config)?;
        check_sparse(num_pert, &self.leaves)?;
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history: Vec<(usize, f32)> = self.history_view();
        let (index, seed, score) = self.engine.sparse_stream(
            self.base,
            &history,
            slot,
            base_seed,
            count,
            num_pert,
            &self.leaves,
            config,
        )?;
        let id = trial_id()?;
        self.pending.push(Pending {
            id,
            slot,
            seed,
            length: config.length,
            materialized: true,
        });
        Ok(Trial {
            id,
            index,
            seed,
            score,
        })
    }

    pub fn regions_stream(
        &mut self,
        num_regions: usize,
        seeds_per_region: usize,
        base_seed: u64,
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        if num_regions == 0 || seeds_per_region == 0 {
            return Err("multi-TR search requires non-zero regions and candidates".to_string());
        }
        let count = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        check_count(count, self.history_bound(), config)?;
        let history: Vec<(usize, f32)> = self.history_view();
        self.engine.regions_stream(
            self.base,
            &history,
            num_regions,
            seeds_per_region,
            base_seed,
            &self.leaves,
            config,
        )
    }

    pub fn centers_stream(
        &mut self,
        num_regions: usize,
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        base_seed: u64,
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        tree::check(centers, region_centers, num_regions)?;
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        let count = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        if seeds_per_region == 0 {
            return Err("multi-TR search requires non-zero regions and candidates".to_string());
        }
        check_count(count, self.history_bound(), config)?;
        let history: Vec<(usize, f32)> = self.history_view();
        self.engine.centers_stream(
            self.base,
            &history,
            seeds_per_region,
            centers,
            region_centers,
            base_seed,
            &self.leaves,
            config,
        )
    }

    pub(crate) fn batch_stream(
        &mut self,
        base_seed: u64,
        arms: usize,
        candidates: usize,
        num_pert: usize,
        config: Ask,
    ) -> Result<Vec<Trial>, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish outstanding trials before batch ask".to_string());
        }
        if arms == 0 || arms > self.pending_capacity {
            return Err(format!(
                "batch arms must be in 1..={}, got {arms}",
                self.pending_capacity
            ));
        }
        check_count(candidates, self.history_bound(), config)?;
        check_sparse(num_pert, &self.leaves)?;
        let mut trials = Vec::with_capacity(arms);
        for arm in 0..arms {
            match self.batch_arm(base_seed, arm, candidates, num_pert, config) {
                Ok(trial) => trials.push(trial),
                Err(error) => {
                    self.pending.clear();
                    return Err(error);
                }
            }
        }
        Ok(trials)
    }

    fn batch_arm(
        &mut self,
        base_seed: u64,
        arm: usize,
        candidates: usize,
        num_pert: usize,
        config: Ask,
    ) -> Result<Trial, String> {
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history: Vec<(usize, f32)> = self.history_view();
        let arm_seed = cpu::seed_at(base_seed, arm as u32);
        let (index, seed, score) = self.engine.sparse_stream(
            self.base,
            &history,
            slot,
            arm_seed,
            candidates,
            num_pert,
            &self.leaves,
            config,
        )?;
        let id = trial_id()?;
        self.pending.push(Pending {
            id,
            slot,
            seed,
            length: config.length,
            materialized: true,
        });
        Ok(Trial {
            id,
            index,
            seed,
            score,
        })
    }
}

fn check_sparse(num_pert: usize, leaves: &[super::Parameter]) -> Result<(), String> {
    let dimensions = leaves.iter().map(|leaf| leaf.length).sum::<usize>();
    if num_pert == 0 || num_pert > dimensions {
        return Err(format!(
            "num_pert must be between one and {dimensions}, got {num_pert}"
        ));
    }
    Ok(())
}
