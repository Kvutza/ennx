use super::*;
use crate::procedural_pool::{ProceduralPool, ProceduralPoolConfig};

#[cfg(test)]
#[path = "procedural_tests.rs"]
mod tests;

impl SearchState {
    pub fn configure_pool(&mut self, pool: ProceduralPool) -> Result<(), String> {
        self.check_idle()?;
        ProceduralPoolConfig {
            arms: pool.arms(),
            candidates_per_arm: pool.slots(),
        }
        .resolve_resident()?;
        if self.started || self.objective_selector.is_some() {
            return Err(
                "Configure procedural layout before asking or configuring vector acquisition"
                    .into(),
            );
        }
        if pool == self.pool_layout {
            return Ok(());
        }
        let source = self.procedural_source(pool);
        let propose = self
            .runtime
            .pipeline(&source, "Procedural search", "bf16_propose")?;
        let initial =
            self.runtime
                .pipeline(&source, "Procedural search", "bf16_propose_initial")?;
        let pool_pipeline =
            self.runtime
                .pipeline(&source, "Procedural search", "bf16_propose_pool")?;
        let short = self
            .runtime
            .precise(&source, "Procedural search", "bf16_replay_short")?;
        let replay = self
            .runtime
            .precise(&source, "Procedural search", "bf16_replay_history")?;
        let select = self
            .runtime
            .precise(&source, "Procedural search", "bf16_select")?;
        let materialize =
            self.runtime
                .pipeline(&source, "Procedural search", "bf16_materialize")?;
        self.propose_pipeline = propose;
        self.initial_pipeline = initial;
        self.pool_pipeline = pool_pipeline;
        self.replay_short_pipeline = short;
        self.replay_pipeline = replay;
        self.selection_pipeline = select;
        self.materialize_pipeline = materialize;
        self.pool_layout = pool;
        Ok(())
    }

    pub(super) fn procedural_seed(&self, root: u64, candidate: usize) -> u64 {
        self.pool_layout
            .seed(root, self.pool_layout.identity(candidate as u32).unwrap())
            .unwrap()
    }

    pub(super) fn procedural_source(&self, pool: ProceduralPool) -> String {
        let source = initialization::search_source(self.independent_fp16, self.perturbation);
        if pool == ProceduralPool::legacy() {
            source
        } else {
            format!("#define PROCEDURAL_SLOTS {}\n{source}", pool.slots())
        }
    }
}
