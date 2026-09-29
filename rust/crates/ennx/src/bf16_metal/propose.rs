use super::*;
use crate::apple_gpu::gpu_interval;

impl SearchState {
    /// Advance the resident proposal state from its bounded history.
    ///
    /// Initialization, neighbor clamping, asynchronous selection, and proposal
    /// publication are one search transaction. `forced` is reserved for a
    /// controlled selection ablation; normal optimization passes `None`.
    pub fn propose(
        &mut self,
        seed: u64,
        mut config: Ask,
        initial_slot: usize,
        forced: Option<usize>,
    ) -> Result<(Buffer, Proposals, bool), String> {
        self.compact_history()?;
        let history = self.history_len()?;
        let initializing = history < config.neighbors;
        let row = if initializing {
            self.begin_initial(seed, initial_slot)?
        } else {
            config.neighbors = config.neighbors.min(history);
            match forced {
                Some(slot) => self.begin_forced(seed, config, slot)?,
                None => self.begin_ask(1, 4, seed, config)?,
            }
        };
        let proposal = self.finish_ask()?;
        Ok((row, proposal, initializing))
    }

    /// Score, select and materialize four correlated BF16 candidates on the GPU.
    pub fn ask_round(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
    ) -> Result<Proposals, String> {
        autoreleasepool(|| self.ask_inner(arms, candidates, seed, config))
    }

    /// Enqueue candidate selection and materialization without waiting. The
    /// caller may append work on the shared Metal queue before finalizing the
    /// proposal. Queue ordering makes the selected row visible to that work.
    pub fn begin_ask(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
    ) -> Result<Buffer, String> {
        self.begin_mode(arms, candidates, seed, config, None)
    }

    /// Selection ablation: preserve the pool, posterior and tell
    /// policy, but choose a precommitted pool index instead of maximizing UCB.
    pub(crate) fn begin_forced(
        &mut self,
        seed: u64,
        config: Ask,
        candidate: usize,
    ) -> Result<Buffer, String> {
        if candidate >= 4 {
            return Err("Diagnostic candidate must be below four".into());
        }
        self.begin_mode(1, 4, seed, config, Some(candidate))
    }

    /// Materialize a counterfactual from an already-scored pool. Diagnostics
    /// must restore round.index before tell; no observation is added here.
    #[cfg(test)]
    pub(crate) fn diagnostic_row(
        &self,
        round: &Proposals,
        root: u64,
        config: Ask,
        candidate: usize,
    ) -> Result<Buffer, String> {
        self.check_round(round)?;
        let decision = read::<Decision>(&self.decision, 1)[0];
        if candidate >= 4 || root != decision.root_seed {
            return Err("Diagnostic pool candidate or root mismatch".into());
        }
        let command = self.runtime.queue.new_command_buffer();
        self.encode_select(command, root, config, Some(candidate));
        self.encode_row(command);
        finish(command)?;
        Ok(self.proposal.clone())
    }

    pub(crate) fn begin_initial(&mut self, seed: u64, candidate: usize) -> Result<Buffer, String> {
        if !self.implicit_history
            || self.initial_observations == 0
            || self.history >= self.initial_observations
        {
            return Err("Metal ENN initialization is complete or not configured".into());
        }
        if candidate >= 4 {
            return Err("Metal initialization candidate must be below four".into());
        }
        if !self.profiling && self.resident_history <= 2 {
            return self.initial_row(seed, candidate);
        }
        self.begin_mode(1, 4, seed, Ask::default(), Some(candidate))
    }

    pub(super) fn initial_row(&mut self, root: u64, candidate: usize) -> Result<Buffer, String> {
        self.check_idle()?;
        self.activate_morbo()?;
        self.compact_history()?;
        let basis_seed = self.prepare_threshold(root)?;
        self.ensure_ref()?;
        unsafe {
            std::ptr::write_bytes(
                self.partials.contents().cast::<u8>(),
                0,
                self.partials.length() as usize,
            );
            std::ptr::write_bytes(
                self.pool_geometry.contents().cast::<u8>(),
                0,
                self.pool_geometry.length() as usize,
            );
        }
        let radius = self.radius(candidate);
        let params = Params {
            seed: self.procedural_seed(root, candidate),
            basis_seed,
            radius,
            alternate_radius: radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            program: u32::from(self.threshold.is_some()),
            base_slot: self.resident_identities[..self.resident_history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .and_then(|slot| u32::try_from(slot).ok())
                .unwrap_or(u32::MAX),
            ..Params::default()
        };
        let command = self.runtime.queue.new_command_buffer();
        self.encode_proposal(command, params);
        self.encode_select(command, root, Ask::default(), Some(candidate));
        command.commit();
        self.async_command = Some(command.to_owned());
        Ok(self.proposal.clone())
    }

    pub(super) fn begin_mode(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
        forced_candidate: Option<usize>,
    ) -> Result<Buffer, String> {
        self.check_idle()?;
        self.activate_morbo()?;
        self.compact_history()?;
        self.prepare_threshold(seed)?;
        if self.history == 0 {
            return Err("Measure the initial Metal BF16 incumbent before asking".into());
        }
        if !((arms == 1 && candidates == 4)
            || (arms == self.pool_layout.arms() as usize
                && candidates == self.pool_layout.slots() as usize))
        {
            return Err("Metal ask shape must match its configured four-candidate procedural layout or use the flat 1x4 view".into());
        }
        check_ask(config)?;
        self.validate_selector()?;
        self.last_profile = None;
        self.ensure_ref()?;
        let analytic_pool = self.analytic_pool();
        if self.profiling {
            self.profile_commands.clear();
            if !analytic_pool {
                let command = self.runtime.queue.new_command_buffer().to_owned();
                self.encode_pool(&command, self.pool_params(seed));
                command.set_label("BF16 proposal pool");
                command.commit();
                self.profile_commands.push(command);
            }
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_select(&command, seed, config, forced_candidate);
            command.set_label("BF16 proposal selection");
            command.commit();
            self.profile_commands.push(command);
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_row(&command);
            command.set_label("BF16 proposal materialization");
            command.commit();
            self.profile_commands.push(command);
            self.async_command = self.profile_commands.last().cloned();
            return Ok(self.proposal.clone());
        }
        let command = self.runtime.queue.new_command_buffer();
        if !analytic_pool
            && !(self.realized_history()
                && self.history > self.resident_history
                && self.history <= 16)
        {
            self.encode_pool(command, self.pool_params(seed));
        }
        self.encode_select(command, seed, config, forced_candidate);
        self.encode_row(command);
        command.set_label("BF16 resident proposal");
        command.commit();
        self.async_command = Some(command.to_owned());
        Ok(self.proposal.clone())
    }

    /// Wait for a previously enqueued ask and publish its immutable handle.
    pub fn finish_ask(&mut self) -> Result<Proposals, String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.poisoned = true;
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_ask()
    }

    /// Abort an asynchronous ask after a downstream evaluator failure.
    pub fn abort_ask(&mut self) {
        if let Some(command) = self.async_command.take() {
            command.wait_until_completed();
        }
        self.poisoned = true;
    }

    pub(super) fn ask_inner(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
    ) -> Result<Proposals, String> {
        let start = Instant::now();
        self.begin_ask(arms, candidates, seed, config)?;
        self.finish_askat(start)
    }

    pub(super) fn finish_askat(&mut self, start: Instant) -> Result<Proposals, String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.poisoned = true;
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_start(start)
    }

    pub(super) fn finish_profile(&mut self) -> Result<(), String> {
        if !self.profiling {
            self.last_profile = None;
            return Ok(());
        }
        let analytic_pool = self.analytic_pool();
        let expected = if analytic_pool { 2 } else { 3 };
        if self.profile_commands.len() != expected {
            return Err("Metal BF16 controller trace is incomplete".into());
        }
        let intervals: Vec<_> = self
            .profile_commands
            .iter()
            .map(|command| {
                if command.status() != MTLCommandBufferStatus::Completed {
                    return Err(format!(
                        "Metal BF16 traced ask failed: {:?}",
                        command.status()
                    ));
                }
                gpu_interval(command)
                    .ok_or_else(|| "Metal did not expose a controller GPU interval".to_string())
            })
            .collect::<Result<_, _>>()?;
        let milliseconds =
            |index: usize| ((intervals[index].1 - intervals[index].0) * 1000.0) as f32;
        let (score_ms, pick_ms, materialize_ms, total_ms) = if analytic_pool {
            (
                0.0,
                milliseconds(0),
                milliseconds(1),
                ((intervals[1].1 - intervals[0].0) * 1000.0) as f32,
            )
        } else {
            (
                milliseconds(0),
                milliseconds(1),
                milliseconds(2),
                ((intervals[2].1 - intervals[0].0) * 1000.0) as f32,
            )
        };
        self.last_profile = Some(AskProfile {
            score_ms,
            pick_ms,
            materialize_ms,
            total_ms,
        });
        self.profile_commands.clear();
        Ok(())
    }
}
