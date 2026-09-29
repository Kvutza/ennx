use super::generation::*;
use crate::traits::Oracle;

pub(super) struct LoopState {
    pub(super) search: SearchState,
    pub(super) updates: UpdateLog,
    pub(super) journal: journal::Journal,
    pub(super) records: File,
    pub(super) walls: Vec<f64>,
    pub(super) minimum_changed_weights: u64,
    pub(super) minimum_generated_tokens: usize,
    pub(super) accepted: u32,
    pub(super) incumbent_version: u64,
    pub(super) incumbent_rollouts: Vec<decode::Rollout>,
    pub(super) regions: RegionState,
    pub(super) temperature: f32,
    pub(super) loop_seconds: f64,
    pub(super) rank_audit: Option<RankAudit>,
    pub(super) causal_value: Option<f32>,
}

pub(super) struct RegionState {
    versions: Vec<u64>,
    rollouts: Vec<Vec<decode::Rollout>>,
    temperatures: Vec<f32>,
}

impl RegionState {
    pub(super) fn initial(
        search: &SearchState,
        model_seed: u64,
        rollouts: &[decode::Rollout],
        temperature: f32,
    ) -> Self {
        let count = search.morbo_regions().max(1);
        let version = crate::hash::splitmix64(model_seed ^ 0x656e_6e78_2d62_6173);
        Self {
            versions: vec![version; count],
            rollouts: vec![rollouts.to_vec(); count],
            temperatures: vec![temperature; count],
        }
    }
}

impl LoopState {
    fn commit(
        &mut self,
        evaluation: &Evaluation,
        accepted: bool,
        region: usize,
        version: u64,
        temperature: f32,
    ) {
        self.minimum_generated_tokens = self.minimum_generated_tokens.min(
            evaluation
                .rollouts
                .iter()
                .map(|rollout| rollout.tokens.len())
                .min()
                .unwrap_or(0),
        );
        if accepted {
            self.accept_region(region, version, &evaluation.rollouts, temperature);
            if let Some(causal) = evaluation.causal {
                self.causal_value = Some(causal.optimizer_value);
            }
        }
        self.temperature = self.regions.temperatures[region];
        self.accepted += u32::from(accepted);
    }

    fn accept_region(
        &mut self,
        region: usize,
        version: u64,
        rollouts: &[decode::Rollout],
        temperature: f32,
    ) {
        self.incumbent_version = version;
        self.incumbent_rollouts = rollouts.to_vec();
        self.regions.versions[region] = version;
        self.regions.rollouts[region] = rollouts.to_vec();
        self.regions.temperatures[region] = temperature;
    }

    pub(super) fn finalize_state(&mut self, record: bool, output: &Path) -> Result<(), String> {
        if let Some(region) = self.search.finalize_morbo()? {
            self.incumbent_version = self.regions.versions[region];
            self.incumbent_rollouts = self.regions.rollouts[region].clone();
            self.temperature = self.regions.temperatures[region];
        }
        if record {
            self.updates.write(&output.join("tensor-updates.jsonl"))?;
        }
        Ok(())
    }

    fn record_rank(&mut self, evaluation: &Evaluation, record: &mut ennx_wire::json::Value) {
        if let (Some(audit), Some(nll)) = (
            self.rank_audit.as_mut(),
            evaluation
                .rollouts
                .first()
                .and_then(|rollout| rollout.free_running_target_nll),
        ) {
            record["model_selection_audit"] = audit.observe(-nll, evaluation.mean);
        }
    }

    fn record_updates(
        &mut self,
        enabled: bool,
        step: u32,
        proposal: &crate::bf16_metal::Proposals,
        accepted: bool,
        changes: Vec<(u64, f64)>,
    ) -> Result<(), String> {
        if enabled {
            self.updates.push_scaled(
                step,
                proposal.seed,
                proposal.length,
                accepted,
                changes,
                proposal.block_scales.clone(),
            )?;
        }
        Ok(())
    }
}

pub(super) struct LoopContext<'a> {
    pub(super) run: &'a ConfigOverrides,
    pub(super) config: &'a GenerationConfig,
    pub(super) output: &'a Path,
    pub(super) runtime: &'a Runtime,
    pub(super) decoder: &'a decode::Decoder,
    pub(super) verifier: &'a Option<block_decode::BlockDecoder>,
    pub(super) weights: &'a CandidateWeights,
    pub(super) sampling_seed: u64,
    pub(super) tokenizer: Option<&'a ByteDecoder>,
    pub(super) evaluator: &'a mut Option<Evaluator>,
    pub(super) validation_tasks: &'a [GenerationTask],
    pub(super) policy: crate::config::ResidentEnnConfig,
    pub(super) proposal_seed: u64,
    pub(super) acquisition_seed: u64,
    pub(super) causal: &'a mut Option<CausalScorer>,
    pub(super) causal_log: &'a mut Option<CausalLog>,
}

impl LoopContext<'_> {
    fn annotate_causal(
        evaluation: &Evaluation,
        record: &mut ennx_wire::json::Value,
    ) -> Result<(), String> {
        if let Some(causal) = evaluation.causal {
            record["causal_pretraining"] =
                ennx_wire::json::to_value(causal).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn validate_causal(&mut self, state: &LoopState, round: u32) -> Result<(), String> {
        if let (Some(scorer), Some(log)) = (self.causal.as_ref(), self.causal_log.as_mut()) {
            let base = state.search.base_buffer();
            log.observe(scorer, self.runtime, self.weights.row(&base)?, round)?;
        }
        Ok(())
    }

    fn evaluate(
        &mut self,
        state: &mut LoopState,
        row: &metal::Buffer,
        proposal: &crate::bf16_metal::Proposals,
        path: &Path,
        initial: bool,
        step: u32,
    ) -> Result<(Evaluation, f32, crate::bf16_metal::NoisyDecision, Duration), String> {
        let region = state.search.morbo_region().unwrap_or(0);
        let temperature = proposal
            .axis_value()
            .unwrap_or(state.regions.temperatures[region]);
        let mut config = self.config.clone();
        config.temperature = temperature;
        let candidate = crate::search::DeviceView::metal(row.clone(), 0, row.length() as usize);
        let incumbent_buffer = state.search.base_buffer();
        let causal = match (self.causal.as_ref(), state.causal_value) {
            (Some(scorer), Some(baseline)) => Some(CausalObjective {
                scorer,
                incumbent: self.weights.row(&incumbent_buffer)?,
                batch: step % scorer.batches(),
                baseline,
            }),
            (None, None) => None,
            _ => return Err("causal pretraining state is incomplete".into()),
        };
        let mut oracle = FbtOracle {
            runtime: self.runtime,
            decoder: self.decoder,
            verifier: self
                .verifier
                .as_ref()
                .map(|verifier| (verifier, Some(state.regions.rollouts[region].as_slice()))),
            weights: self.weights,
            config: &config,
            seed: self.sampling_seed,
            path,
            evaluator: self.evaluator.as_mut(),
            tokenizer: self.tokenizer,
            audit: config.verify.audit_candidate,
            causal,
            vector_control: self.run.objective_acquisition.is_some(),
        };
        let (observation, evaluation) = oracle.observe(candidate)?;
        let tell_start = Instant::now();
        let decision = if initial {
            state.search.initial_objectives(proposal, observation)?
        } else if observation.is_vector() {
            state.search.modeled_objectives(proposal, observation)?
        } else if let Some(causal) = evaluation.causal {
            state.search.paired_modeled(
                proposal,
                causal.optimizer_value,
                causal.variance,
                state.search.best()?,
                state.search.best_variance()?,
                causal.improvement,
                causal.variance,
            )?
        } else {
            state.search.modeled_objectives(proposal, observation)?
        };
        if state.search.sync()? != vec![decision.accepted] {
            return Err("resident search acceptance mismatch".into());
        }
        let tell_time = tell_start.elapsed();
        Ok((evaluation, temperature, decision, tell_time))
    }

    fn propose_step(
        &self,
        state: &mut LoopState,
        step: u32,
    ) -> Result<(metal::Buffer, crate::bf16_metal::Proposals, bool), String> {
        let mut ask = self.policy.ask;
        ask.seed = self.acquisition_seed + u64::from(step);
        let forced = (self.run.selection == Some(crate::config::PretrainSelection::Random))
            .then(|| crate::hash::splitmix64(ask.seed) as usize % 4);
        state.search.propose(
            self.proposal_seed + u64::from(step),
            ask,
            step as usize % 4,
            forced,
        )
    }
    pub(super) fn run(&mut self, state: &mut LoopState) -> Result<(), String> {
        for step in 0..self.run.rounds() {
            let start = Instant::now();
            let (row, proposal, initializing) = self.propose_step(state, step)?;
            let region = state.search.morbo_region().unwrap_or(0);
            let tensor_version = state
                .search
                .tensor_version(state.regions.versions[region], &proposal)?;
            let proposal_seconds = start.elapsed().as_secs_f64();
            let changes = state.search.describe(&proposal)?.remove(0).3;
            let round_path = self.output.join(format!("round-{:04}", step + 1));
            let evaluation_start = Instant::now();
            let (evaluation, candidate_temperature, decision, tell_time) =
                self.evaluate(state, &row, &proposal, &round_path, initializing, step)?;
            let evaluation_seconds =
                evaluation_start.elapsed().as_secs_f64() - tell_time.as_secs_f64();
            let changed_weights = changes.iter().map(|change| change.0).sum::<u64>();
            state.minimum_changed_weights = state.minimum_changed_weights.min(changed_weights);
            let changed_fraction =
                changed_weights as f64 / self.weights.architecture.parameter_count() as f64;
            state.commit(
                &evaluation,
                decision.accepted,
                region,
                tensor_version.id,
                candidate_temperature,
            );
            state.record_updates(
                self.config.record_tensor_updates,
                step + 1,
                &proposal,
                decision.accepted,
                changes,
            )?;
            let text = decoded_completion(self.tokenizer, &evaluation.rollouts[0], &round_path)?;
            let mut record = controller_record(
                step + 1,
                initializing,
                &proposal,
                decision,
                state.search.controller_info()?,
                state.search.reliability_info()?,
                start.elapsed().as_secs_f64(),
                proposal_seconds + tell_time.as_secs_f64(),
            );
            record["generation"] = ennx_wire::json::json!({"reward":evaluation.mean,"variance":evaluation.variance,
            "diagnostics":evaluation.diagnostics,"selection":self.run.selection.unwrap_or_default(),
            "temperature":candidate_temperature,"incumbent_temperature":state.temperature,
            "proposal_seconds":proposal_seconds,"reward_seconds":evaluation.reward_seconds,
            "evaluation_seconds":evaluation_seconds,"tell_seconds":tell_time.as_secs_f64(),
            "draft":evaluation.rollouts.iter().map(|r| &r.draft).collect::<Vec<_>>(),
            "target_evaluated_positions":evaluation.rollouts.iter().map(|r| r.evaluated_positions - r.draft.as_ref().map_or(0, |d| d.evaluated_positions)).sum::<usize>(),
            "target_gpu_seconds":evaluation.rollouts.iter().map(|r| r.gpu_seconds - r.draft.as_ref().map_or(0.0, |d| d.generation_gpu_seconds + d.cache_gpu_seconds)).sum::<f64>(),
            "position_amplification":evaluation.rollouts.iter().map(|r| r.evaluated_positions).sum::<usize>() as f64 / evaluation.rollouts.iter().map(|r| r.tokens.len()).sum::<usize>().max(1) as f64,
            "changed_weights":changed_weights,"changed_fraction":changed_fraction,
            "generated_tokens":evaluation.rollouts.iter().map(|r| r.tokens.len()).sum::<usize>(),
            "evaluated_positions":evaluation.rollouts.iter().map(|r| r.evaluated_positions).sum::<usize>(),
            "committed_tokens":evaluation.rollouts.iter().map(|r| r.committed_tokens).sum::<usize>(),
            "rollout_gpu_seconds":evaluation.rollouts.iter().map(|r| r.gpu_seconds).sum::<f64>(),
            "broad_passes":evaluation.rollouts.iter().map(|r| r.broad_passes).sum::<usize>(),
            "correction_waves":evaluation.rollouts.iter().map(|r| r.correction_waves).sum::<usize>(),
            "repair_batches":evaluation.rollouts.iter().map(|r| r.repair_batches).sum::<usize>(),
            "accepted_tokens":evaluation.rollouts.iter().map(|r| r.accepted_tokens).sum::<usize>(),
            "first_mismatch":evaluation.rollouts.iter().map(|r| r.first_mismatch).collect::<Vec<_>>(),
            "evaluated_lengths":evaluation.rollouts.iter().map(|r| &r.evaluated_lengths).collect::<Vec<_>>(),
            "accepted_lengths":evaluation.rollouts.iter().map(|r| &r.accepted_lengths).collect::<Vec<_>>(),
            "committed_lengths":evaluation.rollouts.iter().map(|r| &r.committed_lengths).collect::<Vec<_>>(),
            "route_samples":evaluation.rollouts.iter().map(|r| &r.route_samples).collect::<Vec<_>>()});
            Self::annotate_causal(&evaluation, &mut record["generation"])?;
            evaluation.annotate(&mut record["generation"])?;
            state.record_rank(&evaluation, &mut record);
            if let Some(report) = state.search.objective_report()? {
                let width = report.width as usize;
                record["objective_acquisition"] = ennx_wire::json::json!({
                    "method": "vector",
                    "values": report.values.map(|values| values[..width].to_vec()),
                    "weights": report.weights[..width].to_vec(),
                    "nondominated_mask": report.nondominated_mask,
                    "selected": report.selected,
                    "region": region,
                    "regions": state.search.morbo_regions(),
                });
            }
            record["tensor_version"] = ennx_wire::json::json!({
                "id": tensor_version.id,
                "parent": tensor_version.parent,
                "seed": tensor_version.seed,
                "radius": tensor_version.radius,
                "distribution": tensor_version.distribution,
                "blocks": tensor_version.blocks.len(),
                "basis_seed": tensor_version.basis_seed,
                "threshold_words": tensor_version.threshold_words,
                "accepted_parent": state.incumbent_version,
            });
            if let Some(bounds) = proposal.bound_report() {
                record["history_bounds"] = ennx_wire::json::json!({
                    "history": bounds.history,
                    "neighbors": bounds.neighbors,
                    "raw_survivors": bounds.raw_survivors,
                    "final_survivors": bounds.final_survivors,
                    "bound_violations": bounds.bound_violations,
                    "rank_violations": bounds.rank_violations,
                    "pruning_enabled": false,
                });
            }
            ennx_wire::json::write_line(&mut state.records, &record).map_err(|e| e.to_string())?;
            state.records.flush().map_err(|e| e.to_string())?;
            let wall = start.elapsed().as_secs_f64();
            state.loop_seconds += wall;
            state.walls.push(wall);
            state.journal.training(
                step + 1,
                wall,
                &evaluation,
                decision.accepted,
                if initializing {
                    "initialization"
                } else {
                    "guided"
                },
            )?;
            eprintln!(
                "ENNX_GENERATION round={} wall_ms={:.3} proposal_ms={:.3} temperature={} incumbent_temperature={} changed_weights={} changed_fraction={:.9} evaluated_positions={} correction_waves={} repair_batches={} first_mismatch={:?} reward={} accepted={}",
                step + 1,
                wall * 1000.0,
                proposal_seconds * 1000.0,
                candidate_temperature,
                state.temperature,
                changed_weights,
                changed_fraction,
                evaluation
                    .rollouts
                    .iter()
                    .map(|rollout| rollout.evaluated_positions)
                    .sum::<usize>(),
                evaluation
                    .rollouts
                    .iter()
                    .map(|rollout| rollout.correction_waves)
                    .sum::<usize>(),
                evaluation
                    .rollouts
                    .iter()
                    .map(|rollout| rollout.repair_batches)
                    .sum::<usize>(),
                evaluation
                    .rollouts
                    .first()
                    .and_then(|rollout| rollout.first_mismatch),
                evaluation.mean,
                decision.accepted
            );
            if let Some(text) = text {
                display_completion(
                    &format!(
                        "ENNX_GENERATED_TEXT round={} tokens={} temperature={} reward={} accepted={}",
                        step + 1,
                        evaluation.rollouts[0].tokens.len(),
                        candidate_temperature,
                        evaluation.mean,
                        decision.accepted,
                    ),
                    &text,
                );
            }
            state
                .journal
                .gate(step + 1, self.config.signal_gate.as_ref())?;
            if self
                .run
                .validation_interval
                .is_some_and(|interval| (step + 1) % interval == 0)
                && step + 1 < self.run.rounds()
            {
                let validation_start = Instant::now();
                let mut validation_config = self.config.clone();
                validation_config.temperature = state.temperature;
                let rewards = validate_generation(
                    &self.runtime,
                    &self.decoder,
                    self.verifier.as_ref(),
                    self.weights,
                    &state.search.base_buffer(),
                    &validation_config,
                    self.sampling_seed,
                    &self.output.join(format!("validation-{:04}", step + 1)),
                    self.evaluator.as_mut(),
                    self.tokenizer,
                    &self.validation_tasks,
                )?;
                state.journal.validation(
                    step + 1,
                    &rewards,
                    validation_start.elapsed().as_secs_f64(),
                )?;
                self.validate_causal(state, step + 1)?;
            }
        }
        Ok(())
    }
}
