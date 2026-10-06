use super::*;
use super::generation::GenerationTask;
use super::stats::RepairStats;
use crate::config::GenerationConfig;

pub(super) fn score_targets(config: &GenerationConfig) -> bool {
    matches!(
        config.reward,
        crate::config::GenerationReward::FreeRunningCrossEntropy
            | crate::config::GenerationReward::CodeObjectives { .. }
    )
}

pub(super) fn loss_window(config: &GenerationConfig) -> usize {
    match &config.reward {
        crate::config::GenerationReward::CodeObjectives { critical_window } => {
            *critical_window as usize
        }
        _ => config.max_tokens as usize,
    }
}

pub(super) fn route_sample(stats: routing::RouteStats) -> decode::RouteSample {
    decode::RouteSample {
        active_experts: stats.active_experts,
        routed_rows: stats.routed_rows,
        routed_tiles: stats.routed_tiles,
    }
}

pub(super) fn repair_bounds(
    row: usize,
    context: u32,
    window: u32,
    prompt: usize,
    limit: usize,
) -> (u32, usize) {
    let start = ((row as u32 / 4) * 4).min(context - window);
    let end = (start as usize + window as usize)
        .saturating_sub(prompt - 1)
        .min(limit);
    (start, end)
}

pub(super) fn adapt_window(
    window: u32,
    config: &GenerationConfig,
    mismatch: bool,
    accepted: usize,
    span: usize,
) -> u32 {
    if !mismatch || (span > 0 && accepted * 2 >= span) {
        (window * 2).min(config.verify.max_window)
    } else if span > 0 && accepted * 4 < span {
        128
    } else {
        (window / 2).max(128)
    }
}

pub(super) fn next_window(
    window: u32,
    config: &GenerationConfig,
    mismatch: bool,
    accepted: usize,
    span: usize,
) -> (bool, u32) {
    (
        mismatch && accepted * 4 < span,
        adapt_window(window, config, mismatch, accepted, span),
    )
}

pub(super) fn expand_draft(
    draft: &decode::Rollout,
    maximum: usize,
    eos: Option<u32>,
) -> Result<Vec<u32>, String> {
    if draft.tokens.len() == maximum {
        return Ok(draft.tokens.clone());
    }
    let Some(eos) = eos else {
        return Err("short incumbent rollout without EOS cannot seed block verification".into());
    };
    if draft.tokens.last() != Some(&eos) || draft.tokens.len() > maximum {
        return Err("incumbent rollout has an invalid generated length".into());
    }
    let mut tokens = draft.tokens.clone();
    tokens.resize(maximum, eos);
    Ok(tokens)
}

pub(super) fn refresh_draft(
    tokens: &mut [u32],
    input: &[u32],
    cursor: usize,
    tile_end: usize,
    prompt: usize,
) {
    for position in cursor..tile_end.saturating_sub(1) {
        tokens[position] = input[prompt + position];
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_repair_tile(
    task: &GenerationTask,
    tokens: &mut [u32],
    input: &mut [u32],
    proposed: &[u32],
    exact_start: usize,
    tile_end: usize,
    context: usize,
    patch: usize,
    eos: Option<u32>,
    stats: &mut RepairStats,
) -> Result<(usize, bool), String> {
    let mut position = exact_start;
    let mut committed = None;
    let mut accepted = 0usize;
    while position < tile_end {
        let prediction = proposed[task.prompt.len() - 1 + position];
        if prediction >= VOCAB {
            return Err("block repair produced an invalid token".into());
        }
        let exact_prediction = committed.is_none();
        if exact_prediction {
            if tokens[position] == prediction {
                accepted += 1;
            } else {
                stats.first_mismatch.get_or_insert(position);
                let patch_end = if patch > 1 {
                    ((position / patch) + 1) * patch
                } else {
                    position + 1
                };
                committed = Some(patch_end.min(tile_end));
            }
        }
        tokens[position] = prediction;
        if task.prompt.len() + position < context {
            input[task.prompt.len() + position] = prediction;
        }
        position += 1;
        if exact_prediction && Some(prediction) == eos {
            tokens[position..].fill(prediction);
            stats.accepted_lengths.push(accepted);
            stats.committed_lengths.push(position - exact_start);
            return Ok((tokens.len(), true));
        }
    }
    let cursor = committed.unwrap_or(tile_end);
    stats.accepted_lengths.push(accepted);
    stats.committed_lengths.push(cursor - exact_start);
    Ok((cursor, false))
}
