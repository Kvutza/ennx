use super::*;
use crate::config::GenerationConfig;

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
