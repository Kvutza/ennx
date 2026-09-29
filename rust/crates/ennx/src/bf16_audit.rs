//! Exact, full-coordinate oracle for the production implicit-history selector.
//! Small enough to retain every row; these are not billion-dimensional results.

use super::*;
use crate::procedural_pool::stream_word;
use ennx_wire::json::json;
use std::io::Write;

fn decode(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    let fraction = f64::from(bits & 1023);
    let value = match exponent {
        0 => fraction * 2.0f64.powi(-24),
        31 => f64::NAN,
        _ => (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25),
    };
    if bits & 0x8000 == 0 { value } else { -value }
}

fn distance(a: &[f64], b: &[f64], blocks: &[ParamBlock]) -> f32 {
    blocks
        .iter()
        .map(|block| {
            let start = block.offset;
            a[start..start + block.len]
                .iter()
                .zip(&b[start..start + block.len])
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                * f64::from(block.weight)
        })
        .sum::<f64>() as f32
}

fn best(scores: &[f32]) -> usize {
    (1..scores.len()).fold(0, |winner, i| {
        if scores[i] > scores[winner] {
            i
        } else {
            winner
        }
    })
}

#[test]
#[ignore = "GPU geometry audit; emits measurements rather than asserting an optimization win"]
fn history_audit() -> Result<(), String> {
    let output = std::env::var("ENNX_GEOMETRY_OUTPUT").map_err(|e| e.to_string())?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    for dimensions in [4096usize, 65536, 1_048_576] {
        for perturbation in [Perturbation::Gaussian, Perturbation::Rademacher] {
            for seed in [123u64, 1009, 2027] {
                autoreleasepool(|| -> Result<(), String> {
                    let base: Vec<u16> = (0..dimensions)
                        .map(|i| 0x3000 + (i % 1024) as u16)
                        .collect();
                    let blocks: Vec<_> = (0..16)
                        .map(|i| {
                            let len = dimensions / 16;
                            let scale = 0.25 * (1 + i % 4) as f32;
                            ParamBlock::new(
                                i as u64,
                                i * len,
                                len,
                                scale,
                                1.0 / (len as f32 * scale * scale),
                            )
                        })
                        .collect::<Result<_, _>>()?;
                    let mut state = SearchState::new_implicit(
                        &base,
                        blocks.clone(),
                        2,
                        TRLengthConfig::new(0.01, 0.0001, 0.1),
                        perturbation,
                    )?;
                    state.set_tolerance(4)?;
                    let mut ask = Ask::default();
                    ask.acquisition = AcquisitionKind::Ucb;
                    ask.neighbors = 10;
                    ask.epistemic_scale = 1.0;
                    ask.aleatoric_scale = 0.05;
                    ask.y_scale = 1.0;
                    ask.beta = 1.0;
                    ask.seed = seed + 456;
                    state.configure_enn(crate::config::ResidentEnnConfig {
                        pool: crate::procedural_pool::ProceduralPool::legacy(),
                        proposal_method: crate::procedural_pool::ProposalMethod::Independent,
                        ask,
                        num_candidates: 30,
                        num_samples: 10,
                        fit_neighbors: false,
                        distance_scaling: crate::config::DistanceScaling::Global,
                        history_geometry: crate::config::HistoryGeometry::Realized,
                        local_scale_neighbors: 8,
                    })?;
                    let objective = |row: &[f64]| -> f32 {
                        -(row
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| {
                                let target = if stream_word(17, 31, i as u32) & 1 == 0 {
                                    0.1
                                } else {
                                    0.4
                                };
                                (x - target).powi(2)
                            })
                            .sum::<f64>()
                            / dimensions as f64) as f32
                    };
                    let initial: Vec<_> = base.iter().map(|&bits| decode(bits)).collect();
                    state.observe_initial(objective(&initial), 0.0)?;
                    let mut archive = vec![(state.base_id, initial)];
                    for step in 0..24u64 {
                        let root = seed + step;
                        let initializing = state.history < ask.neighbors;
                        ask.seed = seed + 456 + step;
                        if initializing {
                            state.begin_initial(root, step as usize % 4)?;
                        } else {
                            state.begin_ask(1, 4, root, ask)?;
                        }
                        let round = state.finish_ask()?;
                        let selected_bits = read::<u16>(&state.proposal, dimensions);
                        let selected: Vec<_> =
                            selected_bits.iter().map(|&bits| decode(bits)).collect();
                        if !initializing {
                            let params = state.select_params(root, ask, None);
                            let mut fitted = ask;
                            fitted.epistemic_scale = params.epistemic_scale;
                            fitted.aleatoric_scale = params.aleatoric_scale;
                            fitted.y_scale = params.y_scale;
                            let approximate: Vec<_> = round
                                .pool
                                .iter()
                                .map(|entry| entry.4.iter().map(|&(_, d)| d).collect::<Vec<_>>())
                                .collect();
                            let mut exact = Vec::new();
                            for candidate in 0..4 {
                                state.diagnostic_row(&round, root, ask, candidate)?;
                                let row: Vec<_> = read::<u16>(&state.proposal, dimensions)
                                    .into_iter()
                                    .map(decode)
                                    .collect();
                                exact.push(
                                    state.identities[..state.history]
                                        .iter()
                                        .map(|id| {
                                            let previous = &archive
                                                .iter()
                                                .find(|(key, _)| key == id)
                                                .expect("every observed vector is archived")
                                                .1;
                                            distance(&row, previous, &blocks)
                                        })
                                        .collect::<Vec<_>>(),
                                );
                            }
                            // Restore the immutable pending proposal before tell.
                            state.diagnostic_row(&round, root, ask, round.index)?;
                            assert_eq!(read::<u16>(&state.proposal, dimensions), selected_bits);
                            let outcomes = &state.outcomes[..state.history];
                            let variances = &state.variances[..state.history];
                            let approx_scores: Vec<_> = approximate
                                .iter()
                                .map(|d| acquisition(d, outcomes, variances, fitted))
                                .collect();
                            let exact_scores: Vec<_> = exact
                                .iter()
                                .map(|d| acquisition(d, outcomes, variances, fitted))
                                .collect();
                            // Check the existing CPU score oracle against the actual GPU score.
                            assert!(
                                (approx_scores[round.index] - round.score).abs()
                                    <= 1e-5 * round.score.abs().max(1.0)
                            );
                            let mut max_relative_error = 0.0f32;
                            let mut max_resident_relative_error = 0.0f32;
                            for candidate in 0..4 {
                                for (i, id) in state.identities[..state.history].iter().enumerate()
                                {
                                    let error = (approximate[candidate][i] - exact[candidate][i])
                                        .abs()
                                        / exact[candidate][i].max(1e-12);
                                    max_relative_error = max_relative_error.max(error);
                                    if state.resident_identities[..state.resident_history]
                                        .contains(id)
                                    {
                                        max_resident_relative_error =
                                            max_resident_relative_error.max(error);
                                    }
                                }
                            }
                            assert!(max_resident_relative_error < 1e-4);
                            let exact_winner = best(&exact_scores);
                            let record = json!({"dimensions": dimensions, "perturbation": perturbation.name(),
                                "seed": seed, "round": step + 1, "history": state.history,
                                "gpu_selected": round.index, "cpu_approx_selected": best(&approx_scores),
                                "exact_selected": exact_winner, "winner_changed": round.index != exact_winner,
                                "radius_changed": round.index % 2 != exact_winner % 2,
                                "max_relative_distance_error": max_relative_error,
                                "max_resident_relative_distance_error": max_resident_relative_error,
                                "exact_score_regret": exact_scores[exact_winner] - exact_scores[round.index],
                                "approx_scores": approx_scores, "exact_scores": exact_scores,
                                "approx_distances": approximate, "exact_distances": exact,
                                "fitted_parameters": [fitted.epistemic_scale, fitted.aleatoric_scale, fitted.y_scale],
                                "scope": "fixed_fitted_parameters; exact_query_distances_only"});
                            ennx_wire::json::write_line(&mut file, &record)
                                .map_err(|e| e.to_string())?;
                        }
                        let reward = objective(&selected);
                        if initializing {
                            state.tell_initial(&round, reward, 0.0)?;
                        } else {
                            state.tell_modeled(&round, reward, 0.0)?;
                        }
                        state.sync()?;
                        archive.push((state.observation, selected));
                    }
                    eprintln!(
                        "GEOMETRY dimensions={dimensions} perturbation={} seed={seed} complete",
                        perturbation.name()
                    );
                    Ok(())
                })?;
            }
        }
    }
    file.flush().map_err(|e| e.to_string())?;
    Ok(())
}
