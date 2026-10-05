//! Isolated micro-kernel evaluation harness for fast, noise-free candidate profiling.

use ennx::config::KernelTrial;
use ennx_wire::json::json;
use std::path::Path;

use super::Search;
use super::artifacts::{Archive, write_json};

pub(super) fn run(
    _root: &Path,
    _parent: &Path,
    search: &Search,
    archive: &Archive,
    trials: &[KernelTrial],
) -> Result<(), String> {
    println!("\n=== ENNX Micro-Kernel Optimization Testbed ===");
    println!(
        "Benchmarking {} candidates with {} iterations each\n",
        search.candidates.len(),
        search.iterations
    );

    let mut reports: Vec<ennx_wire::json::Value> = Vec::new();
    let mut passed_count = 0;

    for (candidate, trial) in search.candidates.iter().zip(trials) {
        #[cfg(target_os = "macos")]
        {
            let metrics = ennx::experimental::benchmark_micro(
                &candidate.operator,
                Some(trial),
                &candidate.name,
                search.iterations,
                search.gate.token_atol,
                search.gate.min_speedup,
            )?;

            println!("------------------------------------------------------------");
            println!(
                "Candidate: {} [{}]",
                metrics.candidate_name, metrics.operator
            );
            println!("  Hypothesis: {}", candidate.hypothesis);
            println!(
                "  Latency: baseline={:.2}us | candidate={:.2}us (speedup: {:.3}x)",
                metrics.baseline_median_us, metrics.candidate_median_us, metrics.speedup
            );
            println!(
                "  Bandwidth: {:.2} GB/s ({:.1}% roofline efficiency)",
                metrics.achieved_bw_gbs, metrics.roofline_pct
            );
            println!(
                "  Parity Error: max_abs_error={:.6} (limit: {:.6})",
                metrics.max_abs_error, search.gate.token_atol
            );
            println!(
                "  Status: {}",
                if metrics.passed_gate {
                    "PASSED"
                } else {
                    "REJECTED"
                }
            );

            if metrics.passed_gate {
                passed_count += 1;
            }

            reports.push(json!({
                "candidate": metrics.candidate_name,
                "operator": metrics.operator,
                "baseline_median_us": metrics.baseline_median_us,
                "candidate_median_us": metrics.candidate_median_us,
                "speedup": metrics.speedup,
                "achieved_bw_gbs": metrics.achieved_bw_gbs,
                "roofline_pct": metrics.roofline_pct,
                "max_abs_error": metrics.max_abs_error,
                "passed": metrics.passed_gate,
            }));
        }

        #[cfg(not(target_os = "macos"))]
        {
            return Err("micro-benchmark mode currently requires macOS with Metal enabled".into());
        }
    }

    write_json(
        &archive.path.join("feedback.json"),
        &json!({
            "schema": "ennx.kernel_feedback.micro.v1",
            "status": "completed",
            "passed_candidates": passed_count,
            "total_candidates": search.candidates.len(),
            "reports": reports,
        }),
    )?;

    println!("------------------------------------------------------------");
    println!(
        "Testbed run finished: {}/{} candidates passed.\n",
        passed_count,
        search.candidates.len()
    );
    Ok(())
}
