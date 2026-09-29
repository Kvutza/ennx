//! Non-compensatory outcomes for free-running coding candidates.

use deser::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const CODING_SCHEMA: &str = "ennx.reward.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct CodingOutcome {
    pub schema: String,
    pub candidate_id: String,
    pub eligibility: CandidateEligibility,
    pub checks: ExecutableChecks,
    pub efficiency: CandidateEfficiency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateEligibility {
    Eligible,
    Ineligible { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct ExecutableChecks {
    pub pass_to_pass_total: u32,
    pub pass_to_pass_passed: u32,
    pub fail_to_pass_total: u32,
    pub fail_to_pass_passed: u32,
    pub hidden_total: u32,
    pub hidden_passed: u32,
    pub safety_violations: u32,
    pub scope_violations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct CandidateEfficiency {
    pub generated_tokens: u32,
    pub tool_calls: u32,
    pub modified_lines: u32,
    pub execution_micros: u64,
}

impl CodingOutcome {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CODING_SCHEMA {
            return Err(format!(
                "unsupported coding reward schema {:?}",
                self.schema
            ));
        }
        if self.candidate_id.is_empty() {
            return Err("candidate identifier must be nonempty".into());
        }
        if let CandidateEligibility::Ineligible { reason } = &self.eligibility
            && reason.is_empty()
        {
            return Err("ineligible candidate requires a reason".into());
        }
        for (passed, total, name) in [
            (
                self.checks.pass_to_pass_passed,
                self.checks.pass_to_pass_total,
                "pass_to_pass",
            ),
            (
                self.checks.fail_to_pass_passed,
                self.checks.fail_to_pass_total,
                "fail_to_pass",
            ),
            (
                self.checks.hidden_passed,
                self.checks.hidden_total,
                "hidden",
            ),
        ] {
            if passed > total {
                return Err(format!("{name} passed count exceeds its total"));
            }
        }
        Ok(())
    }

    /// Compare two candidates from the same task. `Greater` means `self` wins.
    ///
    /// Correctness fields are lexicographic. Efficiency is considered only
    /// after all executable outcomes tie, so no amount of speed can compensate
    /// for a regression or failed task check.
    pub fn compare(&self, other: &Self) -> Result<Ordering, String> {
        self.validate()?;
        other.validate()?;
        let self_eligible = matches!(self.eligibility, CandidateEligibility::Eligible);
        let other_eligible = matches!(other.eligibility, CandidateEligibility::Eligible);
        let eligibility = self_eligible.cmp(&other_eligible);
        if eligibility != Ordering::Equal || !self_eligible {
            return Ok(eligibility);
        }
        if self.check_totals() != other.check_totals() {
            return Err("coding outcomes from different check budgets are not comparable".into());
        }

        let ordering = self
            .regression_failures()
            .cmp(&other.regression_failures())
            .reverse()
            .then_with(|| {
                self.checks
                    .fail_to_pass_passed
                    .cmp(&other.checks.fail_to_pass_passed)
            })
            .then_with(|| self.checks.hidden_passed.cmp(&other.checks.hidden_passed))
            .then_with(|| {
                self.checks
                    .safety_violations
                    .cmp(&other.checks.safety_violations)
                    .reverse()
            })
            .then_with(|| {
                self.checks
                    .scope_violations
                    .cmp(&other.checks.scope_violations)
                    .reverse()
            })
            .then_with(|| {
                self.efficiency
                    .generated_tokens
                    .cmp(&other.efficiency.generated_tokens)
                    .reverse()
            })
            .then_with(|| {
                self.efficiency
                    .tool_calls
                    .cmp(&other.efficiency.tool_calls)
                    .reverse()
            })
            .then_with(|| {
                self.efficiency
                    .modified_lines
                    .cmp(&other.efficiency.modified_lines)
                    .reverse()
            })
            .then_with(|| {
                self.efficiency
                    .execution_micros
                    .cmp(&other.efficiency.execution_micros)
                    .reverse()
            });
        Ok(ordering)
    }

    fn regression_failures(&self) -> u32 {
        self.checks.pass_to_pass_total - self.checks.pass_to_pass_passed
    }

    fn check_totals(&self) -> (u32, u32, u32) {
        (
            self.checks.pass_to_pass_total,
            self.checks.fail_to_pass_total,
            self.checks.hidden_total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome() -> CodingOutcome {
        CodingOutcome {
            schema: CODING_SCHEMA.into(),
            candidate_id: "candidate".into(),
            eligibility: CandidateEligibility::Eligible,
            checks: ExecutableChecks {
                pass_to_pass_total: 100,
                pass_to_pass_passed: 100,
                fail_to_pass_total: 4,
                fail_to_pass_passed: 2,
                hidden_total: 2,
                hidden_passed: 1,
                safety_violations: 0,
                scope_violations: 0,
            },
            efficiency: CandidateEfficiency {
                generated_tokens: 1_000,
                tool_calls: 4,
                modified_lines: 10,
                execution_micros: 100_000,
            },
        }
    }

    #[test]
    fn regression_priority() {
        let safe = outcome();
        let mut regressed = outcome();
        regressed.checks.pass_to_pass_passed -= 1;
        regressed.checks.fail_to_pass_passed = 4;
        regressed.checks.hidden_passed = 2;
        regressed.efficiency.generated_tokens = 1;
        regressed.efficiency.execution_micros = 1;
        assert_eq!(safe.compare(&regressed), Ok(Ordering::Greater));
    }

    #[test]
    fn correctness_priority() {
        let mut correct = outcome();
        correct.checks.fail_to_pass_passed = 3;
        correct.efficiency.generated_tokens = 4_096;
        let fast = outcome();
        assert_eq!(correct.compare(&fast), Ok(Ordering::Greater));
    }

    #[test]
    fn efficiency_tiebreak() {
        let slow = outcome();
        let mut fast = outcome();
        fast.efficiency.execution_micros -= 1;
        assert_eq!(fast.compare(&slow), Ok(Ordering::Greater));
    }

    #[test]
    fn mismatched_budgets() {
        let left = outcome();
        let mut right = outcome();
        right.checks.hidden_total += 1;
        assert_eq!(
            left.compare(&right).unwrap_err(),
            "coding outcomes from different check budgets are not comparable"
        );
    }

    #[test]
    fn ineligible_candidates() {
        let mut left = outcome();
        let mut right = outcome();
        left.eligibility = CandidateEligibility::Ineligible {
            reason: "invalid tool call".into(),
        };
        right.eligibility = CandidateEligibility::Ineligible {
            reason: "budget exceeded".into(),
        };
        assert_eq!(left.compare(&right), Ok(Ordering::Equal));
        assert_eq!(outcome().compare(&left), Ok(Ordering::Greater));
    }
}
