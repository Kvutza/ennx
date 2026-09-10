//! Backend capability declarations.
//!
//! This module is the executable map for accelerator parity work. It should be
//! updated when an operation becomes resident, gains an explicit fallback, or is
//! removed from a backend.

use serde::{Deserialize, Serialize};

/// Execution backend tracked by the capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    OpenCl,
}

/// Library operation tracked for backend parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    ExactDist,
    AnnIndex,
    Posterior,
    Acquisition,
    Candidate,
    WeightSelect,
    TrialSearch,
    ResidentLoop,
}

/// Capability status for one backend/operation pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Support {
    Direct,
    Fallback,
    Missing,
}

/// One row in the backend capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub backend: Backend,
    pub operation: Operation,
    pub support: Support,
}

const BACKENDS: [Backend; 4] = [Backend::Cpu, Backend::Cuda, Backend::Metal, Backend::OpenCl];

const OPS: [Operation; 8] = [
    Operation::ExactDist,
    Operation::AnnIndex,
    Operation::Posterior,
    Operation::Acquisition,
    Operation::Candidate,
    Operation::WeightSelect,
    Operation::TrialSearch,
    Operation::ResidentLoop,
];

/// Returns every backend tracked by the matrix.
pub fn backends() -> &'static [Backend] {
    &BACKENDS
}

/// Returns every operation tracked by the matrix.
pub fn operations() -> &'static [Operation] {
    &OPS
}

/// Returns the declared support for a backend/operation pair.
pub fn support(backend: Backend, operation: Operation) -> Support {
    match (backend, operation) {
        (Backend::Cpu, _) => Support::Direct,
        (_, Operation::ExactDist) => Support::Direct,
        (Backend::Cuda, Operation::AnnIndex | Operation::TrialSearch) => Support::Direct,
        (
            Backend::Metal,
            Operation::AnnIndex | Operation::TrialSearch | Operation::WeightSelect,
        ) => Support::Direct,
        (
            Backend::OpenCl,
            Operation::AnnIndex | Operation::TrialSearch | Operation::WeightSelect,
        ) => Support::Direct,
        (_, Operation::Posterior | Operation::Acquisition | Operation::Candidate) => {
            Support::Fallback
        }
        (_, Operation::ResidentLoop) => Support::Fallback,
        (_, Operation::WeightSelect) => Support::Missing,
    }
}

/// Returns the complete capability matrix.
pub fn matrix() -> Vec<Capability> {
    let mut rows = Vec::with_capacity(BACKENDS.len() * OPS.len());
    for &backend in &BACKENDS {
        for &operation in &OPS {
            rows.push(Capability {
                backend,
                operation,
                support: support(backend, operation),
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::{matrix, support, Backend, Operation, Support, BACKENDS, OPS};

    #[test]
    fn complete() {
        assert_eq!(matrix().len(), BACKENDS.len() * OPS.len());
    }

    #[test]
    fn cpu_direct() {
        for &operation in &OPS {
            assert_eq!(support(Backend::Cpu, operation), Support::Direct);
        }
    }

    #[test]
    fn accel_gaps() {
        for backend in [Backend::Cuda, Backend::Metal, Backend::OpenCl] {
            assert_eq!(support(backend, Operation::Posterior), Support::Fallback);
            assert_eq!(support(backend, Operation::Acquisition), Support::Fallback);
            assert_eq!(support(backend, Operation::Candidate), Support::Fallback);
            assert_eq!(support(backend, Operation::ResidentLoop), Support::Fallback);
        }
    }
}
