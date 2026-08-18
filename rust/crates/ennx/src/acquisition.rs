//! Acquisition function optimizers for TuRBO.

use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;
use thiserror::Error;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct MinHeapItem {
    score: f64,
    index: usize,
}

impl Eq for MinHeapItem {}

impl PartialOrd for MinHeapItem {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapItem {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order so BinaryHeap acts as a Min-Heap (smallest score on top)
        other.score.total_cmp(&self.score)
    }
}

/// Errors in acquisition optimization.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AcquisitionError {
    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Empty candidate set.
    #[error("No candidates available")]
    NoCandidates,
}

/// Result of acquisition optimization.
pub type AcquisitionResult = Result<Vec<usize>, AcquisitionError>;

/// UCB (Upper Confidence Bound) acquisition optimizer.
pub struct UCBAcquisition {
    /// Beta parameter (exploration weight).
    beta: f64,
}

impl UCBAcquisition {
    /// Create new UCB acquisition optimizer.
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    /// Select arms using UCB acquisition.
    ///
    /// # Arguments
    ///
    /// * `mu` - Predictive means, shape (n_candidates,)
    /// * `sigma` - Predictive standard deviations, shape (n_candidates,)
    /// * `num_arms` - Number of arms to select
    /// * `rng` - Random number generator
    pub fn select<R: Rng + ?Sized>(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        _rng: &mut R,
    ) -> AcquisitionResult {
        if mu.is_empty() {
            return Err(AcquisitionError::NoCandidates);
        }

        if num_arms == 0 {
            return Ok(Vec::new());
        }

        let k = num_arms.min(mu.len());
        let mut heap = BinaryHeap::with_capacity(k);

        // Streaming Top-K pass
        for i in 0..mu.len() {
            let score = self.beta.mul_add(sigma[i], mu[i]);

            if heap.len() < k {
                heap.push(MinHeapItem { score, index: i });
            } else if let Some(min_item) = heap.peek() {
                if score > min_item.score {
                    heap.pop();
                    heap.push(MinHeapItem { score, index: i });
                }
            }
        }

        let mut result: Vec<_> = heap.into_iter().collect();
        result.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(result.into_iter().map(|item| item.index).collect())
    }
}

/// Thompson sampling acquisition optimizer.
pub struct ThompsonAcquisition;

impl ThompsonAcquisition {
    /// Create new Thompson sampling acquisition optimizer.
    pub fn new() -> Self {
        Self
    }

    /// Select arms using Thompson sampling.
    ///
    /// # Arguments
    ///
    /// * `mu` - Predictive means, shape (n_candidates,)
    /// * `sigma` - Predictive standard deviations, shape (n_candidates,)
    /// * `num_arms` - Number of arms to select
    /// * `rng` - Random number generator
    pub fn select<R: Rng + ?Sized>(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        if mu.is_empty() {
            return Err(AcquisitionError::NoCandidates);
        }

        if num_arms == 0 {
            return Ok(Vec::new());
        }

        use rand_distr::StandardNormal;

        let k = num_arms.min(mu.len());
        let mut heap = BinaryHeap::with_capacity(k);

        // Fused Ziggurat draw and streaming Top-K in a single O(N) pass
        for i in 0..mu.len() {
            let mean = mu[i];
            let std = sigma[i];
            let z: f64 = rng.sample(StandardNormal);
            let score = z.mul_add(std, mean);

            if heap.len() < k {
                heap.push(MinHeapItem { score, index: i });
            } else if let Some(min_item) = heap.peek() {
                if score > min_item.score {
                    heap.pop();
                    heap.push(MinHeapItem { score, index: i });
                }
            }
        }

        let mut result: Vec<_> = heap.into_iter().collect();
        result.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(result.into_iter().map(|item| item.index).collect())
    }
}

impl Default for ThompsonAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

/// Random acquisition optimizer (baseline).
pub struct RandomAcquisition;

impl RandomAcquisition {
    /// Create new random acquisition optimizer.
    pub fn new() -> Self {
        Self
    }

    /// Select random arms.
    pub fn select<R: Rng + ?Sized>(
        &self,
        n_candidates: usize,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        if n_candidates == 0 {
            return Err(AcquisitionError::NoCandidates);
        }

        let mut indices: Vec<usize> = (0..n_candidates).collect();
        indices.shuffle(rng);
        Ok(indices.into_iter().take(num_arms).collect())
    }
}

impl Default for RandomAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

/// Pareto front acquisition optimizer for multi-objective problems.
pub struct ParetoAcquisition;

impl ParetoAcquisition {
    /// Create new Pareto acquisition optimizer.
    pub fn new() -> Self {
        Self
    }

    /// Select arms from Pareto front.
    ///
    /// For multi-objective, uses non-dominated sorting.
    /// For single-objective, delegates to arms_from_pareto_fronts_2d.
    pub fn select<R: Rng + ?Sized>(
        &self,
        mu: &ArrayView2<f64>,
        se: &ArrayView2<f64>,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        let n_candidates = mu.nrows();
        let n_objectives = mu.ncols();

        if n_candidates == 0 {
            return Err(AcquisitionError::NoCandidates);
        }

        if num_arms == 0 {
            return Ok(Vec::new());
        }

        if mu.shape() != se.shape() {
            return Err(AcquisitionError::InvalidParameter(format!(
                "mu shape {:?} does not match se shape {:?}",
                mu.shape(),
                se.shape()
            )));
        }

        if n_objectives == 1 {
            let mu_1d = mu.column(0).to_owned();
            let sigma_1d = se.column(0).to_owned();
            return Self::arms_from_pareto_fronts_2d(
                &mu_1d.view(),
                &sigma_1d.view(),
                num_arms,
                rng,
            );
        }

        // Multi-objective: use non-dominated sorting (simplified)
        let pareto_fronts = self.non_domin_sort(mu);

        // Select from fronts until we have enough
        let mut selected = Vec::with_capacity(num_arms);
        for front in pareto_fronts {
            if selected.len() >= num_arms {
                break;
            }

            let remaining = num_arms - selected.len();
            if front.len() <= remaining {
                selected.extend(front);
            } else {
                // Random selection from front
                let mut front_indices = front;
                front_indices.shuffle(rng);
                selected.extend(front_indices.into_iter().take(remaining));
            }
        }

        Ok(selected)
    }

    /// Non-dominated sorting (simplified implementation).
    fn non_domin_sort(&self, objectives: &ArrayView2<f64>) -> Vec<Vec<usize>> {
        if objectives.ncols() == 2 {
            return self.non_domin_sort_2d(objectives);
        }

        let n = objectives.nrows();
        let m = objectives.ncols();

        if n == 0 {
            return Vec::new();
        }

        // Domination counts and dominated set
        let mut domination_count: Vec<usize> = vec![0; n];
        let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        // Compute domination
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                // Check if i dominates j (all objectives >= and at least one >)
                let mut all_gte = true;
                let mut any_gt = false;
                for k in 0..m {
                    if objectives[[i, k]] < objectives[[j, k]] {
                        all_gte = false;
                        break;
                    }
                    if objectives[[i, k]] > objectives[[j, k]] {
                        any_gt = true;
                    }
                }

                if all_gte && any_gt {
                    dominated[i].push(j);
                } else if !all_gte {
                    // Check if j dominates i
                    let mut j_gte = true;
                    let mut j_gt = false;
                    for k in 0..m {
                        if objectives[[j, k]] < objectives[[i, k]] {
                            j_gte = false;
                            break;
                        }
                        if objectives[[j, k]] > objectives[[i, k]] {
                            j_gt = true;
                        }
                    }
                    if j_gte && j_gt {
                        domination_count[i] += 1;
                    }
                }
            }
        }

        // First front: points with domination count 0
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front.is_empty() {
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            fronts.push(current_front);
            current_front = next_front;
        }

        fronts
    }

    /// Non-dominated sorting specialized for 2 objectives.
    ///
    /// Uses skyline peeling with objective-0 sort + objective-1 sweep.
    fn non_domin_sort_2d(&self, objectives: &ArrayView2<f64>) -> Vec<Vec<usize>> {
        let n = objectives.nrows();
        if n == 0 {
            return Vec::new();
        }

        let mut remaining: Vec<usize> = (0..n).collect();
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        while !remaining.is_empty() {
            // Sort by objective 0 descending, then objective 1 descending.
            remaining.sort_by(|&i, &j| {
                objectives[[j, 0]]
                    .total_cmp(&objectives[[i, 0]])
                    .then_with(|| objectives[[j, 1]].total_cmp(&objectives[[i, 1]]))
            });

            // Sweep to extract current non-dominated front.
            let mut front: Vec<usize> = Vec::new();
            let mut best_obj1 = f64::NEG_INFINITY;
            let mut last_obj0 = f64::NAN;
            let mut last_obj1 = f64::NAN;

            for &idx in &remaining {
                let obj0 = objectives[[idx, 0]];
                let obj1 = objectives[[idx, 1]];
                if obj1 > best_obj1 {
                    front.push(idx);
                    best_obj1 = obj1;
                    last_obj0 = obj0;
                    last_obj1 = obj1;
                } else if obj1 == best_obj1 && obj0 == last_obj0 && obj1 == last_obj1 {
                    // Equal points do not dominate each other; keep ties.
                    front.push(idx);
                }
            }

            let mut in_front = vec![false; n];
            for &idx in &front {
                in_front[idx] = true;
            }
            remaining.retain(|&idx| !in_front[idx]);
            fronts.push(front);
        }

        fronts
    }

    /// Select arms from 2D Pareto fronts (used for single-objective).
    fn arms_from_pareto_fronts_2d<R: Rng + ?Sized>(
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        let n = mu.len();
        if n == 0 {
            return Err(AcquisitionError::NoCandidates);
        }

        let num_arms = num_arms.min(n);
        let dummy_x = ndarray::Array2::<f64>::zeros((n, 1));
        let seed = rng.r#gen::<u64>();
        Ok(crate::util::arms_from_pareto_fronts(
            &dummy_x.view(),
            mu,
            sigma,
            num_arms,
            seed,
        ))
    }
}

impl Default for ParetoAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_ucb_acquisition() {
        let ucb = UCBAcquisition::new(2.0);
        let mu = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = array![0.1, 0.2, 0.3, 0.4, 0.5];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = ucb.select(&mu.view(), &sigma.view(), 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
        // Highest UCB should be selected (with random tie-breaking)
        assert!(selected.iter().all(|&i| i < 5));
    }

    #[test]
    fn test_thompson_acquisition() {
        let thompson = ThompsonAcquisition::new();
        let mu = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = array![0.1, 0.2, 0.3, 0.4, 0.5];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = thompson
            .select(&mu.view(), &sigma.view(), 3, &mut rng)
            .unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&i| i < 5));
    }

    #[test]
    fn test_random_acquisition() {
        let random = RandomAcquisition::new();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = random.select(10, 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&i| i < 10));
    }

    #[test]
    fn test_pareto_single_objective() {
        let pareto = ParetoAcquisition::new();
        let mu = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let se = array![[0.1], [0.2], [0.3], [0.4], [0.5]];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = pareto.select(&mu.view(), &se.view(), 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_pareto_single_objective_uses_predictive_se_bug_md() {
        let pareto = ParetoAcquisition::new();
        let mu = array![[1.0], [1.0], [2.0]];
        let se = array![[0.01], [10.0], [1.5]];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = pareto.select(&mu.view(), &se.view(), 1, &mut rng).unwrap();

        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn test_pareto_multi_objective() {
        let pareto = ParetoAcquisition::new();
        let mu = array![[1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [1.5, 1.5], [3.0, 3.0]];
        let se = array![[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = pareto.select(&mu.view(), &se.view(), 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.contains(&4));
    }

    #[test]
    fn test_non_domin_sort_2d_three_fronts_golden() {
        let pareto = ParetoAcquisition::new();
        let objectives = array![[3.0, 1.0], [2.0, 2.0], [1.0, 3.0], [0.5, 0.5]];
        let fronts = pareto.non_domin_sort(&objectives.view());
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0], vec![0, 1, 2]);
        assert_eq!(fronts[1], vec![3]);
    }

    #[test]
    fn test_non_domin_sort_2d_with_ties() {
        let pareto = ParetoAcquisition::new();
        // Points 0 and 1 are identical and both should be on first front.
        let objectives = array![[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [0.0, 3.0]];
        let fronts = pareto.non_domin_sort(&objectives.view());
        assert!(!fronts.is_empty());
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&1));
        assert!(fronts[0].contains(&3));
    }

    #[test]
    fn test_ucb_empty() {
        let ucb = UCBAcquisition::new(2.0);
        let mu = array![];
        let sigma = array![];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = ucb.select(&mu.view(), &sigma.view(), 3, &mut rng);

        assert!(matches!(result, Err(AcquisitionError::NoCandidates)));
    }

    #[test]
    fn test_ucb_zero_arms() {
        let ucb = UCBAcquisition::new(2.0);
        let mu = array![1.0, 2.0, 3.0];
        let sigma = array![0.1, 0.2, 0.3];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = ucb.select(&mu.view(), &sigma.view(), 0, &mut rng).unwrap();

        assert!(selected.is_empty());
    }
}
