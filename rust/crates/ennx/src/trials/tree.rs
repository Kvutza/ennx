use crate::util::insert_neighbor;

use super::{make_steps, materialize, score, trial_distance, Ask, Leaf};

const MAX_DEPTH: usize = 8;

/// One node in a persistent perturbation tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Center {
    /// Earlier center node, or `None` for the root model.
    pub parent: Option<usize>,
    /// Deterministic perturbation applied after the parent.
    pub seed: u64,
}

pub(super) fn check(
    centers: &[Center],
    region_centers: &[usize],
    num_regions: usize,
) -> Result<(), String> {
    if num_regions == 0 || region_centers.len() != num_regions {
        return Err("region centers must match the non-zero region count".to_string());
    }
    for (index, center) in centers.iter().enumerate() {
        if center.parent.is_some_and(|parent| parent >= index) {
            return Err(format!("center {index} must reference an earlier parent"));
        }
        let mut depth = 1;
        let mut parent = center.parent;
        while let Some(index) = parent {
            depth += 1;
            if depth > MAX_DEPTH {
                return Err(format!("center chain exceeds depth {MAX_DEPTH}"));
            }
            parent = centers[index].parent;
        }
    }
    if region_centers.iter().any(|&center| center >= centers.len()) {
        return Err("region center index is out of bounds".to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cpu_ask(
    root: &[u8],
    rows: &[u8],
    row_bytes: usize,
    history: &[(usize, f32)],
    seeds_per_region: usize,
    centers: &[Center],
    region_centers: &[usize],
    seeds: &[u64],
    leaves: &[Leaf],
    config: Ask,
) -> Result<Vec<(usize, f32)>, String> {
    let steps = make_steps(leaves, config.length);
    let mut center_rows: Vec<Vec<u8>> = Vec::with_capacity(centers.len());
    for center in centers {
        let parent = center.parent.map_or(root, |index| &center_rows[index]);
        center_rows.push(materialize(parent, leaves, &steps, center.seed));
    }
    let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
    let mut results = Vec::with_capacity(region_centers.len());
    for (region, &center) in region_centers.iter().enumerate() {
        let start = region * seeds_per_region;
        let end = start + seeds_per_region;
        let mut best = (start, f32::NEG_INFINITY);
        let mut nearest = vec![(f32::INFINITY, 0usize); config.neighbors];
        for index in start..end {
            nearest.fill((f32::INFINITY, 0));
            for (observation, &(slot, _)) in history.iter().enumerate() {
                let row = &rows[slot * row_bytes..(slot + 1) * row_bytes];
                let distance =
                    trial_distance(&center_rows[center], row, leaves, &steps, seeds[index]);
                insert_neighbor(&mut nearest, distance, observation);
            }
            let value = score(&nearest, history, draws[index], config);
            if value > best.1 {
                best = (index, value);
            }
        }
        results.push(best);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::{check, Center};

    #[test]
    fn validates_topological_centers() {
        let centers = [
            Center {
                parent: None,
                seed: 1,
            },
            Center {
                parent: Some(0),
                seed: 2,
            },
        ];
        assert!(check(&centers, &[1], 1).is_ok());

        let invalid = [Center {
            parent: Some(0),
            seed: 1,
        }];
        assert!(check(&invalid, &[0], 1).is_err());
    }
}
