//! Exact, count-clipped byte n-gram contrast; a reconstruction proxy, not execution.

use std::collections::HashMap;

pub(crate) struct Profile {
    counts: Vec<HashMap<u32, usize>>,
    totals: Vec<usize>,
}

impl Profile {
    pub(crate) fn new(bytes: &[u8], order: usize) -> Self {
        let bytes = bytes
            .iter()
            .copied()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        let mut counts = Vec::with_capacity(order);
        let mut totals = Vec::with_capacity(order);
        for width in 1..=order {
            let mut table = HashMap::new();
            for gram in bytes.windows(width) {
                let key = gram
                    .iter()
                    .fold(0, |key, byte| (key << 8) | u32::from(*byte));
                *table.entry(key).or_insert(0) += 1;
            }
            totals.push(bytes.len().saturating_sub(width - 1));
            counts.push(table);
        }
        Self { counts, totals }
    }

    pub(crate) fn similarity(&self, other: &Self) -> f64 {
        let mut similarity = 0.0;
        for (index, table) in self.counts.iter().enumerate() {
            let total = self.totals[index] + other.totals[index];
            if total == 0 {
                continue;
            }
            let overlap = table
                .iter()
                .map(|(key, count)| (*count).min(*other.counts[index].get(key).unwrap_or(&0)))
                .sum::<usize>();
            similarity += 2.0 * overlap as f64 / total as f64;
        }
        similarity / self.counts.len() as f64
    }
}

pub(crate) struct Contrast {
    target: Profile,
    decoys: Vec<Profile>,
    order: usize,
}

impl Contrast {
    pub(crate) fn new(target: &[u8], decoys: &[Vec<u8>], order: usize) -> Result<Self, String> {
        if !(1..=4).contains(&order) || target.is_empty() || decoys.is_empty() {
            return Err("code contrast requires 1..4 byte n-gram orders and target/decoys".into());
        }
        Ok(Self {
            target: Profile::new(target, order),
            decoys: decoys
                .iter()
                .map(|bytes| Profile::new(bytes, order))
                .collect(),
            order,
        })
    }

    pub(crate) fn components(&self, bytes: &[u8]) -> (f64, f64) {
        let output = Profile::new(bytes, self.order);
        let target = self.target.similarity(&output);
        let decoy = self
            .decoys
            .iter()
            .map(|profile| profile.similarity(&output))
            .fold(0.0, f64::max);
        (target, decoy)
    }

    pub(crate) fn reward(&self, bytes: &[u8]) -> f64 {
        let (target, decoy) = self.components(bytes);
        target - decoy
    }
}

#[cfg(test)]
#[path = "overlap_tests.rs"]
mod tests;
