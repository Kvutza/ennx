//! Exact unit-cost symbol edit distance, evaluated 64 pattern positions at once.
//! This is a corpus-reconstruction objective, not a semantic code verifier.

use std::collections::HashMap;

pub(crate) struct Reference<T> {
    length: usize,
    masks: HashMap<T, Vec<u64>>,
}

impl<T: Copy + Eq + std::hash::Hash> Reference<T> {
    pub(crate) fn new(tokens: &[T]) -> Self {
        let words = tokens.len().div_ceil(64);
        let mut masks = HashMap::<T, Vec<u64>>::new();
        for (position, &token) in tokens.iter().enumerate() {
            masks.entry(token).or_insert_with(|| vec![0; words])[position / 64] |=
                1 << (position % 64);
        }
        Self {
            length: tokens.len(),
            masks,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.length
    }

    pub(crate) fn distance(&self, tokens: &[T]) -> usize {
        if self.length == 0 {
            return tokens.len();
        }
        let words = self.length.div_ceil(64);
        let highest = 1 << ((self.length - 1) % 64);
        let mut positive = vec![u64::MAX; words];
        let mut negative = vec![0; words];
        let mut distance = self.length;
        for token in tokens {
            let equal = self.masks.get(token);
            let mut carry = Carry::default();
            for (word, (pv, mv)) in positive.iter_mut().zip(&mut negative).enumerate() {
                let eq = equal.map_or(0, |mask| mask[word]);
                let (ph, mh) = advance(pv, mv, eq, &mut carry);
                if word + 1 == words {
                    distance += usize::from(ph & highest != 0);
                    distance -= usize::from(mh & highest != 0);
                }
            }
        }
        distance
    }

    pub(crate) fn reward(&self, tokens: &[T]) -> (usize, f32) {
        let distance = self.distance(tokens);
        let length = self.length.max(tokens.len());
        let reward = if length == 0 {
            1.0
        } else {
            1.0 - distance as f32 / length as f32
        };
        (distance, reward)
    }
}

struct Carry {
    addition: u64,
    positive: u64,
    negative: u64,
}

impl Default for Carry {
    fn default() -> Self {
        Self {
            addition: 0,
            positive: 1,
            negative: 0,
        }
    }
}

fn advance(pv: &mut u64, mv: &mut u64, equal: u64, carry: &mut Carry) -> (u64, u64) {
    let vertical = equal | *mv;
    let (sum, overflow) = (equal & *pv).overflowing_add(*pv);
    let (sum, propagated) = sum.overflowing_add(carry.addition);
    carry.addition = u64::from(overflow || propagated);
    let horizontal = (sum ^ *pv) | equal;
    let positive = *mv | !(horizontal | *pv);
    let negative = *pv & horizontal;
    let shifted_positive = (positive << 1) | carry.positive;
    let shifted_negative = (negative << 1) | carry.negative;
    carry.positive = positive >> 63;
    carry.negative = negative >> 63;
    *pv = shifted_negative | !(vertical | shifted_positive);
    *mv = shifted_positive & vertical;
    (positive, negative)
}

#[cfg(test)]
#[path = "reconstruction_tests.rs"]
mod tests;
