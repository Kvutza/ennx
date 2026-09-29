//! Bayesian spectral programs for dense architecture-aware directions.

use crate::hash::normal_metric;

pub const INPUT_BITS: usize = 16;
pub const TABLE_BITS: usize = 1 << INPUT_BITS;
pub const TABLE_WORDS: usize = TABLE_BITS / 64;
pub const FEATURE_COUNT: usize = feature_count();
// Retain the complete basis so initialization can reach posterior sampling.
const MAX_ROWS: usize = FEATURE_COUNT;
const GROUPS: [(usize, usize); 4] = [(0, 5), (5, 3), (8, 4), (12, 4)];
const INTERACTIONS: [(usize, usize, usize, usize); 5] = [
    (0, 5, 5, 3),
    (0, 5, 8, 4),
    (0, 5, 12, 4),
    (5, 3, 8, 4),
    (5, 3, 12, 4),
];
const fn feature_count() -> usize {
    let mut count = 1;
    let mut i = 0;
    while i < GROUPS.len() {
        count += (1 << GROUPS[i].1) - 1;
        i += 1;
    }
    i = 0;
    while i < INTERACTIONS.len() {
        count += INTERACTIONS[i].1 * INTERACTIONS[i].3;
        i += 1;
    }
    count
}

const FEATURE_MASKS: [usize; FEATURE_COUNT] = build_masks();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThresholdTable {
    words: Box<[u64; TABLE_WORDS]>,
}

impl ThresholdTable {
    pub fn basis(step: usize) -> Self {
        Self::parity(step % FEATURE_COUNT)
    }

    pub fn parity(feature: usize) -> Self {
        assert!(feature < FEATURE_COUNT, "threshold feature is out of range");
        let mut coefficients = [0.0; FEATURE_COUNT];
        coefficients[feature] = 1.0;
        Self::from_coefficients(&coefficients)
    }

    pub fn from_coefficients(coefficients: &[f64; FEATURE_COUNT]) -> Self {
        let mut values = vec![0.0; TABLE_BITS];
        for (mask, coefficient) in feature_masks().zip(coefficients) {
            values[mask] = *coefficient;
        }
        hadamard(&mut values);
        let mut words = Box::new([0; TABLE_WORDS]);
        for (index, value) in values.into_iter().enumerate() {
            if value >= 0.0 {
                words[index / 64] |= 1 << (index % 64);
            }
        }
        Self { words }
    }

    pub fn from_words(words: &[u64]) -> Result<Self, String> {
        let words: Box<[u64; TABLE_WORDS]> = words
            .to_vec()
            .into_boxed_slice()
            .try_into()
            .map_err(|_| format!("threshold table requires {TABLE_WORDS} words"))?;
        Ok(Self { words })
    }

    pub fn words(&self) -> &[u64; TABLE_WORDS] {
        &self.words
    }

    pub fn features(&self) -> [f64; FEATURE_COUNT] {
        let mut values = (0..TABLE_BITS)
            .map(|index| {
                if self.words[index / 64] & (1 << (index % 64)) == 0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>();
        hadamard(&mut values);
        let scale = 1.0 / TABLE_BITS as f64;
        std::array::from_fn(|feature| values[feature_mask(feature)] * scale)
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let differing = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(left, right)| (left ^ right).count_ones())
            .sum::<u32>();
        4.0 * f64::from(differing) / TABLE_BITS as f64
    }
}

#[derive(Clone, Debug)]
struct Observation {
    features: [f64; FEATURE_COUNT],
    improvement: f64,
    variance: f64,
}

#[derive(Clone, Debug)]
pub struct ThresholdModel {
    observations: Vec<Observation>,
    noise_floor: f64,
    radius_scale: Option<f64>,
}

impl ThresholdModel {
    pub fn new(noise_floor: f64) -> Result<Self, String> {
        if !noise_floor.is_finite() || noise_floor <= 0.0 {
            return Err("threshold noise floor must be positive and finite".into());
        }
        Ok(Self {
            observations: Vec::with_capacity(MAX_ROWS),
            noise_floor,
            radius_scale: None,
        })
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn clear(&mut self) {
        self.observations.clear();
        self.radius_scale = None;
    }

    pub fn observe(
        &mut self,
        table: &ThresholdTable,
        radius: f64,
        improvement: f64,
        variance: f64,
    ) -> Result<(), String> {
        if !radius.is_finite()
            || radius <= 0.0
            || !improvement.is_finite()
            || !variance.is_finite()
            || variance < 0.0
        {
            return Err(
                "threshold observation requires a positive radius and finite response".into(),
            );
        }
        if self.observations.len() == MAX_ROWS {
            self.observations.remove(0);
        }
        let radius_scale = *self.radius_scale.get_or_insert(radius);
        let relative_radius = radius / radius_scale;
        let mut features = table.features();
        features
            .iter_mut()
            .for_each(|feature| *feature *= relative_radius);
        self.observations.push(Observation {
            features,
            improvement,
            variance,
        });
        Ok(())
    }

    pub fn draw(&self, seed: u64, stream: u64) -> Result<ThresholdTable, String> {
        self.draw_batch(seed, &[stream])
            .map(|mut tables| tables.remove(0))
    }

    pub fn draw_batch(&self, seed: u64, streams: &[u64]) -> Result<Vec<ThresholdTable>, String> {
        if self.observations.len() < FEATURE_COUNT {
            return Ok(vec![
                ThresholdTable::parity(self.observations.len());
                streams.len()
            ]);
        }
        let mut precision = vec![0.0; FEATURE_COUNT * FEATURE_COUNT];
        let mut rhs = [0.0; FEATURE_COUNT];
        for feature in 0..FEATURE_COUNT {
            precision[feature * FEATURE_COUNT + feature] = prior_precision(feature);
        }
        for observation in &self.observations {
            let weight = 1.0 / (observation.variance + self.noise_floor);
            for row in 0..FEATURE_COUNT {
                rhs[row] += weight * observation.features[row] * observation.improvement;
                for column in 0..=row {
                    precision[row * FEATURE_COUNT + column] +=
                        weight * observation.features[row] * observation.features[column];
                }
            }
        }
        for row in 0..FEATURE_COUNT {
            for column in 0..row {
                precision[column * FEATURE_COUNT + row] = precision[row * FEATURE_COUNT + column];
            }
        }
        cholesky(&mut precision)?;
        let mut mean = rhs;
        solve_lower(&precision, &mut mean);
        solve_upper(&precision, &mut mean);
        Ok(streams
            .iter()
            .map(|stream| {
                let mut sample = std::array::from_fn(|feature| {
                    normal_metric(seed ^ stream.rotate_left(17), feature as i64, 0)
                });
                solve_upper(&precision, &mut sample);
                for (value, center) in sample.iter_mut().zip(mean.iter().copied()) {
                    *value += center;
                }
                ThresholdTable::from_coefficients(&sample)
            })
            .collect())
    }
}

fn prior_precision(_feature: usize) -> f64 {
    1.0
}

fn feature_masks() -> impl Iterator<Item = usize> {
    FEATURE_MASKS.into_iter()
}

fn feature_mask(feature: usize) -> usize {
    FEATURE_MASKS[feature]
}

const fn build_masks() -> [usize; FEATURE_COUNT] {
    let mut masks = [0; FEATURE_COUNT];
    let mut feature = 1;
    let mut group = 0;
    while group < GROUPS.len() {
        let (shift, width) = GROUPS[group];
        let mut contrast = 1;
        while contrast < 1 << width {
            masks[feature] = contrast << shift;
            feature += 1;
            contrast += 1;
        }
        group += 1;
    }
    let mut interaction = 0;
    while interaction < INTERACTIONS.len() {
        let (left_shift, left_width, right_shift, right_width) = INTERACTIONS[interaction];
        let mut left = 0;
        while left < left_width {
            let mut right = 0;
            while right < right_width {
                masks[feature] = (1 << (left_shift + left)) | (1 << (right_shift + right));
                feature += 1;
                right += 1;
            }
            left += 1;
        }
        interaction += 1;
    }
    assert!(feature == FEATURE_COUNT);
    masks
}

fn hadamard(values: &mut [f64]) {
    let mut width = 1;
    while width < values.len() {
        for block in values.chunks_exact_mut(width * 2) {
            for index in 0..width {
                let left = block[index];
                let right = block[index + width];
                block[index] = left + right;
                block[index + width] = left - right;
            }
        }
        width *= 2;
    }
}

fn cholesky(matrix: &mut [f64]) -> Result<(), String> {
    for row in 0..FEATURE_COUNT {
        for column in 0..=row {
            let mut value = matrix[row * FEATURE_COUNT + column];
            for inner in 0..column {
                value -=
                    matrix[row * FEATURE_COUNT + inner] * matrix[column * FEATURE_COUNT + inner];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err("threshold posterior precision is not positive definite".into());
                }
                matrix[row * FEATURE_COUNT + column] = value.sqrt();
            } else {
                matrix[row * FEATURE_COUNT + column] =
                    value / matrix[column * FEATURE_COUNT + column];
            }
        }
        for column in row + 1..FEATURE_COUNT {
            matrix[row * FEATURE_COUNT + column] = 0.0;
        }
    }
    Ok(())
}

fn solve_lower(matrix: &[f64], values: &mut [f64; FEATURE_COUNT]) {
    for row in 0..FEATURE_COUNT {
        for column in 0..row {
            values[row] -= matrix[row * FEATURE_COUNT + column] * values[column];
        }
        values[row] /= matrix[row * FEATURE_COUNT + row];
    }
}

fn solve_upper(matrix: &[f64], values: &mut [f64; FEATURE_COUNT]) {
    for row in (0..FEATURE_COUNT).rev() {
        for column in row + 1..FEATURE_COUNT {
            values[row] -= matrix[column * FEATURE_COUNT + row] * values[column];
        }
        values[row] /= matrix[row * FEATURE_COUNT + row];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_layout() {
        let masks = feature_masks().collect::<Vec<_>>();
        assert_eq!(masks.len(), FEATURE_COUNT);
        assert_eq!(masks[0], 0);
        assert_eq!(masks[1], 1);
        assert!(masks.contains(&0b11111));
        assert!(masks.contains(&(0b111 << 5)));
        assert!(masks.contains(&(0b1111 << 8)));
        assert!(masks.contains(&(0b1111 << 12)));
        assert_eq!(masks.last().copied(), Some((1 << 7) | (1 << 15)));
        let mut sorted = masks.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), FEATURE_COUNT);
    }

    #[test]
    fn linear_threshold() {
        let mut coefficients = [0.0; FEATURE_COUNT];
        coefficients[1] = 1.0;
        let table = ThresholdTable::from_coefficients(&coefficients);
        let features = table.features();
        assert!((features[1] - 1.0).abs() < 1.0e-12);
        assert!(
            features
                .iter()
                .enumerate()
                .all(|(index, value)| index == 1 || value.abs() < 1.0e-12)
        );
    }

    #[test]
    fn interaction_threshold() {
        let mut coefficients = [0.0; FEATURE_COUNT];
        coefficients[53] = 1.0;
        let table = ThresholdTable::from_coefficients(&coefficients);
        let features = table.features();
        assert!((features[53] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn basis_cycles() {
        assert_eq!(ThresholdTable::basis(0), ThresholdTable::parity(0));
        assert_eq!(
            ThresholdTable::basis(FEATURE_COUNT),
            ThresholdTable::parity(0)
        );
    }

    #[test]
    fn posterior_repeats() {
        let mut model = ThresholdModel::new(0.05).unwrap();
        for feature in 0..FEATURE_COUNT {
            let table = model.draw(7, 3).unwrap();
            assert_eq!(table, ThresholdTable::parity(feature));
            model.observe(&table, 1.0, 0.0, 0.0).unwrap();
        }
        let first = model.draw(7, 3).unwrap();
        let second = model.draw(7, 3).unwrap();
        let batch = model.draw_batch(7, &[3, 4]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first, batch[0]);
        assert_eq!(model.draw(7, 4).unwrap(), batch[1]);
        assert!(first.distance(&second) == 0.0);
        assert!(first.distance(&model.draw(7, 4).unwrap()) > 0.0);
    }

    #[test]
    fn posterior_learns() {
        let table = ThresholdTable::parity(1);
        let mut model = ThresholdModel::new(0.001).unwrap();
        for feature in 0..FEATURE_COUNT {
            let basis = ThresholdTable::parity(feature);
            let improvement = if feature == 1 { 100.0 } else { 0.0 };
            model.observe(&basis, 1.0, improvement, 0.0).unwrap();
        }
        let draw = model.draw(11, 9).unwrap();
        assert!(table.distance(&draw) < 2.0);
    }

    #[test]
    fn radius_scaling() {
        let table = ThresholdTable::parity(3);
        let mut model = ThresholdModel::new(0.001).unwrap();
        model.observe(&table, 0.25, 1.0, 0.0).unwrap();
        model.observe(&table, 0.5, 2.0, 0.0).unwrap();
        assert!((model.observations[0].features[3] - 1.0).abs() < 1.0e-12);
        assert!((model.observations[1].features[3] - 2.0).abs() < 1.0e-12);
    }
}
