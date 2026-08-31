use ndarray::{Array2, ArrayView2};

use crate::index::IndexError;

use super::pad_neighbor_cols_to_search_k;

pub(crate) struct ExactBackend {
    rows: Array2<f64>,
    keys: Vec<u64>,
    num_dim: usize,
}

impl ExactBackend {
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        if train_scaled.ncols() != num_dim {
            return Err(IndexError::InvalidShape {
                expected: num_dim,
                got: train_scaled.ncols(),
            });
        }
        Ok(Self {
            rows: train_scaled.to_owned(),
            keys: (0..train_scaled.nrows()).map(|i| i as u64).collect(),
            num_dim,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.nrows()
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.rows.len().saturating_mul(std::mem::size_of::<f64>())
            + self.keys.len().saturating_mul(std::mem::size_of::<u64>())
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        if train_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: train_scaled.ncols(),
            });
        }
        self.rows = train_scaled.to_owned();
        self.keys = (0..train_scaled.nrows()).map(|i| i as u64).collect();
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        if rows_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows_scaled.ncols(),
            });
        }
        if rows_scaled.nrows() == 0 {
            return Ok(());
        }
        self.rows = ndarray::concatenate![ndarray::Axis(0), self.rows.view(), rows_scaled.view()];
        self.keys
            .extend((0..rows_scaled.nrows()).map(|i| start_key + i as u64));
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if queries_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: queries_scaled.ncols(),
            });
        }
        let mut dist2s = Array2::zeros((queries_scaled.nrows(), k_eff));
        let mut indices = Array2::zeros((queries_scaled.nrows(), k_eff));
        for (qi, query) in queries_scaled.outer_iter().enumerate() {
            let mut pairs = self
                .rows
                .outer_iter()
                .zip(self.keys.iter().copied())
                .map(|(row, key)| {
                    let dist2 = row
                        .iter()
                        .zip(query.iter())
                        .map(|(left, right)| {
                            let delta = left - right;
                            delta * delta
                        })
                        .sum::<f64>();
                    (dist2, key)
                })
                .collect::<Vec<_>>();
            pairs.sort_unstable_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            for (ki, (dist2, key)) in pairs.into_iter().take(k_eff).enumerate() {
                dist2s[[qi, ki]] = dist2;
                indices[[qi, ki]] = key as i64;
            }
        }
        Ok(pad_neighbor_cols_to_search_k(dist2s, indices, search_k))
    }
}
