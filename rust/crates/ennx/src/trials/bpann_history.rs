use std::collections::HashSet;
use std::path::PathBuf;

use bpann::BpannBackend;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

/// Stable row identifier returned by the compact BPANN history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObservationId(pub u64);

/// An indexed observation selected for exact full-dimensional reranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexedObservation {
    pub id: ObservationId,
    pub value: f32,
}

/// Disk-backed BPANN index over compact observation descriptors.
///
/// Full quantized model rows deliberately do not live here. A caller stores or
/// regenerates those rows separately and uses the returned [`ObservationId`]s
/// to resolve only the shortlist loaded into `PackedSearch::replace_history`.
pub struct BpannHistory {
    backend: BpannBackend,
    descriptor_dim: usize,
}

impl BpannHistory {
    pub fn new(work_dir: PathBuf, descriptor_dim: usize) -> Result<Self, String> {
        if descriptor_dim == 0 {
            return Err("BPANN history descriptor dimension must be positive".to_string());
        }
        let backend = BpannBackend::new_empty(work_dir, descriptor_dim, 1)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            backend,
            descriptor_dim,
        })
    }

    pub fn len(&self) -> usize {
        self.backend.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn descriptor_dim(&self) -> usize {
        self.descriptor_dim
    }

    /// Append a compact descriptor and objective value.
    ///
    /// The returned ID is the persistent BPANN row number and can be used by a
    /// separate implicit/checkpointed model archive.
    pub fn append(
        &mut self,
        descriptor: &ArrayView1<'_, f64>,
        value: f32,
    ) -> Result<ObservationId, String> {
        if descriptor.len() != self.descriptor_dim {
            return Err(format!(
                "descriptor has {} dimensions, expected {}",
                descriptor.len(),
                self.descriptor_dim
            ));
        }
        if !value.is_finite() {
            return Err("observation value must be finite".to_string());
        }
        let id = ObservationId(
            u64::try_from(self.backend.len())
                .map_err(|_| "BPANN observation ID exceeds u64 range".to_string())?,
        );
        self.backend
            .append_row(
                &descriptor.to_owned(),
                &Array1::from_vec(vec![f64::from(value)]),
                None,
            )
            .map_err(|error| error.to_string())?;
        Ok(id)
    }

    pub fn sync(&mut self) -> Result<(), String> {
        self.backend
            .ensure_index_sync()
            .map_err(|error| error.to_string())
    }

    /// Search compact descriptors and return BPANN row IDs for each query.
    pub fn search(
        &self,
        queries: &ArrayView2<'_, f64>,
        neighbors: usize,
    ) -> Result<Vec<Vec<ObservationId>>, String> {
        if queries.ncols() != self.descriptor_dim {
            return Err(format!(
                "query descriptors have {} dimensions, expected {}",
                queries.ncols(),
                self.descriptor_dim
            ));
        }
        if neighbors == 0 {
            return Err("BPANN history search requires at least one neighbor".to_string());
        }
        if self.is_empty() {
            return Ok(vec![Vec::new(); queries.nrows()]);
        }
        let (_, indices) = self
            .backend
            .search(queries, neighbors, false)
            .map_err(|error| error.to_string())?;
        indices
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .map(|&index| {
                        u64::try_from(index)
                            .map(ObservationId)
                            .map_err(|_| format!("BPANN returned invalid observation ID {index}"))
                    })
                    .collect()
            })
            .collect()
    }

    /// Return a stable, deduplicated union of per-candidate neighbors and their
    /// objective values, suitable for resolving and loading into PackedSearch.
    pub fn shortlist(
        &self,
        queries: &ArrayView2<'_, f64>,
        neighbors_per_query: usize,
        max_observations: usize,
    ) -> Result<Vec<IndexedObservation>, String> {
        if max_observations == 0 {
            return Err("BPANN shortlist capacity must be positive".to_string());
        }
        let per_query = self.search(queries, neighbors_per_query)?;
        let mut seen = HashSet::new();
        let mut ids = Vec::new();
        let max_rank = per_query.iter().map(Vec::len).max().unwrap_or(0);
        'ranks: for rank in 0..max_rank {
            for row in &per_query {
                let Some(&id) = row.get(rank) else {
                    continue;
                };
                if seen.insert(id) {
                    ids.push(id);
                    if ids.len() == max_observations {
                        break 'ranks;
                    }
                }
            }
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let row_indices: Vec<usize> = ids
            .iter()
            .map(|id| {
                usize::try_from(id.0)
                    .map_err(|_| format!("observation ID {} exceeds usize range", id.0))
            })
            .collect::<Result<_, _>>()?;
        let (_, values, _) = self
            .backend
            .train_rows_at(&row_indices)
            .map_err(|error| error.to_string())?;
        Ok(ids
            .into_iter()
            .zip(values.column(0).iter().copied())
            .map(|(id, value)| IndexedObservation {
                id,
                value: value as f32,
            })
            .collect())
    }

    pub fn persist(&mut self) -> Result<(), String> {
        self.backend
            .persist_index_to_disk()
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};
    use tempfile::TempDir;

    #[test]
    fn bpann_history_returns_stable_ids_and_values() {
        let dir = TempDir::new().unwrap();
        let mut history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
        let descriptors = array![[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]];
        let values = [10.0, 20.0, 40.0];
        for (row, value) in descriptors.axis_iter(Axis(0)).zip(values) {
            let expected = history.len() as u64;
            assert_eq!(
                history.append(&row, value).unwrap(),
                ObservationId(expected)
            );
        }
        history.sync().unwrap();

        let queries = array![[0.1, 0.0], [3.9, 0.0]];
        let found = history.search(&queries.view(), 1).unwrap();
        assert_eq!(found, vec![vec![ObservationId(0)], vec![ObservationId(2)]]);

        let shortlist = history.shortlist(&queries.view(), 1, 2).unwrap();
        assert_eq!(
            shortlist,
            vec![
                IndexedObservation {
                    id: ObservationId(0),
                    value: 10.0,
                },
                IndexedObservation {
                    id: ObservationId(2),
                    value: 40.0,
                },
            ]
        );
    }

    #[test]
    fn bpann_history_deduplicates_shortlist_in_query_order() {
        let dir = TempDir::new().unwrap();
        let mut history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
        let descriptors = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        for (index, row) in descriptors.axis_iter(Axis(0)).enumerate() {
            history.append(&row, index as f32).unwrap();
        }
        let queries = array![[0.1, 0.0], [0.2, 0.0], [1.9, 0.0]];
        let shortlist = history.shortlist(&queries.view(), 2, 3).unwrap();
        let ids: Vec<_> = shortlist.iter().map(|item| item.id).collect();
        assert_eq!(
            ids,
            vec![ObservationId(0), ObservationId(2), ObservationId(1)]
        );
    }

    #[test]
    fn bpann_history_validates_descriptor_shapes() {
        let dir = TempDir::new().unwrap();
        let mut history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
        let descriptor = array![1.0, 2.0, 3.0];
        assert!(history.append(&descriptor.view(), 1.0).is_err());
        let queries = array![[1.0, 2.0, 3.0]];
        assert!(history.search(&queries.view(), 1).is_err());
        assert!(history.search(&queries.slice(s![.., ..2]), 0).is_err());
    }

    #[test]
    fn bpann_history_reopens_with_stable_observation_ids() {
        let dir = TempDir::new().unwrap();
        {
            let mut history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
            history.append(&array![0.0, 0.0].view(), 2.5).unwrap();
            history.append(&array![3.0, 0.0].view(), 7.5).unwrap();
            history.sync().unwrap();
            history.persist().unwrap();
        }

        let history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
        assert_eq!(history.len(), 2);
        let shortlist = history.shortlist(&array![[2.9, 0.0]].view(), 1, 1).unwrap();
        assert_eq!(
            shortlist,
            vec![IndexedObservation {
                id: ObservationId(1),
                value: 7.5,
            }]
        );
    }
}
