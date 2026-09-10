use std::fs;
use std::path::PathBuf;

use crate::distance::l2_f32;
use crate::error::BpannError;
use crate::index::build::{BpannIndex, IndexHeader};
use crate::index::search::{refine_store, search_store, search_tree, MmapSearchStore};
use crate::index::LEAF_CAPACITY;
use crate::mmap_store::MmapColumnStore;
use crate::observation as obs;
use crate::tuning::current_tuning;

use crate::index::sync_forest::{
    centroid_rows, load_mmap, range_index, vector_index, IndexBuildContext,
};

const INDEX_MIN: usize = 3;

/// Soft-sync spans in `(2 * structured_limit, MID_MAX]`
/// use one in-RAM vector-leaf forest (leaf size `MID_CAPACITY`) so
/// ask prunes without deep k-means tell. Larger spans use one empty-leaf
/// forest (Internal → row-id leaves of `structured_limit`).
const MID_MAX: usize = 10_000;
/// Vector-leaf size inside the mid-band forest (also the greedy visit size).
const MID_CAPACITY: usize = 256;

fn index_threshold(indexed_rows: usize) -> usize {
    if indexed_rows <= 1000 {
        return 1;
    }
    let t = current_tuning();
    (indexed_rows / t.index_fragment).clamp(INDEX_MIN, t.index_max)
}

fn search_budget(fragment_count: usize, indexed_rows: usize) -> usize {
    if fragment_count <= 2 {
        return fragment_count;
    }
    let t = current_tuning();
    let scaled = (indexed_rows / t.search_fragment).max(2);
    scaled.min(fragment_count).min(t.search_limit)
}

fn search_width(_indexed_rows: usize) -> usize {
    current_tuning().search_width
}

#[derive(Clone)]
pub struct IncrementalIndex {
    pub indices: Vec<BpannIndex>,
    pub indexed_rows: usize,
    pub index_dir: PathBuf,
    pending_centroid_sum: Vec<f64>,
    pending_row_count: usize,
}

impl IncrementalIndex {
    pub fn new(index_dir: PathBuf) -> Self {
        Self {
            indices: Vec::new(),
            indexed_rows: 0,
            index_dir,
            pending_centroid_sum: Vec::new(),
            pending_row_count: 0,
        }
    }

    pub fn note_rows(&mut self, x: &ndarray::ArrayView2<f64>, scale_x: bool, x_scale: &[f64]) {
        if x.nrows() == 0 {
            return;
        }
        if self.pending_centroid_sum.len() != x.ncols() {
            self.pending_centroid_sum = vec![0.0; x.ncols()];
        }
        let col_sums = x.sum_axis(ndarray::Axis(0));
        for (j, &s) in col_sums.iter().enumerate() {
            self.pending_centroid_sum[j] += if scale_x { s / x_scale[j] } else { s };
        }
        self.pending_row_count += x.nrows();
    }

    fn take_centroid(&mut self, num_dim: usize) -> Option<Vec<f32>> {
        if self.pending_row_count == 0 {
            return None;
        }
        if self.pending_centroid_sum.len() != num_dim {
            return None;
        }
        let count = self.pending_row_count as f64;
        let centroid = self
            .pending_centroid_sum
            .iter()
            .map(|&s| (s / count) as f32)
            .collect();
        self.pending_centroid_sum.fill(0.0);
        self.pending_row_count = 0;
        Some(centroid)
    }

    pub fn reset(&mut self) {
        self.indices.clear();
        self.indexed_rows = 0;
        self.pending_centroid_sum.clear();
        self.pending_row_count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_backend(
        &mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
    ) -> Result<(), BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.index_through(&ctx, end)
    }

    pub fn persist_backend(
        &mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
    ) -> Result<(), BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.persist_index(&ctx)
    }

    pub fn needs_rewrite(&self, index_dirty: bool, nrows: usize) -> bool {
        if self.indices.len() > 1 {
            return true;
        }
        if index_dirty {
            return true;
        }
        let on_disk = match disk_rows(&self.index_dir) {
            Ok(on_disk) => on_disk,
            Err(_) => return true,
        };
        if on_disk != nrows {
            return true;
        }
        match self.indices.first() {
            Some(index) => !index.disk_index().unwrap_or(false),
            None => on_disk != 0 || nrows != 0,
        }
    }

    fn persist_index(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        self.index_through(ctx, ctx.train_x.nrows)?;
        if self.indices.is_empty() && self.indexed_rows == 0 {
            return Ok(());
        }
        if self.indices.len() > 1 {
            let merged =
                BpannIndex::concat_merge(self.indices.clone(), self.index_dir.clone(), false)?;
            merged.persist()?;
        } else if let Some(index) = self.indices.first() {
            index.persist()?;
        }
        obs::write_metadata(
            ctx.work_dir,
            ctx.train_x.nrows,
            ctx.num_dim,
            ctx.num_metrics,
            ctx.scale_x,
            self.indexed_rows,
        )?;
        obs::write_rows(ctx.work_dir, self.indexed_rows)?;
        Ok(())
    }

    fn index_through(&mut self, ctx: &IndexBuildContext<'_>, end: usize) -> Result<(), BpannError> {
        if self.indexed_rows >= end {
            return Ok(());
        }
        let limit = current_tuning().structured_limit;
        let pending = end - self.indexed_rows;
        // Soft-sync spans just above `limit` used to take one structured
        // `build_rows` tree (better ask, slower tell). Very large spans
        // become one empty-leaf forest (Internal → row-id leaves of `limit`).
        // Mid-band spans become one in-RAM vector-leaf forest so ask prunes
        // without k-means tell.
        let chunk_large_spans = pending > limit.saturating_mul(2);
        if chunk_large_spans && pending > MID_MAX {
            let _ = self.take_centroid(ctx.num_dim);
            self.build_ranges(ctx, self.indexed_rows, end, limit)?;
            self.maybe_compact(ctx)?;
        } else if chunk_large_spans {
            let _ = self.take_centroid(ctx.num_dim);
            self.build_vectors(ctx, self.indexed_rows, end, MID_CAPACITY)?;
            self.maybe_compact(ctx)?;
        } else {
            self.build_batch(ctx, self.indexed_rows, end)?;
            self.maybe_compact(ctx)?;
        }
        obs::write_rows(ctx.work_dir, self.indexed_rows)?;
        Ok(())
    }

    /// RAM amalgamation only: no `persist()`, no metadata rewrite.
    fn maybe_compact(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        let max_fragments = index_threshold(self.indexed_rows);
        let compact_limit = max_fragments.saturating_mul(2).max(max_fragments + 1);
        if self.indices.len() > compact_limit {
            self.compact(ctx)?;
        }
        Ok(())
    }

    fn compact(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        if self.indexed_rows == 0 {
            self.indices.clear();
            return Ok(());
        }
        let max_fragments = index_threshold(self.indexed_rows);
        while self.indices.len() > max_fragments {
            let over = self.indices.len() - max_fragments;
            let merge_n = over.clamp(2, 4).min(self.indices.len());
            self.amalgamate_run(ctx, merge_n)?;
        }
        Ok(())
    }

    fn amalgamate_run(
        &mut self,
        _ctx: &IndexBuildContext<'_>,
        merge_n: usize,
    ) -> Result<(), BpannError> {
        if self.indices.len() < 2 {
            return Ok(());
        }
        let merge_n = merge_n.min(self.indices.len());
        let mut best_i = 0usize;
        let mut best_rows = usize::MAX;
        let small_limit = current_tuning().fragment_rows;
        for i in 0..=self.indices.len().saturating_sub(merge_n) {
            let slice = &self.indices[i..i + merge_n];
            let rows: usize = slice.iter().map(|index| index.header.indexed_rows).sum();
            let all_small = slice.windows(2).all(|pair| {
                pair[0].header.indexed_rows <= small_limit
                    && pair[1].header.indexed_rows <= small_limit
            });
            let rank = if all_small {
                rows
            } else {
                rows + usize::MAX / 2
            };
            if rank < best_rows {
                best_rows = rank;
                best_i = i;
            }
        }
        let removed: Vec<BpannIndex> = self.indices.drain(best_i..best_i + merge_n).collect();
        let merged = BpannIndex::concat_merge(removed, self.index_dir.clone(), false)?;
        self.indices.insert(best_i, merged);
        Ok(())
    }

    #[allow(dead_code)]
    fn amalgamate_pair(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        self.amalgamate_run(ctx, 2)
    }

    /// Large soft-sync: one fragment of empty row-id leaves under an Internal root.
    fn build_ranges(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
        leaf_rows: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let index = range_index(ctx, start, end, leaf_rows, self.index_dir.clone())?;
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    /// Mid-band: one fragment of in-RAM vector leaves (no k-means).
    fn build_vectors(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
        leaf_rows: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let index = vector_index(ctx, start, end, leaf_rows, self.index_dir.clone())?;
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    fn build_batch(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let tuning = current_tuning();
        let seed = tuning.build_seed.unwrap_or(start as u64);
        let row_ids: Vec<u32> = (start..end).map(|i| i as u32).collect();
        let batch_len = end - start;
        let index = if batch_len <= tuning.structured_limit {
            let centroid = self
                .take_centroid(ctx.num_dim)
                .unwrap_or_else(|| centroid_rows(ctx, start, end).expect("centroid_rows"));
            BpannIndex::build_centroid(
                &row_ids,
                centroid,
                ctx.num_dim,
                self.index_dir.clone(),
                false,
            )?
        } else {
            // Structured leaf path takes pending for placement; the large path
            // builds its own tree from rows but must still drain the accumulator
            // so the next small sync is not poisoned by already-indexed mass.
            let _ = self.take_centroid(ctx.num_dim);
            let vectors = load_mmap(ctx, start, end)?;
            BpannIndex::build_tree(
                &row_ids,
                &vectors,
                ctx.num_dim,
                LEAF_CAPACITY,
                seed,
                self.index_dir.clone(),
                false,
            )?
        };
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    pub fn search_candidates(
        &self,
        query_f32: &[f32],
        k: usize,
        store: Option<&MmapSearchStore<'_>>,
    ) -> Result<Vec<(u32, f32)>, BpannError> {
        let budget = search_budget(self.indices.len(), self.indexed_rows);
        let indices_to_search: Vec<&BpannIndex> = if self.indices.len() <= budget {
            self.indices.iter().collect()
        } else {
            let mut ranked: Vec<(f32, &BpannIndex)> = self
                .indices
                .iter()
                .map(|index| {
                    let centroid = index.root_centroid();
                    let dist = if centroid.is_empty() {
                        f32::INFINITY
                    } else {
                        l2_f32(query_f32, &centroid)
                    };
                    (dist, index)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            ranked
                .into_iter()
                .take(budget)
                .map(|(_, index)| index)
                .collect()
        };
        let per_fragment_k = k
            .saturating_mul(self.indices.len())
            .div_ceil(indices_to_search.len().max(1))
            .max(k);
        let mut merged: Vec<(u32, f32)> = Vec::new();
        for index in indices_to_search {
            merged.extend(search_candidates(index, query_f32, per_fragment_k, store)?);
        }
        merged.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        merged.truncate(k);
        Ok(merged)
    }

    pub fn index_bytes(&self) -> usize {
        self.indices.iter().map(|i| i.index_bytes()).sum()
    }
}

fn search_candidates(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    store: Option<&MmapSearchStore<'_>>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let rows = index.header.indexed_rows;
    // Mode cliffs are call-time tuning. On-disk skip edges reflect build-time limits
    // and may be empty if limits changed since the fragment was built.
    let t = current_tuning();
    if t.exhaustive_search(rows) {
        search_store(index, query, k, store)
    } else {
        let beam = search_width(rows);
        let mut visited = Vec::new();
        if t.refine_search(rows) {
            refine_store(index, query, k, beam, &mut visited, store)
        } else {
            search_tree(index, query, k, beam, store)
        }
    }
}

fn disk_rows(index_dir: &std::path::Path) -> Result<usize, BpannError> {
    let header_path = index_dir.join("header.json");
    if !header_path.exists() {
        return Ok(0);
    }
    let text = fs::read_to_string(&header_path)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let header: IndexHeader =
        serde_json::from_str(&text).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    Ok(header.indexed_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmap_store::MmapColumnStore;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn search_max() {
        // Default search_limit=1: with many fragments, search only one.
        assert_eq!(search_budget(1, 100_000), 1);
        assert_eq!(search_budget(2, 100_000), 2);
        assert_eq!(search_budget(8, 100_000), 1);
        assert_eq!(search_budget(32, 800_000), 1);
    }

    #[test]
    fn sync_behavioral() {
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        for batch in 0..3 {
            let row0 = batch as f64 * 2.0;
            let chunk = array![[row0, 0.0], [row0 + 1.0, 0.0]];
            store.mmap_append(&chunk.view()).unwrap();
            let start = batch * 2;
            let end = start + 2;
            idx.ensure_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, end)
                .unwrap();
        }
        idx.persist_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1)
            .unwrap();
        assert!(dir.path().join("index/header.json").exists());
        assert_eq!(idx.indexed_rows, 6);
    }

    #[test]
    fn sync_behavioral2() {
        let dir = TempDir::new().unwrap();
        let index_dir = dir.path().join("index");

        assert_eq!(disk_rows(&index_dir).unwrap(), 0);

        let mut idx = IncrementalIndex::new(index_dir.clone());
        assert!(!idx.needs_rewrite(false, 0));
        assert!(idx.needs_rewrite(true, 0));

        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let chunk = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        store.mmap_append(&chunk.view()).unwrap();
        idx.ensure_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, 3)
            .unwrap();
        idx.persist_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1)
            .unwrap();

        assert_eq!(disk_rows(&index_dir).unwrap(), 3);
        assert!(idx.needs_rewrite(false, 999));
    }

    #[test]
    fn api_behavioral() {
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().to_path_buf());
        let x = array![[0.0, 1.0], [1.0, 0.0]];
        idx.note_rows(&x.view(), false, &[1.0, 1.0]);
        assert!(idx.take_centroid(2).is_some());
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        store.mmap_append(&x.view()).unwrap();
        idx.ensure_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, 2)
            .unwrap();
        let results = idx.search_candidates(&[0.0, 0.0], 1, None).unwrap();
        let _ = results;
        assert_eq!(idx.indexed_rows, 2);
        assert!(!idx.indices.is_empty());
        // Soft sync leaves pages on disk unwritten; hard persist materializes files.
        assert_eq!(idx.index_bytes(), 0);
        idx.persist_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1)
            .unwrap();
        assert!(idx.index_bytes() > 0);
    }

    #[test]
    fn sync_behavioral3() {
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        for batch in 0..8 {
            let row0 = batch as f64 * 2.0;
            let chunk = array![[row0, 0.0], [row0 + 1.0, 0.0]];
            store.mmap_append(&chunk.view()).unwrap();
            let start = batch * 2;
            let end = start + 2;
            idx.ensure_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, end)
                .unwrap();
        }
        assert!(idx.indices.len() <= 4);
        let _ = idx.search_candidates(&[1.0, 0.0], 2, None).unwrap();
    }

    /// Regression: soft-sync spans larger than `structured_limit` must still
    /// consume pending-centroid state (drained before chunked leaf builds). Otherwise
    /// the next structured fragment is placed with a contaminated centroid.
    #[test]
    fn soft_centroid() {
        let limit = current_tuning().structured_limit;
        let n = limit + 76; // force multi-chunk soft-sync
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];

        let mut large = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            large[[i, 0]] = (i as f64) * 0.01;
            large[[i, 1]] = 1.0;
        }
        store.mmap_append(&large.view()).unwrap();
        idx.note_rows(&large.view(), false, &scale);
        idx.ensure_backend(&store, 2, false, &scale, dir.path(), 1, n)
            .unwrap();

        assert_eq!(
            idx.pending_row_count, 0,
            "large soft-sync batch must clear pending_row_count"
        );
        assert!(
            idx.take_centroid(2).is_none(),
            "large soft-sync batch must leave no pending centroid to take"
        );

        // Metamorphic: next singleton fragment must place at the new row, not a blend
        // with the already-indexed large batch.
        let far = array![[10_000.0, 0.0]];
        store.mmap_append(&far.view()).unwrap();
        idx.note_rows(&far.view(), false, &scale);
        idx.ensure_backend(&store, 2, false, &scale, dir.path(), 1, n + 1)
            .unwrap();
        let centroid = idx
            .indices
            .last()
            .expect("singleton fragment")
            .root_centroid();
        assert!(
            (centroid[0] - 10_000.0).abs() < 1.0,
            "post-large-sync fragment must place near the new row, got {centroid:?}"
        );
    }

    #[test]
    fn soft_builds() {
        let limit = current_tuning().structured_limit;
        // Must exceed MID_MAX to take the empty-leaf chunk path.
        let n = MID_MAX + limit + 10;
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];
        let mut large = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            large[[i, 0]] = i as f64;
            large[[i, 1]] = 0.0;
        }
        store.mmap_append(&large.view()).unwrap();
        idx.note_rows(&large.view(), false, &scale);
        idx.ensure_backend(&store, 2, false, &scale, dir.path(), 1, n)
            .unwrap();
        assert_eq!(idx.indexed_rows, n);
        assert_eq!(
            idx.indices.len(),
            1,
            "expected one empty-leaf forest fragment"
        );
        assert!(
            idx.indices[0].pages.len() >= 3,
            "expected multi-page forest, got {} pages",
            idx.indices[0].pages.len()
        );
    }

    #[test]
    fn midband_forest() {
        let limit = current_tuning().structured_limit;
        let n = limit * 2 + 100;
        assert!(n > limit * 2 && n <= MID_MAX);
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];
        let mut mid = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            mid[[i, 0]] = i as f64;
        }
        store.mmap_append(&mid.view()).unwrap();
        idx.note_rows(&mid.view(), false, &scale);
        idx.ensure_backend(&store, 2, false, &scale, dir.path(), 1, n)
            .unwrap();
        assert_eq!(idx.indexed_rows, n);
        assert_eq!(
            idx.indices.len(),
            1,
            "expected one vector-leaf forest fragment"
        );
        assert_eq!(idx.indices[0].header.leaf_capacity, MID_CAPACITY);
        let has_vectors = idx.indices[0].pages.iter().any(
            |p| matches!(p, crate::index::page::Page::Leaf { vectors, .. } if !vectors.is_empty()),
        );
        assert!(has_vectors);
        assert!(!idx
            .search_candidates(&[1.0, 0.0], 3, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn midsize_fragment() {
        let limit = current_tuning().structured_limit;
        // Between limit and 2*limit: structured single-build path.
        let n = limit + limit / 2;
        assert!(n > limit && n <= limit * 2);
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];
        let mut mid = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            mid[[i, 0]] = i as f64;
            mid[[i, 1]] = 0.0;
        }
        store.mmap_append(&mid.view()).unwrap();
        idx.note_rows(&mid.view(), false, &scale);
        idx.ensure_backend(&store, 2, false, &scale, dir.path(), 1, n)
            .unwrap();
        assert_eq!(idx.indexed_rows, n);
        assert_eq!(
            idx.indices.len(),
            1,
            "mid-size sync should stay one structured fragment"
        );
        assert_eq!(idx.indices[0].header.indexed_rows, n);
    }
}
