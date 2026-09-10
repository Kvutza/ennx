use std::collections::HashSet;

use bpann::index::build::{BpannIndex, LEAF_CAPACITY};
use bpann::index::search::{self, TraversalLog};
use bpann::mmap_store::MmapColumnStore;
use ndarray::ArrayView1;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

#[test]
fn tree_blocks() {
    let vectors = synth(256, 8, 42);
    let dir = TempDir::new().unwrap();
    let index =
        BpannIndex::build_vectors(&vectors, 8, LEAF_CAPACITY, 0, dir.path().join("index")).unwrap();
    let total_leaves = index.leaf_page().len();
    assert!(total_leaves > 1);
    let query = &vectors[0];
    let mut log = TraversalLog::new();
    bpann::index::search::search_refinement(&index, query, 10, 2, &mut log.visited_pages);
    assert!(!log.visited_pages.is_empty());
    assert!(log.visited_pages.len() <= total_leaves);
    assert!(log.visited_pages.len() < 256);
}

#[test]
fn skip_greedy() {
    let vectors = synth(128, 16, 99);
    let dir = TempDir::new().unwrap();
    let index = BpannIndex::build_vectors(&vectors, 16, LEAF_CAPACITY, 0, dir.path().join("index"))
        .unwrap();
    let queries = synth(5, 16, 1);
    let k = 10;
    let mut greedy_total = 0.0;
    let mut skip_total = 0.0;
    for q in &queries {
        let bf = bpann::index::search::bpann_topk(&vectors, q, k);
        let bf_set: HashSet<u32> = bf.iter().map(|(id, _)| *id).collect();
        let greedy = bpann::index::search::search_only(&index, q, k, 2);
        let skip = bpann::index::search::search_refinement(&index, q, k, 2, &mut Vec::new());
        greedy_total +=
            greedy.iter().filter(|(id, _)| bf_set.contains(id)).count() as f64 / k as f64;
        skip_total += skip.iter().filter(|(id, _)| bf_set.contains(id)).count() as f64 / k as f64;
    }
    assert!(skip_total + 1e-9 >= greedy_total);
}

#[test]
fn leaf_pairwise() {
    let vectors = synth(16, 4, 7);
    let query = &vectors[3];
    let batched = bpann::distance::sq_f32(query, &vectors);
    for (i, v) in vectors.iter().enumerate() {
        let pairwise = bpann::distance::l2_f32(query, v);
        assert!((batched[i] - pairwise).abs() < 1e-6);
    }
}

#[test]
fn search_neighbors() {
    let vectors = synth(64, 8, 42);
    let dir = TempDir::new().unwrap();
    let index =
        BpannIndex::build_vectors(&vectors, 8, LEAF_CAPACITY, 0, dir.path().join("index")).unwrap();
    let mut visited = Vec::new();
    let results =
        search::search_index(&index, &vectors[0], 5, 2, true, &mut visited, None).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results.len(), 5);
    assert!(!visited.is_empty());
    assert_eq!(search::search_leaves(&index, &vectors[0], 5).len(), 5);
}

#[test]
fn mean_empty() {
    let vectors = synth(8, 4, 1);
    let dir = TempDir::new().unwrap();
    let index =
        BpannIndex::build_vectors(&vectors, 4, LEAF_CAPACITY, 0, dir.path().join("index")).unwrap();
    assert_eq!(bpann::index::search::bpann_k(&vectors, &[], 3, &index), 0.0);
}

#[test]
fn sq_l22() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::open_mmap(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(
            &ndarray::Array2::from_shape_fn((2, 2), |(i, j)| match (i, j) {
                (0, 0) => 4.0,
                (0, 1) => 8.0,
                (1, 0) => 0.0,
                (1, 1) => 0.0,
                _ => unreachable!(),
            })
            .view(),
        )
        .unwrap();
    let query = [2.0, 4.0];
    let x_scale = [2.0, 4.0];
    let top = bpann::index::search::bpann_mmap(&store, 0, 2, &query, 2, true, &x_scale).unwrap();
    for (row_id, dist) in top {
        let row = store.row_slice(row_id as usize).unwrap();
        let expected = bpann::distance::sq_l2(
            ArrayView1::from(&query),
            ArrayView1::from(row),
            true,
            ArrayView1::from(&x_scale),
        ) as f32;
        assert!(
            (dist - expected).abs() < 1e-5,
            "row {row_id}: got {dist}, expected {expected}"
        );
    }
}

#[test]
fn brute_paths() {
    let vectors = synth(4, 3, 2);
    let q = &vectors[1];
    let top = bpann::index::search::bpann_topk(&vectors, q, 2);
    assert_eq!(top[0].0, 1);
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::open_mmap(dir.path().join("x.bin"), 3, None).unwrap();
    store
        .mmap_append(&ndarray::Array2::from_shape_fn((4, 3), |(i, j)| vectors[i][j] as f64).view())
        .unwrap();
    let scaled =
        bpann::index::search::bpann_mmap(&store, 0, 4, &[0.0, 0.0, 0.0], 2, true, &[1.0, 1.0, 1.0])
            .unwrap();
    assert_eq!(scaled.len(), 2);
}
