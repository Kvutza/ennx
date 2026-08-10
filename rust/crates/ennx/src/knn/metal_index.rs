use std::ffi::c_void;
use std::sync::Arc;
use std::time::{Duration, Instant};

use metal::{ComputePipelineState, MTLSize};
use ndarray::{Array2, ArrayView2};

use super::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k};
use super::{KnnPlan, KnnProfile};
use crate::apple_gpu::Runtime;
use crate::index::IndexError;
use crate::knn::metal_plan::Plan;

const THREADS: u64 = 256;
const TILE_ROWS: usize = 1024;
const GRAM_ROWS: usize = 64;
const GRAM_QUERIES: usize = 8;
const MAX_K: usize = 2048;
const SOURCE: &str = include_str!("metal_index.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    dim: u32,
    queries: u32,
    tile_start: u32,
    tile_rows: u32,
    k: u32,
    lanes: u32,
    tiles: u32,
    lists: u32,
    groups: u32,
}

pub(crate) struct MetalIndex {
    runtime: Arc<Runtime>,
    agx: bool,
    distance: Vec<ComputePipelineState>,
    distance_choice: Option<usize>,
    topk16: ComputePipelineState,
    fused16: ComputePipelineState,
    tree16: ComputePipelineState,
    tiled16: ComputePipelineState,
    simd16: ComputePipelineState,
    gram16: ComputePipelineState,
    reduce16: ComputePipelineState,
    wide: ComputePipelineState,
    fold: ComputePipelineState,
    local_topk: ComputePipelineState,
    merge: ComputePipelineState,
    init_results: ComputePipelineState,
    rows: metal::Buffer,
    row_norms: metal::Buffer,
    host_rows: Vec<f32>,
    host_norms: Vec<f32>,
    row_capacity: usize,
    scratch: Scratch,
    num_dim: usize,
    plan: Plan,
    request: KnnPlan,
    profile: Option<KnnProfile>,
}

struct Scratch {
    query: metal::Buffer,
    query_norms: metal::Buffer,
    tile_distances: metal::Buffer,
    local_distances: metal::Buffer,
    local_indices: metal::Buffer,
    result_distances: metal::Buffer,
    result_indices: metal::Buffer,
    tree_distances: metal::Buffer,
    tree_indices: metal::Buffer,
    tree_alt_distances: metal::Buffer,
    tree_alt_indices: metal::Buffer,
    query_capacity: usize,
    k_capacity: usize,
    tree_query_capacity: usize,
    tree_k_capacity: usize,
    tile_capacity: usize,
    group_capacity: usize,
}

impl Scratch {
    fn new(runtime: &Runtime) -> Self {
        let empty = || runtime.buffer::<u32>(1);
        Self {
            query: empty(),
            query_norms: empty(),
            tile_distances: empty(),
            local_distances: empty(),
            local_indices: empty(),
            result_distances: empty(),
            result_indices: empty(),
            tree_distances: empty(),
            tree_indices: empty(),
            tree_alt_distances: empty(),
            tree_alt_indices: empty(),
            query_capacity: 0,
            k_capacity: 0,
            tree_query_capacity: 0,
            tree_k_capacity: 0,
            tile_capacity: 0,
            group_capacity: 0,
        }
    }

    fn ensure(&mut self, runtime: &Runtime, dim: usize, queries: usize, k: usize) {
        if queries <= self.query_capacity && k <= self.k_capacity {
            return;
        }
        self.query_capacity = next_capacity(queries);
        self.k_capacity = next_capacity(k);
        self.query = runtime.buffer::<f32>(self.query_capacity * dim);
        self.query_norms = runtime.buffer::<f32>(self.query_capacity);
        self.tile_distances = runtime.buffer::<f32>(self.query_capacity * TILE_ROWS);
        self.local_distances = runtime.buffer::<f32>(self.query_capacity * self.k_capacity);
        self.local_indices = runtime.buffer::<u32>(self.query_capacity * self.k_capacity);
        self.result_distances = runtime.buffer::<f32>(self.query_capacity * self.k_capacity);
        self.result_indices = runtime.buffer::<u32>(self.query_capacity * self.k_capacity);
    }

    fn ensure_tree(
        &mut self,
        runtime: &Runtime,
        queries: usize,
        k: usize,
        tiles: usize,
        fan: usize,
    ) {
        let groups = tiles.div_ceil(fan);
        if queries <= self.tree_query_capacity
            && k <= self.tree_k_capacity
            && tiles <= self.tile_capacity
            && groups <= self.group_capacity
        {
            return;
        }
        self.tree_query_capacity = next_capacity(queries);
        self.tree_k_capacity = next_capacity(k);
        self.tile_capacity = next_capacity(tiles);
        self.group_capacity = next_capacity(groups);
        let lists = self.tree_query_capacity * self.tile_capacity * self.tree_k_capacity;
        let alt = self.tree_query_capacity * self.group_capacity * self.tree_k_capacity;
        self.tree_distances = runtime.buffer::<f32>(lists);
        self.tree_indices = runtime.buffer::<u32>(lists);
        self.tree_alt_distances = runtime.buffer::<f32>(alt);
        self.tree_alt_indices = runtime.buffer::<u32>(alt);
    }
}

impl MetalIndex {
    pub(crate) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::new_plan(num_dim, train, KnnPlan::Measured)
    }

    pub(crate) fn new_agx(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::new_agx_plan(num_dim, train, KnnPlan::Measured)
    }

    pub(crate) fn new_plan(
        num_dim: usize,
        train: &ArrayView2<f64>,
        plan: KnnPlan,
    ) -> Result<Self, IndexError> {
        Self::with_agx(num_dim, train, false, plan)
    }

    pub(crate) fn new_agx_plan(
        num_dim: usize,
        train: &ArrayView2<f64>,
        plan: KnnPlan,
    ) -> Result<Self, IndexError> {
        Self::with_agx(num_dim, train, true, plan)
    }

    fn with_agx(
        num_dim: usize,
        train: &ArrayView2<f64>,
        agx: bool,
        request: KnnPlan,
    ) -> Result<Self, IndexError> {
        let runtime = Runtime::shared().map_err(IndexError::InvalidParameter)?;
        let pipeline = |name: &str| -> Result<ComputePipelineState, IndexError> {
            let result = if agx {
                runtime.agx_pipeline(SOURCE, "index", name)
            } else {
                runtime.pipeline(SOURCE, "index", name)
            };
            result.map_err(IndexError::InvalidParameter)
        };
        let names = &[
            "distance_rows",
            "distance_rows_2",
            "distance_rows_4",
            "distance_simd",
        ];
        let distance = names
            .iter()
            .map(|name| pipeline(name))
            .collect::<Result<Vec<_>, _>>()?;
        let topk16 = pipeline("topk_16")?;
        let fused16 = pipeline("l2_topk_16")?;
        let tree16 = pipeline("l2_topk_16_batch")?;
        let tiled16 = pipeline("l2_topk_16_tiled")?;
        let simd16 = pipeline("l2_topk_16_simd")?;
        let gram16 = pipeline("l2_topk_16_gram")?;
        let reduce16 = pipeline("reduce_topk_16")?;
        let wide = pipeline("l2_topk_batch")?;
        let fold = pipeline("fold_topk")?;
        let local_topk = pipeline("local_topk")?;
        let merge = pipeline("merge_topk")?;
        let init_results = pipeline("init_results")?;
        let scratch = Scratch::new(&runtime);
        let mut index = Self {
            rows: runtime.buffer::<u32>(1),
            row_norms: runtime.buffer::<u32>(1),
            runtime,
            agx,
            distance,
            distance_choice: None,
            topk16,
            fused16,
            tree16,
            tiled16,
            simd16,
            gram16,
            reduce16,
            wide,
            fold,
            local_topk,
            merge,
            init_results,
            host_rows: Vec::new(),
            host_norms: Vec::new(),
            row_capacity: 0,
            scratch,
            num_dim,
            plan: Plan::Split,
            request,
            profile: None,
        };
        index.rebuild(train)?;
        Ok(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.host_rows.len() / self.num_dim
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.row_capacity
            .saturating_mul(self.num_dim)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(crate) fn rebuild(&mut self, train: &ArrayView2<f64>) -> Result<(), IndexError> {
        self.check_rows(train)?;
        self.host_rows = arr2_rows_to_f32(train);
        self.host_norms = norms(&self.host_rows, self.num_dim);
        self.upload_rows();
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        self.check_rows(rows)?;
        let start = self.len();
        let values = arr2_rows_to_f32(rows);
        let norms = norms(&values, self.num_dim);
        self.host_rows.extend_from_slice(&values);
        self.host_norms.extend_from_slice(&norms);
        if self.len() > self.row_capacity {
            self.upload_rows();
        } else {
            self.write_f32(&self.rows, start * self.num_dim, &values);
            self.write_f32(&self.row_norms, start, &norms);
        }
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        self.check_rows(queries)?;
        if k_eff == 0 || k_eff > MAX_K {
            return Err(IndexError::InvalidParameter(format!(
                "Metal index supports 1..={MAX_K} neighbors, got {k_eff}"
            )));
        }
        if queries.nrows() == 0 || self.len() == 0 {
            return Ok(pad_neighbor_cols_to_search_k(
                Array2::from_elem((queries.nrows(), 0), f64::INFINITY),
                Array2::zeros((queries.nrows(), 0)),
                search_k,
            ));
        }

        let query_values = {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.metal.prepare"));
            span.emit_value(queries.nrows() as u64);
            arr2_rows_to_f32(queries)
        };
        {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.metal.upload"));
            span.emit_value(queries.nrows() as u64);
            self.scratch
                .ensure(&self.runtime, self.num_dim, queries.nrows(), k_eff);
            self.write_f32(&self.scratch.query, 0, &query_values);
            self.write_f32(
                &self.scratch.query_norms,
                0,
                &norms(&query_values, self.num_dim),
            );
        }
        let plan = {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.metal.calibrate"));
            span.emit_value(queries.nrows() as u64);
            self.calibrate_distance(queries.nrows())?;
            let plan = self.tile_plan(queries.nrows(), k_eff)?;
            span.emit_value(plan.id());
            plan
        };
        self.plan = plan;

        let command_buffer = self.runtime.queue.new_command_buffer();
        let lists = plan_lists(plan, self.len());
        let passes = plan.graph().passes(lists, k_eff);
        let mut gpu = self
            .runtime
            .trace(passes)
            .map_err(IndexError::InvalidParameter)?;
        let init_params = Params {
            rows: to_u32(self.len(), "row count")?,
            dim: to_u32(self.num_dim, "dimension")?,
            queries: to_u32(queries.nrows(), "query count")?,
            tile_start: 0,
            tile_rows: 0,
            k: to_u32(k_eff, "neighbor count")?,
            lanes: 1,
            tiles: 0,
            lists: 0,
            groups: 0,
        };
        {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.metal.encode"));
            span.emit_value(queries.nrows() as u64);
            if matches!(
                plan,
                Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
            ) {
                self.encode_tree(plan, command_buffer, &mut gpu, queries.nrows(), k_eff)
                    .map_err(IndexError::InvalidParameter)?;
            } else {
                let encoder = gpu
                    .encoder(command_buffer, "knn.init")
                    .map_err(IndexError::InvalidParameter)?;
                encoder.set_compute_pipeline_state(&self.init_results);
                encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
                encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
                set_params(&encoder, 2, &init_params);
                dispatch(&encoder, queries.nrows() * k_eff);
                drop(encoder);

                for tile_start in (0..self.len()).step_by(TILE_ROWS) {
                    let tile_rows = (self.len() - tile_start).min(TILE_ROWS);
                    let params = Params {
                        rows: to_u32(self.len(), "row count")?,
                        dim: to_u32(self.num_dim, "dimension")?,
                        queries: to_u32(queries.nrows(), "query count")?,
                        tile_start: to_u32(tile_start, "tile start")?,
                        tile_rows: to_u32(tile_rows, "tile rows")?,
                        k: to_u32(k_eff, "neighbor count")?,
                        lanes: self.distance_lanes(),
                        tiles: 0,
                        lists: 0,
                        groups: 0,
                    };

                    match plan {
                        Plan::Split => {
                            let encoder = gpu
                                .encoder(command_buffer, "knn.distance")
                                .map_err(IndexError::InvalidParameter)?;
                            encoder.set_compute_pipeline_state(
                                &self.distance
                                    [self.distance_choice.expect("distance schedule calibrated")],
                            );
                            encoder.set_buffer(0, Some(&self.rows), 0);
                            encoder.set_buffer(1, Some(&self.scratch.query), 0);
                            encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
                            set_params(&encoder, 3, &params);
                            self.dispatch_distance(&encoder, queries.nrows());
                            drop(encoder);

                            let encoder = gpu
                                .encoder(command_buffer, "knn.topk")
                                .map_err(IndexError::InvalidParameter)?;
                            let topk = if k_eff <= 16 {
                                &self.topk16
                            } else {
                                &self.local_topk
                            };
                            encoder.set_compute_pipeline_state(topk);
                            encoder.set_buffer(0, Some(&self.scratch.tile_distances), 0);
                            encoder.set_buffer(1, Some(&self.scratch.local_distances), 0);
                            encoder.set_buffer(2, Some(&self.scratch.local_indices), 0);
                            set_params(&encoder, 3, &params);
                            encoder.dispatch_thread_groups(
                                MTLSize {
                                    width: queries.nrows() as u64,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            drop(encoder);
                        }
                        Plan::Fused => {
                            let encoder = gpu
                                .encoder(command_buffer, "knn.l2_topk")
                                .map_err(IndexError::InvalidParameter)?;
                            encoder.set_compute_pipeline_state(&self.fused16);
                            encoder.set_buffer(0, Some(&self.rows), 0);
                            encoder.set_buffer(1, Some(&self.scratch.query), 0);
                            encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
                            encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
                            set_params(&encoder, 4, &params);
                            encoder.dispatch_thread_groups(
                                MTLSize {
                                    width: queries.nrows() as u64,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            drop(encoder);
                        }
                        Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide => {
                            unreachable!("tree plan encoded above")
                        }
                    }

                    let encoder = gpu
                        .encoder(command_buffer, "knn.merge")
                        .map_err(IndexError::InvalidParameter)?;
                    encoder.set_compute_pipeline_state(&self.merge);
                    encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
                    encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
                    encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
                    encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
                    set_params(&encoder, 4, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: queries.nrows() as u64,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: THREADS,
                            height: 1,
                            depth: 1,
                        },
                    );
                    drop(encoder);
                }
            }
        }
        gpu.resolve(command_buffer);
        {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.metal.execute"));
            span.emit_value(queries.nrows() as u64);
            span.emit_text(&format!(
                "rows={} queries={} dims={} k={} plan={}",
                self.len(),
                queries.nrows(),
                self.num_dim,
                k_eff,
                plan.name()
            ));
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let stages = gpu.stages().map_err(IndexError::InvalidParameter)?;
            let elapsed = stages.iter().map(|(_, elapsed)| *elapsed).sum();
            let mut profile = KnnProfile {
                rows: self.len(),
                queries: queries.nrows(),
                dims: self.num_dim,
                k: k_eff,
                plan: plan.name(),
                gpu: elapsed,
                scan: Duration::ZERO,
                select: Duration::ZERO,
                reduce: Duration::ZERO,
            };
            for (name, elapsed) in stages {
                if name.contains("distance") || name.contains("scan") || name.contains("l2_topk") {
                    profile.scan += elapsed;
                } else if name.contains("topk") {
                    profile.select += elapsed;
                } else if name.contains("merge") || name.contains("reduce") || name.contains("fold")
                {
                    profile.reduce += elapsed;
                }
            }
            crate::tracy::knn(&profile);
            self.profile = Some(profile);
            gpu.upload().map_err(IndexError::InvalidParameter)?;
        }

        let output = {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.metal.readback"));
            span.emit_value(queries.nrows() as u64);
            let distances = unsafe {
                std::slice::from_raw_parts(
                    self.scratch.result_distances.contents().cast::<f32>(),
                    queries.nrows() * k_eff,
                )
            };
            let indices = unsafe {
                std::slice::from_raw_parts(
                    self.scratch.result_indices.contents().cast::<u32>(),
                    queries.nrows() * k_eff,
                )
            };
            let mut out_dist = Array2::zeros((queries.nrows(), k_eff));
            let mut out_idx = Array2::zeros((queries.nrows(), k_eff));
            for q in 0..queries.nrows() {
                for k in 0..k_eff {
                    out_dist[[q, k]] = f64::from(distances[q * k_eff + k]);
                    out_idx[[q, k]] = i64::from(indices[q * k_eff + k]);
                }
            }
            pad_neighbor_cols_to_search_k(out_dist, out_idx, search_k)
        };
        Ok(output)
    }

    pub(crate) fn plan(&self) -> &'static str {
        self.plan.name()
    }

    pub(crate) fn profile(&self) -> Option<KnnProfile> {
        self.profile.clone()
    }

    fn encode_tree(
        &self,
        plan: Plan,
        command: &metal::CommandBufferRef,
        gpu: &mut crate::tracy_metal::Batch,
        queries: usize,
        k: usize,
    ) -> Result<(), String> {
        let scan_lists = plan_lists(plan, self.len());
        let params = Params {
            rows: to_u32(self.len(), "row count").map_err(|error| error.to_string())?,
            dim: to_u32(self.num_dim, "dimension").map_err(|error| error.to_string())?,
            queries: to_u32(queries, "query count").map_err(|error| error.to_string())?,
            tile_start: 0,
            tile_rows: 0,
            k: to_u32(k, "neighbor count").map_err(|error| error.to_string())?,
            lanes: self.distance_lanes(),
            tiles: to_u32(scan_lists, "list count").map_err(|error| error.to_string())?,
            lists: 0,
            groups: 0,
        };
        let (scan_distances, scan_indices) = if scan_lists == 1 {
            (&self.scratch.result_distances, &self.scratch.result_indices)
        } else {
            (&self.scratch.tree_distances, &self.scratch.tree_indices)
        };
        let name = match plan {
            Plan::Tree => "knn.tree.scan",
            Plan::Tiled => "knn.tiled.scan",
            Plan::Simd => "knn.simd.scan",
            Plan::Gram => "knn.gram.scan",
            Plan::Wide => "knn.wide.scan",
            Plan::Split | Plan::Fused => unreachable!("serial plan cannot encode a tree"),
        };
        let encoder = gpu.encoder(command, name)?;
        encoder.set_compute_pipeline_state(match plan {
            Plan::Tree => &self.tree16,
            Plan::Tiled => &self.tiled16,
            Plan::Simd => &self.simd16,
            Plan::Gram => &self.gram16,
            Plan::Wide => &self.wide,
            Plan::Split | Plan::Fused => unreachable!("serial plan cannot encode a tree"),
        });
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.query), 0);
        encoder.set_buffer(2, Some(scan_distances), 0);
        encoder.set_buffer(3, Some(scan_indices), 0);
        if plan == Plan::Gram {
            encoder.set_buffer(4, Some(&self.row_norms), 0);
            encoder.set_buffer(5, Some(&self.scratch.query_norms), 0);
            set_params(&encoder, 6, &params);
        } else {
            set_params(&encoder, 4, &params);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: if plan == Plan::Gram {
                    (queries.div_ceil(GRAM_QUERIES) * scan_lists) as u64
                } else {
                    (queries * scan_lists) as u64
                },
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        let mut lists = scan_lists;
        let mut tree_input = true;
        while lists > 1 {
            let fan = if plan == Plan::Wide { 2 } else { plan_fan(k) };
            let groups = lists.div_ceil(fan);
            let final_pass = groups == 1;
            let (input_distances, input_indices) = if tree_input {
                (&self.scratch.tree_distances, &self.scratch.tree_indices)
            } else {
                (
                    &self.scratch.tree_alt_distances,
                    &self.scratch.tree_alt_indices,
                )
            };
            let (output_distances, output_indices) = if final_pass {
                (&self.scratch.result_distances, &self.scratch.result_indices)
            } else if tree_input {
                (
                    &self.scratch.tree_alt_distances,
                    &self.scratch.tree_alt_indices,
                )
            } else {
                (&self.scratch.tree_distances, &self.scratch.tree_indices)
            };
            let params = Params {
                lists: to_u32(lists, "list count").map_err(|error| error.to_string())?,
                groups: to_u32(groups, "merge groups").map_err(|error| error.to_string())?,
                ..params
            };
            let name = if plan == Plan::Wide {
                "knn.wide.fold"
            } else {
                "knn.tree.reduce"
            };
            let encoder = gpu.encoder(command, name)?;
            encoder.set_compute_pipeline_state(if plan == Plan::Wide {
                &self.fold
            } else {
                &self.reduce16
            });
            encoder.set_buffer(0, Some(input_distances), 0);
            encoder.set_buffer(1, Some(input_indices), 0);
            encoder.set_buffer(2, Some(output_distances), 0);
            encoder.set_buffer(3, Some(output_indices), 0);
            set_params(&encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (queries * groups) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            drop(encoder);
            lists = groups;
            tree_input = !tree_input;
        }
        Ok(())
    }

    fn calibrate_distance(&mut self, queries: usize) -> Result<(), IndexError> {
        if self.distance_choice.is_some() {
            return Ok(());
        }
        let query_count = queries;
        let params = Params {
            rows: to_u32(self.len(), "row count")?,
            dim: to_u32(self.num_dim, "dimension")?,
            queries: to_u32(query_count, "query count")?,
            tile_start: 0,
            tile_rows: to_u32(self.len().min(TILE_ROWS), "tile rows")?,
            k: 1,
            lanes: 1,
            tiles: 0,
            lists: 0,
            groups: 0,
        };
        self.run_distance(0, &params);
        let reference = self.distance_values(query_count);
        let runtime = Arc::clone(&self.runtime);
        let choice = runtime
            .race(
                "knn.distance.v2",
                &[self.num_dim, query_count],
                self.distance.len(),
                5,
                |schedule| {
                    let start = Instant::now();
                    self.run_distance(schedule, &params);
                    let elapsed = start.elapsed();
                    if !same_distances(&reference, &self.distance_values(query_count)) {
                        return Ok(None);
                    }
                    Ok(Some(elapsed))
                },
            )
            .map_err(IndexError::InvalidParameter)?;
        self.distance_choice = Some(choice);
        Ok(())
    }

    fn tile_plan(&mut self, queries: usize, k: usize) -> Result<Plan, IndexError> {
        let direct = match self.request {
            KnnPlan::Measured => None,
            KnnPlan::Split => Some(Plan::Split),
            KnnPlan::Fused if k <= 16 => Some(Plan::Fused),
            KnnPlan::Tree if k <= 16 => Some(Plan::Tree),
            KnnPlan::Tiled if k <= 16 => Some(Plan::Tiled),
            KnnPlan::Simd if k <= 16 => Some(Plan::Simd),
            KnnPlan::Gram if k <= 16 => Some(Plan::Gram),
            KnnPlan::Wide => Some(Plan::Wide),
            KnnPlan::Fused | KnnPlan::Tree | KnnPlan::Tiled | KnnPlan::Simd | KnnPlan::Gram => {
                return Err(IndexError::InvalidParameter(
                    "specialized KNN plans support at most 16 neighbors".to_string(),
                ));
            }
        };
        if let Some(plan) = direct {
            if matches!(
                plan,
                Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
            ) {
                self.scratch.ensure_tree(
                    &self.runtime,
                    queries,
                    k,
                    plan_lists(plan, self.len()),
                    if plan == Plan::Wide { 2 } else { plan_fan(k) },
                );
            }
            return Ok(plan);
        }
        let gram = k <= 16 && queries >= GRAM_QUERIES;
        let tree_lists = if gram {
            self.len().div_ceil(GRAM_ROWS)
        } else {
            self.len().div_ceil(TILE_ROWS)
        };
        self.scratch.ensure_tree(
            &self.runtime,
            queries,
            k,
            tree_lists,
            if k <= 16 { plan_fan(k) } else { 2 },
        );
        let candidates: &[Plan] = if gram {
            &[
                Plan::Split,
                Plan::Fused,
                Plan::Tree,
                Plan::Tiled,
                Plan::Simd,
                Plan::Gram,
            ]
        } else if k <= 16 {
            &[
                Plan::Split,
                Plan::Fused,
                Plan::Tree,
                Plan::Tiled,
                Plan::Simd,
            ]
        } else {
            &[Plan::Split, Plan::Wide]
        };
        let query_count = queries;
        let mut reference = None;
        let runtime = Arc::clone(&self.runtime);
        let family = if self.agx {
            "knn.plan.agx.v5"
        } else {
            "knn.plan.metal.v5"
        };
        let choice = runtime
            .race(
                family,
                &[self.num_dim, self.len(), query_count, k],
                candidates.len(),
                5,
                |candidate| {
                    let plan = candidates[candidate];
                    if !plan.graph().valid() {
                        return Ok(None);
                    }
                    let elapsed = self.run_plan(plan, query_count, k)?;
                    let values = self.result_values(query_count, k);
                    if let Some(reference) = &reference {
                        if !same_topk(reference, &values, query_count, k, self.len()) {
                            return Ok(None);
                        }
                    } else {
                        reference = Some(values);
                    }
                    Ok(Some(elapsed))
                },
            )
            .map_err(IndexError::InvalidParameter)?;
        Ok(candidates[choice])
    }

    fn run_plan(&self, plan: Plan, queries: usize, k: usize) -> Result<Duration, String> {
        let span = match plan {
            Plan::Split => crate::tracy::zone(tracy_client::span_location!("knn.plan.split")),
            Plan::Fused => crate::tracy::zone(tracy_client::span_location!("knn.plan.fused")),
            Plan::Tree => crate::tracy::zone(tracy_client::span_location!("knn.plan.tree")),
            Plan::Tiled => crate::tracy::zone(tracy_client::span_location!("knn.plan.tiled")),
            Plan::Simd => crate::tracy::zone(tracy_client::span_location!("knn.plan.simd")),
            Plan::Gram => crate::tracy::zone(tracy_client::span_location!("knn.plan.gram")),
            Plan::Wide => crate::tracy::zone(tracy_client::span_location!("knn.plan.wide")),
        };
        let start = Instant::now();
        let command_buffer = self.runtime.queue.new_command_buffer();
        let lists = plan_lists(plan, self.len());
        let passes = plan.graph().passes(lists, k);
        let mut gpu = self.runtime.trace(passes)?;
        if matches!(
            plan,
            Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
        ) {
            self.encode_tree(plan, command_buffer, &mut gpu, queries, k)?;
        } else {
            let init = Params {
                rows: to_u32(self.len(), "row count").map_err(|error| error.to_string())?,
                dim: to_u32(self.num_dim, "dimension").map_err(|error| error.to_string())?,
                queries: to_u32(queries, "query count").map_err(|error| error.to_string())?,
                tile_start: 0,
                tile_rows: 0,
                k: to_u32(k, "neighbor count").map_err(|error| error.to_string())?,
                lanes: 1,
                tiles: 0,
                lists: 0,
                groups: 0,
            };
            let encoder = gpu.encoder(command_buffer, "knn.plan.init")?;
            encoder.set_compute_pipeline_state(&self.init_results);
            encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
            encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
            set_params(&encoder, 2, &init);
            dispatch(&encoder, queries * k);
            drop(encoder);
            for tile_start in (0..self.len()).step_by(TILE_ROWS) {
                let params = Params {
                    rows: to_u32(self.len(), "row count").map_err(|error| error.to_string())?,
                    dim: to_u32(self.num_dim, "dimension").map_err(|error| error.to_string())?,
                    queries: to_u32(queries, "query count").map_err(|error| error.to_string())?,
                    tile_start: to_u32(tile_start, "tile start")
                        .map_err(|error| error.to_string())?,
                    tile_rows: to_u32((self.len() - tile_start).min(TILE_ROWS), "tile rows")
                        .map_err(|error| error.to_string())?,
                    k: to_u32(k, "neighbor count").map_err(|error| error.to_string())?,
                    lanes: self.distance_lanes(),
                    tiles: 0,
                    lists: 0,
                    groups: 0,
                };
                match plan {
                    Plan::Split => {
                        let encoder = gpu.encoder(command_buffer, "knn.plan.distance")?;
                        encoder.set_compute_pipeline_state(
                            &self.distance
                                [self.distance_choice.expect("distance schedule calibrated")],
                        );
                        encoder.set_buffer(0, Some(&self.rows), 0);
                        encoder.set_buffer(1, Some(&self.scratch.query), 0);
                        encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
                        set_params(&encoder, 3, &params);
                        self.dispatch_distance(&encoder, params.queries as usize);
                        drop(encoder);

                        let encoder = gpu.encoder(command_buffer, "knn.plan.topk")?;
                        encoder.set_compute_pipeline_state(if k <= 16 {
                            &self.topk16
                        } else {
                            &self.local_topk
                        });
                        encoder.set_buffer(0, Some(&self.scratch.tile_distances), 0);
                        encoder.set_buffer(1, Some(&self.scratch.local_distances), 0);
                        encoder.set_buffer(2, Some(&self.scratch.local_indices), 0);
                        set_params(&encoder, 3, &params);
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: u64::from(params.queries),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: THREADS,
                                height: 1,
                                depth: 1,
                            },
                        );
                        drop(encoder);
                    }
                    Plan::Fused => {
                        let encoder = gpu.encoder(command_buffer, "knn.plan.l2_topk")?;
                        encoder.set_compute_pipeline_state(&self.fused16);
                        encoder.set_buffer(0, Some(&self.rows), 0);
                        encoder.set_buffer(1, Some(&self.scratch.query), 0);
                        encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
                        encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
                        set_params(&encoder, 4, &params);
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: u64::from(params.queries),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: THREADS,
                                height: 1,
                                depth: 1,
                            },
                        );
                        drop(encoder);
                    }
                    Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide => {
                        unreachable!("tree plan encoded above")
                    }
                }
                let encoder = gpu.encoder(command_buffer, "knn.plan.merge")?;
                encoder.set_compute_pipeline_state(&self.merge);
                encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
                encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
                encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
                encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
                set_params(&encoder, 4, &params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: queries as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
                drop(encoder);
            }
        }
        gpu.resolve(command_buffer);
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let gpu_elapsed = gpu.duration()?;
        gpu.upload()?;
        span.emit_value(gpu_elapsed.as_nanos().min(u128::from(u64::MAX)) as u64);
        Ok(start.elapsed())
    }

    fn result_values(&self, queries: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
        unsafe {
            (
                std::slice::from_raw_parts(
                    self.scratch.result_distances.contents().cast::<f32>(),
                    queries * k,
                )
                .to_vec(),
                std::slice::from_raw_parts(
                    self.scratch.result_indices.contents().cast::<u32>(),
                    queries * k,
                )
                .to_vec(),
            )
        }
    }

    fn run_distance(&self, schedule: usize, params: &Params) {
        let command_buffer = self.runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance[schedule]);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.query), 0);
        encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
        set_params(encoder, 3, params);
        self.dispatch_distance_for(encoder, schedule, params.queries as usize);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn distance_values(&self, queries: usize) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                self.scratch.tile_distances.contents().cast::<f32>(),
                queries * TILE_ROWS,
            )
            .to_vec()
        }
    }

    fn distance_lanes(&self) -> u32 {
        [1, 2, 4, 4][self.distance_choice.expect("distance schedule calibrated")]
    }

    fn dispatch_distance(&self, encoder: &metal::ComputeCommandEncoderRef, queries: usize) {
        self.dispatch_distance_for(
            encoder,
            self.distance_choice.expect("distance schedule calibrated"),
            queries,
        );
    }

    fn dispatch_distance_for(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        schedule: usize,
        queries: usize,
    ) {
        if schedule + 1 == self.distance.len() {
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (queries * TILE_ROWS).div_ceil(8) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            dispatch(encoder, queries * TILE_ROWS);
        }
    }

    fn check_rows(&self, rows: &ArrayView2<f64>) -> Result<(), IndexError> {
        if rows.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows.ncols(),
            });
        }
        Ok(())
    }

    fn upload_rows(&mut self) {
        self.row_capacity = next_capacity(self.len());
        self.rows = self.runtime.buffer::<f32>(self.row_capacity * self.num_dim);
        self.row_norms = self.runtime.buffer::<f32>(self.row_capacity);
        self.write_f32(&self.rows, 0, &self.host_rows);
        self.write_f32(&self.row_norms, 0, &self.host_norms);
    }

    fn write_f32(&self, buffer: &metal::Buffer, offset: usize, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents().cast::<f32>().add(offset),
                values.len(),
            );
        }
    }
}

fn same_distances(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .all(|(&left, &right)| same_distance(left, right))
}

fn same_distance(left: f32, right: f32) -> bool {
    left == right
        || (left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= 1e-4 * (1.0 + left.abs()))
}

fn same_topk(
    left: &(Vec<f32>, Vec<u32>),
    right: &(Vec<f32>, Vec<u32>),
    queries: usize,
    k: usize,
    rows: usize,
) -> bool {
    if left.0.len() != queries * k
        || left.1.len() != queries * k
        || right.0.len() != queries * k
        || right.1.len() != queries * k
        || !same_distances(&left.0, &right.0)
    {
        return false;
    }
    for query in 0..queries {
        let range = query * k..(query + 1) * k;
        let left_distances = &left.0[range.clone()];
        let left_indices = &left.1[range.clone()];
        let right_distances = &right.0[range.clone()];
        let right_indices = &right.1[range];
        if left_indices
            .iter()
            .chain(right_indices)
            .any(|&index| index != u32::MAX && index as usize >= rows)
            || left_indices
                .iter()
                .enumerate()
                .any(|(rank, &index)| index != u32::MAX && left_indices[..rank].contains(&index))
            || right_indices
                .iter()
                .enumerate()
                .any(|(rank, &index)| index != u32::MAX && right_indices[..rank].contains(&index))
        {
            return false;
        }
        for (&distance, &index) in left_distances.iter().zip(left_indices) {
            if index == u32::MAX {
                if distance.is_finite() {
                    return false;
                }
                continue;
            }
            if let Some(rank) = right_indices.iter().position(|&right| right == index) {
                if !same_distance(distance, right_distances[rank]) {
                    return false;
                }
            } else if !same_distance(distance, left_distances[k - 1])
                || !same_distance(distance, right_distances[k - 1])
            {
                return false;
            }
        }
        for (&distance, &index) in right_distances.iter().zip(right_indices) {
            if index == u32::MAX {
                if distance.is_finite() {
                    return false;
                }
                continue;
            }
            if let Some(rank) = left_indices.iter().position(|&left| left == index) {
                if !same_distance(distance, left_distances[rank]) {
                    return false;
                }
            } else if !same_distance(distance, right_distances[k - 1])
                || !same_distance(distance, left_distances[k - 1])
            {
                return false;
            }
        }
    }
    true
}

fn next_capacity(value: usize) -> usize {
    value
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(value.max(1))
}

fn norms(values: &[f32], dim: usize) -> Vec<f32> {
    values
        .chunks_exact(dim)
        .map(|row| {
            row.iter()
                .fold(0.0, |sum, &value| value.mul_add(value, sum))
        })
        .collect()
}

fn plan_lists(plan: Plan, rows: usize) -> usize {
    if plan == Plan::Gram {
        rows.div_ceil(GRAM_ROWS)
    } else {
        rows.div_ceil(TILE_ROWS)
    }
}

fn plan_fan(k: usize) -> usize {
    if k <= 16 {
        TILE_ROWS / k
    } else {
        2
    }
}

fn set_params(encoder: &metal::ComputeCommandEncoderRef, slot: u64, params: &Params) {
    encoder.set_bytes(
        slot,
        std::mem::size_of::<Params>() as u64,
        (params as *const Params).cast::<c_void>(),
    );
}

fn dispatch(encoder: &metal::ComputeCommandEncoderRef, count: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: count.div_ceil(THREADS as usize) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
}

fn to_u32(value: usize, name: &str) -> Result<u32, IndexError> {
    u32::try_from(value)
        .map_err(|_| IndexError::InvalidParameter(format!("{name} exceeds u32 range")))
}
