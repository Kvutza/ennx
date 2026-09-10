use std::ptr;

use ndarray::{Array2, ArrayView2};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_mem_flags, CL_BLOCKING};

use super::{flatten_f32, pad_k};
use crate::index::IndexError;

const THREADS: usize = 256;
const TILE_ROWS: usize = 1024;
const MAX_K: usize = 1024;
const SOURCE: &str = include_str!("opencl_index.cl");

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    dim: u32,
    queries: u32,
    tile_start: u32,
    tile_rows: u32,
    k: u32,
}

pub(crate) struct OpenClIndex {
    context: Context,
    queue: CommandQueue,
    rows: Buffer<f32>,
    host_rows: Vec<f32>,
    num_dim: usize,
    distance: Kernel,
    local_topk: Kernel,
    merge: Kernel,
}

impl OpenClIndex {
    pub(crate) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .unwrap_or_default()
            .into_iter()
            .next()
            .or_else(|| {
                get_all_devices(CL_DEVICE_TYPE_CPU)
                    .ok()
                    .and_then(|devices| devices.into_iter().next())
            })
            .ok_or_else(|| {
                IndexError::InvalidParameter("no OpenCL GPU or CPU device found".to_string())
            })?;
        let device = Device::new(device_id);
        let context = Context::from_device(&device).map_err(|error| {
            IndexError::InvalidParameter(format!("OpenCL index context: {error}"))
        })?;
        let queue = CommandQueue::create_default(&context, 0).map_err(|error| {
            IndexError::InvalidParameter(format!("OpenCL index queue: {error}"))
        })?;
        let program =
            Program::create_and_build_from_source(&context, SOURCE, "").map_err(|error| {
                IndexError::InvalidParameter(format!("OpenCL index compile: {error}"))
            })?;
        let kernel = |name: &str| -> Result<Kernel, IndexError> {
            Kernel::create(&program, name).map_err(|error| {
                IndexError::InvalidParameter(format!("OpenCL index kernel {name}: {error}"))
            })
        };
        let mut index = Self {
            rows: Self::buffer(&context, 1, CL_MEM_READ_ONLY, "rows")?,
            context,
            queue,
            host_rows: Vec::new(),
            num_dim,
            distance: kernel("distance_rows")?,
            local_topk: kernel("local_topk")?,
            merge: kernel("merge_topk")?,
        };
        index.rebuild(train)?;
        Ok(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.host_rows.len() / self.num_dim
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.host_rows
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(crate) fn rebuild(&mut self, train: &ArrayView2<f64>) -> Result<(), IndexError> {
        self.check_rows(train)?;
        self.host_rows = flatten_f32(train);
        self.upload_rows()?;
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        self.check_rows(rows)?;
        self.host_rows.extend(flatten_f32(rows));
        self.upload_rows()?;
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
                "OpenCL index supports 1..={MAX_K} neighbors, got {k_eff}"
            )));
        }
        if queries.nrows() == 0 || self.len() == 0 {
            return Ok(pad_k(
                Array2::from_elem((queries.nrows(), 0), f64::INFINITY),
                Array2::zeros((queries.nrows(), 0)),
                search_k,
            ));
        }

        let query_values = flatten_f32(queries);
        let mut query_buffer: Buffer<f32> = Self::buffer(
            &self.context,
            query_values.len().max(1),
            CL_MEM_READ_ONLY,
            "queries",
        )?;
        let tile_distance_buffer: Buffer<f32> = Self::buffer(
            &self.context,
            queries.nrows() * TILE_ROWS,
            CL_MEM_READ_WRITE,
            "tile distances",
        )?;
        let local_dist_buffer: Buffer<f32> = Self::buffer(
            &self.context,
            queries.nrows() * k_eff,
            CL_MEM_READ_WRITE,
            "local distances",
        )?;
        let local_idx_buffer: Buffer<u32> = Self::buffer(
            &self.context,
            queries.nrows() * k_eff,
            CL_MEM_READ_WRITE,
            "local indices",
        )?;
        let initial_dist = vec![f32::INFINITY; queries.nrows() * k_eff];
        let initial_idx = vec![0u32; queries.nrows() * k_eff];
        let mut result_dist_buffer: Buffer<f32> = Self::buffer(
            &self.context,
            initial_dist.len(),
            CL_MEM_READ_WRITE,
            "result distances",
        )?;
        let mut result_idx_buffer: Buffer<u32> = Self::buffer(
            &self.context,
            initial_idx.len(),
            CL_MEM_READ_WRITE,
            "result indices",
        )?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut query_buffer, CL_BLOCKING, 0, &query_values, &[])
                .map_err(|error| {
                    IndexError::InvalidParameter(format!("OpenCL query upload: {error}"))
                })?;
            self.queue
                .enqueue_write_buffer(&mut result_dist_buffer, CL_BLOCKING, 0, &initial_dist, &[])
                .map_err(|error| {
                    IndexError::InvalidParameter(format!("OpenCL result init: {error}"))
                })?;
            self.queue
                .enqueue_write_buffer(&mut result_idx_buffer, CL_BLOCKING, 0, &initial_idx, &[])
                .map_err(|error| {
                    IndexError::InvalidParameter(format!("OpenCL index init: {error}"))
                })?;
        }

        for tile_start in (0..self.len()).step_by(TILE_ROWS) {
            let tile_rows = (self.len() - tile_start).min(TILE_ROWS);
            let params = Params {
                rows: to_u32(self.len(), "row count")?,
                dim: to_u32(self.num_dim, "dimension")?,
                queries: to_u32(queries.nrows(), "query count")?,
                tile_start: to_u32(tile_start, "tile start")?,
                tile_rows: to_u32(tile_rows, "tile rows")?,
                k: to_u32(k_eff, "neighbor count")?,
            };
            unsafe {
                ExecuteKernel::new(&self.distance)
                    .set_arg(&self.rows)
                    .set_arg(&query_buffer)
                    .set_arg(&tile_distance_buffer)
                    .set_arg(&params)
                    .set_global_work_size(queries.nrows() * TILE_ROWS)
                    .set_local_work_size(THREADS)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| {
                        IndexError::InvalidParameter(format!("OpenCL distances: {error}"))
                    })?;
                ExecuteKernel::new(&self.local_topk)
                    .set_arg(&tile_distance_buffer)
                    .set_arg(&local_dist_buffer)
                    .set_arg(&local_idx_buffer)
                    .set_arg(&params)
                    .set_global_work_size(queries.nrows() * THREADS)
                    .set_local_work_size(THREADS)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| {
                        IndexError::InvalidParameter(format!("OpenCL local top-k: {error}"))
                    })?;
                ExecuteKernel::new(&self.merge)
                    .set_arg(&result_dist_buffer)
                    .set_arg(&result_idx_buffer)
                    .set_arg(&local_dist_buffer)
                    .set_arg(&local_idx_buffer)
                    .set_arg(&params)
                    .set_global_work_size(queries.nrows() * THREADS)
                    .set_local_work_size(THREADS)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| {
                        IndexError::InvalidParameter(format!("OpenCL merge: {error}"))
                    })?;
            }
        }
        self.queue
            .finish()
            .map_err(|error| IndexError::InvalidParameter(format!("OpenCL search: {error}")))?;

        let mut distances = vec![0.0f32; queries.nrows() * k_eff];
        let mut indices = vec![0u32; queries.nrows() * k_eff];
        unsafe {
            self.queue
                .enqueue_read_buffer(&result_dist_buffer, CL_BLOCKING, 0, &mut distances, &[])
                .map_err(|error| {
                    IndexError::InvalidParameter(format!("OpenCL distance read: {error}"))
                })?;
            self.queue
                .enqueue_read_buffer(&result_idx_buffer, CL_BLOCKING, 0, &mut indices, &[])
                .map_err(|error| {
                    IndexError::InvalidParameter(format!("OpenCL index read: {error}"))
                })?;
        }
        let mut out_dist = Array2::zeros((queries.nrows(), k_eff));
        let mut out_idx = Array2::zeros((queries.nrows(), k_eff));
        for q in 0..queries.nrows() {
            for k in 0..k_eff {
                out_dist[[q, k]] = f64::from(distances[q * k_eff + k]);
                out_idx[[q, k]] = i64::from(indices[q * k_eff + k]);
            }
        }
        Ok(pad_k(out_dist, out_idx, search_k))
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

    fn upload_rows(&mut self) -> Result<(), IndexError> {
        self.rows = Self::buffer(
            &self.context,
            self.host_rows.len().max(1),
            CL_MEM_READ_ONLY,
            "rows",
        )?;
        if !self.host_rows.is_empty() {
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut self.rows, CL_BLOCKING, 0, &self.host_rows, &[])
                    .map_err(|error| {
                        IndexError::InvalidParameter(format!("OpenCL row upload: {error}"))
                    })?;
            }
        }
        Ok(())
    }

    fn buffer<T>(
        context: &Context,
        len: usize,
        flags: cl_mem_flags,
        name: &str,
    ) -> Result<Buffer<T>, IndexError> {
        unsafe {
            Buffer::<T>::create(context, flags, len.max(1), ptr::null_mut()).map_err(|error| {
                IndexError::InvalidParameter(format!("OpenCL {name} buffer: {error}"))
            })
        }
    }
}

fn to_u32(value: usize, name: &str) -> Result<u32, IndexError> {
    u32::try_from(value)
        .map_err(|_| IndexError::InvalidParameter(format!("{name} exceeds u32 range")))
}
