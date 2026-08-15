use std::ffi::{CStr, OsStr, c_char, c_int, c_void};
use std::mem::{MaybeUninit, size_of, size_of_val};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig, LaunchConfig1D,
};
use cuda_host::embedded::{ArtifactPayloadKind, EmbeddedModuleError, OwnedArtifactBundle};
use ennx_cuda_kernels::trials;
pub use ennx_cuda_kernels::{
    CenterStep, DenseLeaf, DenseLinearParams, DenseTerm, DenseTile, Leaf, MAX_CENTER_DEPTH,
    MAX_HISTORY, Seed, Selection, THREADS, Tile,
};

pub type CudaResult<T> = Result<T, String>;

static TRIAL_BUNDLE: OnceLock<Result<OwnedArtifactBundle, String>> = OnceLock::new();
static TRACY: OnceLock<tracy_client::Client> = OnceLock::new();

#[repr(C)]
struct DlInfo {
    filename: *const c_char,
    _base: *mut c_void,
    _symbol: *const c_char,
    _symbol_address: *mut c_void,
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
}

#[inline(never)]
fn containing_binary() -> CudaResult<PathBuf> {
    let mut info = MaybeUninit::<DlInfo>::zeroed();
    // SAFETY: `info` is writable and the function address remains valid while
    // its executable or shared library is loaded.
    if unsafe {
        dladdr(
            containing_binary as *const () as *const c_void,
            info.as_mut_ptr(),
        )
    } == 0
    {
        return Err("dladdr could not locate the ENNx CUDA host library".to_string());
    }
    // SAFETY: a successful dladdr call initialized `info`.
    let info = unsafe { info.assume_init() };
    if info.filename.is_null() {
        return Err("dladdr returned no ENNx CUDA host library path".to_string());
    }
    // SAFETY: dladdr owns a NUL-terminated filename for the loaded object.
    let bytes = unsafe { CStr::from_ptr(info.filename) }.to_bytes();
    Ok(PathBuf::from(OsStr::from_bytes(bytes)))
}

fn trial_bundle() -> CudaResult<&'static OwnedArtifactBundle> {
    TRIAL_BUNDLE
        .get_or_init(|| {
            let path = containing_binary()?;
            cuda_host::embedded::artifact_bundles_from_binary_path(&path)
                .map_err(|error| {
                    format!(
                        "failed reading CUDA artifacts from {}: {error}",
                        path.display()
                    )
                })?
                .into_iter()
                .find(|bundle| bundle.name == ennx_cuda_kernels::MODULE_NAME)
                .ok_or_else(|| {
                    format!(
                        "embedded CUDA module {:?} was not found in {}",
                        ennx_cuda_kernels::MODULE_NAME,
                        path.display()
                    )
                })
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn load_trial_module(context: &Arc<CudaContext>) -> CudaResult<trials::LoadedModule> {
    // SAFETY: this crate launches only the embedded module generated from the
    // matching kernel crate and uses checked prepared launches below.
    match unsafe { trials::load(context) } {
        Ok(module) => return Ok(module),
        Err(EmbeddedModuleError::ModuleNotFound { .. } | EmbeddedModuleError::NoModules) => {}
        Err(error) => return Err(cuda_error(error)),
    }

    let bundle = trial_bundle()?;
    let image = if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        cubin.to_vec()
    } else if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        ptx.to_vec()
    } else if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        cuda_host::ltoir::build_cubin_from_nvvm_ir_with_compile_options(
            nvvm_ir,
            &bundle.name,
            &bundle.target,
            bundle.compile_options,
        )
        .map_err(cuda_error)?
    } else if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        cuda_host::ltoir::link_ltoir_to_cubin_with_compile_options(
            ltoir,
            &bundle.name,
            &bundle.target,
            bundle.compile_options,
        )
        .map_err(cuda_error)?
    } else {
        return Err(format!(
            "embedded CUDA module {:?} has no supported payload",
            bundle.name
        ));
    };
    let module = context.load_module_from_image(&image).map_err(cuda_error)?;
    // SAFETY: the fallback image is extracted from the same embedded kernel
    // bundle and therefore has the generated module's exact ABI.
    unsafe { trials::from_module(module) }.map_err(cuda_error)
}

struct Runtime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: trials::LoadedModule,
}

impl Runtime {
    fn new() -> CudaResult<Self> {
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.default_stream();
        let module = load_trial_module(&context)?;
        Ok(Self {
            context,
            stream,
            module,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ask {
    pub neighbors: usize,
    pub acquisition: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AskProfile {
    pub score_ms: f32,
    pub pick_ms: f32,
    pub materialize_ms: f32,
    pub total_ms: f32,
}

struct AskEvents {
    score_start: CudaEvent,
    score_end: CudaEvent,
    pick_end: CudaEvent,
    materialize_end: Option<CudaEvent>,
}

struct Scratch {
    history_capacity: usize,
    candidate_capacity: usize,
    center_capacity: usize,
    history_slots: DeviceBuffer<u32>,
    outcomes: DeviceBuffer<f32>,
    seeds: DeviceBuffer<Seed>,
    draws: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    selection: DeviceBuffer<Selection>,
    centers: DeviceBuffer<CenterStep>,
    candidate_centers: DeviceBuffer<u32>,
}

impl Scratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            history_capacity: 1,
            candidate_capacity: 1,
            center_capacity: 1,
            history_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            draws: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scores: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            selection: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            centers: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            candidate_centers: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(&mut self, stream: &CudaStream, history: usize, candidates: usize) -> CudaResult<()> {
        let history_capacity = history
            .checked_next_power_of_two()
            .ok_or("CUDA trial history capacity overflow")?;
        let candidate_capacity = candidates
            .checked_next_power_of_two()
            .ok_or("CUDA trial candidate capacity overflow")?;
        if history_capacity > self.history_capacity {
            self.history_slots =
                DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.outcomes = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.history_capacity = history_capacity;
        }
        if candidate_capacity > self.candidate_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.draws = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.scores = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.selection =
                DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_centers =
                DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_capacity = candidate_capacity;
        }
        Ok(())
    }

    fn ensure_centers(&mut self, stream: &CudaStream, centers: usize) -> CudaResult<()> {
        let center_capacity = centers
            .max(1)
            .checked_next_power_of_two()
            .ok_or("CUDA trial center capacity overflow")?;
        if center_capacity > self.center_capacity {
            self.centers = DeviceBuffer::zeroed(stream, center_capacity).map_err(cuda_error)?;
            self.center_capacity = center_capacity;
        }
        Ok(())
    }
}

pub struct TrialEngine {
    runtime: Runtime,
    rows: DeviceBuffer<u8>,
    leaves: DeviceBuffer<Leaf>,
    tiles: DeviceBuffer<Tile>,
    row_bytes: usize,
    slots: usize,
    scratch: Scratch,
    profiling: bool,
    last_profile: Option<AskProfile>,
}

impl TrialEngine {
    pub fn new(base: &[u8], leaves: &[Leaf], tiles: &[Tile], slots: usize) -> CudaResult<Self> {
        if base.is_empty() {
            return Err("CUDA trial base row must not be empty".to_string());
        }
        if leaves.is_empty() || tiles.is_empty() {
            return Err("CUDA trial layout must contain leaves and tiles".to_string());
        }
        if slots < 2 {
            return Err("CUDA trial engine requires at least two row slots".to_string());
        }
        validate_trial_layout(base.len(), leaves, tiles)?;
        let runtime = Runtime::new()?;
        let row_bytes = base.len();
        let total_bytes = slots
            .checked_mul(row_bytes)
            .ok_or("CUDA resident row byte count overflow")?;
        let rows = DeviceBuffer::zeroed(&runtime.stream, total_bytes).map_err(cuda_error)?;
        copy_prefix(&rows, base, &runtime.stream)?;
        let leaves = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
        let tiles = DeviceBuffer::from_host(&runtime.stream, tiles).map_err(cuda_error)?;
        let scratch = Scratch::new(&runtime.stream)?;
        Ok(Self {
            runtime,
            rows,
            leaves,
            tiles,
            row_bytes,
            slots,
            scratch,
            profiling: false,
            last_profile: None,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if !enabled {
            self.last_profile = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> CudaResult<(usize, f32)> {
        let (selections, _) = self.ask_impl(
            base_slot,
            history_slots,
            outcomes,
            trial_slot,
            seeds,
            draws,
            1,
            seeds.len(),
            &[],
            &[],
            leaves,
            config,
            materialize_row,
            false,
        )?;
        let selection = selections[0];
        Ok((selection.index as usize, selection.score))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_with_scores(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> CudaResult<(usize, Vec<f32>)> {
        let (selections, scores) = self.ask_impl(
            base_slot,
            history_slots,
            outcomes,
            trial_slot,
            seeds,
            draws,
            1,
            seeds.len(),
            &[],
            &[],
            leaves,
            config,
            materialize_row,
            true,
        )?;
        let selection = selections[0];
        Ok((selection.index as usize, scores))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_multi(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        regions: usize,
        candidates_per_region: usize,
        seeds: &[u64],
        draws: &[f32],
        centers: &[CenterStep],
        region_centers: &[u32],
        leaves: &[Leaf],
        config: Ask,
    ) -> CudaResult<Vec<(usize, f32)>> {
        let (selections, _) = self.ask_impl(
            base_slot,
            history_slots,
            outcomes,
            base_slot,
            seeds,
            draws,
            regions,
            candidates_per_region,
            centers,
            region_centers,
            leaves,
            config,
            false,
            false,
        )?;
        Ok(selections
            .into_iter()
            .map(|selection| (selection.index as usize, selection.score))
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn ask_impl(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        regions: usize,
        candidates_per_region: usize,
        centers: &[CenterStep],
        region_centers: &[u32],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
        read_scores: bool,
    ) -> CudaResult<(Vec<Selection>, Vec<f32>)> {
        self.last_profile = None;
        let client = TRACY.get_or_init(tracy_client::Client::start);
        let _zone = client
            .clone()
            .span(tracy_client::span_location!("ennx.cuda.trial.ask"), 0);
        self.check_slot(base_slot)?;
        self.check_slot(trial_slot)?;
        if materialize_row && base_slot == trial_slot {
            return Err("CUDA trial base and destination slots must differ".to_string());
        }
        if history_slots.is_empty() || history_slots.len() > MAX_HISTORY {
            return Err(format!(
                "CUDA trial history must contain 1..={MAX_HISTORY} rows"
            ));
        }
        if history_slots.len() != outcomes.len() {
            return Err("CUDA trial history slots and outcomes differ in length".to_string());
        }
        if seeds.is_empty() || seeds.len() != draws.len() {
            return Err(
                "CUDA trial seeds and draws must be non-empty and equal length".to_string(),
            );
        }
        if regions == 0 || candidates_per_region == 0 {
            return Err("CUDA multi-region search requires non-zero dimensions".to_string());
        }
        let expected_candidates = regions
            .checked_mul(candidates_per_region)
            .ok_or("CUDA multi-region candidate count overflow")?;
        if seeds.len() != expected_candidates {
            return Err(format!(
                "CUDA expected {expected_candidates} candidates for {regions} regions, got {}",
                seeds.len()
            ));
        }
        validate_centers(centers, region_centers, regions)?;
        if materialize_row && (regions != 1 || !centers.is_empty()) {
            return Err("CUDA can materialize only a single root-based trial".to_string());
        }
        if config.neighbors == 0 || config.neighbors > history_slots.len() {
            return Err("CUDA trial neighbor count exceeds resident history".to_string());
        }
        if leaves.len() != self.leaves.len() {
            return Err("CUDA trial leaf layout changed after engine creation".to_string());
        }
        validate_trial_leaves(self.row_bytes, leaves)?;
        if outcomes.iter().any(|value| !value.is_finite())
            || draws.iter().any(|value| !value.is_finite())
        {
            return Err("CUDA trial outcomes and draws must be finite".to_string());
        }
        if config.acquisition > 2
            || !config.epistemic_scale.is_finite()
            || !config.aleatoric_scale.is_finite()
            || !config.y_scale.is_finite()
            || !config.beta.is_finite()
        {
            return Err("CUDA trial acquisition configuration is invalid".to_string());
        }
        for &slot in history_slots {
            self.check_slot(slot as usize)?;
        }

        let history = to_u32(history_slots.len(), "history rows")?;
        let candidates = to_u32(seeds.len(), "candidate count")?;
        self.scratch
            .ensure(&self.runtime.stream, history_slots.len(), seeds.len())?;
        self.scratch
            .ensure_centers(&self.runtime.stream, centers.len())?;
        let packed_seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        copy_prefix(
            &self.scratch.history_slots,
            history_slots,
            &self.runtime.stream,
        )?;
        copy_prefix(&self.scratch.outcomes, outcomes, &self.runtime.stream)?;
        copy_prefix(&self.scratch.seeds, &packed_seeds, &self.runtime.stream)?;
        copy_prefix(&self.scratch.draws, draws, &self.runtime.stream)?;
        copy_prefix(&self.leaves, leaves, &self.runtime.stream)?;
        if !centers.is_empty() {
            let candidate_centers = region_centers
                .iter()
                .flat_map(|&center| std::iter::repeat_n(center, candidates_per_region))
                .collect::<Vec<_>>();
            copy_prefix(&self.scratch.centers, centers, &self.runtime.stream)?;
            copy_prefix(
                &self.scratch.candidate_centers,
                &candidate_centers,
                &self.runtime.stream,
            )?;
        }

        let score_launch = self
            .runtime
            .module
            .prepare_score_trials(LaunchConfig1D::new(candidates, THREADS, 0))
            .map_err(cuda_error)?;
        let pick_launch = self
            .runtime
            .module
            .prepare_pick_trial(LaunchConfig1D::new(
                to_u32(regions, "region count")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let profile = self.profiling
            || tracy_client::Client::is_connected()
            || std::env::var_os("ENNX_CUDA_PROFILE").is_some();
        let score_start = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;

        // SAFETY: each candidate block owns its score output, each pick thread
        // participates in the tree reduction, and all resident buffers cover the
        // validated launch dimensions.
        self.runtime
            .module
            .score_trials(
                &self.runtime.stream,
                &score_launch,
                &self.rows,
                &self.scratch.history_slots,
                &self.scratch.outcomes,
                &self.scratch.seeds,
                &self.scratch.draws,
                &self.leaves,
                &self.scratch.centers,
                &self.scratch.candidate_centers,
                &mut self.scratch.scores,
                to_u32(self.row_bytes, "row bytes")?,
                history,
                candidates,
                to_u32(base_slot, "base slot")?,
                to_u32(centers.len(), "center count")?,
                to_u32(config.neighbors, "neighbors")?,
                config.acquisition,
                config.epistemic_scale,
                config.aleatoric_scale,
                config.y_scale,
                config.beta,
            )
            .map_err(cuda_error)?;
        let score_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime
            .module
            .pick_trial(
                &self.runtime.stream,
                &pick_launch,
                &self.scratch.scores,
                &mut self.scratch.selection,
                to_u32(regions, "region count")?,
                to_u32(candidates_per_region, "candidates per region")?,
            )
            .map_err(cuda_error)?;
        let pick_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        if materialize_row {
            self.launch_write(base_slot, trial_slot)?;
        }
        let materialize_end = if profile && materialize_row {
            Some(timing_event(&self.runtime.stream)?)
        } else {
            None
        };
        self.runtime.context.check_err().map_err(cuda_error)?;
        let selections = read_prefix(&self.scratch.selection, &self.runtime.stream, regions)?;
        for (region, selection) in selections.iter().enumerate() {
            let first = region * candidates_per_region;
            let end = first + candidates_per_region;
            if !(first..end).contains(&(selection.index as usize)) {
                return Err(format!(
                    "CUDA region {region} selected invalid trial index {}",
                    selection.index
                ));
            }
        }
        let scores = if read_scores {
            read_prefix(&self.scratch.scores, &self.runtime.stream, seeds.len())?
        } else {
            Vec::new()
        };
        self.last_profile = match (score_start, score_end, pick_end) {
            (Some(score_start), Some(score_end), Some(pick_end)) => {
                let events = AskEvents {
                    score_start,
                    score_end,
                    pick_end,
                    materialize_end,
                };
                let profile = events.profile()?;
                publish_profile(client, profile);
                Some(profile)
            }
            _ => None,
        };
        Ok((selections, scores))
    }

    pub fn last_profile(&self) -> Option<AskProfile> {
        self.last_profile
    }

    pub fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        leaves: &[Leaf],
    ) -> CudaResult<()> {
        self.check_slot(base_slot)?;
        self.check_slot(trial_slot)?;
        if base_slot == trial_slot {
            return Err("CUDA trial base and destination slots must differ".to_string());
        }
        if leaves.len() != self.leaves.len() {
            return Err("CUDA trial leaf layout changed after engine creation".to_string());
        }
        let seeds = [Seed {
            low: seed as u32,
            high: (seed >> 32) as u32,
        }];
        copy_prefix(&self.scratch.seeds, &seeds, &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.selection,
            &[Selection {
                index: 0,
                score: 0.0,
            }],
            &self.runtime.stream,
        )?;
        copy_prefix(&self.leaves, leaves, &self.runtime.stream)?;
        // SAFETY: the validated base and trial slots are distinct row regions,
        // and write_trial gives each tile thread a unique output byte.
        self.launch_write(base_slot, trial_slot)?;
        self.runtime.stream.synchronize().map_err(cuda_error)
    }

    pub fn read(&self, slot: usize) -> CudaResult<Vec<u8>> {
        self.check_slot(slot)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        let mut row = vec![0_u8; self.row_bytes];
        let source = self.rows.cu_deviceptr() + (slot * self.row_bytes) as u64;
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                row.as_mut_ptr(),
                source,
                self.row_bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        Ok(row)
    }

    pub fn write(&mut self, slot: usize, row: &[u8]) -> CudaResult<()> {
        self.check_slot(slot)?;
        if row.len() != self.row_bytes {
            return Err(format!(
                "CUDA row has {} bytes, expected {}",
                row.len(),
                self.row_bytes
            ));
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        let destination = self.rows.cu_deviceptr() + (slot * self.row_bytes) as u64;
        unsafe {
            cuda_core::memory::memcpy_htod_sync(destination, row.as_ptr(), row.len())
                .map_err(cuda_error)
        }
    }

    fn launch_write(&mut self, base_slot: usize, trial_slot: usize) -> CudaResult<()> {
        let launch = self
            .runtime
            .module
            .prepare_write_trial(LaunchConfig1D::new(
                to_u32(self.tiles.len(), "tile count")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .write_trial(
                &self.runtime.stream,
                &launch,
                &mut self.rows,
                &self.scratch.seeds,
                &self.scratch.selection,
                &self.leaves,
                &self.tiles,
                to_u32(self.row_bytes, "row bytes")?,
                to_u32(base_slot, "base slot")?,
                to_u32(trial_slot, "trial slot")?,
            )
            .map_err(cuda_error)
    }

    fn check_slot(&self, slot: usize) -> CudaResult<()> {
        if slot >= self.slots {
            Err(format!(
                "CUDA row slot {slot} exceeds capacity {}",
                self.slots
            ))
        } else {
            Ok(())
        }
    }
}

impl AskEvents {
    fn profile(&self) -> CudaResult<AskProfile> {
        let end = self.materialize_end.as_ref().unwrap_or(&self.pick_end);
        Ok(AskProfile {
            score_ms: self
                .score_start
                .elapsed_ms(&self.score_end)
                .map_err(cuda_error)?,
            pick_ms: self
                .score_end
                .elapsed_ms(&self.pick_end)
                .map_err(cuda_error)?,
            materialize_ms: self
                .materialize_end
                .as_ref()
                .map(|materialize_end| self.pick_end.elapsed_ms(materialize_end))
                .transpose()
                .map_err(cuda_error)?
                .unwrap_or(0.0),
            total_ms: self.score_start.elapsed_ms(end).map_err(cuda_error)?,
        })
    }
}

fn timing_event(stream: &CudaStream) -> CudaResult<CudaEvent> {
    stream
        .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .map_err(cuda_error)
}

fn publish_profile(client: &tracy_client::Client, profile: AskProfile) {
    for (name, milliseconds) in [
        (
            tracy_client::plot_name!("ennx.cuda.trial.score_ns"),
            profile.score_ms,
        ),
        (
            tracy_client::plot_name!("ennx.cuda.trial.pick_ns"),
            profile.pick_ms,
        ),
        (
            tracy_client::plot_name!("ennx.cuda.trial.materialize_ns"),
            profile.materialize_ms,
        ),
        (
            tracy_client::plot_name!("ennx.cuda.trial.total_ns"),
            profile.total_ms,
        ),
    ] {
        client.plot(name, f64::from(milliseconds) * 1_000_000.0);
    }
}

const DENSE_TILE_ELEMENTS: usize = 65_536;

pub fn dense_apply(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
) -> CudaResult<Vec<f32>> {
    if base.is_empty() || leaves.is_empty() || terms.is_empty() {
        return Err("CUDA dense apply requires base values, leaves, and terms".to_string());
    }
    let mut tiles = Vec::new();
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let leaf_index = to_u32(leaf_index, "dense leaf count")?;
        let length = usize::try_from(leaf.length)
            .map_err(|_| "CUDA dense leaf length exceeds usize".to_string())?;
        let mut start = 0;
        while start < length {
            let tile_length = (length - start).min(DENSE_TILE_ELEMENTS);
            tiles.push(DenseTile {
                leaf: leaf_index,
                start: to_u32(start, "dense leaf length")?,
                length: to_u32(tile_length, "dense tile length")?,
                pad: 0,
            });
            start += tile_length;
        }
    }
    if tiles.is_empty() {
        return Err("CUDA dense apply requires non-empty leaves".to_string());
    }

    let runtime = Runtime::new()?;
    let base_buffer = DeviceBuffer::from_host(&runtime.stream, base).map_err(cuda_error)?;
    let leaf_buffer = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
    let term_buffer = DeviceBuffer::from_host(&runtime.stream, terms).map_err(cuda_error)?;
    let tile_buffer = DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&runtime.stream, base.len()).map_err(cuda_error)?;
    let config = LaunchConfig {
        grid_dim: (to_u32(tiles.len(), "dense tile count")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: validation in ENNx guarantees complete, disjoint leaves. Each
    // CUDA block owns one tile and each tile thread owns distinct outputs.
    unsafe {
        runtime
            .module
            .apply_dense(
                &runtime.stream,
                config,
                &base_buffer,
                &leaf_buffer,
                &term_buffer,
                &tile_buffer,
                &mut output,
                to_u32(terms.len(), "dense term count")?,
            )
            .map_err(cuda_error)?;
    }
    runtime.context.check_err().map_err(cuda_error)?;
    read_prefix(&output, &runtime.stream, base.len())
}

pub struct DenseLinearEngine {
    runtime: Runtime,
    input: DeviceBuffer<f32>,
    weight: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    terms: DeviceBuffer<DenseTerm>,
    output: DeviceBuffer<f32>,
    term_capacity: usize,
    params: DenseLinearParams,
}

impl DenseLinearEngine {
    pub fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        mut params: DenseLinearParams,
    ) -> CudaResult<Self> {
        if params.rows == 0 || params.columns == 0 {
            return Err("CUDA dense linear requires non-zero rows and columns".to_string());
        }
        let expected = params.rows as usize * params.columns as usize;
        if weight.len() != expected {
            return Err(format!(
                "CUDA dense linear has {} weights, expected {expected}",
                weight.len()
            ));
        }
        if bias.is_some_and(|values| values.len() != params.rows as usize) {
            return Err("CUDA dense linear bias length must equal rows".to_string());
        }
        params.has_bias = u32::from(bias.is_some());
        params.term_count = 0;
        let runtime = Runtime::new()?;
        let no_bias = [0.0_f32];
        let input =
            DeviceBuffer::zeroed(&runtime.stream, params.columns as usize).map_err(cuda_error)?;
        let weight = DeviceBuffer::from_host(&runtime.stream, weight).map_err(cuda_error)?;
        let bias = DeviceBuffer::from_host(&runtime.stream, bias.unwrap_or(&no_bias))
            .map_err(cuda_error)?;
        let terms = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let output =
            DeviceBuffer::zeroed(&runtime.stream, params.rows as usize).map_err(cuda_error)?;
        Ok(Self {
            runtime,
            input,
            weight,
            bias,
            terms,
            output,
            term_capacity: 1,
            params,
        })
    }

    pub fn eval(&mut self, input: &[f32], terms: &[DenseTerm]) -> CudaResult<Vec<f32>> {
        if input.len() != self.params.columns as usize {
            return Err(format!(
                "CUDA dense linear input has {} values, expected {}",
                input.len(),
                self.params.columns
            ));
        }
        if terms.is_empty() {
            return Err("CUDA dense linear requires perturbation terms".to_string());
        }
        if terms.len() > self.term_capacity {
            self.term_capacity = terms
                .len()
                .checked_next_power_of_two()
                .ok_or("CUDA dense linear term capacity overflow")?;
            self.terms = DeviceBuffer::zeroed(&self.runtime.stream, self.term_capacity)
                .map_err(cuda_error)?;
        }
        copy_prefix(&self.input, input, &self.runtime.stream)?;
        copy_prefix(&self.terms, terms, &self.runtime.stream)?;
        self.params.term_count = to_u32(terms.len(), "dense linear term count")?;
        let launch = self
            .runtime
            .module
            .prepare_dense_linear(LaunchConfig1D::new(self.params.rows, THREADS, 0))
            .map_err(cuda_error)?;
        // SAFETY: every row owns one block, each block reduces exactly one
        // output, and all resident buffers match the validated dimensions.
        self.runtime
            .module
            .dense_linear(
                &self.runtime.stream,
                &launch,
                &self.input,
                &self.weight,
                &self.bias,
                &self.terms,
                &mut self.output,
                self.params,
            )
            .map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        read_prefix(
            &self.output,
            &self.runtime.stream,
            self.params.rows as usize,
        )
    }
}

fn copy_prefix<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    values: &[T],
    stream: &CudaStream,
) -> CudaResult<()> {
    if values.len() > buffer.len() {
        return Err(format!(
            "CUDA input has {} elements, buffer capacity is {}",
            values.len(),
            buffer.len()
        ));
    }
    if values.is_empty() {
        return Ok(());
    }
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buffer.cu_deviceptr(),
            values.as_ptr(),
            size_of_val(values),
            stream.cu_stream(),
        )
        .map_err(cuda_error)
    }
}

fn read_prefix<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    stream: &CudaStream,
    len: usize,
) -> CudaResult<Vec<T>> {
    if len > buffer.len() {
        return Err(format!(
            "CUDA output requests {len} elements from capacity {}",
            buffer.len()
        ));
    }
    let mut output = Vec::<T>::with_capacity(len);
    if len == 0 {
        return Ok(output);
    }
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            buffer.cu_deviceptr(),
            len * size_of::<T>(),
            stream.cu_stream(),
        )
        .map_err(cuda_error)?;
    }
    stream.synchronize().map_err(cuda_error)?;
    unsafe {
        output.set_len(len);
    }
    Ok(output)
}

fn to_u32(value: usize, name: &str) -> CudaResult<u32> {
    value
        .try_into()
        .map_err(|_| format!("CUDA {name} exceeds u32"))
}

fn validate_centers(
    centers: &[CenterStep],
    region_centers: &[u32],
    regions: usize,
) -> CudaResult<()> {
    if centers.is_empty() {
        if !region_centers.is_empty() {
            return Err("CUDA root-based search must not provide region centers".to_string());
        }
        return Ok(());
    }
    if region_centers.len() != regions {
        return Err(format!(
            "CUDA expected {regions} region centers, got {}",
            region_centers.len()
        ));
    }
    for (index, center) in centers.iter().enumerate() {
        if center.parent != u32::MAX && center.parent as usize >= index {
            return Err(format!(
                "CUDA center {index} must reference an earlier parent"
            ));
        }
        let mut depth = 1_usize;
        let mut parent = center.parent;
        while parent != u32::MAX {
            depth += 1;
            if depth > MAX_CENTER_DEPTH {
                return Err(format!(
                    "CUDA center chain exceeds depth {MAX_CENTER_DEPTH}"
                ));
            }
            parent = centers[parent as usize].parent;
        }
    }
    if region_centers
        .iter()
        .any(|&center| center as usize >= centers.len())
    {
        return Err("CUDA region center index is out of bounds".to_string());
    }
    Ok(())
}

fn validate_trial_leaves(row_bytes: usize, leaves: &[Leaf]) -> CudaResult<()> {
    if leaves.is_empty() || row_bytes == 0 || row_bytes > u32::MAX as usize {
        return Err("CUDA trial layout requires non-empty u32-sized rows and leaves".to_string());
    }
    let mut expected_byte_offset = 0usize;
    let mut expected_element_offset = 0u32;
    for (index, leaf) in leaves.iter().enumerate() {
        if leaf.length == 0 || !matches!(leaf.bits, 4 | 8) {
            return Err(format!("CUDA trial leaf {index} has an invalid shape"));
        }
        let encoding_matches = matches!((leaf.bits, leaf.encoding), (4, 0 | 2) | (8, 1 | 3 | 4));
        if !encoding_matches
            || !leaf.scale.is_finite()
            || leaf.scale <= 0.0
            || !leaf.weight.is_finite()
            || leaf.weight <= 0.0
        {
            return Err(format!(
                "CUDA trial leaf {index} has an invalid encoding or scale"
            ));
        }
        if leaf.byte_offset as usize != expected_byte_offset
            || leaf.element_offset != expected_element_offset
        {
            return Err(format!("CUDA trial leaf {index} is not contiguous"));
        }
        let bytes = if leaf.bits == 4 {
            leaf.length.div_ceil(2)
        } else {
            leaf.length
        };
        expected_byte_offset = expected_byte_offset
            .checked_add(bytes as usize)
            .ok_or("CUDA trial row byte count overflow")?;
        expected_element_offset = expected_element_offset
            .checked_add(leaf.length)
            .ok_or("CUDA trial element count overflow")?;
        let max_code = (1u32 << leaf.bits) - 1;
        if leaf.whole > max_code {
            return Err(format!(
                "CUDA trial leaf {index} perturbation exceeds its encoding"
            ));
        }
    }
    if expected_byte_offset != row_bytes {
        return Err(format!(
            "CUDA trial leaves cover {expected_byte_offset} bytes, expected {row_bytes}"
        ));
    }
    Ok(())
}

fn validate_trial_layout(row_bytes: usize, leaves: &[Leaf], tiles: &[Tile]) -> CudaResult<()> {
    validate_trial_leaves(row_bytes, leaves)?;
    if tiles.is_empty() {
        return Err("CUDA trial layout requires at least one tile".to_string());
    }
    let mut covered = vec![0u32; leaves.len()];
    for (index, tile) in tiles.iter().enumerate() {
        let leaf_index = tile.leaf as usize;
        let Some(leaf) = leaves.get(leaf_index) else {
            return Err(format!(
                "CUDA trial tile {index} references an invalid leaf"
            ));
        };
        if tile.length == 0 || tile.start != covered[leaf_index] {
            return Err(format!(
                "CUDA trial tile {index} is empty, overlapping, or out of order"
            ));
        }
        let end = tile
            .start
            .checked_add(tile.length)
            .ok_or("CUDA trial tile extent overflow")?;
        if end > leaf.length || (leaf.bits == 4 && tile.start & 1 != 0) {
            return Err(format!(
                "CUDA trial tile {index} exceeds or misaligns its leaf"
            ));
        }
        covered[leaf_index] = end;
    }
    if covered
        .iter()
        .zip(leaves)
        .any(|(&length, leaf)| length != leaf.length)
    {
        return Err("CUDA trial tiles do not cover every leaf exactly once".to_string());
    }
    Ok(())
}

fn cuda_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}
