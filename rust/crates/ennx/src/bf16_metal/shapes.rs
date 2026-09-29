use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Tile {
    pub(super) leaf: u32,
    pub(super) start: u32,
    pub(super) length: u32,
    pub(super) pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct Params {
    pub(super) seed: u64,
    pub(super) stream_seed: u64,
    pub(super) basis_seed: u64,
    pub(super) radius: f32,
    pub(super) alternate_radius: f32,
    pub(super) candidate: u32,
    pub(super) tiles: u32,
    pub(super) history: u32,
    pub(super) initialize: u32,
    pub(super) mode: u32,
    pub(super) base_slot: u32,
    pub(super) blocks: u32,
    pub(super) program: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ReplayStep {
    pub(super) seed: u64,
    pub(super) radius: f32,
    pub(super) accepted: u32,
    pub(super) candidate: u32,
    pub(super) mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SelectionParams {
    pub(super) root_seed: u64,
    pub(super) basis_seed: u64,
    pub(super) outcomes: [f32; MAX_HISTORY],
    pub(super) variances: [f32; MAX_HISTORY],
    pub(super) draws: [f32; MAX_HISTORY],
    pub(super) base_distances: [f32; MAX_HISTORY],
    pub(super) local_scales: [f32; MAX_HISTORY],
    pub(super) latent_norms: [f32; 4],
    pub(super) axis_history: [f32; MAX_HISTORY],
    pub(super) axis_candidates: [f32; 4],
    pub(super) axis_base: f32,
    pub(super) axis_weight: f32,
    pub(super) axis_enabled: u32,
    pub(super) epistemic_scale: f32,
    pub(super) aleatoric_scale: f32,
    pub(super) y_scale: f32,
    pub(super) beta: f32,
    pub(super) radius: f32,
    pub(super) alternate_radius: f32,
    pub(super) neighbors: u32,
    pub(super) history: u32,
    pub(super) acquisition: u32,
    pub(super) tiles: u32,
    pub(super) mode: u32,
    pub(super) program: u32,
    pub(super) resident_history: u32,
    pub(super) resident_indices: [u32; 2],
    pub(super) implicit_history: u32,
    pub(super) exact_history: u32,
    pub(super) latent_history: u32,
    pub(super) forced_candidate: u32,
    pub(super) distance_scaling: u32,
    pub(super) local_scale_neighbors: u32,
    pub(super) incumbent_index: u32,
    pub(super) candidate_floor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct Decision {
    pub(super) index: u32,
    pub(super) valid: u32,
    pub(super) root_seed: u64,
    pub(super) seed: u64,
    pub(super) basis_seed: u64,
    pub(super) radius: f32,
    pub(super) score: f32,
    pub(super) mode: u32,
    pub(super) program: u32,
    pub(super) predicted_mean: f32,
    pub(super) predicted_standard_error: f32,
    pub(super) incumbent_mean: f32,
    pub(super) incumbent_standard_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Partial {
    pub(super) anchor: f32,
    pub(super) rejected: f32,
    pub(super) squared: f32,
    pub(super) changed: u32,
    pub(super) invalid: u32,
}
