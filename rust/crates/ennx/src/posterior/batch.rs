use super::{
    compute_posterior_internals, compute_weighted_posterior, empty_posterior_internals,
    get_neighbor_data, WeightedPosteriorData,
};
use crate::draw::DrawInternals;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};
use ndarray::{Array3, ArrayView2, Axis};

#[allow(clippy::too_many_arguments)]
pub(super) fn shared_batch(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    se_epi_all: &mut Array3<f64>,
    se_ale_all: &mut Array3<f64>,
) -> Result<(), ENNError> {
    let neighbor_data = get_neighbor_data(
        model,
        x,
        &paramss[0],
        flags.exclude_nearest,
        flags.tie_break_neighbors,
    )?;

    if let Some(data) = neighbor_data {
        let wp_data = WeightedPosteriorData {
            dist2s: &data.dist2s.view(),
            idx: &data.idx,
            y_neighbors: &data.y_neighbors.view(),
            params: &paramss[0],
            observation_noise: flags.observation_noise,
            yvar_neighbors_override: None,
        };

        for (i, params) in paramss.iter().enumerate() {
            let data_with_params = WeightedPosteriorData { params, ..wp_data };
            let internals = compute_weighted_posterior(model, data_with_params, None)?;
            assign_result(&internals, mu_all, se_all, se_epi_all, se_ale_all, i);
        }
    } else {
        let batch_size = x.nrows();
        let internals = empty_posterior_internals(model, batch_size);
        for i in 0..paramss.len() {
            assign_result(&internals, mu_all, se_all, se_epi_all, se_ale_all, i);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn separate_batch(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    se_epi_all: &mut Array3<f64>,
    se_ale_all: &mut Array3<f64>,
) -> Result<(), ENNError> {
    for (i, params) in paramss.iter().enumerate() {
        let internals = compute_posterior_internals(model, x, params, flags)?;
        assign_result(&internals, mu_all, se_all, se_epi_all, se_ale_all, i);
    }
    Ok(())
}

pub(super) fn assign_result(
    internals: &DrawInternals,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    se_epi_all: &mut Array3<f64>,
    se_ale_all: &mut Array3<f64>,
    index: usize,
) {
    let slice = ndarray::Slice::from(index..index + 1);
    mu_all
        .slice_axis_mut(Axis(0), slice)
        .assign(&internals.mu.slice_axis(Axis(0), ndarray::Slice::from(..)));
    se_all
        .slice_axis_mut(Axis(0), slice)
        .assign(&internals.se.slice_axis(Axis(0), ndarray::Slice::from(..)));
    se_epi_all.slice_axis_mut(Axis(0), slice).assign(
        &internals
            .se_epi
            .slice_axis(Axis(0), ndarray::Slice::from(..)),
    );
    se_ale_all.slice_axis_mut(Axis(0), slice).assign(
        &internals
            .se_ale
            .slice_axis(Axis(0), ndarray::Slice::from(..)),
    );
}
