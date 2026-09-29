//! Observation shape and output bounds.

use super::*;

impl ENN {
    pub fn len(&self) -> usize {
        self.num_obs
    }

    pub fn is_empty(&self) -> bool {
        self.num_obs == 0
    }

    pub fn num_outputs(&self) -> usize {
        self.num_metrics
    }

    pub fn y_bounds(&self) -> &Array2<f64> {
        &self.y_bounds
    }

    pub fn bounded_outputs(&self) -> bool {
        self.bounded_outputs
    }

    pub(crate) fn output_scale(&self) -> &Array1<f64> {
        &self.y_scale
    }

    pub fn x_scale(&self) -> Array2<f64> {
        self.x_scale.clone().insert_axis(ndarray::Axis(0))
    }

    pub fn y_scale(&self) -> Array2<f64> {
        self.y_scale.clone().insert_axis(ndarray::Axis(0))
    }

    pub(crate) fn num_obs(&self) -> usize {
        self.num_obs
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn has_yvar(&self) -> bool {
        self.num_obs > 0 && self.rows().row_yvar(0).ok().flatten().is_some()
    }
}
