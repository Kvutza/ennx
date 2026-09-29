//! Optional upstream AUTO metric on ordinary disk ENN, not full model-weight rows.
use super::*;
use crate::metric_auto::{AutoMetric, MetricSnapshot, ScaleUpdate};

impl ENN {
    /// Enable the pinned upstream dependence estimator, Algorithm R reservoir,
    /// LOO gate and geometric refit schedule. The caller supplies the seed.
    /// This does not select the resident optimizer's tied-family metric policy.
    pub fn learn_metric(&mut self, tied: Vec<Vec<usize>>, seed: u64) -> Result<(), ENNError> {
        if !matches!(self.backend, EnnBackend::Disk(_)) || self.scale_x || self.metric.is_some() {
            return Err(ENNError::InvalidParameter(
                "Enable AUTO once on unscaled disk BPANN".into(),
            ));
        }
        let mut metric = AutoMetric::new(self.num_dim, self.num_metrics, tied, seed)?;
        // Observe in bounded chunks; never gather the whole persisted dataset.
        for start in (0..self.num_obs).step_by(64) {
            let ids = (start..(start + 64).min(self.num_obs)).collect::<Vec<_>>();
            let (x, y, _) = self.backend.train_rows(&ids)?;
            metric.observe(x.as_slice().unwrap(), y.as_slice().unwrap(), ids.len())?;
        }
        let weights = metric.weights();
        let scale = Array1::from_iter(weights.iter().map(|v| 1.0 / v.sqrt()));
        self.backend.set_metric(&scale, true)?;
        self.x_scale = scale;
        self.scale_x = true;
        self.metric = Some(metric);
        Ok(())
    }

    pub fn metric_snapshot(&self) -> Option<MetricSnapshot> {
        self.metric.as_ref().map(AutoMetric::snapshot)
    }

    pub fn metric_config(
        &mut self,
        growth: Option<f64>,
        drift: Option<f64>,
        seed: Option<u64>,
        capacity: Option<usize>,
    ) -> Result<(), ENNError> {
        self.metric
            .as_mut()
            .ok_or_else(|| ENNError::InvalidParameter("AUTO metric is not enabled".into()))?
            .configure(growth, drift, seed, capacity)
    }

    pub fn metric_weights(&mut self, weights: &[f64]) -> Result<(), ENNError> {
        let update = self
            .metric
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("AUTO metric is not enabled".into()))?
            .preview(weights)?;
        self.apply_metric(update)?;
        self.metric.as_mut().unwrap().set_weights(weights)?;
        Ok(())
    }

    pub(super) fn observe_metric(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        if self.metric.is_none() {
            return Ok(());
        }
        let x = x.to_owned();
        let y = if self.bounded_outputs {
            crate::y_bounds::warp_y(*y, &self.y_bounds)?
        } else {
            y.to_owned()
        };
        let ready = self.metric.as_mut().unwrap().ingest(
            x.as_slice().unwrap(),
            y.as_slice().unwrap(),
            x.nrows(),
        )?;
        if ready {
            if let Some(weights) = self.metric.as_mut().unwrap().propose()? {
                self.metric_weights(&weights)?;
            }
        }
        Ok(())
    }

    fn apply_metric(&mut self, update: ScaleUpdate) -> Result<(), ENNError> {
        let scale = Array1::from(update.x_scale);
        self.backend.set_metric(&scale, update.rebuild)?;
        self.x_scale = scale;
        Ok(())
    }
}
