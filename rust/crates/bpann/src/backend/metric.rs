//! Diagonal metric changes preserve raw observation rows and forest membership.
use super::*;

impl BpannBackend {
    pub fn set_metric(&mut self, scale: &Array1<f64>, rebuild: bool) -> Result<(), BpannError> {
        if scale.len() != self.num_dim || scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(BpannError::InvalidParameter(
                "metric scales must be positive, finite and dimension-matched".into(),
            ));
        }
        if self.scale_x && self.x_scale == *scale {
            return Ok(());
        }
        let factors = (0..self.num_dim)
            .map(|i| {
                let old = if self.scale_x { self.x_scale[i] } else { 1.0 };
                old / scale[i]
            })
            .collect::<Vec<_>>();
        if !rebuild {
            self.index.check_rescale(&factors)?;
        }
        metric_pending(&self.work_dir)?;
        if rebuild {
            self.reset_index();
        } else {
            self.index.rescale(&factors);
        }
        self.scale_x = true;
        self.x_scale = scale.clone();
        *self.small_n_x_cache.lock().expect("small_n_x_cache") = None;
        *self.index_dirty.lock().expect("index_dirty") = true;
        Ok(())
    }
}

pub(super) fn metric_matches(dir: &std::path::Path, scaled: bool, scale: &Array1<f64>) -> bool {
    if dir.join("metric.pending").exists() {
        return false;
    }
    match fs::read(dir.join("metric.bin")) {
        Ok(bytes) => {
            bytes.len() == scale.len() * 8
                && bytes.chunks_exact(8).enumerate().all(|(i, b)| {
                    f64::from_le_bytes(b.try_into().unwrap()) == if scaled { scale[i] } else { 1.0 }
                })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            !scaled || scale.iter().all(|&v| v == 1.0)
        }
        Err(_) => false,
    }
}

pub(super) fn metric_pending(dir: &std::path::Path) -> Result<(), BpannError> {
    fs::write(dir.join("metric.pending"), [])
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub(super) fn persist_metric(
    dir: &std::path::Path,
    scaled: bool,
    scale: &Array1<f64>,
) -> Result<(), BpannError> {
    let bytes = scale
        .iter()
        .flat_map(|&v| (if scaled { v } else { 1.0 }).to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(dir.join("metric.tmp"), bytes)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    fs::rename(dir.join("metric.tmp"), dir.join("metric.bin"))
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    fs::remove_file(dir.join("metric.pending"))
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}
