use crate::error::BpannError;
use crate::index::page::Page;
use crate::index::sync::IncrementalIndex;
#[cfg(test)]
#[path = "metric_tests.rs"]
mod tests;

fn page_vectors(page: &Page) -> Vec<&[f32]> {
    match page {
        Page::Internal { centroids, .. } => centroids.iter().map(Vec::as_slice).collect(),
        Page::Leaf {
            vectors,
            stored_centroid,
            ..
        } => vectors
            .iter()
            .map(Vec::as_slice)
            .chain(stored_centroid.iter().map(Vec::as_slice))
            .collect(),
    }
}

fn scale_vector(vector: &mut [f32], factors: &[f64]) {
    for (v, &factor) in vector.iter_mut().zip(factors) {
        *v = (f64::from(*v) * factor) as f32;
    }
}

impl IncrementalIndex {
    pub(crate) fn check_rescale(&self, factors: &[f64]) -> Result<(), BpannError> {
        let valid = factors.iter().all(|v| v.is_finite() && *v > 0.0)
            && self
                .indices
                .iter()
                .flat_map(|index| &index.pages)
                .flat_map(page_vectors)
                .all(|row| {
                    row.len() == factors.len()
                        && row
                            .iter()
                            .zip(factors)
                            .all(|(&v, &f)| ((v as f64 * f) as f32).is_finite())
                });
        if !valid {
            return Err(BpannError::InvalidParameter(
                "metric rescale exceeds FP32 index range".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rescale(&mut self, factors: &[f64]) {
        self.rescale_pending(factors);
        for index in &mut self.indices {
            for page in &mut index.pages {
                match page {
                    Page::Internal { centroids, .. } => {
                        for vector in centroids {
                            scale_vector(vector, factors);
                        }
                    }
                    Page::Leaf {
                        vectors,
                        stored_centroid,
                        ..
                    } => {
                        for vector in vectors {
                            scale_vector(vector, factors);
                        }
                        if let Some(vector) = stored_centroid {
                            scale_vector(vector, factors);
                        }
                    }
                }
            }
        }
    }
}
