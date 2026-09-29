/// Weight-tied depth recurrence inside a physical layer stack.
///
/// Layers before and after this half-open range execute once. Layers in the
/// range execute `visits` times without re-embedding or re-running readout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentCore {
    pub first_layer: usize,
    pub layer_count: usize,
    pub training_visits: usize,
    pub min_inference_visits: usize,
    pub max_inference_visits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer: usize,
    pub visit: usize,
    pub execution: usize,
    pub recurrent: bool,
}

impl RecurrentCore {
    pub fn selective_fbt() -> Self {
        Self {
            first_layer: 1,
            layer_count: 2,
            training_visits: 2,
            min_inference_visits: 1,
            max_inference_visits: 4,
        }
    }

    pub fn validate(self, physical_layers: usize) -> Result<(), String> {
        let end = self
            .first_layer
            .checked_add(self.layer_count)
            .ok_or("recurrent core layer range overflow")?;
        if physical_layers == 0
            || self.layer_count == 0
            || self.first_layer >= physical_layers
            || end > physical_layers
            || self.training_visits == 0
            || self.min_inference_visits == 0
            || self.min_inference_visits > self.training_visits
            || self.training_visits > self.max_inference_visits
        {
            return Err("recurrent core requires a nonempty in-range layer span and 1 <= min-inference-visits <= training-visits <= max-inference-visits".into());
        }
        Ok(())
    }

    pub fn layer_visits(
        self,
        physical_layers: usize,
        visits: usize,
    ) -> Result<Vec<LayerVisit>, String> {
        self.validate(physical_layers)?;
        if visits < self.min_inference_visits || visits > self.max_inference_visits {
            return Err(format!(
                "inference visits must be in {}..={}, got {visits}",
                self.min_inference_visits, self.max_inference_visits
            ));
        }
        let end = self.first_layer + self.layer_count;
        let mut sequence = Vec::with_capacity(
            physical_layers + self.layer_count.saturating_mul(visits.saturating_sub(1)),
        );
        for layer in 0..self.first_layer {
            sequence.push(LayerVisit {
                layer,
                visit: 0,
                execution: sequence.len(),
                recurrent: false,
            });
        }
        for visit in 0..visits {
            for layer in self.first_layer..end {
                sequence.push(LayerVisit {
                    layer,
                    visit,
                    execution: sequence.len(),
                    recurrent: true,
                });
            }
        }
        for layer in end..physical_layers {
            sequence.push(LayerVisit {
                layer,
                visit: 0,
                execution: sequence.len(),
                recurrent: false,
            });
        }
        Ok(sequence)
    }
}
