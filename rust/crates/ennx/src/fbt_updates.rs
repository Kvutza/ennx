use deser::Serialize;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub(super) struct Tensor {
    pub name: String,
    pub family: &'static str,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub elements: usize,
    pub initial_rms: f64,
    pub proposal_scale: f32,
    pub trust_multiplier: f32,
}

#[derive(Debug, Clone)]
struct Round {
    index: u32,
    seed: u64,
    radius: f32,
    accepted: bool,
    changes: Vec<(u64, f64)>,
    scales: Vec<f32>,
}

#[derive(Serialize)]
struct Metrics {
    changed_weights: u64,
    changed_fraction: f64,
    squared_change: f64,
    rms_change: f64,
    relative_rms: Option<f64>,
    realized_requested_ratio: f64,
}

fn metrics(tensor: &Tensor, radius: f32, change: (u64, f64)) -> Metrics {
    let rms = (change.1 / tensor.elements as f64).sqrt();
    Metrics {
        changed_weights: change.0,
        changed_fraction: change.0 as f64 / tensor.elements as f64,
        squared_change: change.1,
        rms_change: rms,
        relative_rms: (tensor.initial_rms > 0.0).then(|| rms / tensor.initial_rms),
        realized_requested_ratio: rms / f64::from(tensor.proposal_scale * radius),
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct UpdateLog {
    pub tensors: Vec<Tensor>,
    rounds: Vec<Round>,
}

impl UpdateLog {
    pub fn push(
        &mut self,
        index: u32,
        seed: u64,
        radius: f32,
        accepted: bool,
        changes: Vec<(u64, f64)>,
    ) -> Result<(), String> {
        let scales = self.tensors.iter().map(|t| t.proposal_scale).collect();
        self.push_scaled(index, seed, radius, accepted, changes, scales)
    }

    pub fn push_scaled(
        &mut self,
        index: u32,
        seed: u64,
        radius: f32,
        accepted: bool,
        changes: Vec<(u64, f64)>,
        scales: Vec<f32>,
    ) -> Result<(), String> {
        if changes.len() != self.tensors.len()
            || scales.len() != self.tensors.len()
            || scales.iter().any(|s| !s.is_finite() || *s <= 0.0)
            || !radius.is_finite()
            || radius <= 0.0
            || self
                .tensors
                .iter()
                .zip(&changes)
                .any(|(tensor, &(changed, squared))| {
                    changed > tensor.elements as u64 || !squared.is_finite() || squared < 0.0
                })
        {
            return Err("Invalid per-tensor update statistics".into());
        }
        self.rounds.push(Round {
            index,
            seed,
            radius,
            accepted,
            changes,
            scales,
        });
        Ok(())
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer.flush().map_err(|e| e.to_string())
    }

    fn write_to(&self, writer: &mut impl Write) -> Result<(), String> {
        #[derive(Serialize)]
        struct Record<'a> {
            round: u32,
            seed: u64,
            radius: f32,
            accepted: bool,
            tensor_id: usize,
            #[deser(flatten)]
            tensor: &'a Tensor,
            #[deser(flatten)]
            metrics: Metrics,
        }
        for round in &self.rounds {
            for (tensor_id, (tensor, &change)) in
                self.tensors.iter().zip(&round.changes).enumerate()
            {
                let mut tensor = tensor.clone();
                tensor.proposal_scale = round.scales[tensor_id];
                tensor.trust_multiplier =
                    tensor.proposal_scale / tensor.initial_rms.max(1e-6) as f32;
                ennx_wire::json::to_writer(
                    &mut *writer,
                    &Record {
                        round: round.index,
                        seed: round.seed,
                        radius: round.radius,
                        accepted: round.accepted,
                        tensor_id,
                        tensor: &tensor,
                        metrics: metrics(&tensor, round.radius, change),
                    },
                )
                .map_err(|e| e.to_string())?;
                writer.write_all(b"\n").map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_metrics() {
        let tensor = Tensor {
            name: "layer.0.expert.0.down".into(),
            family: "expert_down",
            layer: Some(0),
            expert: Some(0),
            elements: 4,
            initial_rms: 2.0,
            proposal_scale: 2.0,
            trust_multiplier: 1.0,
        };
        // Realized changes [0, 0, 0.5, -0.5].
        let mut log = UpdateLog {
            tensors: vec![tensor],
            ..Default::default()
        };
        log.push(1, 17, 0.25, false, vec![(2, 0.5)]).unwrap();
        log.push(2, 18, 0.25, true, vec![(0, 0.0)]).unwrap();
        let mut bytes = Vec::new();
        log.write_to(&mut bytes).unwrap();
        let rows: Vec<ennx_wire::json::Value> = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| ennx_wire::json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["changed_fraction"], 0.5);
        assert_eq!(rows[0]["accepted"], false);
        let rms = rows[0]["rms_change"].as_f64().unwrap();
        assert!((rms - 0.125f64.sqrt()).abs() < 1e-12);
        assert_eq!(rows[0]["relative_rms"].as_f64().unwrap(), rms / 2.0);
        assert_eq!(
            rows[0]["realized_requested_ratio"].as_f64().unwrap(),
            rms / 0.5
        );
        assert_eq!(rows[1]["accepted"], true);
        assert_eq!(rows[1]["rms_change"], 0.0);
        assert!(log.push(3, 19, 0.25, false, vec![(5, 0.0)]).is_err());
        log.tensors[0].initial_rms = 0.0;
        assert!(
            metrics(&log.tensors[0], 0.25, (0, 0.0))
                .relative_rms
                .is_none()
        );
    }
}
