//! Explicit versioned independent observations. Reporting rewards never define control.
use super::generation::*;
use crate::objective_observation::{MAX_OBJECTIVES, ObjectiveEstimate, ObjectiveObservation};

#[derive(Clone, Copy, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub(crate) struct Estimate {
    mean: f32,
    variance: f32,
}

impl Estimate {
    fn measured(self) -> ObjectiveEstimate {
        ObjectiveEstimate {
            mean: self.mean,
            variance: self.variance,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub(crate) struct VectorResult {
    schema: String,
    /// Task rewards are reporting data, not implicitly aggregated into control.
    #[deser(skip_serializing)]
    rewards: Vec<f32>,
    #[deser(default, skip_serializing_if = Vec::is_empty)]
    names: Vec<String>,
    objectives: Vec<Estimate>,
    control: Estimate,
}

pub(crate) struct RewardResult {
    pub(crate) rewards: Vec<f32>,
    pub(crate) vector: Option<VectorResult>,
}

impl RewardResult {
    pub(crate) fn observation(&self) -> Result<ObjectiveObservation, String> {
        if let Some(vector) = &self.vector {
            return vector.observation();
        }
        let (mean, variance) = if self.rewards.len() > 1 {
            mean_variance(&self.rewards)
        } else {
            (self.rewards[0], 0.0)
        };
        ObjectiveObservation::scalar(mean, variance)
    }
}

impl VectorResult {
    fn observation(&self) -> Result<ObjectiveObservation, String> {
        if self.schema != "ennx.generation_objectives.v2"
            || !(2..=MAX_OBJECTIVES).contains(&self.objectives.len())
            || (!self.names.is_empty()
                && (self.names.len() != self.objectives.len()
                    || self.names.iter().any(|name| name.trim().is_empty())
                    || self
                        .names
                        .iter()
                        .enumerate()
                        .any(|(index, name)| self.names[..index].contains(name))))
        {
            return Err(
                "vector response requires ennx.generation_objectives.v2 and 2..=8 objectives"
                    .into(),
            );
        }
        let mut estimates = [ObjectiveEstimate::default(); MAX_OBJECTIVES];
        for (slot, source) in estimates.iter_mut().zip(&self.objectives) {
            *slot = source.measured();
        }
        ObjectiveObservation::new(&estimates[..self.objectives.len()], self.control.measured())
    }
}

pub(crate) fn parse_vector(path: &Path, tasks: usize) -> Result<RewardResult, String> {
    let vector: VectorResult =
        ennx_wire::json::from_reader(File::open(path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid objective response: {e}"))?;
    validate_vector(vector, tasks)
}

pub(crate) fn native_vector(
    rewards: Vec<f32>,
    objectives: &[Vec<f32>],
    names: &[&str],
) -> Result<RewardResult, String> {
    if !(2..=MAX_OBJECTIVES).contains(&objectives.len())
        || objectives
            .iter()
            .any(|values| values.len() != rewards.len() || values.iter().any(|v| !v.is_finite()))
        || rewards.is_empty()
        || rewards.iter().any(|value| !value.is_finite())
        || names.len() != objectives.len()
        || names.iter().any(|name| name.trim().is_empty())
    {
        return Err("native vector reward requires aligned finite task observations".into());
    }
    let estimate = |values: &[f32]| {
        let (mean, variance) = if values.len() > 1 {
            mean_variance(values)
        } else {
            (values[0], 0.0)
        };
        Estimate { mean, variance }
    };
    let vector = VectorResult {
        schema: "ennx.generation_objectives.v2".into(),
        names: names.iter().map(|name| (*name).into()).collect(),
        control: estimate(&rewards),
        objectives: objectives.iter().map(|values| estimate(values)).collect(),
        rewards,
    };
    validate_vector(vector, objectives[0].len())
}

fn validate_vector(mut vector: VectorResult, tasks: usize) -> Result<RewardResult, String> {
    vector.observation()?;
    if vector.rewards.len() != tasks || vector.rewards.iter().any(|value| !value.is_finite()) {
        return Err("vector response requires one finite reporting reward per task".into());
    }
    Ok(RewardResult {
        rewards: std::mem::take(&mut vector.rewards),
        vector: Some(vector),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector() -> VectorResult {
        ennx_wire::json::from_str(r#"{"schema":"ennx.generation_objectives.v2","rewards":[1000.0],"objectives":[{"mean":-100.0,"variance":0.25},{"mean":100.0,"variance":0.5}],"control":{"mean":-2.0,"variance":0.125}}"#).unwrap()
    }

    #[test]
    fn explicit_control() {
        let result = validate_vector(vector(), 1).unwrap();
        let observation = result.observation().unwrap();
        assert_eq!(result.rewards, [1000.0]);
        assert_eq!(observation.control().mean, -2.0);
        assert_eq!(observation.control().variance, 0.125);
        assert_eq!(observation.estimates()[0].mean, -100.0);
        assert_eq!(observation.estimates()[1].variance, 0.5);
    }

    #[test]
    fn protocol_guards() {
        let mut wrong = vector();
        wrong.schema = "ennx.generation_reward.v1".into();
        assert!(validate_vector(wrong, 1).is_err());
        assert!(validate_vector(vector(), 2).is_err());
        for width in [0, 1, 9] {
            let mut wrong = vector();
            wrong.objectives.resize(
                width,
                Estimate {
                    mean: 0.0,
                    variance: 0.0,
                },
            );
            assert!(validate_vector(wrong, 1).is_err());
        }
        for value in [-1.0, f32::INFINITY, f32::NAN] {
            let mut wrong = vector();
            wrong.control.variance = value;
            assert!(validate_vector(wrong, 1).is_err());
            let mut wrong = vector();
            wrong.objectives[0].variance = value;
            assert!(validate_vector(wrong, 1).is_err());
        }
    }

    #[test]
    fn scalar_compatibility() {
        for rewards in [vec![1.0], vec![-2.0, 3.0, 4.0]] {
            let expected = if rewards.len() == 1 {
                (rewards[0], 0.0)
            } else {
                mean_variance(&rewards)
            };
            let result = RewardResult {
                rewards,
                vector: None,
            };
            let control = result.observation().unwrap().control();
            assert_eq!(control.mean.to_bits(), expected.0.to_bits());
            assert_eq!(control.variance.to_bits(), expected.1.to_bits());
        }
        assert!(ennx_wire::json::from_str::<VectorResult>(r#"{"rewards":[0.0]}"#).is_err());
    }

    #[test]
    fn native_objectives() {
        let result = native_vector(
            vec![0.25, 0.75],
            &[vec![-2.0, -1.0], vec![0.25, 0.75]],
            &["likelihood", "quality"],
        )
        .unwrap();
        let observation = result.observation().unwrap();
        assert_eq!(result.rewards, [0.25, 0.75]);
        assert_eq!(observation.control().mean, 0.5);
        assert_eq!(observation.estimates()[0].mean, -1.5);
        assert_eq!(observation.estimates()[1].mean, 0.5);
        assert!(native_vector(vec![0.0], &[vec![0.0], Vec::new()], &["a", "b"]).is_err());
    }
}
