//! Four-group metric fitting; canonical components never change units after a fit.
use super::*;

pub(super) const FAMILIES: usize = 4;

pub(super) struct FamilyHistory {
    pub groups: Vec<usize>,
    pub original: Vec<ParamBlock>,
    pub components: Vec<[f32; FAMILIES]>,
    pub weights: [f32; FAMILIES],
}

impl FamilyHistory {
    pub fn new(groups: Vec<usize>, blocks: &[ParamBlock]) -> Result<Self, String> {
        if groups.len() != blocks.len()
            || groups.iter().any(|&g| g >= FAMILIES)
            || (0..FAMILIES).any(|g| !groups.contains(&g))
        {
            return Err("Learned shape requires all four families and one group per block".into());
        }
        Ok(Self {
            groups,
            original: blocks.to_vec(),
            components: vec![[0.0; FAMILIES]; MAX_HISTORY * MAX_HISTORY],
            weights: [1.0; FAMILIES],
        })
    }

    pub fn scales(&self) -> [f32; FAMILIES] {
        let raw = self.weights.map(|w| 1.0 / w.sqrt());
        // Each canonical block has unit expected metric energy. Keep total
        // expected canonical energy fixed, independently of the global radius.
        let rms = (self.groups.iter().map(|&g| raw[g] * raw[g]).sum::<f32>()
            / self.groups.len() as f32)
            .sqrt();
        raw.map(|s| s / rms)
    }

    pub fn aggregate(&self, weights: [f32; FAMILIES], n: usize) -> ndarray::Array2<f64> {
        ndarray::Array2::from_shape_fn((n, n), |(i, j)| {
            self.components[i * MAX_HISTORY + j]
                .iter()
                .zip(weights)
                .map(|(&d, w)| f64::from(d) * f64::from(w))
                .sum()
        })
    }

    pub fn fit(
        &mut self,
        y: &ndarray::ArrayView2<f64>,
        variance: &ndarray::ArrayView2<f64>,
        params: ENNParams,
        samples: usize,
        seed: u64,
        local_scale_neighbors: Option<usize>,
    ) -> Result<(), String> {
        // Conditional coordinate search. All shapes see the same held-out row
        // IDs. The current shape wins weak evidence; accepted moves are damped.
        let mut candidates = vec![self.weights];
        for group in 0..FAMILIES {
            for step in [-0.25f32, 0.25] {
                let mut logs = self.weights.map(f32::ln);
                logs[group] += step;
                let mean = logs.iter().sum::<f32>() / FAMILIES as f32;
                logs.iter_mut().for_each(|v| *v -= mean);
                if logs.iter().all(|v| v.abs() <= 4.0f32.ln()) {
                    candidates.push(logs.map(f32::exp));
                }
            }
        }
        let mut best = f64::NEG_INFINITY;
        let mut selected = self.weights;
        for weights in candidates {
            let raw_distances = self.aggregate(weights, y.nrows());
            let distances = match local_scale_neighbors {
                Some(k) => crate::fit::self_tuned_distances(&raw_distances.view(), k)
                    .map_err(|error| error.to_string())?,
                None => raw_distances,
            };
            let ll = crate::fit::distance_loglik(
                &distances.view(),
                y,
                Some(variance),
                &[params],
                samples,
                &mut StdRng::seed_from_u64(seed),
                None,
            )
            .map_err(|e| e.to_string())?[0];
            let penalty = weights
                .iter()
                .map(|w| f64::from(*w).ln().powi(2))
                .sum::<f64>()
                / FAMILIES as f64;
            let score = ll / samples.min(y.nrows()) as f64 - 0.1 * penalty;
            if score.is_finite() && score > best + 0.005 {
                best = score;
                selected = weights;
            }
        }
        if !best.is_finite() {
            return Err("No finite family-shape likelihood".into());
        }
        let old = self.weights.map(f32::ln);
        let target = selected.map(f32::ln);
        let mut blended = std::array::from_fn(|g| 0.5 * old[g] + 0.5 * target[g]);
        let mean = blended.iter().sum::<f32>() / FAMILIES as f32;
        blended.iter_mut().for_each(|value| *value -= mean);
        self.weights = blended.map(f32::exp);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> FamilyHistory {
        let blocks = (0..4)
            .map(|g| ParamBlock::new(g, g as usize, 1, 1.0, 1.0).unwrap())
            .collect::<Vec<_>>();
        FamilyHistory::new(vec![0, 1, 2, 3], &blocks).unwrap()
    }

    #[test]
    fn family_energy_and_sensitivity() {
        let mut h = fixture();
        h.weights = [4.0, 1.0, 1.0, 0.25];
        let scales = h.scales();
        assert!(scales[0] < scales[1] && scales[1] < scales[3]);
        assert!((scales.iter().map(|s| s * s).sum::<f32>() - 4.0).abs() < 1e-6);
        assert!(FamilyHistory::new(vec![0; 4], &h.original).is_err());
    }

    #[test]
    fn family_components_keep_units() {
        let mut h = fixture();
        h.components[1] = [1.0, 2.0, 3.0, 4.0];
        h.components[MAX_HISTORY] = h.components[1];
        let d = h.aggregate([4.0, 1.0, 1.0, 0.25], 2);
        assert_eq!(d[[0, 1]], 10.0);
        assert_eq!(d[[0, 1]], d[[1, 0]]);
        assert_eq!(h.components[1], [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn family_fit_learns_informative_axis() {
        use rand::Rng;
        let mut h = fixture();
        let n = 64;
        let mut rng = StdRng::seed_from_u64(914);
        let x: Vec<[f32; FAMILIES]> = (0..n)
            .map(|_| std::array::from_fn(|_| rng.gen_range(-1.0..1.0)))
            .collect();
        for i in 0..n {
            for j in 0..n {
                h.components[i * MAX_HISTORY + j] =
                    std::array::from_fn(|g| (x[i][g] - x[j][g]).powi(2));
            }
        }
        let y = ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(x[i][0]));
        let v = ndarray::Array2::zeros((n, 1));
        let params = ENNParams::new(4, 0.1, 0.001).unwrap();
        for _ in 0..24 {
            h.fit(&y.view(), &v.view(), params, n, 12, None).unwrap();
        }
        assert!(h.weights[0] > 1.0, "{:?}", h.weights);
        assert!(h.scales()[0] < h.scales()[1], "{:?}", h.weights);
        assert!((h.weights.iter().product::<f32>() - 1.0).abs() < 1e-5);
        assert!(h.weights.iter().all(|w| (0.25..=4.0).contains(w)));
        let previous = h.weights;
        h.fit(&y.view(), &v.view(), params, 0, 12, None)
            .unwrap_err();
        assert_eq!(previous, h.weights);
    }

    #[test]
    fn family_gpu_history_replay_and_reset() -> Result<(), String> {
        autoreleasepool(|| {
            let len = 65_537;
            let base = vec![0x3400u16; 4 * len];
            let blocks = (0..4)
                .map(|g| ParamBlock::new(g as u64, g * len, len, 0.25, 16.0 / len as f32))
                .collect::<Result<Vec<_>, _>>()?;
            let mut s = SearchState::new_fp16_implicit(
                &base,
                blocks,
                2,
                TRLengthConfig::new(0.01, 0.0001, 0.1),
                Perturbation::Rademacher,
            )?;
            s.enable_family_shape(vec![0, 1, 2, 3])?;
            let config = crate::config::ConfigOverrides {
                k_neighbors: Some(4),
                num_candidates: Some(2),
                num_samples: Some(4),
                distance_scaling: Some(crate::config::DistanceScaling::SelfTuning),
                local_scale_neighbors: Some(2),
                ..Default::default()
            }
            .resident_enn(42)?;
            s.configure_implicit_enn(config)?;
            s.observe_initial(0.0, 0.001)?;
            for step in 0..10 {
                if step == 5 {
                    s.family.as_mut().unwrap().weights = [4.0, 1.0, 1.0, 0.25];
                    s.apply_family_shape()?;
                }
                let initializing = s.history < 4;
                let root = 123 + step;
                if initializing {
                    s.begin_initial(root, step as usize % 4)?;
                } else {
                    s.begin_ask(1, 4, root, config.ask)?;
                }
                let proposal = s.finish_ask()?;
                let original = read::<u16>(&s.proposal, base.len());
                s.diagnostic_row(&proposal, root, config.ask, proposal.index)?;
                assert_eq!(original, read::<u16>(&s.proposal, base.len()));
                let f = s.family.as_ref().unwrap();
                for slot in 0..s.resident_history {
                    let history = read::<u16>(&s.history_rows[slot], base.len());
                    let row = s.identities[..s.history]
                        .iter()
                        .position(|&id| id == s.resident_identities[slot])
                        .unwrap();
                    for g in 0..4 {
                        let decode = |bits: u16| {
                            let exp = (bits >> 10) & 31;
                            let frac = f64::from(bits & 1023);
                            let v = if exp == 0 {
                                frac * 2f64.powi(-24)
                            } else {
                                (1024.0 + frac) * 2f64.powi(i32::from(exp) - 25)
                            };
                            if bits & 0x8000 == 0 { v } else { -v }
                        };
                        let exact = original[g * len..(g + 1) * len]
                            .iter()
                            .zip(&history[g * len..(g + 1) * len])
                            .map(|(&a, &b)| (decode(a) - decode(b)).powi(2))
                            .sum::<f64>()
                            * f64::from(f.original[g].weight);
                        let got = f64::from(proposal.family_distances.as_ref().unwrap()[row][g]);
                        assert!((got - exact).abs() < 1e-6 * (1.0 + exact), "{got} {exact}");
                    }
                }
                let reward = step as f32 * 0.01;
                if initializing {
                    s.tell_initial(&proposal, reward, 0.001)?;
                } else {
                    s.tell_model_aware(&proposal, reward, 0.001)?;
                }
                s.sync()?;
                let f = s.family.as_ref().unwrap();
                let d = f.aggregate(f.weights, s.history);
                for i in 0..s.history {
                    for j in 0..s.history {
                        assert!(
                            (s.pairwise_distances[i * MAX_HISTORY + j] - d[[i, j]] as f32).abs()
                                < 1e-6
                        );
                    }
                }
            }
            s.history = MAX_HISTORY;
            s.compact_implicit_history()?;
            assert_eq!(s.history, 1);
            assert_eq!(s.family.as_ref().unwrap().weights, [1.0; 4]);
            assert!(
                s.family
                    .as_ref()
                    .unwrap()
                    .components
                    .iter()
                    .flatten()
                    .all(|&x| x == 0.0)
            );
            assert!(s.blocks.iter().all(|b| b.scale == 0.25));
            Ok(())
        })
    }
}
