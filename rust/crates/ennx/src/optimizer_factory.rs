//! Factory functions for creating optimizers with preset configs.

use ndarray::Array2;
use rand::RngCore;

use crate::config::{
    lhd_only, turbo_enn, turbo_zero, ConfigOverrides, InitStrategy, SurrogateConfig,
};
use crate::error::ENNError;
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;

/// Create an optimizer for TuRBO-ENN.
pub fn enn_optimizer(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    enn_overrides(bounds, k, num_init, rng, None)
}

/// Create an experimental multi-trust-region optimizer for TuRBO-ENN.
pub fn enn_tr(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    region_overrides(bounds, k, num_init, num_regions, rng, None)
}

/// Create TuRBO-ENN with optional config overrides (for future Python pass-through).
pub fn enn_overrides(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_enn();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    Optimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create experimental multi-trust-region TuRBO-ENN with optional config overrides.
pub fn region_overrides(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_enn();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::experimental(bounds.nrows(), num_regions, num_init, rng)?;
    Optimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for TuRBO-ZERO.
pub fn create_optimizer(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    create_overrides(bounds, num_init, rng, None)
}

/// Create TuRBO-ZERO with optional config overrides.
pub fn create_overrides(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_zero();
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    Optimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for LHD-only.
pub fn create_lhd(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    lhd_overrides(bounds, num_init, rng, None)
}

/// Create LHD-only with optional config overrides.
pub fn lhd_overrides(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = lhd_only();
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::init(InitStrategy::LHD, num_init);
    Optimizer::new_strategy(bounds, config, strategy, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn create_smoke() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(101);

        let mut enn = enn_optimizer(bounds.clone(), 3, 2, &mut rng).unwrap();
        let _ = enn.ask(1, &mut rng).unwrap();

        let mut exp = enn_tr(bounds.clone(), 3, 2, 3, &mut rng).unwrap();
        let _ = exp.ask(2, &mut rng).unwrap();

        let mut zero = create_optimizer(bounds.clone(), 2, &mut rng).unwrap();
        let _ = zero.ask(1, &mut rng).unwrap();

        let mut lhd = create_lhd(bounds, 2, &mut rng).unwrap();
        let _ = lhd.ask(1, &mut rng).unwrap();
    }
}
