use ennx::config::turbo_zero_config;
use ennx::optimizer::obs_access::{build_obs_array2, ObsAccess};
use ennx::optimizer::Optimizer;
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn observation_access() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let mut optimizer = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();
    let access: ObsAccess<'_> = optimizer.obs_access();
    assert!(access.observations_empty());

    optimizer
        .add_observations(&array![[0.1, 0.2]].view(), &array![[0.5, 1.5]].view())
        .unwrap();
    let access = optimizer.obs_access();
    assert!(!access.observations_empty());
    assert_eq!(access.obs_row_x(0).unwrap(), array![0.1, 0.2]);
    assert_eq!(access.obs_row_y(0).unwrap(), array![0.5, 1.5]);

    let rows = build_obs_array2(&[array![1.0, 2.0]]);
    assert_eq!(rows.shape(), &[1, 2]);
}
