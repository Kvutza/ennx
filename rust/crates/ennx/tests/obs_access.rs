use ennx::config::turbo_zero;
use ennx::optimizer::obs_access::{stack_rows, ObsAccess};
use ennx::optimizer::Optimizer;
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn observation_access() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let mut optimizer = Optimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let access: ObsAccess<'_> = optimizer.obs_access();
    assert!(access.observations_empty());

    optimizer
        .add_observations(&array![[0.1, 0.2]].view(), &array![[0.5, 1.5]].view())
        .unwrap();
    let access = optimizer.obs_access();
    assert!(!access.observations_empty());
    assert_eq!(access.x_row(0).unwrap(), array![0.1, 0.2]);
    assert_eq!(access.y_row(0).unwrap(), array![0.5, 1.5]);

    let rows = stack_rows(&[array![1.0, 2.0]]);
    assert_eq!(rows.shape(), &[1, 2]);
}
