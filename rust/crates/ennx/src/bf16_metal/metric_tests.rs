use super::*;
use crate::params::ENNParams;

fn fixture(n: usize) -> (FamilyHistory, ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let blocks = (0..FAMILIES)
        .map(|i| ParamBlock::new(i as u64, i, 1, 1.0, 1.0).unwrap())
        .collect::<Vec<_>>();
    let mut family = FamilyHistory::new((0..FAMILIES).collect(), &blocks).unwrap();
    let x = (0..n)
        .map(|i| {
            std::array::from_fn::<_, FAMILIES, _>(|g| ((i * (g + 3) * 17 % 97) as f32 / 97.0).sin())
        })
        .collect::<Vec<_>>();
    for i in 0..n {
        for j in 0..n {
            family.components[i * MAX_HISTORY + j] =
                std::array::from_fn(|g| (x[i][g] - x[j][g]).powi(2));
        }
    }
    let y = ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(x[i][0]));
    let variance = ndarray::Array2::from_shape_fn((n, 1), |(i, _)| i as f64 * 0.0001);
    (family, y, variance)
}

#[test]
fn metric_agreement() -> Result<(), String> {
    autoreleasepool(|| {
        let mut gpu = MetricGpu::new(Runtime::shared()?)?;
        for n in [2, 17, MAX_HISTORY] {
            let (mut family, y, variance) = fixture(n);
            family.weights = [4.0, 1.0, 1.0, 0.25];
            let candidates = family.candidates();
            for local in [None, Some(1), Some(n + 1)] {
                for samples in [1, n] {
                    let params = ENNParams::new((n + 1) as i32, 0.01, 0.003).unwrap();
                    let got = gpu.score(
                        &family,
                        &candidates,
                        &y.view(),
                        &variance.view(),
                        params,
                        samples,
                        173,
                        local,
                    )?;
                    let expected = family.reference(
                        &candidates,
                        &y.view(),
                        &variance.view(),
                        params,
                        samples,
                        173,
                        local,
                    )?;
                    for (&a, &b) in got.iter().zip(&expected) {
                        assert!(
                            (a - b).abs() < 3e-5 * (1.0 + b.abs()),
                            "n={n} local={local:?}: {a} != {b}"
                        );
                    }
                    let mut cpu = fixture(n).0;
                    cpu.weights = family.weights;
                    let mut device = fixture(n).0;
                    device.weights = family.weights;
                    cpu.select(&candidates, &expected, samples)?;
                    device.select(&candidates, &got, samples)?;
                    assert_eq!(cpu.weights, device.weights);
                }
            }
        }
        Ok(())
    })
}

#[test]
fn metric_duplicates() -> Result<(), String> {
    autoreleasepool(|| {
        let mut gpu = MetricGpu::new(Runtime::shared()?)?;
        let (mut family, _, _) = fixture(17);
        family.components.fill([0.0; FAMILIES]);
        let candidates = family.candidates();
        for value in [0.0, 1.0] {
            let y = ndarray::Array2::from_elem((17, 1), value);
            let variance = ndarray::Array2::zeros((17, 1));
            let params = ENNParams::new(4, 0.0, 0.0).unwrap();
            let got = gpu.score(
                &family,
                &candidates,
                &y.view(),
                &variance.view(),
                params,
                17,
                914,
                Some(2),
            )?;
            let expected = family.reference(
                &candidates,
                &y.view(),
                &variance.view(),
                params,
                17,
                914,
                Some(2),
            )?;
            for (&a, &b) in got.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-5 * (1.0 + b.abs()));
            }
        }
        let y = ndarray::Array2::from_shape_fn((17, 1), |(i, _)| i as f64 / 17.0);
        let variance = ndarray::Array2::from_shape_fn((17, 1), |(i, _)| i as f64 / 17000.0);
        let params = ENNParams::new(4, 0.1, 0.001).unwrap();
        let got = gpu.score(
            &family,
            &candidates,
            &y.view(),
            &variance.view(),
            params,
            17,
            719,
            None,
        )?;
        let expected = family.reference(
            &candidates,
            &y.view(),
            &variance.view(),
            params,
            17,
            719,
            None,
        )?;
        for (&a, &b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 3e-5 * (1.0 + b.abs()));
        }
        let y = ndarray::Array2::zeros((17, 1));
        assert!(
            gpu.score(
                &family,
                &candidates,
                &y.view(),
                &y.view(),
                ENNParams::new(4, 0.1, 0.01).unwrap(),
                0,
                914,
                None
            )
            .is_err()
        );
        Ok(())
    })
}

#[test]
#[ignore = "paired metric benchmark; requires Low Power Mode off"]
fn metric_timing() -> Result<(), String> {
    autoreleasepool(|| {
        crate::apple_gpu::require_power()?;
        let mut gpu = MetricGpu::new(Runtime::shared()?)?;
        let params = ENNParams::new(8, 0.1, 0.001).unwrap();
        for n in [16, 64, MAX_HISTORY] {
            let (family, y, variance) = fixture(n);
            let candidates = family.candidates();
            for local in [None, Some(8)] {
                gpu.score(
                    &family,
                    &candidates,
                    &y.view(),
                    &variance.view(),
                    params,
                    8,
                    179,
                    local,
                )?;
                let mut cpu = Vec::new();
                let mut device = Vec::new();
                for _ in 0..9 {
                    let start = Instant::now();
                    std::hint::black_box(family.reference(
                        &candidates,
                        &y.view(),
                        &variance.view(),
                        params,
                        8,
                        179,
                        local,
                    )?);
                    cpu.push(start.elapsed().as_secs_f64());
                    let start = Instant::now();
                    std::hint::black_box(gpu.score(
                        &family,
                        &candidates,
                        &y.view(),
                        &variance.view(),
                        params,
                        8,
                        179,
                        local,
                    )?);
                    device.push(start.elapsed().as_secs_f64());
                }
                cpu.sort_by(f64::total_cmp);
                device.sort_by(f64::total_cmp);
                eprintln!(
                    "metric fit n={n} local={local:?}: CPU={:.3} ms Metal={:.3} ms",
                    cpu[4] * 1000.0,
                    device[4] * 1000.0
                );
            }
        }
        Ok(())
    })
}
