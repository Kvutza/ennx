const STREAMS: usize = 4;
const SINKHORN_ITERATIONS: usize = 20;

type Matrix = [[f32; STREAMS]; STREAMS];

fn sigmoid(value: f32) -> f32 {
    let e = (-value.abs()).exp();
    if value >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

fn sinkhorn(raw: Matrix) -> Matrix {
    let maximum = raw
        .iter()
        .flatten()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut matrix = raw.map(|row| row.map(|value| (value - maximum).max(-80.0).exp()));
    for _ in 0..SINKHORN_ITERATIONS {
        for column in 0..STREAMS {
            let sum = matrix.iter().map(|row| row[column]).sum::<f32>();
            for row in &mut matrix {
                row[column] /= sum;
            }
        }
        for row in &mut matrix {
            let sum = row.iter().sum::<f32>();
            for value in row {
                *value /= sum;
            }
        }
    }
    matrix
}

fn transport(raw: Matrix, raw_lambda: f32) -> Matrix {
    let lambda = raw_lambda.clamp(0.0, 1.0);
    let projected = sinkhorn(raw);
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let identity = f32::from(row == column);
            (1.0 - lambda) * identity + lambda * projected[row][column]
        })
    })
}

fn input_coefficients(raw: [f32; STREAMS]) -> [f32; STREAMS] {
    let mut coefficients = raw.map(sigmoid);
    let total = coefficients.iter().sum::<f32>();
    for coefficient in &mut coefficients {
        *coefficient /= total;
    }
    coefficients
}

fn output_coefficients(raw: [f32; STREAMS]) -> [f32; STREAMS] {
    raw.map(|value| 2.0 * sigmoid(value))
}

fn mix<const WIDTH: usize>(
    streams: &[[f32; WIDTH]; STREAMS],
    coefficients: &[f32; STREAMS],
) -> [f32; WIDTH] {
    std::array::from_fn(|column| {
        (0..STREAMS)
            .map(|stream| coefficients[stream] * streams[stream][column])
            .sum()
    })
}

fn update<const WIDTH: usize>(
    streams: &[[f32; WIDTH]; STREAMS],
    branch: &[f32; WIDTH],
    residual: &Matrix,
    output: &[f32; STREAMS],
) -> [[f32; WIDTH]; STREAMS] {
    let mut next = [[0.0; WIDTH]; STREAMS];
    for destination in 0..STREAMS {
        for column in 0..WIDTH {
            let carried = (0..STREAMS)
                .map(|source| residual[destination][source] * streams[source][column])
                .sum::<f32>();
            next[destination][column] = carried + output[destination] * branch[column];
        }
    }
    next
}

fn multiply(left: &Matrix, right: &Matrix) -> Matrix {
    let mut product = [[0.0; STREAMS]; STREAMS];
    for row in 0..STREAMS {
        for column in 0..STREAMS {
            product[row][column] = (0..STREAMS)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    product
}

fn spectral_norm(matrix: &Matrix) -> f32 {
    let mut vector = [0.5, -0.25, 0.75, 1.0];
    for _ in 0..64 {
        let applied = std::array::from_fn::<_, STREAMS, _>(|row| {
            (0..STREAMS)
                .map(|column| matrix[row][column] * vector[column])
                .sum::<f32>()
        });
        let normal = std::array::from_fn::<_, STREAMS, _>(|column| {
            (0..STREAMS)
                .map(|row| matrix[row][column] * applied[row])
                .sum::<f32>()
        });
        let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
        vector = normal.map(|value| value / length);
    }
    let applied = std::array::from_fn::<_, STREAMS, _>(|row| {
        (0..STREAMS)
            .map(|column| matrix[row][column] * vector[column])
            .sum::<f32>()
    });
    applied
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn assert_transport(matrix: &Matrix, tolerance: f32) {
    for row in matrix {
        assert!(row.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!((row.iter().sum::<f32>() - 1.0).abs() <= tolerance);
    }
    for column in 0..STREAMS {
        let sum = matrix.iter().map(|row| row[column]).sum::<f32>();
        assert!((sum - 1.0).abs() <= tolerance, "column {column}: {sum}");
    }
}

#[test]
fn base_equivalence() {
    const WIDTH: usize = 8;
    let state = [0.5, -0.25, 1.0, 2.0, -3.0, 0.125, 4.0, -1.5];
    let branch = [1.0, 0.5, -0.5, 2.0, 1.5, -2.0, 0.25, 3.0];
    let streams = [state; STREAMS];
    let read = input_coefficients([-3.0f32.ln(); STREAMS]);
    let write = output_coefficients([0.0; STREAMS]);
    let residual = transport([[0.0; STREAMS]; STREAMS], 0.0);
    let mixed = mix(&streams, &read);
    let next = update(&streams, &branch, &residual, &write);

    for column in 0..WIDTH {
        assert!((mixed[column] - state[column]).abs() <= 1.0e-6);
        for stream in &next {
            assert!((stream[column] - state[column] - branch[column]).abs() <= 1.0e-6);
        }
    }
}

#[test]
fn manifold_composition() {
    let first = transport(
        std::array::from_fn(|row| {
            std::array::from_fn(|column| (row * STREAMS + column) as f32 * 0.17 - 0.9)
        }),
        1.0,
    );
    let second = transport(
        std::array::from_fn(|row| {
            std::array::from_fn(|column| ((row + 2 * column) % STREAMS) as f32 - 1.5)
        }),
        0.35,
    );
    let composite = multiply(&second, &first);

    assert_transport(&first, 2.0e-5);
    assert_transport(&second, 2.0e-5);
    assert_transport(&composite, 4.0e-5);
    assert!(spectral_norm(&first) <= 1.0 + 2.0e-5);
    assert!(spectral_norm(&second) <= 1.0 + 2.0e-5);
    assert!(spectral_norm(&composite) <= 1.0 + 4.0e-5);
}

#[test]
fn extreme_coefficients() {
    let raw = std::array::from_fn(|row| {
        std::array::from_fn(|column| if (row + column) % 2 == 0 { 80.0 } else { -80.0 })
    });
    let projected = transport(raw, 1.0);
    assert_transport(&projected, 2.0e-5);
    assert!(
        input_coefficients([-80.0, -1.0, 1.0, 80.0])
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    );
    assert!(
        output_coefficients([-80.0, -1.0, 1.0, 80.0])
            .iter()
            .all(|value| value.is_finite() && (0.0..=2.0).contains(value))
    );
}
