use ennx::experimental::{
    apply_dense, dense_linear, ComputeBackend, DenseLeaf, DenseLinear, DenseTerm, DenseView,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    compare_materialization()?;
    compare_linear()?;
    println!("CUDA_DENSE ok=true materialization=true linear=true resident=true target=sm_75");
    Ok(())
}

fn compare_materialization() -> Result<(), String> {
    let first = 70_003;
    let second = 521;
    let base = (0..first + second)
        .map(|index| ((index * 17 + 5) % 257) as f32 / 31.0 - 4.0)
        .collect::<Vec<_>>();
    let leaves = [
        DenseLeaf::new(11, 0, first, 0.015625)?,
        DenseLeaf::new(29, first, second, 0.03125)?,
    ];
    let terms = [
        DenseTerm::new(0x1234_5678_9abc_def0, 0.5)?,
        DenseTerm::new(91, -0.125)?,
        DenseTerm::new(7, 0.03125)?,
    ];
    let cpu = apply_dense(&base, &leaves, &terms, ComputeBackend::Cpu)?;
    let cuda = apply_dense(&base, &leaves, &terms, ComputeBackend::Cuda)?;
    compare(&cpu.values, &cuda.values, 2.0e-6, "dense materialization")
}

fn compare_linear() -> Result<(), String> {
    let columns = 513;
    let rows = 7;
    let input = (0..columns)
        .map(|index| ((index * 13 + 3) % 101) as f32 / 50.0 - 1.0)
        .collect::<Vec<_>>();
    let weight = (0..rows * columns)
        .map(|index| ((index * 19 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect::<Vec<_>>();
    let bias = (0..rows)
        .map(|index| index as f32 * 0.03125 - 0.0625)
        .collect::<Vec<_>>();
    let weight_view = DenseView::new(41, 1_000, 0.0078125)?;
    let bias_view = DenseView::new(43, 9_000, 0.00390625)?;
    let first_terms = [
        DenseTerm::new(17, 0.5)?,
        DenseTerm::new(0xfeed_face_cafe_beef, -0.25)?,
    ];
    let second_terms = [
        DenseTerm::new(23, -0.75)?,
        DenseTerm::new(29, 0.125)?,
        DenseTerm::new(31, 0.0625)?,
    ];

    let cpu = dense_linear(
        &input,
        &weight,
        Some(&bias),
        weight_view,
        Some(bias_view),
        &first_terms,
        ComputeBackend::Cpu,
    )?;
    let cuda = dense_linear(
        &input,
        &weight,
        Some(&bias),
        weight_view,
        Some(bias_view),
        &first_terms,
        ComputeBackend::Cuda,
    )?;
    compare(&cpu, &cuda, 2.0e-5, "dense linear")?;

    let mut cpu = DenseLinear::new(
        weight.clone(),
        columns,
        Some(bias.clone()),
        weight_view,
        Some(bias_view),
        ComputeBackend::Cpu,
    )?;
    let mut cuda = DenseLinear::new(
        weight,
        columns,
        Some(bias),
        weight_view,
        Some(bias_view),
        ComputeBackend::Cuda,
    )?;
    compare(
        &cpu.eval(&input, &first_terms)?,
        &cuda.eval(&input, &first_terms)?,
        2.0e-5,
        "resident dense linear first evaluation",
    )?;
    compare(
        &cpu.eval(&input, &second_terms)?,
        &cuda.eval(&input, &second_terms)?,
        2.0e-5,
        "resident dense linear second evaluation",
    )
}

fn compare(left: &[f32], right: &[f32], relative: f32, name: &str) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!("{name} lengths differ"));
    }
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        let tolerance = relative * left.abs().max(1.0);
        if (left - right).abs() > tolerance {
            return Err(format!(
                "{name} mismatch at {index}: CPU={left} CUDA={right} tolerance={tolerance}"
            ));
        }
    }
    Ok(())
}
