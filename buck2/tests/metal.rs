use ennx::experimental::{
    AcquisitionKind, ComputeDevice, DenseLeaf, DenseTerm, ForwardProgram, ParamBuffer,
    KdaControlRequest, KdaForwardRequest, KdaMoeLayerRequest, KdaMoeMetalArena,
    KdaMoeMetalExecutor, KdaMoeMetalKdaVectors, KdaMoeMetalModel, KdaMoeMetalWeights,
    KdaPackedLinear, KdaTensorLayout, ResidentBoState, SearchConfig, PackedLeaf, PackedSearch,
};
use ennx::{
    compute_posterior_internals, ENNParams, EpistemicNearestNeighbors, IndexDriver, PosteriorFlags,
};
use ndarray::Array2;

fn leaves() -> Vec<PackedLeaf> {
    vec![
        PackedLeaf::new(0, 257, 4, 0.25, 1.0, 0.75).unwrap(),
        PackedLeaf::new(257, 263, 8, 0.5, 0.5, 1.0).unwrap(),
    ]
}

fn base() -> Vec<u8> {
    let row_bytes = 257usize.div_ceil(2) + 263;
    (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

fn ask(device: ComputeDevice) -> Result<(usize, f32, Vec<u8>), String> {
    let base = base();
    let mut search = PackedSearch::new(&base, 0.25, leaves(), 4, device)?;
    let warm = search.ask(
        &[17],
        SearchConfig {
            neighbors: 1,
            length: 1.0,
            ..SearchConfig::default()
        },
    )?;
    search.tell(warm, 0.75, true)?;
    let trial = search.ask(
        &[19, 23, 29, 31],
        SearchConfig {
            neighbors: 2,
            length: 0.65,
            beta: 1.3,
            acquisition: AcquisitionKind::Ucb,
            seed: 41,
            ..SearchConfig::default()
        },
    )?;
    Ok((trial.index, trial.score, search.row(trial)?))
}

#[test]
fn metal_matches_cpu() {
    let cpu = ask(ComputeDevice::Cpu).unwrap();
    let metal = ask(ComputeDevice::Metal).unwrap();
    assert_eq!(metal.0, cpu.0);
    assert!((metal.1 - cpu.1).abs() <= 1.0e-5);
    assert_eq!(metal.2, cpu.2);
}

#[test]
fn agx_matches_cpu() {
    let cpu = ask(ComputeDevice::Cpu).unwrap();
    let agx = match ask(ComputeDevice::Agx) {
        Ok(result) => result,
        Err(error) if error.contains("binary archive contains no items eligible") => {
            eprintln!("AGX archive serialization is unavailable: {error}");
            return;
        }
        Err(error) => panic!("AGX weight search failed: {error}"),
    };
    assert_eq!(agx.0, cpu.0);
    assert!((agx.1 - cpu.1).abs() <= 1.0e-5);
    assert_eq!(agx.2, cpu.2);
}

#[test]
fn knn_matches_exact() {
    let rows = Array2::from_shape_fn((137, 7), |(i, j)| {
        ((i * 37 + j * 19 + 5) % 509) as f64 / 509.0
    });
    let values = Array2::from_shape_fn((137, 3), |(i, j)| {
        ((i * 13 + j * 23 + 7) % 521) as f64 / 97.0
    });
    let queries = Array2::from_shape_fn((29, 7), |(i, j)| {
        ((i * 31 + j * 11 + 3) % 503) as f64 / 503.0
    });
    let flags = PosteriorFlags::new().with_tie_break_neighbors(false);
    let exact_model = EpistemicNearestNeighbors::new(
        rows.clone(),
        values.clone(),
        None,
        false,
        IndexDriver::Exact,
    )
    .unwrap();

    for neighbors in [1, 3, 8, 10, 16, 17, 31, 64] {
        let params = ENNParams::new(neighbors, 0.7, 0.13).unwrap();
        let expected =
            compute_posterior_internals(&exact_model, &queries.view(), &params, &flags).unwrap();
        for driver in [IndexDriver::Metal, IndexDriver::Agx] {
            let model =
                EpistemicNearestNeighbors::new(rows.clone(), values.clone(), None, false, driver)
                    .unwrap();
            let actual =
                compute_posterior_internals(&model, &queries.view(), &params, &flags).unwrap();
            assert_eq!(actual.idx, expected.idx, "{driver:?}, k={neighbors}");
            assert!(actual
                .mu
                .iter()
                .zip(expected.mu.iter())
                .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-5));
            assert!(actual
                .se
                .iter()
                .zip(expected.se.iter())
                .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-5));
        }
    }
}

#[test]
fn bf16_pytree_matches_cpu() {
    let base = [0.5f32, -1.0, 2.0, 0.25, 4.0, -2.0, 0.75, -0.125]
        .map(|value| (value.to_bits() >> 16) as u16);
    let leaves = vec![
        DenseLeaf::new(11, 0, 4, 0.5).unwrap(),
        DenseLeaf::new(29, 4, 4, 1.25).unwrap(),
    ];
    let terms = [
        DenseTerm::new(0x1234_5678_9abc_def0, 0.01).unwrap(),
        DenseTerm::new(91, -0.0025).unwrap(),
    ];
    let mut cpu = ParamBuffer::new(base.to_vec(), leaves.clone(), ComputeDevice::Cpu).unwrap();
    assert_eq!(cpu.candidate().unwrap(), base);
    cpu.materialize(&terms).unwrap();
    for device in [ComputeDevice::Metal, ComputeDevice::Agx] {
        let mut tree = ParamBuffer::new(base.to_vec(), leaves.clone(), device).unwrap();
        assert_eq!(tree.candidate().unwrap(), base);
        tree.materialize(&terms).unwrap();
        assert_eq!(tree.candidate().unwrap(), cpu.candidate().unwrap());
    }
}

#[test]
fn bf16_pytree_preserves_sub_ulp_directions() {
    let base = [1.0f32, -2.0, 4.0, -8.0].map(|value| (value.to_bits() >> 16) as u16);
    let leaves = vec![DenseLeaf::new(11, 0, base.len(), 1.0e-6).unwrap()];
    let terms = [DenseTerm::new(17, 1.0e-6).unwrap()];
    let mut cpu = ParamBuffer::new(base.to_vec(), leaves.clone(), ComputeDevice::Cpu).unwrap();
    cpu.materialize(&terms).unwrap();
    assert!(cpu
        .candidate()
        .unwrap()
        .iter()
        .zip(base)
        .all(|(candidate, base)| *candidate != base));
    for device in [ComputeDevice::Metal, ComputeDevice::Agx] {
        let mut tree = ParamBuffer::new(base.to_vec(), leaves.clone(), device).unwrap();
        tree.materialize(&terms).unwrap();
        assert_eq!(tree.candidate().unwrap(), cpu.candidate().unwrap());
    }
}

fn linear(input_width: usize, output_width: usize, bits: u8) -> KdaPackedLinear {
    KdaPackedLinear {
        byte_offset: 0,
        scale_offset: 0,
        bias_offset: 0,
        input_width,
        output_width,
        bits,
        group_size: 1,
        element_offset: 0,
        perturb_whole: 1,
        perturb_threshold: 0,
    }
}

fn kda_layer() -> (KdaMoeLayerRequest, KdaControlRequest) {
    let tensor = KdaTensorLayout::new(1, 16_384, 1, 4, 4).unwrap();
    let qkv = linear(4, 12, 8);
    let control = linear(4, 3, 8);
    let output = linear(4, 4, 8);
    let expert_gate = linear(4, 4, 8);
    let expert_up = linear(4, 4, 8);
    let expert_down = linear(4, 4, 8);
    (
        KdaMoeLayerRequest {
            kda: KdaForwardRequest {
                tensor,
                qkv,
                control,
                output,
                seed: 0,
            },
            attention_norm: linear(4, 1, 8),
            moe_norm: linear(4, 1, 8),
            router: linear(4, 2, 8),
            expert_gate: vec![expert_gate; 2],
            expert_up: vec![expert_up; 2],
            expert_down: vec![expert_down; 2],
            top_k: 1,
            residual_scale: 1.0,
            rms_epsilon: 1.0e-6,
        },
        KdaControlRequest {
            qkv_conv: KdaPackedLinear {
                group_size: 4,
                ..linear(4, 12, 8)
            },
            control,
            forget: linear(1, 4, 8),
            output_gate: linear(1, 4, 8),
            decay: linear(1, 1, 8),
            time_bias: linear(4, 1, 8),
            output_norm: linear(4, 1, 8),
            output,
            gate_rank: 1,
        },
    )
}

#[test]
fn resident_kda_binds_the_search_row() {
    let (request, control) = kda_layer();
    let packed = [0_u8; 1];
    let scales = [1.0_f32];
    let biases = [0.0_f32];
    let vectors = KdaMoeMetalKdaVectors {
        decay: &[1.0],
        time_bias: &[0.0; 4],
        output_norm: &[1.0; 4],
    };
    let mut executor = KdaMoeMetalExecutor::new(
        &request,
        control,
        vectors,
        ComputeDevice::Metal,
        KdaMoeMetalWeights {
            packed: &packed,
            scales: &scales,
            biases: &biases,
        },
    )
    .unwrap();
    let mut state = ResidentBoState::new(
        &[0],
        0.0,
        vec![PackedLeaf::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()],
        2,
        ComputeDevice::Metal,
        ForwardProgram::kda().unwrap(),
    )
    .unwrap();
    let round = state
        .ask(
            &[17],
            SearchConfig {
                neighbors: 1,
                ..SearchConfig::default()
            },
        )
        .unwrap();
    state.bind_pending_metal_row(&round, &mut executor).unwrap();
}

fn new_kda_executor(
    request: &KdaMoeLayerRequest,
    control: KdaControlRequest,
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    decay: &[f32],
    time_bias: &[f32],
    output_norm: &[f32],
) -> KdaMoeMetalExecutor {
    KdaMoeMetalExecutor::new(
        request,
        control,
        KdaMoeMetalKdaVectors {
            decay,
            time_bias,
            output_norm,
        },
        ComputeDevice::Metal,
        KdaMoeMetalWeights {
            packed,
            scales,
            biases,
        },
    )
    .unwrap()
}

fn run_kda(executor: &mut KdaMoeMetalExecutor) -> Vec<u16> {
    let hidden = vec![0x3c00_u16; executor.memory().hidden_elements];
    executor.upload_hidden(&hidden).unwrap();
    executor
        .attention_rms_norm(&[0x3c00; 4], 1, 16_384, 1.0e-6)
        .unwrap();
    executor.kda(17).unwrap();
    unsafe {
        std::slice::from_raw_parts(
            executor.hidden.contents().cast::<u16>(),
            executor.memory().hidden_elements,
        )
        .to_vec()
    }
}

#[test]
fn materialized_row_matches_seeded_forward() {
    let (request, control) = kda_layer();
    let packed = vec![0_u8; 48];
    let scales = vec![1.0_f32; 48];
    let biases = vec![0.0_f32; 48];
    let decay = [1.0_f32];
    let time_bias = [0.0_f32; 4];
    let output_norm = [1.0_f32; 4];

    let mut seeded = new_kda_executor(
        &request,
        control,
        &packed,
        &scales,
        &biases,
        &decay,
        &time_bias,
        &output_norm,
    );
    let seeded_hidden = run_kda(&mut seeded);

    let mut state = ResidentBoState::new(
        &packed,
        0.0,
        vec![PackedLeaf::new(0, 48, 8, 1.0, 1.0, 1.0).unwrap()],
        2,
        ComputeDevice::Metal,
        ForwardProgram::kda().unwrap(),
    )
    .unwrap();
    let round = state
        .ask(
            &[17],
            SearchConfig {
                neighbors: 1,
                length: 1.0,
                ..SearchConfig::default()
            },
        )
        .unwrap();
    let mut materialized = new_kda_executor(
        &request,
        control,
        &packed,
        &scales,
        &biases,
        &decay,
        &time_bias,
        &output_norm,
    );
    state
        .bind_pending_metal_row(&round, &mut materialized)
        .unwrap();
    let materialized_hidden = run_kda(&mut materialized);
    assert_eq!(materialized_hidden, seeded_hidden);
}

#[test]
fn kda_decode_advances_persistent_state() {
    let (request, control) = kda_layer();
    let packed = vec![0_u8; 48];
    let scales = vec![1.0_f32; 48];
    let biases = vec![0.0_f32; 48];
    let mut executor = new_kda_executor(
        &request,
        control,
        &packed,
        &scales,
        &biases,
        &[1.0],
        &[0.0; 4],
        &[1.0; 4],
    );
    executor.reset_recurrence_state();

    // Four value columns use the same scalar recurrence on key row zero:
    // s' = s + 0.5 * (2 - s), with q = k = [1, 0, 0, 0].
    let vector = [0x3c00, 0, 0, 0]; // IEEE f16 [1, 0, 0, 0]
    executor
        .upload_kda_decode_inputs(&vector, &vector, &[0x4000; 4], &[0.0; 4], &[0.5])
        .unwrap();
    executor.kda_decode_step().unwrap();
    assert_eq!(
        executor.recurrence_state(),
        [vec![1.0; 4], vec![0.0; 12]].concat()
    );
    assert!(executor
        .kda_decode_output()
        .iter()
        .all(|&value| value == 0x3800)); // IEEE f16 0.5 after the 1/sqrt(key_width) query scale

    executor.kda_decode_step().unwrap();
    assert_eq!(
        executor.recurrence_state(),
        [vec![1.5; 4], vec![0.0; 12]].concat()
    );
    assert!(executor
        .kda_decode_output()
        .iter()
        .all(|&value| value == 0x3a00)); // IEEE f16 0.75 after the 1/sqrt(key_width) query scale
}

#[test]
fn single_token_layer_decode_stays_resident() {
    let (mut request, control) = kda_layer();
    request.kda.tensor.sequence_length = 1;
    let packed = vec![0_u8; 48];
    let scales = vec![1.0_f32; 48];
    let biases = vec![0.0_f32; 48];
    let mut executor = new_kda_executor(
        &request,
        control,
        &packed,
        &scales,
        &biases,
        &[1.0],
        &[0.0; 4],
        &[1.0; 4],
    );
    executor.upload_hidden(&[0x3c00; 4]).unwrap();
    executor.upload_norms(&[0x3c00; 4], &[0x3c00; 4]).unwrap();
    executor.reset_decode_state();
    executor.prepare_candidate(17).unwrap();
    executor.decode_layer(17).unwrap();

    let hidden = unsafe {
        std::slice::from_raw_parts(
            executor.hidden.contents().cast::<u16>(),
            executor.memory().hidden_elements,
        )
    };
    assert!(hidden.iter().all(|value| value & 0x7c00 != 0x7c00));
    assert!(executor
        .recurrence_state()
        .iter()
        .all(|value| value.is_finite()));

    let mut model = KdaMoeMetalModel::new(vec![executor]).unwrap();
    model
        .attach_causal_head(linear(4, 8, 8), linear(4, 1, 8), 1.0)
        .unwrap();
    model.reset_decode_state();
    model.prepare_candidate(17).unwrap();
    assert!(model.decode_token(0, 17).unwrap() < 8);
}

fn production_linear(input_width: usize, output_width: usize, bits: u8) -> KdaPackedLinear {
    KdaPackedLinear {
        byte_offset: 0,
        scale_offset: 0,
        bias_offset: 0,
        input_width,
        output_width,
        bits,
        group_size: 64,
        element_offset: 0,
        perturb_whole: 0,
        perturb_threshold: 0,
    }
}

#[test]
#[ignore = "production-shape Metal microbenchmark"]
fn production_decode_layer_benchmark() {
    let hidden = 1_024;
    let expert_width = 512;
    let heads = 8;
    let head_width = 128;
    let experts = 32;
    let mut byte_offset = 0;
    let mut scale_offset = 0;
    let mut element_offset = 0;
    let mut allocate = |input_width, output_width, bits, group_size| {
        let descriptor = KdaPackedLinear {
            byte_offset,
            scale_offset,
            bias_offset: scale_offset,
            input_width,
            output_width,
            bits,
            group_size,
            element_offset,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        byte_offset += descriptor.packed_bytes();
        scale_offset += descriptor.groups_per_row() * output_width;
        element_offset += input_width * output_width;
        descriptor
    };
    let qkv = allocate(hidden, 3 * heads * head_width, 4, 64);
    let qkv_conv = allocate(4, 3 * heads * head_width, 8, 4);
    let control_projection = allocate(hidden, 2 * 128 + heads, 4, 64);
    let forget = allocate(128, hidden, 4, 64);
    let output_gate = allocate(128, hidden, 4, 64);
    let output = allocate(hidden, hidden, 4, 64);
    let router = allocate(hidden, experts, 8, 64);
    let expert_gate = (0..experts)
        .map(|_| allocate(hidden, expert_width, 4, 64))
        .collect::<Vec<_>>();
    let expert_up = (0..experts)
        .map(|_| allocate(hidden, expert_width, 4, 64))
        .collect::<Vec<_>>();
    let expert_down = (0..experts)
        .map(|_| allocate(expert_width, hidden, 8, 64))
        .collect::<Vec<_>>();
    drop(allocate);
    let request = KdaMoeLayerRequest {
        kda: KdaForwardRequest {
            tensor: KdaTensorLayout::new(1, 1, heads, head_width, head_width).unwrap(),
            qkv,
            control: control_projection,
            output,
            seed: 0,
        },
        attention_norm: production_linear(hidden, 1, 8),
        moe_norm: production_linear(hidden, 1, 8),
        router,
        expert_gate,
        expert_up,
        expert_down,
        top_k: 8,
        residual_scale: 0.22,
        rms_epsilon: 1.0e-6,
    };
    let control = KdaControlRequest {
        qkv_conv,
        control: control_projection,
        forget,
        output_gate,
        decay: production_linear(heads, 1, 8),
        time_bias: production_linear(hidden, 1, 8),
        output_norm: production_linear(head_width, 1, 8),
        output,
        gate_rank: 128,
    };
    let packed = vec![0_u8; byte_offset];
    let scales = vec![0.001_f32; scale_offset];
    let biases = vec![0.0_f32; scale_offset];
    let mut executor = new_kda_executor(
        &request,
        control,
        &packed,
        &scales,
        &biases,
        &[0.0; 8],
        &[0.0; 1_024],
        &[1.0; 128],
    );
    let resident = executor.packed.to_owned();
    executor
        .bind_resident_row(&resident, 0, packed.len())
        .unwrap();
    executor.upload_hidden(&[0x3c00; 1_024]).unwrap();
    executor
        .upload_norms(&[0x3c00; 1_024], &[0x3c00; 1_024])
        .unwrap();
    executor.reset_decode_state();
    executor.decode_layer(0).unwrap();

    let iterations = 5;
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        executor.decode_layer(0).unwrap();
    }
    let elapsed = started.elapsed();
    let layer_ms = elapsed.as_secs_f64() * 1_000.0 / f64::from(iterations);
    println!(
        "production KDA-MoE layer: {layer_ms:.3} ms; projected 24-layer rate: {:.3} tok/s",
        1_000.0 / (24.0 * layer_ms)
    );

    executor
        .moe_rms_norm(&[0x3c00; 1_024], 1, 1, 1.0e-6)
        .unwrap();
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        executor.moe(0).unwrap();
    }
    let moe_ms = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations);

    executor.reset_recurrence_state();
    executor
        .upload_kda_decode_inputs(
            &[0; 1_024],
            &[0; 1_024],
            &[0; 1_024],
            &[0.0; 1_024],
            &[0.5; 8],
        )
        .unwrap();
    let recurrence_iterations = 50;
    let started = std::time::Instant::now();
    for _ in 0..recurrence_iterations {
        executor.kda_decode_step().unwrap();
    }
    let recurrence_ms =
        started.elapsed().as_secs_f64() * 1_000.0 / f64::from(recurrence_iterations);
    println!(
        "kernel families: selected-expert MoE {moe_ms:.3} ms; KDA recurrence {recurrence_ms:.3} ms; other projections {:.3} ms",
        (layer_ms - moe_ms).max(0.0)
    );

    let arena = KdaMoeMetalArena::new(KdaMoeMetalWeights {
        packed: &packed,
        scales: &scales,
        biases: &biases,
    })
    .unwrap();
    let mut layers = Vec::with_capacity(24);
    for _ in 0..24 {
        let mut layer = KdaMoeMetalExecutor::new_with_arena(
            &request,
            control,
            KdaMoeMetalKdaVectors {
                decay: &[0.0; 8],
                time_bias: &[0.0; 1_024],
                output_norm: &[1.0; 128],
            },
            ComputeDevice::Metal,
            &arena,
        )
        .unwrap();
        layer
            .upload_norms(&[0x3c00; 1_024], &[0x3c00; 1_024])
            .unwrap();
        layers.push(layer);
    }
    let mut model = KdaMoeMetalModel::new(layers).unwrap();
    let resident = arena.packed_buffer();
    model
        .bind_resident_row(&resident, 0, arena.packed_bytes())
        .unwrap();
    model.upload_hidden(&[0x3c00; 1_024]).unwrap();
    model.reset_decode_state();
    model.prepare_candidate(0).unwrap();
    model.decode(0).unwrap();
    let model_iterations = 5;
    let started = std::time::Instant::now();
    for _ in 0..model_iterations {
        model.decode(0).unwrap();
    }
    let model_ms = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(model_iterations);
    println!(
        "resident 24-layer command buffer: {model_ms:.3} ms/token ({:.3} tok/s)",
        1_000.0 / model_ms
    );
}
