use crate::{CudaResult, cuda_error, read_prefix};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, simt::LaunchConfig2D};
use ennx_cuda_kernels::{MoeShape, RouteShape, RoutedTile, fbt_model};

pub(super) fn check() -> CudaResult<()> {
    const ROWS: usize = 400;
    const EXPERTS: usize = 625;
    const ASSIGNMENTS: usize = ROWS * 3;
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    // SAFETY: load the generated binding's matching embedded CUDA module.
    let module = unsafe { fbt_model::load(&context) }.map_err(cuda_error)?;
    let mut scores = vec![0_u16; ROWS * EXPERTS];
    for row in (0..ROWS).step_by(2) {
        for (expert, value) in [(624, 0x4000), (623, 0x3c00), (622, 0x3800)] {
            scores[row * EXPERTS + expert] = value;
        }
    }
    let scores = DeviceBuffer::from_host(&stream, &scores).map_err(cuda_error)?;
    let mut experts = DeviceBuffer::zeroed(&stream, ASSIGNMENTS).map_err(cuda_error)?;
    let mut weights = DeviceBuffer::<u16>::zeroed(&stream, ASSIGNMENTS).map_err(cuda_error)?;
    let mut margins = DeviceBuffer::<f32>::zeroed(&stream, ROWS).map_err(cuda_error)?;
    let shape = RouteShape {
        rows: ROWS as u32,
        experts: EXPERTS as u32,
        top_k: 3,
    };
    let launch = module
        .prepare_routetopk(LaunchConfig1D::new(ROWS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .routetopk(
            &stream,
            &launch,
            &scores,
            &mut experts,
            &mut weights,
            &mut margins,
            shape,
        )
        .map_err(cuda_error)?;
    let routes = read_prefix(&experts, &stream, ASSIGNMENTS)?;
    for (row, route) in routes.chunks_exact(3).enumerate() {
        let expected = if row % 2 == 0 {
            [624, 623, 622]
        } else {
            [0, 1, 2]
        };
        if route != expected {
            return Err(format!("625-expert route mismatch at row {row}: {route:?}"));
        }
    }
    let input = (0..ROWS * 512)
        .map(|index| 0x3000 + ((index / 512) % 16) as u16 * 64 + (index % 4) as u16)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &input).map_err(cuda_error)?;
    let mut counts = DeviceBuffer::zeroed(&stream, EXPERTS).map_err(cuda_error)?;
    let mut prefixes =
        DeviceBuffer::zeroed(&stream, EXPERTS * ASSIGNMENTS.div_ceil(1024)).map_err(cuda_error)?;
    let mut offsets = DeviceBuffer::zeroed(&stream, EXPERTS).map_err(cuda_error)?;
    let capacity = ASSIGNMENTS.div_ceil(16) + EXPERTS;
    let mut tiles = DeviceBuffer::<RoutedTile>::zeroed(&stream, capacity).map_err(cuda_error)?;
    let mut mapping = DeviceBuffer::zeroed(&stream, ASSIGNMENTS).map_err(cuda_error)?;
    let mut packed = DeviceBuffer::<u16>::zeroed(&stream, ASSIGNMENTS * 512).map_err(cuda_error)?;
    let launch = module
        .prepare_route_counts(LaunchConfig1D::new(EXPERTS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .route_counts(
            &stream,
            &launch,
            &experts,
            &mut counts,
            &mut prefixes,
            ASSIGNMENTS as u32,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_route_layout(LaunchConfig1D::new(1, 32, 0))
        .map_err(cuda_error)?;
    module
        .route_layout(
            &stream,
            &launch,
            &counts,
            &mut offsets,
            &mut tiles,
            EXPERTS as u32,
            capacity as u32,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_route_pack(LaunchConfig1D::new(ASSIGNMENTS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .route_pack(
            &stream,
            &launch,
            &input,
            &experts,
            &offsets,
            &prefixes,
            &mut mapping,
            &mut packed,
            ASSIGNMENTS as u32,
        )
        .map_err(cuda_error)?;
    let counts = read_prefix(&counts, &stream, EXPERTS)?;
    let mapping = read_prefix(&mapping, &stream, ASSIGNMENTS)?;
    let packed_host = read_prefix(&packed, &stream, ASSIGNMENTS * 512)?;
    let mut offsets = vec![0_usize; EXPERTS];
    let mut ranks = vec![0_usize; EXPERTS];
    let mut total = 0;
    for expert in 0..EXPERTS {
        let expected = if [0, 1, 2, 622, 623, 624].contains(&expert) {
            ROWS / 2
        } else {
            0
        };
        if counts[expert] as usize != expected {
            return Err(format!("expert {expert} count mismatch"));
        }
        offsets[expert] = total;
        total += expected;
    }
    for (assignment, &expert) in routes.iter().enumerate() {
        let expert = expert as usize;
        let target = offsets[expert] + ranks[expert];
        ranks[expert] += 1;
        if mapping[assignment] as usize != target
            || packed_host[target * 512..(target + 1) * 512]
                .iter()
                .enumerate()
                .any(|(column, &v)| {
                    v != 0x3000 + ((assignment / 3) % 16) as u16 * 64 + (column % 4) as u16
                })
        {
            return Err(format!(
                "stable expert packing mismatch at assignment {assignment}"
            ));
        }
    }
    moe(&module, &stream, &packed, &tiles, capacity)?;
    sampling(&module, &stream)?;
    context.check_err().map_err(cuda_error)?;
    println!(
        "ROUTE_PARITY ok=true experts=625 assignments=1200 high-ids=622,623,624 ties=stable chunk-boundary=1024"
    );
    Ok(())
}

fn moe(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    tiles: &DeviceBuffer<RoutedTile>,
    capacity: usize,
) -> CudaResult<()> {
    let rows = input.len() / 512;
    let mut gate = vec![0_u16; 626 * 512 * 432];
    let mut down = vec![0_u16; 626 * 216 * 512];
    for expert in [0, 1, 2, 3, 623, 624, 625] {
        for inner in 0..512 {
            for column in 0..432 {
                gate[expert * 512 * 432 + inner * 432 + column] = if column < 216 {
                    0x1800 + (expert % 3) as u16 * 1024
                } else {
                    0x1400 + (inner % 3) as u16 * 1024
                };
            }
        }
        for inner in 0..216 {
            for column in 0..512 {
                down[expert * 216 * 512 + inner * 512 + column] =
                    0x1800 + (column % 3) as u16 * 1024;
            }
        }
    }
    let gate = DeviceBuffer::from_host(stream, &gate).map_err(cuda_error)?;
    let down = DeviceBuffer::from_host(stream, &down).map_err(cuda_error)?;
    let mut reference_activation = DeviceBuffer::zeroed(stream, rows * 224).map_err(cuda_error)?;
    let mut reference_output = DeviceBuffer::zeroed(stream, rows * 512).map_err(cuda_error)?;
    let mut projection = DeviceBuffer::zeroed(stream, rows * 432).map_err(cuda_error)?;
    let mut activation = DeviceBuffer::zeroed(stream, rows * 224).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(stream, rows * 512).map_err(cuda_error)?;
    let shape = MoeShape {
        row_tiles: 4,
        ..MoeShape::fbt()
    };
    let launch = module
        .prepare_routed_gate(LaunchConfig2D::new(
            (216_u32.div_ceil(16), capacity as u32 * 4),
            (16, 16),
            0,
        ))
        .map_err(cuda_error)?;
    module
        .routed_gate(
            stream,
            &launch,
            input,
            &gate,
            tiles,
            cuda_host::RowWidth::new(&mut reference_activation, 224),
            shape,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_routed_down(LaunchConfig2D::new((32, capacity as u32 * 4), (16, 16), 0))
        .map_err(cuda_error)?;
    module
        .routed_down(
            stream,
            &launch,
            &reference_activation,
            &down,
            tiles,
            cuda_host::RowWidth::new(&mut reference_output, 512),
            shape,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_routed_project(LaunchConfig2D::new((27, capacity as u32), (256, 1), 0))
        .map_err(cuda_error)?;
    module
        .routed_project(
            stream,
            &launch,
            input,
            &gate,
            tiles,
            cuda_host::RowWidth::new(&mut projection, 432),
            0,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_routed_activate(LaunchConfig1D::new(
            (rows as u32 * 216).div_ceil(256),
            256,
            0,
        ))
        .map_err(cuda_error)?;
    module
        .routed_activate(stream, &launch, &projection, &mut activation, rows as u32)
        .map_err(cuda_error)?;
    let launch = module
        .prepare_routed_project(LaunchConfig2D::new((32, capacity as u32), (256, 1), 0))
        .map_err(cuda_error)?;
    module
        .routed_project(
            stream,
            &launch,
            &activation,
            &down,
            tiles,
            cuda_host::RowWidth::new(&mut output, 512),
            1,
        )
        .map_err(cuda_error)?;
    let reference = read_prefix(&reference_output, stream, rows * 512)?;
    let actual = read_prefix(&output, stream, rows * 512)?;
    let mut maximum = 0.0_f32;
    for (&expected, &value) in reference.iter().zip(&actual) {
        let expected = widen(expected);
        let value = widen(value);
        let error = (expected - value).abs();
        if !value.is_finite() || error > 0.0001 + expected.abs() * 0.001 {
            return Err(format!("tensor-core MoE mismatch: {expected} vs {value}"));
        }
        maximum = maximum.max(error);
    }
    if reference.iter().all(|&value| value == 0) {
        return Err("MoE parity fixture did not exercise a nonzero branch".into());
    }
    super::expert_project(
        module,
        stream,
        input,
        &gate,
        tiles,
        &mut projection,
        capacity,
        0,
        true,
    )?;
    let launch = module
        .prepare_routed_activate(LaunchConfig1D::new(
            (rows as u32 * 216).div_ceil(256),
            256,
            0,
        ))
        .map_err(cuda_error)?;
    module
        .routed_activate(stream, &launch, &projection, &mut activation, rows as u32)
        .map_err(cuda_error)?;
    super::expert_project(
        module,
        stream,
        &activation,
        &down,
        tiles,
        &mut output,
        capacity,
        1,
        true,
    )?;
    if actual != read_prefix(&output, stream, rows * 512)? {
        return Err("16-row MoE projection differs from 64-row reference".into());
    }
    println!(
        "MOE_PARITY ok=true rows={rows} width=512 expert-width=216 tiles=16,64 tensor-cores=sm75 max-abs-error={maximum}"
    );
    Ok(())
}

fn widen(value: u16) -> f32 {
    let exponent = (value >> 10) & 31;
    let fraction = value & 1023;
    let sign = if value & 0x8000 == 0 { 1.0 } else { -1.0 };
    match exponent {
        0 => sign * f32::from(fraction) * 2.0_f32.powi(-24),
        31 => f32::from_bits(
            (u32::from(value & 0x8000) << 16) | 0x7f800000 | (u32::from(fraction) << 13),
        ),
        _ => sign * f32::from(1024 + fraction) * 2.0_f32.powi(i32::from(exponent) - 25),
    }
}

fn sampling(module: &fbt_model::LoadedModule, stream: &CudaStream) -> CudaResult<()> {
    let logits =
        DeviceBuffer::from_host(stream, &vec![0x6000_u16; 32 * 8192]).map_err(cuda_error)?;
    let mut sampled = DeviceBuffer::zeroed(stream, 32).map_err(cuda_error)?;
    let launch = module
        .prepare_sample(LaunchConfig1D::new(32, 256, 0))
        .map_err(cuda_error)?;
    let mut reference = Vec::new();
    for temperature in [0.8, 1.0e-8] {
        module
            .sample(
                stream,
                &launch,
                &logits,
                &mut sampled,
                32,
                temperature,
                17,
                0,
            )
            .map_err(cuda_error)?;
        let actual = read_prefix(&sampled, stream, 32)?;
        if reference.is_empty() {
            reference = actual;
        } else if actual != reference {
            return Err("uniform sampling changed at small positive temperature".into());
        }
    }
    if reference.iter().all(|&token| token == reference[0]) {
        return Err("uniform stochastic sampler collapsed".into());
    }
    module
        .sample(stream, &launch, &logits, &mut sampled, 32, 0.0, 17, 0)
        .map_err(cuda_error)?;
    if read_prefix(&sampled, stream, 32)?
        .iter()
        .any(|&token| token != 0)
    {
        return Err("greedy sampling tie break changed".into());
    }
    println!("SAMPLING_PARITY ok=true temperature=0,0.8,1e-8 finite-offset=512 stable-seed=17");
    Ok(())
}
