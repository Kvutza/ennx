use crate::weights::ComputeDevice;

use super::{dense_next, sign, DenseTerm};

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod cuda;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseView {
    pub key: u64,
    pub start: u64,
    pub scale: f32,
}

impl DenseView {
    pub fn new(key: u64, start: u64, scale: f32) -> Result<Self, String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("dense view scale must be finite and positive".into());
        }
        Ok(Self { key, start, scale })
    }
}

pub struct DenseLinear {
    columns: usize,
    rows: usize,
    engine: LinearResident,
}

enum LinearResident {
    Cpu {
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        weight_view: DenseView,
        bias_view: Option<DenseView>,
    },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Resident),
    #[cfg(feature = "opencl")]
    OpenCl(opencl::Resident),
    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    Cuda(cuda::Resident),
}

impl DenseLinear {
    pub fn new(
        weight: Vec<f32>,
        columns: usize,
        bias: Option<Vec<f32>>,
        weight_view: DenseView,
        bias_view: Option<DenseView>,
        device: ComputeDevice,
    ) -> Result<Self, String> {
        let rows = validate_model(&weight, columns, bias.as_deref(), weight_view, bias_view)?;
        let engine = match device {
            ComputeDevice::Cpu => LinearResident::Cpu {
                weight,
                bias,
                weight_view,
                bias_view,
            },
            ComputeDevice::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    LinearResident::Metal(metal::Resident::new(
                        &weight,
                        columns,
                        bias.as_deref(),
                        weight_view,
                        bias_view,
                        false,
                    )?)
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err("Metal dense linear is not available in this build".into());
                }
            }
            ComputeDevice::Agx => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    LinearResident::Metal(metal::Resident::new(
                        &weight,
                        columns,
                        bias.as_deref(),
                        weight_view,
                        bias_view,
                        true,
                    )?)
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err("AGX dense linear is not available in this build".into());
                }
            }
            ComputeDevice::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    LinearResident::OpenCl(opencl::Resident::new(
                        &weight,
                        columns,
                        bias.as_deref(),
                        weight_view,
                        bias_view,
                    )?)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err("OpenCL dense linear is not available in this build".into());
                }
            }
            ComputeDevice::Cuda => {
                #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
                {
                    LinearResident::Cuda(cuda::Resident::new(
                        &weight,
                        columns,
                        bias.as_deref(),
                        weight_view,
                        bias_view,
                    )?)
                }
                #[cfg(not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64")))]
                {
                    return Err("CUDA dense linear is not available in this build".into());
                }
            }
            ComputeDevice::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    LinearResident::Metal(
                        metal::Resident::new(
                            &weight,
                            columns,
                            bias.as_deref(),
                            weight_view,
                            bias_view,
                            true,
                        )
                        .or_else(|_| {
                            metal::Resident::new(
                                &weight,
                                columns,
                                bias.as_deref(),
                                weight_view,
                                bias_view,
                                false,
                            )
                        })?,
                    )
                }
                #[cfg(all(
                    feature = "cuda",
                    target_os = "linux",
                    target_arch = "x86_64",
                    not(all(target_os = "macos", feature = "metal"))
                ))]
                {
                    LinearResident::Cuda(cuda::Resident::new(
                        &weight,
                        columns,
                        bias.as_deref(),
                        weight_view,
                        bias_view,
                    )?)
                }
                #[cfg(all(
                    feature = "opencl",
                    not(all(target_os = "macos", feature = "metal")),
                    not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))
                ))]
                {
                    LinearResident::OpenCl(opencl::Resident::new(
                        &weight,
                        columns,
                        bias.as_deref(),
                        weight_view,
                        bias_view,
                    )?)
                }
                #[cfg(not(any(
                    all(target_os = "macos", feature = "metal"),
                    all(feature = "cuda", target_os = "linux", target_arch = "x86_64"),
                    feature = "opencl"
                )))]
                {
                    LinearResident::Cpu {
                        weight,
                        bias,
                        weight_view,
                        bias_view,
                    }
                }
            }
        };
        Ok(Self {
            columns,
            rows,
            engine,
        })
    }

    pub fn eval(&mut self, input: &[f32], terms: &[DenseTerm]) -> Result<Vec<f32>, String> {
        validate_eval(input, self.columns, terms)?;
        let values = match &mut self.engine {
            LinearResident::Cpu {
                weight,
                bias,
                weight_view,
                bias_view,
            } => linear_cpu(
                input,
                weight,
                bias.as_deref(),
                *weight_view,
                *bias_view,
                terms,
                self.rows,
            )?,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            LinearResident::Metal(engine) => engine.eval(input, terms)?,
            #[cfg(feature = "opencl")]
            LinearResident::OpenCl(engine) => engine.eval(input, terms)?,
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            LinearResident::Cuda(engine) => engine.eval(input, terms)?,
        };
        if values.iter().any(|value| !value.is_finite()) {
            return Err("dense linear overflowed FP32".into());
        }
        Ok(values)
    }

    pub fn input_size(&self) -> usize {
        self.columns
    }

    pub fn output_size(&self) -> usize {
        self.rows
    }
}

pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
    device: ComputeDevice,
) -> Result<Vec<f32>, String> {
    let rows = linear_validate(input, weight, bias, weight_view, bias_view, terms)?;
    let values = match device {
        ComputeDevice::Cpu => linear_cpu(input, weight, bias, weight_view, bias_view, terms, rows)?,
        ComputeDevice::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::linear(
                    input,
                    weight,
                    bias,
                    weight_view,
                    bias_view,
                    terms,
                    rows,
                    false,
                )?
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("Metal dense linear is not available in this build".into());
            }
        }
        ComputeDevice::Agx => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::linear(
                    input,
                    weight,
                    bias,
                    weight_view,
                    bias_view,
                    terms,
                    rows,
                    true,
                )?
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("AGX dense linear is not available in this build".into());
            }
        }
        ComputeDevice::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                opencl::linear(input, weight, bias, weight_view, bias_view, terms, rows)?
            }
            #[cfg(not(feature = "opencl"))]
            {
                return Err("OpenCL dense linear is not available in this build".into());
            }
        }
        ComputeDevice::Cuda => {
            #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
            {
                cuda::linear(input, weight, bias, weight_view, bias_view, terms, rows)?
            }
            #[cfg(not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64")))]
            {
                return Err("CUDA dense linear is not available in this build".into());
            }
        }
        ComputeDevice::Auto => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::linear(
                    input,
                    weight,
                    bias,
                    weight_view,
                    bias_view,
                    terms,
                    rows,
                    true,
                )
                .or_else(|_| {
                    metal::linear(
                        input,
                        weight,
                        bias,
                        weight_view,
                        bias_view,
                        terms,
                        rows,
                        false,
                    )
                })?
            }
            #[cfg(all(
                feature = "cuda",
                target_os = "linux",
                target_arch = "x86_64",
                not(all(target_os = "macos", feature = "metal"))
            ))]
            {
                cuda::linear(input, weight, bias, weight_view, bias_view, terms, rows)?
            }
            #[cfg(all(
                feature = "opencl",
                not(all(target_os = "macos", feature = "metal")),
                not(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))
            ))]
            {
                opencl::linear(input, weight, bias, weight_view, bias_view, terms, rows)?
            }
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                all(feature = "cuda", target_os = "linux", target_arch = "x86_64"),
                feature = "opencl"
            )))]
            {
                linear_cpu(input, weight, bias, weight_view, bias_view, terms, rows)?
            }
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err("dense linear overflowed FP32".into());
    }
    Ok(values)
}

fn linear_validate(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
) -> Result<usize, String> {
    let rows = validate_model(weight, input.len(), bias, weight_view, bias_view)?;
    validate_eval(input, input.len(), terms)?;
    Ok(rows)
}

fn validate_model(
    weight: &[f32],
    columns: usize,
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
) -> Result<usize, String> {
    if columns == 0 || weight.is_empty() || !weight.len().is_multiple_of(columns) {
        return Err("dense linear weight must contain complete non-empty rows".into());
    }
    let rows = weight.len() / columns;
    if bias.is_some_and(|values| values.len() != rows) {
        return Err("dense linear bias length must equal the weight row count".into());
    }
    if bias.is_some() != bias_view.is_some() {
        return Err("dense linear bias and bias view must be supplied together".into());
    }
    if weight
        .iter()
        .chain(bias.into_iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err("dense linear weights must be finite".into());
    }
    if !weight_view.scale.is_finite() || weight_view.scale <= 0.0 {
        return Err("dense linear weight scale must be finite and positive".into());
    }
    if bias_view.is_some_and(|view| !view.scale.is_finite() || view.scale <= 0.0) {
        return Err("dense linear bias scale must be finite and positive".into());
    }
    u64::try_from(weight.len())
        .ok()
        .and_then(|len| weight_view.start.checked_add(len))
        .ok_or("dense linear weight coordinates overflow u64")?;
    if let Some(view) = bias_view {
        view.start
            .checked_add(u64::try_from(rows).map_err(|_| "dense linear row count exceeds u64")?)
            .ok_or("dense linear bias coordinates overflow u64")?;
    }
    Ok(rows)
}

fn validate_eval(input: &[f32], columns: usize, terms: &[DenseTerm]) -> Result<(), String> {
    if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "dense linear input must contain {columns} finite values"
        ));
    }
    if terms.is_empty() || terms.iter().any(|term| !term.coefficient.is_finite()) {
        return Err("dense linear requires finite perturbation terms".into());
    }
    if !super::has_direction(terms) {
        return Err("dense linear terms cancel to zero".into());
    }
    Ok(())
}

#[cfg(feature = "zig-dense")]
fn linear_cpu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
    rows: usize,
) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0; rows];
    let status = unsafe {
        ennx_dense_linear_f32(
            input.as_ptr(),
            input.len(),
            weight.as_ptr(),
            rows,
            bias.map_or(std::ptr::null(), |values| values.as_ptr()),
            weight_view,
            bias_view.unwrap_or(DenseView {
                key: 0,
                start: 0,
                scale: 1.0,
            }),
            terms.as_ptr(),
            terms.len(),
            out.as_mut_ptr(),
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err("Zig dense linear rejected its inputs".into())
    }
}

#[cfg(not(feature = "zig-dense"))]
fn linear_cpu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
    rows: usize,
) -> Result<Vec<f32>, String> {
    let columns = input.len();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0f32;
        for column in 0..columns {
            let index = row * columns + column;
            let value = perturbed(
                weight[index],
                weight_view,
                u64::try_from(index).expect("validated dense weight index fits u64"),
                terms,
            )?;
            sum = input[column].mul_add(value, sum);
        }
        if let (Some(values), Some(view)) = (bias, bias_view) {
            sum += perturbed(
                values[row],
                view,
                u64::try_from(row).expect("validated dense bias index fits u64"),
                terms,
            )?;
        }
        out.push(sum);
    }
    Ok(out)
}

fn perturbed(base: f32, view: DenseView, element: u64, terms: &[DenseTerm]) -> Result<f32, String> {
    let mut sum = 0.0f32;
    let mut strongest = 0.0f32;
    let mut positive = true;
    let coordinate = view
        .start
        .checked_add(element)
        .ok_or("dense linear coordinate overflow")?;
    for term in terms {
        if term.coefficient == 0.0 {
            continue;
        }
        let direction = sign(term.seed, view.key, coordinate);
        sum += term.coefficient * direction;
        if term.coefficient.abs() > strongest {
            strongest = term.coefficient.abs();
            positive = (term.coefficient > 0.0) == (direction > 0.0);
        }
    }
    let candidate = base + view.scale * sum;
    if sum == 0.0 || candidate == base {
        Ok(dense_next(base, positive))
    } else if candidate.is_finite() {
        Ok(candidate)
    } else {
        Err("dense perturbation overflowed FP32".into())
    }
}

#[cfg(feature = "zig-dense")]
unsafe extern "C" {
    fn ennx_dense_linear_f32(
        input: *const f32,
        columns: usize,
        weight: *const f32,
        rows: usize,
        bias: *const f32,
        weight_view: DenseView,
        bias_view: DenseView,
        terms: *const DenseTerm,
        num_terms: usize,
        out: *mut f32,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_linear_matches_materialized_weights() {
        let input = [0.25, -0.5, 1.5, 2.0];
        let weight = [0.5, -1.0, 0.75, 0.25, -0.5, 2.0, 1.25, -0.75];
        let bias = [0.125, -0.25];
        let weight_view = DenseView::new(11, 0, 0.02).unwrap();
        let bias_view = DenseView::new(29, 0, 0.01).unwrap();
        let terms = [
            DenseTerm::new(0x1234_5678_9abc_def0, 0.5).unwrap(),
            DenseTerm::new(91, -0.125).unwrap(),
        ];
        let actual = linear(
            &input,
            &weight,
            Some(&bias),
            weight_view,
            Some(bias_view),
            &terms,
            ComputeDevice::Cpu,
        )
        .unwrap();

        let moved_weight = (0..weight.len())
            .map(|index| perturbed(weight[index], weight_view, index as u64, &terms).unwrap())
            .collect::<Vec<_>>();
        let moved_bias = (0..bias.len())
            .map(|index| perturbed(bias[index], bias_view, index as u64, &terms).unwrap())
            .collect::<Vec<_>>();
        let expected = moved_weight
            .chunks_exact(input.len())
            .zip(moved_bias)
            .map(|(row, bias)| {
                row.iter()
                    .zip(input)
                    .fold(bias, |sum, (weight, input)| input.mul_add(*weight, sum))
            })
            .collect::<Vec<_>>();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() <= 2.0 * f32::EPSILON);
        }
    }
}
