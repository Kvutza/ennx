use super::DenseView;
use crate::dense::DenseTerm;

pub(super) struct Resident {
    inner: ennx_cuda::DenseLinearEngine,
}

pub(super) fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
    rows: usize,
) -> Result<Vec<f32>, String> {
    let mut engine = ennx_cuda::DenseLinearEngine::new(
        weight,
        bias,
        params(input.len(), rows, weight_view, bias_view)?,
    )?;
    engine.eval(input, &cuda_terms(terms))
}

impl Resident {
    pub(super) fn new(
        weight: &[f32],
        columns: usize,
        bias: Option<&[f32]>,
        weight_view: DenseView,
        bias_view: Option<DenseView>,
    ) -> Result<Self, String> {
        let rows = weight.len() / columns;
        Ok(Self {
            inner: ennx_cuda::DenseLinearEngine::new(
                weight,
                bias,
                params(columns, rows, weight_view, bias_view)?,
            )?,
        })
    }

    pub(super) fn eval(&mut self, input: &[f32], terms: &[DenseTerm]) -> Result<Vec<f32>, String> {
        self.inner.eval(input, &cuda_terms(terms))
    }
}

fn params(
    columns: usize,
    rows: usize,
    weight: DenseView,
    bias: Option<DenseView>,
) -> Result<ennx_cuda::DenseLinearParams, String> {
    let bias = bias.unwrap_or(DenseView {
        key: 0,
        start: 0,
        scale: 1.0,
    });
    Ok(ennx_cuda::DenseLinearParams {
        rows: u32::try_from(rows).map_err(|_| "CUDA dense linear rows exceed u32")?,
        columns: u32::try_from(columns).map_err(|_| "CUDA dense linear columns exceed u32")?,
        has_bias: 0,
        term_count: 0,
        weight_key: weight.key,
        weight_start: weight.start,
        bias_key: bias.key,
        bias_start: bias.start,
        weight_scale: weight.scale,
        bias_scale: bias.scale,
        pad0: 0,
        pad1: 0,
    })
}

fn cuda_terms(terms: &[DenseTerm]) -> Vec<ennx_cuda::DenseTerm> {
    terms
        .iter()
        .map(|term| ennx_cuda::DenseTerm {
            seed: term.seed,
            coefficient: term.coefficient,
            pad: 0,
        })
        .collect()
}
