use super::{DenseLeaf, DenseTerm};

pub(super) fn apply(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
) -> Result<Vec<f32>, String> {
    let leaves = leaves
        .iter()
        .map(|leaf| ennx_cuda::DenseLeaf {
            key: leaf.key,
            offset: leaf.offset as u64,
            length: leaf.len as u64,
            scale: leaf.scale,
            pad: 0,
        })
        .collect::<Vec<_>>();
    let terms = terms
        .iter()
        .map(|term| ennx_cuda::DenseTerm {
            seed: term.seed,
            coefficient: term.coefficient,
            pad: 0,
        })
        .collect::<Vec<_>>();
    ennx_cuda::dense_apply(base, &leaves, &terms)
}
