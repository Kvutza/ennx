use super::{DenseLeaf, DenseTerm};

pub(super) struct Resident {
    inner: ennx_cuda::Bf16Engine,
}

impl Resident {
    pub(super) fn new(base: &[u16], leaves: &[DenseLeaf]) -> Result<Self, String> {
        let leaves = cuda_leaves(leaves);
        Ok(Self {
            inner: ennx_cuda::Bf16Engine::new(base, &leaves)?,
        })
    }

    pub(super) unsafe fn from_device(
        pointer: u64,
        len: usize,
        leaves: &[DenseLeaf],
    ) -> Result<Self, String> {
        let leaves = cuda_leaves(leaves);
        Ok(Self {
            inner: unsafe { ennx_cuda::Bf16Engine::from_device(pointer, len, &leaves)? },
        })
    }

    pub(super) fn materialize(&mut self, terms: &[DenseTerm]) -> Result<(), String> {
        let terms = terms
            .iter()
            .map(|term| ennx_cuda::DenseTerm {
                seed: term.seed,
                coefficient: term.coefficient,
                pad: 0,
            })
            .collect::<Vec<_>>();
        self.inner.materialize(&terms)
    }

    pub(super) fn candidate(&self) -> Result<Vec<u16>, String> {
        self.inner.candidate()
    }

    pub(super) fn device_ptr(&self, stream: Option<i64>) -> Result<(u64, usize, usize), String> {
        self.inner.device_ptr(stream)
    }
}

fn cuda_leaves(leaves: &[DenseLeaf]) -> Vec<ennx_cuda::DenseLeaf> {
    leaves
        .iter()
        .map(|leaf| ennx_cuda::DenseLeaf {
            key: leaf.key,
            offset: leaf.offset as u64,
            length: leaf.len as u64,
            scale: leaf.scale,
            pad: 0,
        })
        .collect()
}
