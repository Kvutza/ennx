use super::{make_steps, make_tiles, Ask, Center, Leaf, Step, Tile};

pub(super) struct Engine {
    inner: ennx_cuda::TrialEngine,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Leaf], slots: usize) -> Result<Self, String> {
        let steps = make_steps(leaves, 0.0);
        let tiles = make_tiles(leaves);
        Ok(Self {
            inner: ennx_cuda::TrialEngine::new(
                base,
                &steps.iter().copied().map(cuda_leaf).collect::<Vec<_>>(),
                &tiles.iter().copied().map(cuda_tile).collect::<Vec<_>>(),
                slots,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        let history_slots = history
            .iter()
            .map(|&(slot, _)| {
                u32::try_from(slot).map_err(|_| format!("CUDA history slot {slot} exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = history.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let steps = make_steps(leaves, config.length)
            .into_iter()
            .map(cuda_leaf)
            .collect::<Vec<_>>();
        let (index, score) = self.inner.ask(
            base_slot,
            &history_slots,
            &outcomes,
            trial_slot,
            seeds,
            &draws,
            &steps,
            ennx_cuda::Ask {
                neighbors: config.neighbors,
                acquisition: crate::weights::acquisition_code(config.acquisition),
                epistemic_scale: config.epistemic_scale,
                aleatoric_scale: config.aleatoric_scale,
                y_scale: config.y_scale,
                beta: config.beta,
            },
            materialize_row,
        )?;
        Ok((index, score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_multi_tr(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.ask_multi_tr_impl(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            seeds,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_multi_tr_tree(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.ask_multi_tr_impl(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            seeds,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ask_multi_tr_impl(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        tree: Option<(&[Center], &[usize])>,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        let history_slots = history
            .iter()
            .map(|&(slot, _)| {
                u32::try_from(slot).map_err(|_| format!("CUDA history slot {slot} exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = history.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let steps = make_steps(leaves, config.length)
            .into_iter()
            .map(cuda_leaf)
            .collect::<Vec<_>>();
        let (centers, region_centers) = match tree {
            Some((centers, region_centers)) => {
                let centers = centers
                    .iter()
                    .map(|center| {
                        Ok(ennx_cuda::CenterStep {
                            parent: center
                                .parent
                                .map(|parent| {
                                    u32::try_from(parent)
                                        .map_err(|_| "CUDA center parent exceeds u32".to_string())
                                })
                                .transpose()?
                                .unwrap_or(u32::MAX),
                            seed: ennx_cuda::Seed {
                                low: center.seed as u32,
                                high: (center.seed >> 32) as u32,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let region_centers = region_centers
                    .iter()
                    .map(|&center| {
                        u32::try_from(center)
                            .map_err(|_| "CUDA region center exceeds u32".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (centers, region_centers)
            }
            None => (Vec::new(), Vec::new()),
        };
        self.inner.ask_multi(
            base_slot,
            &history_slots,
            &outcomes,
            num_regions,
            seeds_per_region,
            seeds,
            &draws,
            &centers,
            &region_centers,
            &steps,
            ennx_cuda::Ask {
                neighbors: config.neighbors,
                acquisition: crate::weights::acquisition_code(config.acquisition),
                epistemic_scale: config.epistemic_scale,
                aleatoric_scale: config.aleatoric_scale,
                y_scale: config.y_scale,
                beta: config.beta,
            },
        )
    }

    pub(super) fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        steps: &[Step],
    ) -> Result<(), String> {
        let steps = steps.iter().copied().map(cuda_leaf).collect::<Vec<_>>();
        self.inner.materialize(base_slot, trial_slot, seed, &steps)
    }

    pub(super) fn read(&self, slot: usize) -> Result<Vec<u8>, String> {
        self.inner.read(slot)
    }

    pub(super) fn write(&mut self, slot: usize, row: &[u8]) -> Result<(), String> {
        self.inner.write(slot, row)
    }
}

fn cuda_leaf(step: Step) -> ennx_cuda::Leaf {
    ennx_cuda::Leaf {
        byte_offset: step.byte_offset,
        element_offset: step.element_offset,
        length: step.length,
        bits: step.bits,
        encoding: step.encoding,
        scale: step.scale,
        weight: step.weight,
        whole: step.whole,
        threshold: step.threshold,
    }
}

fn cuda_tile(tile: Tile) -> ennx_cuda::Tile {
    ennx_cuda::Tile {
        leaf: tile.leaf,
        start: tile.start,
        length: tile.length,
        pad: tile.pad,
    }
}
