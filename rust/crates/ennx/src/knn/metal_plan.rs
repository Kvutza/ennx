const TILE_ROWS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Plan {
    Split,
    Fused,
    Tree,
    Tiled,
    Simd,
    Gram,
    Wide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Obj {
    Input,
    Dist,
    Lists,
    Result,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Op {
    Init,
    L2,
    Topk,
    Fuse,
    Merge,
    Batch,
    Tile,
    Simd,
    Gram,
    Reduce,
    Fold,
}

#[derive(Clone, Copy)]
struct Step {
    from: Obj,
    op: Op,
    to: Obj,
}

pub(super) struct Graph(&'static [Step]);

const SPLIT: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Init,
        to: Obj::Result,
    },
    Step {
        from: Obj::Input,
        op: Op::L2,
        to: Obj::Dist,
    },
    Step {
        from: Obj::Dist,
        op: Op::Topk,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Merge,
        to: Obj::Result,
    },
]);

const FUSED: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Init,
        to: Obj::Result,
    },
    Step {
        from: Obj::Input,
        op: Op::Fuse,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Merge,
        to: Obj::Result,
    },
]);

const TREE: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Batch,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Reduce,
        to: Obj::Result,
    },
]);

const TILED: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Tile,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Reduce,
        to: Obj::Result,
    },
]);

const SIMD: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Simd,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Reduce,
        to: Obj::Result,
    },
]);

const GRAM: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Gram,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Reduce,
        to: Obj::Result,
    },
]);

const WIDE: Graph = Graph(&[
    Step {
        from: Obj::Input,
        op: Op::Batch,
        to: Obj::Lists,
    },
    Step {
        from: Obj::Lists,
        op: Op::Fold,
        to: Obj::Result,
    },
]);

impl Plan {
    pub(super) fn graph(self) -> &'static Graph {
        match self {
            Self::Split => &SPLIT,
            Self::Fused => &FUSED,
            Self::Tree => &TREE,
            Self::Tiled => &TILED,
            Self::Simd => &SIMD,
            Self::Gram => &GRAM,
            Self::Wide => &WIDE,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Fused => "fused",
            Self::Tree => "tree",
            Self::Tiled => "tiled",
            Self::Simd => "simd",
            Self::Gram => "gram",
            Self::Wide => "wide",
        }
    }

    pub(super) fn id(self) -> u64 {
        match self {
            Self::Split => 0,
            Self::Fused => 1,
            Self::Tree => 2,
            Self::Tiled => 3,
            Self::Simd => 4,
            Self::Gram => 5,
            Self::Wide => 6,
        }
    }
}

impl Graph {
    pub(super) fn passes(&self, tiles: usize, k: usize) -> usize {
        self.0
            .iter()
            .map(|step| match step.op {
                Op::Init | Op::Batch | Op::Tile | Op::Simd | Op::Gram => 1,
                Op::L2 | Op::Topk | Op::Fuse | Op::Merge => tiles,
                Op::Reduce => levels(tiles, k),
                Op::Fold => fold_levels(tiles),
            })
            .sum()
    }

    pub(super) fn valid(&self) -> bool {
        !self.0.is_empty()
            && self.0.iter().all(|step| {
                matches!(
                    (step.from, step.op, step.to),
                    (Obj::Input, Op::Init, Obj::Result)
                        | (Obj::Input, Op::L2, Obj::Dist)
                        | (Obj::Dist, Op::Topk, Obj::Lists)
                        | (Obj::Input, Op::Fuse, Obj::Lists)
                        | (Obj::Lists, Op::Merge, Obj::Result)
                        | (Obj::Input, Op::Batch, Obj::Lists)
                        | (Obj::Input, Op::Tile, Obj::Lists)
                        | (Obj::Input, Op::Simd, Obj::Lists)
                        | (Obj::Input, Op::Gram, Obj::Lists)
                        | (Obj::Lists, Op::Reduce, Obj::Result)
                        | (Obj::Lists, Op::Fold, Obj::Result)
                )
            })
            && self.0.last().is_some_and(|step| step.to == Obj::Result)
    }
}

fn levels(mut lists: usize, k: usize) -> usize {
    let mut levels = 0;
    while lists > 1 {
        lists = lists.div_ceil(TILE_ROWS / k);
        levels += 1;
    }
    levels
}

fn fold_levels(mut lists: usize) -> usize {
    let mut levels = 0;
    while lists > 1 {
        lists = lists.div_ceil(2);
        levels += 1;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::Plan;

    #[test]
    fn graphs() {
        for plan in [
            Plan::Split,
            Plan::Fused,
            Plan::Tree,
            Plan::Tiled,
            Plan::Simd,
            Plan::Gram,
            Plan::Wide,
        ] {
            assert!(plan.graph().valid());
        }
        assert_eq!(Plan::Split.graph().passes(8, 16), 25);
        assert_eq!(Plan::Fused.graph().passes(8, 16), 17);
        assert_eq!(Plan::Tree.graph().passes(8, 16), 2);
        assert_eq!(Plan::Tree.graph().passes(65, 16), 3);
        assert_eq!(Plan::Wide.graph().passes(8, 2048), 4);
    }
}
