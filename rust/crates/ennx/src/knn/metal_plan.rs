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
}

impl Graph {
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
    }
}
