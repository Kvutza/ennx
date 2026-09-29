use super::graph::{BackendCapability, Exactness, FusionBoundary, SemanticNode, SemanticOp};
use super::recurrence::{LayerVisit, RecurrentCore};
use super::tensor::{Extent, TensorDType, TensorLayout, TensorLifetime, TensorSpec, TensorStorage};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgramKind {
    Prefill,
    Decode,
    Draft,
    Verify,
    Score,
    Objective,
    ExperimentRound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramFunction {
    pub name: &'static str,
    pub kind: ProgramKind,
    pub nodes: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightArenaSpec {
    pub name: &'static str,
    pub overlay: &'static str,
    pub dtype: TensorDType,
    pub exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProgram {
    pub name: &'static str,
    pub version: u32,
    pub tensors: Vec<TensorSpec>,
    pub nodes: Vec<SemanticNode>,
    pub functions: Vec<ProgramFunction>,
    pub weight_arenas: Vec<WeightArenaSpec>,
    pub requirements: Vec<BackendCapability>,
    pub physical_layers: usize,
    pub recurrent_core: Option<RecurrentCore>,
}

impl ModelProgram {
    pub const VERSION: u32 = 2;

    pub fn fbt_pisa() -> Self {
        use BackendCapability as Capability;
        use Exactness::{Approximate, Exact};
        use FusionBoundary::{CloseAfter, Open};
        use ProgramKind as Function;
        use SemanticOp as Op;

        let tensor = |name, shape, dtype, layout, storage, lifetime| TensorSpec {
            name,
            shape,
            dtype,
            layout,
            storage,
            lifetime,
        };
        let node = |name, op, inputs, outputs, exactness, fusion| SemanticNode {
            name,
            op,
            inputs,
            outputs,
            exactness,
            fusion,
        };
        let sequence = vec![Extent::Symbol("tokens")];
        let hidden = vec![Extent::Symbol("tokens"), Extent::Fixed(512)];
        let qkv = vec![Extent::Symbol("tokens"), Extent::Fixed(640)];
        let routed = vec![Extent::Symbol("routed_rows"), Extent::Fixed(224)];
        Self {
            name: "fbt-pisa1",
            version: Self::VERSION,
            tensors: vec![
                tensor(
                    "tokens",
                    sequence.clone(),
                    TensorDType::U32,
                    TensorLayout::Sequence,
                    TensorStorage::Host,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "reference",
                    sequence.clone(),
                    TensorDType::U32,
                    TensorLayout::Sequence,
                    TensorStorage::Host,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "hidden",
                    hidden.clone(),
                    TensorDType::F16,
                    TensorLayout::SequenceHidden,
                    TensorStorage::Device,
                    TensorLifetime::Sequence,
                ),
                tensor(
                    "normalized",
                    hidden.clone(),
                    TensorDType::F16,
                    TensorLayout::SequenceHidden,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "qkv",
                    qkv,
                    TensorDType::F16,
                    TensorLayout::SequenceHidden,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "pyramid",
                    vec![
                        Extent::Symbol("sequences"),
                        Extent::Symbol("pisa_nodes"),
                        Extent::Fixed(64),
                    ],
                    TensorDType::F16,
                    TensorLayout::HeadSequenceWidth,
                    TensorStorage::State,
                    TensorLifetime::Sequence,
                ),
                tensor(
                    "blocks",
                    vec![Extent::Symbol("tokens"), Extent::Fixed(8)],
                    TensorDType::U32,
                    TensorLayout::Sequence,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "attention",
                    hidden.clone(),
                    TensorDType::F16,
                    TensorLayout::SequenceHidden,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "routes",
                    vec![Extent::Symbol("tokens"), Extent::Fixed(3)],
                    TensorDType::U32,
                    TensorLayout::RoutedRows,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "route-weights",
                    vec![Extent::Symbol("tokens"), Extent::Fixed(3)],
                    TensorDType::F16,
                    TensorLayout::RoutedRows,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "route-margin",
                    sequence.clone(),
                    TensorDType::F32,
                    TensorLayout::Sequence,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "packed-rows",
                    vec![Extent::Symbol("tokens"), Extent::Fixed(3)],
                    TensorDType::U32,
                    TensorLayout::RoutedRows,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "routed-tiles",
                    vec![Extent::Symbol("route_tiles")],
                    TensorDType::Packed,
                    TensorLayout::Opaque,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "packed-hidden",
                    vec![Extent::Symbol("routed_rows"), Extent::Fixed(512)],
                    TensorDType::F16,
                    TensorLayout::RoutedRows,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "routed",
                    routed,
                    TensorDType::F16,
                    TensorLayout::RoutedRows,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "routed-output",
                    vec![Extent::Symbol("routed_rows"), Extent::Fixed(512)],
                    TensorDType::F16,
                    TensorLayout::RoutedRows,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "logits",
                    vec![Extent::Symbol("tokens"), Extent::Fixed(8192)],
                    TensorDType::F32,
                    TensorLayout::SequenceHidden,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "draft_tokens",
                    sequence.clone(),
                    TensorDType::U32,
                    TensorLayout::Sequence,
                    TensorStorage::Scratch,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "verified_tokens",
                    sequence.clone(),
                    TensorDType::U32,
                    TensorLayout::Sequence,
                    TensorStorage::Device,
                    TensorLifetime::Round,
                ),
                tensor(
                    "reward-channels",
                    vec![Extent::Symbol("objectives")],
                    TensorDType::F32,
                    TensorLayout::Sequence,
                    TensorStorage::Device,
                    TensorLifetime::Invocation,
                ),
                tensor(
                    "objective",
                    vec![Extent::Fixed(1)],
                    TensorDType::F32,
                    TensorLayout::Scalar,
                    TensorStorage::Device,
                    TensorLifetime::Round,
                ),
            ],
            nodes: vec![
                node(
                    "embed",
                    Op::Embed,
                    vec!["tokens"],
                    vec!["hidden"],
                    Exact,
                    Open,
                ),
                node(
                    "attention-norm",
                    Op::Normalize,
                    vec!["hidden"],
                    vec!["normalized"],
                    Exact,
                    Open,
                ),
                node(
                    "qkv",
                    Op::ProjectQkv,
                    vec!["normalized"],
                    vec!["qkv"],
                    Exact,
                    Open,
                ),
                node(
                    "pisa-summaries",
                    Op::PisaSummaries,
                    vec!["qkv"],
                    vec!["pyramid"],
                    Exact,
                    Open,
                ),
                node(
                    "pisa-select",
                    Op::PisaSelect,
                    vec!["qkv", "pyramid"],
                    vec!["blocks"],
                    Exact,
                    Open,
                ),
                node(
                    "pisa-attention",
                    Op::PisaAttention,
                    vec!["qkv", "blocks"],
                    vec!["attention"],
                    Exact,
                    Open,
                ),
                node(
                    "attention-project",
                    Op::ProjectAttention,
                    vec!["attention"],
                    vec!["normalized"],
                    Exact,
                    Open,
                ),
                node(
                    "attention-residual",
                    Op::Residual,
                    vec!["hidden", "normalized"],
                    vec!["hidden"],
                    Exact,
                    CloseAfter,
                ),
                node(
                    "moe-norm",
                    Op::Normalize,
                    vec!["hidden"],
                    vec!["normalized"],
                    Exact,
                    Open,
                ),
                node(
                    "route",
                    Op::RouteTopK,
                    vec!["normalized"],
                    vec!["routes", "route-weights", "route-margin"],
                    Exact,
                    Open,
                ),
                node(
                    "pack-routes",
                    Op::PackRoutes,
                    vec!["normalized", "routes", "route-weights"],
                    vec!["packed-hidden", "packed-rows", "routed-tiles"],
                    Exact,
                    Open,
                ),
                node(
                    "gate-up",
                    Op::RoutedGateUp,
                    vec!["packed-hidden", "routed-tiles"],
                    vec!["routed"],
                    Exact,
                    Open,
                ),
                node(
                    "down",
                    Op::RoutedDown,
                    vec!["routed", "routed-tiles"],
                    vec!["routed-output"],
                    Exact,
                    Open,
                ),
                node(
                    "combine-routes",
                    Op::CombineRoutes,
                    vec!["routed-output", "packed-rows", "route-weights"],
                    vec!["normalized"],
                    Exact,
                    Open,
                ),
                node(
                    "moe-residual",
                    Op::Residual,
                    vec!["hidden", "normalized"],
                    vec!["hidden"],
                    Exact,
                    CloseAfter,
                ),
                node(
                    "feedback",
                    Op::Feedback,
                    vec!["hidden"],
                    vec!["hidden"],
                    Exact,
                    CloseAfter,
                ),
                node(
                    "readout",
                    Op::Readout,
                    vec!["hidden"],
                    vec!["logits"],
                    Exact,
                    Open,
                ),
                node(
                    "sample",
                    Op::Sample,
                    vec!["logits"],
                    vec!["verified_tokens"],
                    Exact,
                    CloseAfter,
                ),
                node(
                    "draft",
                    Op::Draft,
                    vec!["hidden"],
                    vec!["draft_tokens"],
                    Approximate,
                    CloseAfter,
                ),
                node(
                    "verify",
                    Op::Verify,
                    vec!["draft_tokens", "hidden"],
                    vec!["verified_tokens"],
                    Exact,
                    CloseAfter,
                ),
                node(
                    "score-generation",
                    Op::ScoreGeneration,
                    vec!["verified_tokens", "reference"],
                    vec!["reward-channels"],
                    Exact,
                    Open,
                ),
                node(
                    "objective",
                    Op::Objective,
                    vec!["reward-channels", "verified_tokens"],
                    vec!["objective"],
                    Exact,
                    CloseAfter,
                ),
            ],
            functions: vec![
                ProgramFunction {
                    name: "prefill",
                    kind: Function::Prefill,
                    nodes: vec![
                        "embed",
                        "attention-norm",
                        "qkv",
                        "pisa-summaries",
                        "pisa-select",
                        "pisa-attention",
                        "attention-project",
                        "attention-residual",
                        "moe-norm",
                        "route",
                        "pack-routes",
                        "gate-up",
                        "down",
                        "combine-routes",
                        "moe-residual",
                        "feedback",
                    ],
                },
                ProgramFunction {
                    name: "decode",
                    kind: Function::Decode,
                    nodes: vec![
                        "attention-norm",
                        "qkv",
                        "pisa-summaries",
                        "pisa-select",
                        "pisa-attention",
                        "attention-project",
                        "attention-residual",
                        "moe-norm",
                        "route",
                        "pack-routes",
                        "gate-up",
                        "down",
                        "combine-routes",
                        "moe-residual",
                        "feedback",
                        "readout",
                        "sample",
                    ],
                },
                ProgramFunction {
                    name: "draft",
                    kind: Function::Draft,
                    nodes: vec!["draft"],
                },
                ProgramFunction {
                    name: "verify",
                    kind: Function::Verify,
                    nodes: vec!["verify"],
                },
                ProgramFunction {
                    name: "score",
                    kind: Function::Score,
                    nodes: vec!["score-generation"],
                },
                ProgramFunction {
                    name: "objective",
                    kind: Function::Objective,
                    nodes: vec!["objective"],
                },
                ProgramFunction {
                    name: "experiment-round",
                    kind: Function::ExperimentRound,
                    nodes: vec![
                        "embed",
                        "attention-norm",
                        "qkv",
                        "pisa-summaries",
                        "pisa-select",
                        "pisa-attention",
                        "attention-project",
                        "attention-residual",
                        "moe-norm",
                        "route",
                        "pack-routes",
                        "gate-up",
                        "down",
                        "combine-routes",
                        "moe-residual",
                        "feedback",
                        "readout",
                        "draft",
                        "verify",
                        "score-generation",
                        "objective",
                    ],
                },
            ],
            weight_arenas: vec![WeightArenaSpec {
                name: "weights",
                overlay: "candidate",
                dtype: TensorDType::F16,
                exact: true,
            }],
            requirements: vec![
                Capability::Fp16,
                Capability::Fp32Accumulate,
                Capability::PackedWeightOverlay,
                Capability::ResidentState,
                Capability::IndirectDispatch,
                Capability::SubgroupMatrix,
            ],
            physical_layers: 5,
            recurrent_core: Some(RecurrentCore::selective_fbt()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.version != Self::VERSION {
            return Err("model program identity is invalid".into());
        }
        let tensors = self
            .tensors
            .iter()
            .map(|tensor| (tensor.name, tensor))
            .collect::<HashMap<_, _>>();
        if tensors.len() != self.tensors.len()
            || self.tensors.iter().any(|tensor| {
                tensor.name.is_empty()
                    || tensor.shape.is_empty()
                    || tensor.shape.contains(&Extent::Fixed(0))
            })
        {
            return Err("model program tensors must have unique names and nonzero shapes".into());
        }
        let nodes = self
            .nodes
            .iter()
            .map(|node| (node.name, node))
            .collect::<HashMap<_, _>>();
        if nodes.len() != self.nodes.len() || self.nodes.iter().any(|node| node.name.is_empty()) {
            return Err("model program nodes must have unique names".into());
        }
        for node in &self.nodes {
            if node
                .inputs
                .iter()
                .chain(&node.outputs)
                .any(|name| !tensors.contains_key(name))
            {
                return Err(format!("node {} references an unknown tensor", node.name));
            }
        }
        let mut kinds = HashSet::new();
        for function in &self.functions {
            if function.name.is_empty()
                || function.nodes.is_empty()
                || !kinds.insert(function.kind)
                || function.nodes.iter().any(|name| !nodes.contains_key(name))
            {
                return Err(format!("function {} is invalid", function.name));
            }
        }
        let requirements = self.requirements.iter().copied().collect::<HashSet<_>>();
        if requirements.len() != self.requirements.len() {
            return Err("model program backend requirements must be unique".into());
        }
        if self.physical_layers == 0 {
            return Err("model program must contain physical layers".into());
        }
        if let Some(core) = self.recurrent_core {
            core.validate(self.physical_layers)?;
        }
        Ok(())
    }

    pub fn layer_visits(&self, visits: usize) -> Result<Vec<LayerVisit>, String> {
        match self.recurrent_core {
            Some(core) => core.layer_visits(self.physical_layers, visits),
            None if visits == 1 => Ok((0..self.physical_layers)
                .map(|layer| LayerVisit {
                    layer,
                    visit: 0,
                    execution: layer,
                    recurrent: false,
                })
                .collect()),
            None => Err("model program has no recurrent core".into()),
        }
    }

    pub fn function(&self, kind: ProgramKind) -> Option<&ProgramFunction> {
        self.functions.iter().find(|function| function.kind == kind)
    }

    pub fn node(&self, name: &str) -> Option<&SemanticNode> {
        self.nodes.iter().find(|node| node.name == name)
    }
}
