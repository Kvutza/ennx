use ennx::experimental::{
    AcquisitionKind, ComputeBackend, ForwardProgram, PackedModel, ResidentBoState, ResidentRound,
    WeightAsk, WeightLeaf,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use ennx::experimental::{
    KdaControlRequest, KdaForwardRequest, KdaMoeLayerRequest, KdaMoeMetalArena,
    KdaMoeMetalExecutor, KdaMoeMetalKdaVectors, KdaMoeMetalModel, KdaMoeMetalWeights,
    KdaPackedLinear, KdaTensorLayout,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

/// An ENNX-native quantized model package.
///
/// Opening this object parses and validates the package in Rust. It does not
/// import JAX, NumPy, or model tensors into Python.
#[pyclass(name = "ModelPackage", unsendable)]
pub struct PyModelPackage {
    inner: PackedModel,
}

#[pymethods]
impl PyModelPackage {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PackedModel::open(path).map_err(err)?,
        })
    }

    /// Return the packed descriptor required by a native forward program.
    fn linear(
        &self,
        name: &str,
    ) -> PyResult<(
        usize,
        usize,
        usize,
        usize,
        usize,
        u8,
        usize,
        usize,
        u32,
        u32,
    )> {
        let linear = self.inner.linear(name).map_err(err)?;
        Ok((
            linear.byte_offset,
            linear.scale_offset,
            linear.bias_offset,
            linear.input_width,
            linear.output_width,
            linear.bits,
            linear.group_size,
            linear.element_offset,
            linear.perturb_whole,
            linear.perturb_threshold,
        ))
    }

    #[getter]
    fn weight_bytes(&self) -> usize {
        self.inner.packed().len()
    }

    #[getter]
    fn scale_count(&self) -> usize {
        self.inner.scales().len()
    }

    #[getter]
    fn bias_count(&self) -> usize {
        self.inner.biases().len()
    }

    #[pyo3(signature=(base_value,capacity,backend="auto",scale=1.0,weight=1.0,radius=1.0))]
    fn resident_session(
        &self,
        base_value: f32,
        capacity: usize,
        backend: &str,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> PyResult<PyResidentBoSession> {
        Ok(PyResidentBoSession {
            inner: ResidentBoState::new(
                self.inner.packed(),
                base_value,
                self.inner
                    .trial_leaves(scale, weight, radius)
                    .map_err(err)?,
                capacity,
                ComputeBackend::parse(backend).map_err(err)?,
                ForwardProgram::kda().map_err(err)?,
            )
            .map_err(err)?,
            pending: None,
        })
    }
}

#[pyclass(name = "ResidentBoSession", unsendable)]
pub struct PyResidentBoSession {
    inner: ResidentBoState,
    pending: Option<ResidentRound>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
type NativeLinear = (
    String,
    usize,
    usize,
    usize,
    usize,
    usize,
    u8,
    usize,
    usize,
    u32,
    u32,
);

#[cfg(all(target_os = "macos", feature = "metal"))]
fn native_linear(raw: NativeLinear) -> (String, KdaPackedLinear) {
    let (name, byte, scale, bias, input, output, bits, group, element, whole, threshold) = raw;
    (
        name,
        KdaPackedLinear {
            byte_offset: byte,
            scale_offset: scale,
            bias_offset: bias,
            input_width: input,
            output_width: output,
            bits,
            group_size: group,
            element_offset: element,
            perturb_whole: whole,
            perturb_threshold: threshold,
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn named_linear(
    linears: &std::collections::BTreeMap<String, KdaPackedLinear>,
    name: &str,
) -> PyResult<KdaPackedLinear> {
    linears
        .get(name)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("native model has no linear {name:?}")))
}

/// Complete resident KDA-MoE greedy decoder for Apple Metal.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[pyclass(name = "NativeKdaModel", unsendable)]
pub struct PyNativeKdaModel {
    pub(crate) inner: KdaMoeMetalModel,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[pymethods]
impl PyNativeKdaModel {
    #[new]
    #[pyo3(signature=(packed,scales,biases,linears,num_layers,hidden_size,heads,head_width,gate_rank,num_experts,top_k,expert_width,residual_scale,rms_epsilon,embedding_scale,backend="metal"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        packed: PyReadonlyArray1<'_, u8>,
        scales: PyReadonlyArray1<'_, f32>,
        biases: PyReadonlyArray1<'_, f32>,
        linears: Vec<NativeLinear>,
        num_layers: usize,
        hidden_size: usize,
        heads: usize,
        head_width: usize,
        gate_rank: usize,
        num_experts: usize,
        top_k: usize,
        expert_width: usize,
        residual_scale: f32,
        rms_epsilon: f32,
        embedding_scale: f32,
        backend: &str,
    ) -> PyResult<Self> {
        let packed = packed
            .as_slice()
            .map_err(|_| PyValueError::new_err("packed model arena must be contiguous"))?;
        let scales = scales
            .as_slice()
            .map_err(|_| PyValueError::new_err("scale arena must be contiguous"))?;
        let biases = biases
            .as_slice()
            .map_err(|_| PyValueError::new_err("bias arena must be contiguous"))?;
        let linears = linears
            .into_iter()
            .map(native_linear)
            .collect::<std::collections::BTreeMap<_, _>>();
        let arena = KdaMoeMetalArena::new(KdaMoeMetalWeights {
            packed,
            scales,
            biases,
        })
        .map_err(err)?;
        let backend = ComputeBackend::parse(backend).map_err(err)?;
        let decay = vec![0.0_f32; heads];
        let time_bias = vec![0.0_f32; heads * head_width];
        let output_norm = vec![1.0_f32; head_width];
        let mut layers = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            let prefix = format!("layers.{index}");
            let attention = format!("{prefix}.attention");
            let moe = format!("{prefix}.moe");
            let qkv = named_linear(&linears, &format!("{attention}.qkv"))?;
            let control_projection = named_linear(&linears, &format!("{attention}.control"))?;
            let output = named_linear(&linears, &format!("{attention}.output"))?;
            let expert_gate = (0..num_experts)
                .map(|expert| named_linear(&linears, &format!("{moe}.experts.{expert}.gate")))
                .collect::<PyResult<Vec<_>>>()?;
            let expert_up = (0..num_experts)
                .map(|expert| named_linear(&linears, &format!("{moe}.experts.{expert}.up")))
                .collect::<PyResult<Vec<_>>>()?;
            let expert_down = (0..num_experts)
                .map(|expert| named_linear(&linears, &format!("{moe}.experts.{expert}.down")))
                .collect::<PyResult<Vec<_>>>()?;
            let request = KdaMoeLayerRequest {
                kda: KdaForwardRequest {
                    tensor: KdaTensorLayout::new(1, 1, heads, head_width, head_width)
                        .map_err(err)?,
                    qkv,
                    control: control_projection,
                    output,
                    seed: 0,
                },
                attention_norm: named_linear(&linears, &format!("{prefix}.attention_norm"))?,
                moe_norm: named_linear(&linears, &format!("{prefix}.moe_norm"))?,
                router: named_linear(&linears, &format!("{moe}.router"))?,
                expert_gate,
                expert_up,
                expert_down,
                top_k,
                residual_scale,
                rms_epsilon,
            };
            let control = KdaControlRequest {
                qkv_conv: named_linear(&linears, &format!("{attention}.qkv_conv"))?,
                control: control_projection,
                forget: named_linear(&linears, &format!("{attention}.forget_b"))?,
                output_gate: named_linear(&linears, &format!("{attention}.gate_b"))?,
                decay: named_linear(&linears, &format!("{attention}.decay"))?,
                time_bias: named_linear(&linears, &format!("{attention}.time_bias"))?,
                output_norm: named_linear(&linears, &format!("{attention}.output_norm"))?,
                output,
                gate_rank,
            };
            layers.push(
                KdaMoeMetalExecutor::new_with_arena(
                    &request,
                    control,
                    KdaMoeMetalKdaVectors {
                        decay: &decay,
                        time_bias: &time_bias,
                        output_norm: &output_norm,
                    },
                    backend,
                    &arena,
                )
                .map_err(err)?,
            );
        }
        let mut inner = KdaMoeMetalModel::new(layers).map_err(err)?;
        inner
            .attach_causal_head(
                named_linear(&linears, "embedding")?,
                named_linear(&linears, "final_norm")?,
                embedding_scale,
            )
            .map_err(err)?;
        inner.prepare_candidate(0).map_err(err)?;
        Ok(Self { inner })
    }

    fn reset(&mut self) {
        self.inner.reset_decode_state();
    }

    #[pyo3(signature=(token,seed=0))]
    fn decode_token(&mut self, token: u32, seed: u64) -> PyResult<u32> {
        self.inner.decode_token(token, seed).map_err(err)
    }

    fn logit_bits(&self) -> PyResult<Vec<u16>> {
        self.inner.logit_bits().map_err(err)
    }

    #[pyo3(signature=(prompt,max_new_tokens,seed=0))]
    fn generate(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        seed: u64,
    ) -> PyResult<Vec<u32>> {
        self.inner
            .generate(&prompt, max_new_tokens, seed)
            .map_err(err)
    }
}

#[pymethods]
impl PyResidentBoSession {
    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,backend="auto"))]
    fn new(
        base: Vec<u8>,
        base_value: f32,
        leaves: Vec<(usize, usize, u8, f32, f32, f32)>,
        capacity: usize,
        backend: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ResidentBoState::new(
                &base,
                base_value,
                trial_leaves(leaves)?,
                capacity,
                ComputeBackend::parse(backend).map_err(err)?,
                ForwardProgram::kda().map_err(err)?,
            )
            .map_err(err)?,
            pending: None,
        })
    }

    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        seeds: Vec<u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<(usize, u64, f32, u32)> {
        let round = self
            .inner
            .ask(
                &seeds,
                ask_config(
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition,
                    seed,
                )?,
            )
            .map_err(err)?;
        let out = (
            round.trial.index,
            round.trial.seed,
            round.trial.score,
            round.program_version,
        );
        self.pending = Some(round);
        Ok(out)
    }

    fn tell(&mut self, reward: f32, accept: bool) -> PyResult<()> {
        let round = self
            .pending
            .take()
            .ok_or_else(|| PyValueError::new_err("there is no pending resident BO round"))?;
        self.inner.tell(round, reward, accept).map_err(err)
    }

    #[getter]
    fn rewards(&self) -> Vec<f32> {
        self.inner.rewards().to_vec()
    }
}

fn ask_config(
    length: f32,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    acquisition: &str,
    seed: u64,
) -> PyResult<WeightAsk> {
    Ok(WeightAsk {
        length,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
        seed,
    })
}

fn trial_leaves(raw: Vec<(usize, usize, u8, f32, f32, f32)>) -> PyResult<Vec<WeightLeaf>> {
    raw.into_iter()
        .map(|(offset, length, bits, scale, weight, radius)| {
            WeightLeaf::new(offset, length, bits, scale, weight, radius).map_err(err)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{PyModelPackage, PyResidentBoSession};

    #[test]
    fn package_binding_is_linked() {
        let _ = std::mem::size_of::<PyModelPackage>();
    }

    #[test]
    fn resident_session_binding_is_linked() {
        let _ = std::mem::size_of::<PyResidentBoSession>();
    }
}
