use ennx::experimental::{
    apply_dense, apply_sparse, blocks_for_words, dense_dist2, dense_linear, draw_sparse,
    merge_values, missing_words, select_weights, sparse_union, sparse_xor, take_words,
    AcquisitionKind, BpannHistory, ComputeDevice, DenseLeaf, DenseLinear, DenseTerm, DenseView,
    PackedLeaf, PackedSearch, PackedTrial, PackedTurbo, SearchConfig, TurboTrial as CoreTrial,
    WeightBlock, WeightSelectConfig,
};
use ennx::TRLengthConfig;
use numpy::{
    Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
type PyObject = Py<PyAny>;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use ennx::experimental::ParamBuffer;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
#[pyclass(name = "ParamBuffer", unsendable)]
pub struct PyParamBuffer {
    inner: ParamBuffer,
    exported: Arc<AtomicBool>,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
#[pymethods]
impl PyParamBuffer {
    #[new]
    fn new(
        base: &Bound<'_, PyAny>,
        blocks: Vec<PyRef<'_, crate::py_bf16::PyParamBlock>>,
    ) -> PyResult<Self> {
        let input = crate::dlpack::Input::new(base)?;
        let leaves = blocks
            .iter()
            .map(|block| {
                let block = block.inner;
                DenseLeaf::new(block.key, block.offset, block.len, block.scale).map_err(err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: unsafe { ParamBuffer::from_device(input.pointer, input.len, leaves) }
                .map_err(err)?,
            exported: Arc::new(AtomicBool::new(false)),
        })
    }

    fn materialize(&mut self, terms: Vec<(u64, f32)>) -> PyResult<()> {
        if self.exported.load(Ordering::Acquire) {
            return Err(PyValueError::new_err(
                "cannot materialize while a JAX candidate is alive",
            ));
        }
        self.inner.materialize(&dense_terms(terms)?).map_err(err)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, 0)
    }

    #[pyo3(signature=(stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        slf: PyRef<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        if copy == Some(true) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "ParamBuffer does not export copies",
            ));
        }
        if dl_device.is_some_and(|device| device != (2, 0)) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "ParamBuffer cannot export to another device",
            ));
        }
        if stream.is_some_and(|value| value == 0 || value < -1) {
            return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
        }
        if slf
            .exported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(PyValueError::new_err(
                "the current BF16 candidate is already exported",
            ));
        }
        let py = slf.py();
        let (pointer, _, _) = match slf.inner.device_ptr(stream) {
            Ok(value) => value,
            Err(error) => {
                slf.exported.store(false, Ordering::Release);
                return Err(err(error));
            }
        };
        let len = slf.inner.len();
        let lease = Arc::clone(&slf.exported);
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        match crate::dlpack::export(py, owner, Arc::clone(&lease), pointer, len, max_version) {
            Ok(capsule) => Ok(capsule),
            Err(error) => {
                lease.store(false, Ordering::Release);
                Err(error)
            }
        }
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

fn array1_vec<T: Copy + Element>(array: PyReadonlyArray1<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

fn array2_vec<T: Copy + Element>(array: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

fn int4_blocks(raw: Vec<(usize, usize, f32, f32, f32)>) -> PyResult<Vec<WeightBlock>> {
    raw.into_iter()
        .map(
            |(offset, length, quantization_scale, metric_scale, weight)| {
                WeightBlock::new(offset, length, 4, quantization_scale, metric_scale, weight)
                    .map_err(err)
            },
        )
        .collect()
}

fn mixed_blocks(raw: Vec<(usize, usize, u8, f32, f32, f32)>) -> PyResult<Vec<WeightBlock>> {
    raw.into_iter()
        .map(
            |(offset, length, bits, quantization_scale, metric_scale, weight)| {
                WeightBlock::new(
                    offset,
                    length,
                    bits,
                    quantization_scale,
                    metric_scale,
                    weight,
                )
                .map_err(err)
            },
        )
        .collect()
}

fn trial_leaves(raw: Vec<(usize, usize, u8, f32, f32, f32)>) -> PyResult<Vec<PackedLeaf>> {
    raw.into_iter()
        .map(|(offset, length, bits, scale, weight, radius)| {
            PackedLeaf::new(offset, length, bits, scale, weight, radius).map_err(err)
        })
        .collect()
}

fn dense_leaves(raw: Vec<(u64, usize, usize, f32)>) -> PyResult<Vec<DenseLeaf>> {
    raw.into_iter()
        .map(|(key, offset, len, scale)| DenseLeaf::new(key, offset, len, scale).map_err(err))
        .collect()
}

fn dense_terms(raw: Vec<(u64, f32)>) -> PyResult<Vec<DenseTerm>> {
    raw.into_iter()
        .map(|(seed, coefficient)| DenseTerm::new(seed, coefficient).map_err(err))
        .collect()
}

fn dense_view(raw: (u64, u64, f32)) -> PyResult<DenseView> {
    DenseView::new(raw.0, raw.1, raw.2).map_err(err)
}

#[pyfunction(name = "dense_apply")]
#[pyo3(signature=(base,leaves,terms,device="auto"))]
pub fn dense_apply_py<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'_, f32>,
    leaves: Vec<(u64, usize, usize, f32)>,
    terms: Vec<(u64, f32)>,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<f32>>, usize)> {
    let result = apply_dense(
        &array1_vec(base),
        &dense_leaves(leaves)?,
        &dense_terms(terms)?,
        ComputeDevice::parse(device).map_err(err)?,
    )
    .map_err(err)?;
    Ok((result.values.into_pyarray(py), result.changed))
}

#[pyfunction(name = "dense_dist2")]
pub fn dense_dist2_py(
    leaves: Vec<(u64, usize, usize, f32)>,
    left: Vec<(u64, f32)>,
    right: Vec<(u64, f32)>,
) -> PyResult<f64> {
    dense_dist2(
        &dense_leaves(leaves)?,
        &dense_terms(left)?,
        &dense_terms(right)?,
    )
    .map_err(err)
}

#[pyfunction(name = "dense_linear")]
#[pyo3(signature=(input,weight,weight_view,terms,bias=None,bias_view=None,device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn dense_linear_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f32>,
    weight: PyReadonlyArray2<'_, f32>,
    weight_view: (u64, u64, f32),
    terms: Vec<(u64, f32)>,
    bias: Option<PyReadonlyArray1<'_, f32>>,
    bias_view: Option<(u64, u64, f32)>,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let input = array1_vec(input);
    let weight = array2_vec(&weight);
    let bias = bias.map(array1_vec);
    let bias_view = bias_view.map(dense_view).transpose()?;
    dense_linear(
        &input,
        &weight,
        bias.as_deref(),
        dense_view(weight_view)?,
        bias_view,
        &dense_terms(terms)?,
        ComputeDevice::parse(device).map_err(err)?,
    )
    .map(|values| values.into_pyarray(py))
    .map_err(err)
}

#[pyclass(name = "DenseLinear", unsendable)]
pub struct PyDenseLinear {
    inner: DenseLinear,
}

#[pymethods]
impl PyDenseLinear {
    #[new]
    #[pyo3(signature=(weight,weight_view,bias=None,bias_view=None,device="auto"))]
    fn new(
        weight: PyReadonlyArray2<'_, f32>,
        weight_view: (u64, u64, f32),
        bias: Option<PyReadonlyArray1<'_, f32>>,
        bias_view: Option<(u64, u64, f32)>,
        device: &str,
    ) -> PyResult<Self> {
        let columns = weight.as_array().ncols();
        let bias = bias.map(array1_vec);
        Ok(Self {
            inner: DenseLinear::new(
                array2_vec(&weight),
                columns,
                bias,
                dense_view(weight_view)?,
                bias_view.map(dense_view).transpose()?,
                ComputeDevice::parse(device).map_err(err)?,
            )
            .map_err(err)?,
        })
    }

    fn eval<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
        terms: Vec<(u64, f32)>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner
            .eval(&array1_vec(input), &dense_terms(terms)?)
            .map(|values| values.into_pyarray(py))
            .map_err(err)
    }

    #[getter]
    fn input_size(&self) -> usize {
        self.inner.input_size()
    }

    #[getter]
    fn output_size(&self) -> usize {
        self.inner.output_size()
    }
}

#[pyclass(name = "PackedSearch", unsendable)]
pub struct PyPackedSearch {
    pub(crate) inner: PackedSearch,
    pub(crate) pending: Option<PackedTrial>,
}

#[pymethods]
impl PyPackedSearch {
    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,device="auto"))]
    fn new(
        base: PyReadonlyArray1<'_, u8>,
        base_value: f32,
        leaves: Vec<(usize, usize, u8, f32, f32, f32)>,
        capacity: usize,
        device: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PackedSearch::new(
                &array1_vec(base),
                base_value,
                trial_leaves(leaves)?,
                capacity,
                ComputeDevice::parse(device).map_err(err)?,
            )
            .map_err(err)?,
            pending: None,
        })
    }

    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<(usize, u64, f32)> {
        let trial = self
            .inner
            .ask(
                &array1_vec(seeds),
                SearchConfig {
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                },
            )
            .map_err(err)?;
        self.pending = Some(trial);
        Ok((trial.index, trial.seed, trial.score))
    }

    /// Select a perturbation seed without writing the full candidate row.
    /// The evaluator can regenerate the row from the returned seed.
    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask_lazy(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<(usize, u64, f32)> {
        let trial = self
            .inner
            .ask_lazy(
                &array1_vec(seeds),
                SearchConfig {
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                },
            )
            .map_err(err)?;
        self.pending = Some(trial);
        Ok((trial.index, trial.seed, trial.score))
    }

    fn row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let trial = self
            .pending
            .ok_or_else(|| PyValueError::new_err("there is no pending trial"))?;
        Ok(self.inner.row(trial).map_err(err)?.into_pyarray(py))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_row(&self) -> PyResult<(u64, usize, usize)> {
        let trial = self
            .pending
            .ok_or_else(|| PyValueError::new_err("there is no pending trial"))?;
        self.inner.device_row(trial).map_err(err)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn bind_pending_model(
        &mut self,
        mut model: PyRefMut<'_, crate::py_experimental::PyNativeKdaModel>,
    ) -> PyResult<()> {
        let trial = self
            .pending
            .ok_or_else(|| PyValueError::new_err("there is no pending trial"))?;
        self.inner.materialize_pending(trial).map_err(err)?;
        model
            .inner
            .bind_pending_search(&self.inner, trial)
            .map_err(err)?;
        model.inner.prepare_candidate(0).map_err(err)
    }

    fn tell(&mut self, value: f32, accept: bool) -> PyResult<()> {
        let trial = self
            .pending
            .ok_or_else(|| PyValueError::new_err("there is no pending trial"))?;
        self.inner.tell(trial, value, accept).map_err(err)?;
        self.pending = None;
        Ok(())
    }

    fn replace_history(
        &mut self,
        rows: PyReadonlyArray2<'_, u8>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        self.inner
            .replace_history(&array2_vec(&rows), &array1_vec(values))
            .map_err(err)
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }

    #[getter]
    fn history_capacity(&self) -> usize {
        self.inner.history_capacity()
    }

    #[getter]
    fn row_bytes(&self) -> usize {
        self.inner.row_bytes()
    }
}

#[pyclass(name = "PackedTurbo", unsendable)]
pub struct PyPackedTurbo {
    inner: PackedTurbo,
    pending: Vec<CoreTrial>,
}

#[pyclass(name = "TurboTrial", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTurboTrial {
    inner: CoreTrial,
}

#[pymethods]
impl PyTurboTrial {
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    #[getter]
    fn length(&self) -> f32 {
        self.inner.length
    }

    #[getter]
    fn probability(&self) -> f32 {
        self.inner.probability
    }
}

#[pymethods]
impl PyPackedTurbo {
    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,device="auto",num_pert=20,length_init=0.8,length_min=0.0078125,length_max=1.6,max_pending=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base: PyReadonlyArray1<'_, u8>,
        base_value: f32,
        leaves: Vec<(usize, usize, u8, f32, f32, f32)>,
        capacity: usize,
        device: &str,
        num_pert: usize,
        length_init: f64,
        length_min: f64,
        length_max: f64,
        max_pending: usize,
    ) -> PyResult<Self> {
        let device = ComputeDevice::parse(device).map_err(err)?;
        Ok(Self {
            inner: PackedTurbo::new_batch(
                &array1_vec(base),
                base_value,
                trial_leaves(leaves)?,
                capacity,
                device,
                num_pert,
                TRLengthConfig::new(length_init, length_min, length_max),
                max_pending,
            )
            .map_err(err)?,
            pending: Vec::with_capacity(max_pending),
        })
    }

    #[pyo3(signature=(seeds,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<(usize, u64, f32, f32, f32)> {
        let trial = self
            .inner
            .ask(
                &array1_vec(seeds),
                SearchConfig {
                    length: 0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                },
            )
            .map_err(err)?;
        self.pending.push(trial);
        Ok((
            trial.index,
            trial.seed,
            trial.score,
            trial.length,
            trial.probability,
        ))
    }

    #[pyo3(signature=(seeds,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask_batch(
        &mut self,
        seeds: PyReadonlyArray2<'_, u64>,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<Vec<PyTurboTrial>> {
        let shape = seeds.shape();
        let arms = shape[0];
        let trials = self
            .inner
            .ask_batch(
                &array2_vec(&seeds),
                arms,
                SearchConfig {
                    length: 0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                },
            )
            .map_err(err)?;
        self.pending.extend(trials.iter().copied());
        Ok(trials
            .into_iter()
            .map(|inner| PyTurboTrial { inner })
            .collect())
    }

    fn row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let trial = self.only_pending()?;
        Ok(self.inner.row(trial).map_err(err)?.into_pyarray(py))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_row(&self) -> PyResult<(u64, usize, usize)> {
        let trial = self.only_pending()?;
        self.inner.device_row(trial).map_err(err)
    }

    fn row_trial<'py>(
        &self,
        py: Python<'py>,
        trial: PyRef<'_, PyTurboTrial>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        Ok(self.inner.row(trial.inner).map_err(err)?.into_pyarray(py))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_trial(&self, trial: PyRef<'_, PyTurboTrial>) -> PyResult<(u64, usize, usize)> {
        self.inner.device_row(trial.inner).map_err(err)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_batch(
        &self,
        trials: Vec<PyRef<'_, PyTurboTrial>>,
    ) -> PyResult<Vec<(u64, usize, usize)>> {
        let inner = trials.iter().map(|trial| trial.inner).collect::<Vec<_>>();
        self.inner.device_batch(&inner).map_err(err)
    }

    fn tell(&mut self, value: f32) -> PyResult<bool> {
        let trial = self.only_pending()?;
        let accepted = self.inner.tell(trial, value).map_err(err)?;
        self.pending.clear();
        Ok(accepted)
    }

    fn tell_trial(&mut self, trial: PyRef<'_, PyTurboTrial>, value: f32) -> PyResult<bool> {
        let accepted = self.inner.tell(trial.inner, value).map_err(err)?;
        self.pending.retain(|candidate| *candidate != trial.inner);
        Ok(accepted)
    }

    fn tell_batch(
        &mut self,
        trials: Vec<PyRef<'_, PyTurboTrial>>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Vec<bool>> {
        let inner = trials.iter().map(|trial| trial.inner).collect::<Vec<_>>();
        let accepted = self
            .inner
            .tell_batch(&inner, &array1_vec(values))
            .map_err(err)?;
        self.pending.retain(|candidate| !inner.contains(candidate));
        Ok(accepted)
    }

    #[getter]
    fn length(&self) -> f64 {
        self.inner.length()
    }

    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    #[getter]
    fn best(&self) -> f32 {
        self.inner.best()
    }

    #[getter]
    fn restarts(&self) -> usize {
        self.inner.restarts()
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }
}

impl PyPackedTurbo {
    fn only_pending(&self) -> PyResult<CoreTrial> {
        match self.pending.as_slice() {
            [trial] => Ok(*trial),
            [] => Err(PyValueError::new_err("there is no pending trial")),
            _ => Err(PyValueError::new_err(
                "multiple trials are outstanding; use the trial-specific method",
            )),
        }
    }
}

#[pyclass(name = "BpannHistory", unsendable)]
pub struct PyBpannHistory {
    inner: BpannHistory,
}

#[pymethods]
impl PyBpannHistory {
    #[new]
    fn new(work_dir: String, descriptor_dim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BpannHistory::new(work_dir.into(), descriptor_dim).map_err(err)?,
        })
    }

    fn append(&mut self, descriptor: PyReadonlyArray1<'_, f64>, value: f32) -> PyResult<u64> {
        self.inner
            .append(&descriptor.as_array(), value)
            .map(|id| id.0)
            .map_err(err)
    }

    fn search(
        &self,
        descriptors: PyReadonlyArray2<'_, f64>,
        neighbors: usize,
    ) -> PyResult<Vec<Vec<u64>>> {
        self.inner
            .search(&descriptors.as_array(), neighbors)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.into_iter().map(|id| id.0).collect())
                    .collect()
            })
            .map_err(err)
    }

    fn shortlist(
        &self,
        descriptors: PyReadonlyArray2<'_, f64>,
        neighbors_per_candidate: usize,
        max_observations: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        self.inner
            .shortlist(
                &descriptors.as_array(),
                neighbors_per_candidate,
                max_observations,
            )
            .map(|items| {
                items
                    .into_iter()
                    .map(|item| (item.id.0, item.value))
                    .collect()
            })
            .map_err(err)
    }

    fn sync(&mut self) -> PyResult<()> {
        self.inner.sync().map_err(err)
    }

    fn persist(&mut self) -> PyResult<()> {
        self.inner.persist().map_err(err)
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn descriptor_dim(&self) -> usize {
        self.inner.descriptor_dim()
    }
}

fn weight_ucb<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<WeightBlock>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    let observation_count = observations.as_array().nrows();
    let candidate_count = candidates.as_array().nrows();
    let observation_bytes = array2_vec(&observations);
    let candidate_bytes = array2_vec(&candidates);
    let outcome_vec = array1_vec(outcomes);
    let result = select_weights(
        &observation_bytes,
        observation_count,
        &outcome_vec,
        &candidate_bytes,
        candidate_count,
        &blocks,
        WeightSelectConfig {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            y_scale,
            beta,
            acquisition: AcquisitionKind::Ucb,
            seed: 0,
            device: ComputeDevice::parse(device).map_err(err)?,
        },
    )
    .map_err(err)?;
    let row_bytes = candidates.as_array().ncols();
    let start = result.index * row_bytes;
    let selected = candidate_bytes[start..start + row_bytes].to_vec();
    Ok((selected.into_pyarray(py), result.index, result.score))
}

#[pyfunction(name = "weight_int4_select_ucb")]
#[pyo3(signature=(observations,outcomes,candidates,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,device="auto"))]
pub fn weight_int4_select_ucb_py<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<(usize, usize, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    weight_ucb(
        py,
        observations,
        outcomes,
        candidates,
        int4_blocks(blocks)?,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        device,
    )
}

#[pyfunction(name = "weight_select_ucb")]
#[pyo3(signature=(observations,outcomes,candidates,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,device="auto"))]
pub fn weight_select_ucb_py<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    weight_ucb(
        py,
        observations,
        outcomes,
        candidates,
        mixed_blocks(blocks)?,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        device,
    )
}

#[pyfunction(name = "sparse_union")]
pub fn sparse_union_py<'py>(
    py: Python<'py>,
    rows: Vec<PyReadonlyArray1<'_, u32>>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let owned: Vec<Vec<u32>> = rows.into_iter().map(array1_vec).collect();
    let refs: Vec<&[u32]> = owned.iter().map(Vec::as_slice).collect();
    Ok(sparse_union(&refs).into_pyarray(py))
}

#[pyfunction(name = "sparse_xor")]
pub fn sparse_xor_py<'py>(
    py: Python<'py>,
    left_words: PyReadonlyArray1<'_, u32>,
    left_masks: PyReadonlyArray1<'_, u32>,
    right_words: PyReadonlyArray1<'_, u32>,
    right_masks: PyReadonlyArray1<'_, u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (words, masks) = sparse_xor(
        &array1_vec(left_words),
        &array1_vec(left_masks),
        &array1_vec(right_words),
        &array1_vec(right_masks),
    )
    .map_err(err)?;
    Ok((words.into_pyarray(py), masks.into_pyarray(py)))
}

#[pyfunction(name = "sparse_missing")]
pub fn sparse_missing_py<'py>(
    py: Python<'py>,
    cached: PyReadonlyArray1<'_, u32>,
    query: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(missing_words(&array1_vec(cached), &array1_vec(query)).into_pyarray(py))
}

#[pyfunction(name = "sparse_merge")]
pub fn sparse_merge_py<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    extra_words: PyReadonlyArray1<'_, u32>,
    extra_values: PyReadonlyArray1<'_, u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (words, values) = merge_values(
        &array1_vec(words),
        &array1_vec(values),
        &array1_vec(extra_words),
        &array1_vec(extra_values),
    )
    .map_err(err)?;
    Ok((words.into_pyarray(py), values.into_pyarray(py)))
}

#[pyfunction(name = "sparse_take")]
pub fn sparse_take_py<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    query: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(
        take_words(&array1_vec(words), &array1_vec(values), &array1_vec(query))
            .map_err(err)?
            .into_pyarray(py),
    )
}

#[pyfunction(name = "sparse_apply")]
pub fn sparse_apply_py<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    move_words: PyReadonlyArray1<'_, u32>,
    move_masks: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(apply_sparse(
        &array1_vec(words),
        &array1_vec(values),
        &array1_vec(move_words),
        &array1_vec(move_masks),
    )
    .map_err(err)?
    .into_pyarray(py))
}

#[pyfunction(name = "sparse_blocks")]
pub fn sparse_blocks_py(
    words: PyReadonlyArray1<'_, u32>,
    word_ends: PyReadonlyArray1<'_, u32>,
    widths: PyReadonlyArray1<'_, u8>,
) -> PyResult<(Vec<(usize, usize, u8)>, usize)> {
    blocks_for_words(
        &array1_vec(words),
        &array1_vec(word_ends),
        &array1_vec(widths),
    )
    .map_err(err)
}

#[pyfunction(name = "sparse_draw")]
#[pyo3(signature=(count,size,dimension,parameter_ends,parameter_starts,word_offsets,widths,seed))]
pub fn sparse_draw_py<'py>(
    py: Python<'py>,
    count: usize,
    size: usize,
    dimension: u64,
    parameter_ends: PyReadonlyArray1<'_, u64>,
    parameter_starts: PyReadonlyArray1<'_, u64>,
    word_offsets: PyReadonlyArray1<'_, u32>,
    widths: PyReadonlyArray1<'_, u8>,
    seed: u64,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let rows = draw_sparse(
        count,
        size,
        dimension,
        &array1_vec(parameter_starts),
        &array1_vec(parameter_ends),
        &array1_vec(word_offsets),
        &array1_vec(widths),
        seed,
    )
    .map_err(err)?;
    let word_rows = PyList::empty(py);
    let mask_rows = PyList::empty(py);
    for (words, masks) in rows {
        word_rows.append(words.into_pyarray(py))?;
        mask_rows.append(masks.into_pyarray(py))?;
    }
    Ok((word_rows, mask_rows))
}

#[allow(clippy::too_many_arguments)]
fn sparse_select_impl(
    base: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    observation_words: Vec<PyReadonlyArray1<'_, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'_, u32>>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidate_words: Vec<PyReadonlyArray1<'_, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'_, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    acquisition: &str,
    seed: u64,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(usize, f32)> {
    if observation_words.len() != observation_masks.len()
        || candidate_words.len() != candidate_masks.len()
    {
        return Err(PyValueError::new_err(
            "sparse word and mask row counts must match",
        ));
    }
    let base = array1_vec(base);
    let indices = array1_vec(indices);
    let observation_words: Vec<Vec<u32>> = observation_words.into_iter().map(array1_vec).collect();
    let observation_masks: Vec<Vec<u32>> = observation_masks.into_iter().map(array1_vec).collect();
    let candidate_words: Vec<Vec<u32>> = candidate_words.into_iter().map(array1_vec).collect();
    let candidate_masks: Vec<Vec<u32>> = candidate_masks.into_iter().map(array1_vec).collect();
    let observation_bytes =
        pack_sparse_rows(&base, &indices, &observation_words, &observation_masks)?;
    let candidate_bytes = pack_sparse_rows(&base, &indices, &candidate_words, &candidate_masks)?;
    let outcome_vec = array1_vec(outcomes);
    let result = select_weights(
        &observation_bytes,
        observation_words.len(),
        &outcome_vec,
        &candidate_bytes,
        candidate_words.len(),
        &mixed_blocks(blocks)?,
        WeightSelectConfig {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            y_scale,
            beta,
            acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
            seed,
            device: ComputeDevice::parse(device).map_err(err)?,
        },
    )
    .map_err(err)?;
    Ok((result.index, result.score))
}

#[pyfunction(name = "sparse_select")]
#[pyo3(signature=(base,indices,observation_words,observation_masks,outcomes,candidate_words,candidate_masks,blocks,acquisition,seed,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_select_py(
    base: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    observation_words: Vec<PyReadonlyArray1<'_, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'_, u32>>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidate_words: Vec<PyReadonlyArray1<'_, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'_, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    acquisition: &str,
    seed: u64,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(usize, f32)> {
    sparse_select_impl(
        base,
        indices,
        observation_words,
        observation_masks,
        outcomes,
        candidate_words,
        candidate_masks,
        blocks,
        acquisition,
        seed,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        device,
    )
}

#[pyfunction(name = "sparse_select_ucb")]
#[pyo3(signature=(base,indices,observation_words,observation_masks,outcomes,candidate_words,candidate_masks,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_select_ucb_py(
    base: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    observation_words: Vec<PyReadonlyArray1<'_, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'_, u32>>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidate_words: Vec<PyReadonlyArray1<'_, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'_, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
) -> PyResult<(usize, f32)> {
    sparse_select_impl(
        base,
        indices,
        observation_words,
        observation_masks,
        outcomes,
        candidate_words,
        candidate_masks,
        blocks,
        "ucb",
        0,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        "auto",
    )
}

fn pack_sparse_rows(
    base: &[u32],
    indices: &[u32],
    words: &[Vec<u32>],
    masks: &[Vec<u32>],
) -> PyResult<Vec<u8>> {
    if base.len() != indices.len() {
        return Err(PyValueError::new_err(
            "base and sparse index arrays must have the same length",
        ));
    }
    let mut bytes = Vec::with_capacity(words.len() * base.len() * std::mem::size_of::<u32>());
    for (row_words, row_masks) in words.iter().zip(masks) {
        if row_words.len() != row_masks.len() {
            return Err(PyValueError::new_err(
                "sparse row words and masks must have the same length",
            ));
        }
        let mut row = base.to_vec();
        for (&word, &mask) in row_words.iter().zip(row_masks) {
            let position = indices
                .binary_search(&word)
                .map_err(|_| PyValueError::new_err(format!("sparse word {word} is not indexed")))?;
            row[position] ^= mask;
        }
        for value in row {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(bytes)
}
