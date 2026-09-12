//! Python binding for the resident Metal Qwen2 dense evaluator.

use std::path::Path;

use ennx::experimental::MetalQwenEvaluator;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict};

use crate::py_metal::PyMetalWeights;

fn err(message: String) -> PyErr {
    PyValueError::new_err(message)
}

#[pyclass(name = "MetalQwenEvaluator", module = "ennx.experimental", unsendable)]
pub struct PyMetalQwenEvaluator {
    pub(crate) inner: MetalQwenEvaluator,
}

#[pymethods]
impl PyMetalQwenEvaluator {
    #[new]
    fn new(max_tokens: &Bound<'_, PyAny>) -> PyResult<Self> {
        if max_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_tokens must be an integer, not bool",
            ));
        }
        let max_tokens = max_tokens
            .extract::<u32>()
            .map_err(|_| PyTypeError::new_err("max_tokens must be a nonnegative integer"))?;
        Ok(Self {
            inner: MetalQwenEvaluator::new(max_tokens).map_err(err)?,
        })
    }

    #[getter]
    fn weights_len(&self) -> usize {
        self.inner.weights_len()
    }

    #[getter]
    fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }

    #[getter]
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }

    fn load_weights(&mut self, path: String) -> PyResult<PyMetalWeights> {
        let buffer = self.inner.load_weights(Path::new(&path)).map_err(err)?;
        Ok(PyMetalWeights::owned(buffer, self.inner.weights_len()))
    }

    fn blocks(&self) -> Vec<(u64, usize, usize, f32, f32)> {
        self.inner.blocks()
    }

    fn logits(&mut self, weights: &Bound<'_, PyAny>, tokens: Vec<i32>) -> PyResult<Vec<Vec<f32>>> {
        let buffer = crate::py_metal::weight_buffer(weights)?;
        let values = self.inner.logits(&buffer, &tokens).map_err(err)?;
        Ok(values.chunks(self.inner.vocab()).map(Vec::from).collect())
    }

    fn next_logits(&mut self, weights: &Bound<'_, PyAny>, tokens: Vec<i32>) -> PyResult<Vec<f32>> {
        let buffer = crate::py_metal::weight_buffer(weights)?;
        self.inner.next_logits(&buffer, &tokens).map_err(err)
    }

    #[pyo3(signature=(weights,tokens,max_new_tokens))]
    fn generate(
        &mut self,
        weights: &Bound<'_, PyAny>,
        tokens: Vec<i32>,
        max_new_tokens: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<i32>> {
        if max_new_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_new_tokens must be an integer, not bool",
            ));
        }
        let max_new_tokens = max_new_tokens
            .extract::<usize>()
            .map_err(|_| PyTypeError::new_err("max_new_tokens must be positive"))?;
        let buffer = crate::py_metal::weight_buffer(weights)?;
        self.inner
            .generate(&buffer, &tokens, max_new_tokens)
            .map_err(err)
    }

    #[pyo3(signature=(weights,tokens,max_new_tokens,mode="greedy"))]
    fn bench_generate<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'_, PyAny>,
        tokens: Vec<i32>,
        max_new_tokens: &Bound<'_, PyAny>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if max_new_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_new_tokens must be an integer, not bool",
            ));
        }
        let max_new_tokens = max_new_tokens
            .extract::<usize>()
            .map_err(|_| PyTypeError::new_err("max_new_tokens must be positive"))?;
        let buffer = crate::py_metal::weight_buffer(weights)?;
        let profile = self
            .inner
            .bench_generate(&buffer, &tokens, max_new_tokens, mode)
            .map_err(err)?;
        let dict = PyDict::new(py);
        dict.set_item("prompt_tokens", profile.prompt_tokens)?;
        dict.set_item(
            "requested_generated_tokens",
            profile.requested_generated_tokens,
        )?;
        dict.set_item("generated_tokens", profile.generated_tokens)?;
        dict.set_item("prefill_ms", profile.prefill_ms)?;
        dict.set_item("first_token_ms", profile.first_token_ms)?;
        dict.set_item("decode_ms_after_first", profile.decode_ms_after_first)?;
        dict.set_item("total_ms", profile.total_ms)?;
        dict.set_item("tiled_attention", profile.tile_attn)?;
        dict.set_item(
            "steady_decode_tokens_per_second",
            profile.steady_decode_tokens_per_second,
        )?;
        dict.set_item(
            "end_to_end_generated_tokens_per_second",
            profile.end_to_end_generated_tokens_per_second,
        )?;
        dict.set_item("host_overhead_ms", profile.host_overhead_ms)?;
        dict.set_item("command_buffer_count", profile.command_buffer_count)?;
        dict.set_item("logits_read_to_cpu", profile.logits_read_to_cpu)?;
        dict.set_item("token_selection_on_gpu", profile.token_selection_on_gpu)?;
        dict.set_item("kv_cache_bytes", profile.kv_cache_bytes)?;
        dict.set_item("device_name", profile.device_name)?;
        dict.set_item("max_tokens", profile.max_tokens)?;
        dict.set_item("decode_kernel_trace", profile.decode_kernel_trace)?;
        dict.set_item("decode_kernel_times_ms", profile.decode_kernel_times_ms)?;
        dict.set_item(
            "decode_bottleneck_candidates",
            profile.decode_bottleneck_candidates,
        )?;
        dict.set_item("lm_head_argmax_decision", profile.lm_head_argmax_decision)?;
        Ok(dict)
    }

    #[pyo3(signature=(weights,prompts,max_new_tokens))]
    fn generate_batch(
        &mut self,
        weights: &Bound<'_, PyAny>,
        prompts: Vec<Vec<i32>>,
        max_new_tokens: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<i32>>> {
        if max_new_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_new_tokens must be an integer, not bool",
            ));
        }
        let max_new_tokens = max_new_tokens
            .extract::<usize>()
            .map_err(|_| PyTypeError::new_err("max_new_tokens must be positive"))?;
        let buffer = crate::py_metal::weight_buffer(weights)?;
        self.inner
            .generate_batch(&buffer, &prompts, max_new_tokens)
            .map_err(err)
    }

    #[pyo3(name = "generate_sampled", signature=(weights,tokens,max_new_tokens,temperature,top_p,top_k,seed))]
    fn sample(
        &mut self,
        weights: &Bound<'_, PyAny>,
        tokens: Vec<i32>,
        max_new_tokens: &Bound<'_, PyAny>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    ) -> PyResult<Vec<i32>> {
        if max_new_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_new_tokens must be an integer, not bool",
            ));
        }
        let max_new_tokens = max_new_tokens
            .extract::<usize>()
            .map_err(|_| PyTypeError::new_err("max_new_tokens must be positive"))?;
        let buffer = crate::py_metal::weight_buffer(weights)?;
        self.inner
            .sample(
                &buffer,
                &tokens,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                seed,
            )
            .map_err(err)
    }

    #[pyo3(name = "generate_sampled_batch", signature=(weights,prompts,max_new_tokens,temperature,top_p,top_k,seeds))]
    fn sample_batch(
        &mut self,
        weights: &Bound<'_, PyAny>,
        prompts: Vec<Vec<i32>>,
        max_new_tokens: &Bound<'_, PyAny>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seeds: Vec<u64>,
    ) -> PyResult<Vec<Vec<i32>>> {
        if max_new_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_new_tokens must be an integer, not bool",
            ));
        }
        let max_new_tokens = max_new_tokens
            .extract::<usize>()
            .map_err(|_| PyTypeError::new_err("max_new_tokens must be positive"))?;
        let buffer = crate::py_metal::weight_buffer(weights)?;
        self.inner
            .sample_batch(
                &buffer,
                &prompts,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                &seeds,
            )
            .map_err(err)
    }

    fn losses(
        &mut self,
        weights: &Bound<'_, PyAny>,
        tokens: Vec<Vec<i32>>,
        masks: Vec<Vec<bool>>,
    ) -> PyResult<Vec<f32>> {
        let buffer = crate::py_metal::weight_buffer(weights)?;
        self.inner.losses(&buffer, &tokens, &masks).map_err(err)
    }

    #[getter]
    #[pyo3(name = "last_loss_profile")]
    fn loss_profile<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(profile) = self.inner.loss_profile() else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("rows", profile.rows)?;
        dict.set_item("tokens", profile.tokens)?;
        dict.set_item("scored_tokens", profile.scored_tokens)?;
        dict.set_item("cached_tokens", profile.cached_tokens)?;
        dict.set_item("kv_cache_bytes", profile.kv_cache_bytes)?;
        dict.set_item("write_ms", profile.write_ms)?;
        dict.set_item("forward_ms", profile.forward_ms)?;
        dict.set_item("output_ms", profile.output_ms)?;
        dict.set_item("total_ms", profile.total_ms)?;
        if let Some(stages) = profile.stages {
            let detail = PyDict::new(py);
            detail.set_item("embedding_gpu_ms", stages.embed_ms)?;
            detail.set_item("qkv_gpu_ms", stages.qkv_ms)?;
            detail.set_item("attention_gpu_ms", stages.attn_ms)?;
            detail.set_item("attention_output_gpu_ms", stages.attn_out_ms)?;
            detail.set_item("mlp_expand_gpu_ms", stages.mlp_expand_ms)?;
            detail.set_item("mlp_reduce_gpu_ms", stages.mlp_reduce_ms)?;
            detail.set_item("final_norm_gpu_ms", stages.final_norm_ms)?;
            detail.set_item("total_gpu_ms", stages.gpu_ms)?;
            detail.set_item("command_buffers", stages.commands)?;
            detail.set_item("missing_timestamps", stages.missing)?;
            detail.set_item("chunks", stages.chunks)?;
            dict.set_item("stages", detail)?;
        }
        Ok(Some(dict))
    }
}
