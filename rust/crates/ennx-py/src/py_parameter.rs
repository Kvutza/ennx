use ennx::experimental::EncodingType;
use ennx::search::Parameter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Immutable description of a contiguous range of encoded parameters.
#[pyclass(name = "Parameter", module = "ennx.search", frozen)]
pub struct PyParameter {
    pub(crate) inner: Parameter,
}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature=(*,offset,length,encoding="int8",scale=1.0,weight=1.0,radius=1.0))]
    fn new(
        offset: usize,
        length: usize,
        encoding: &str,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> PyResult<Self> {
        let (bits, encoding) = match encoding {
            "int4" => (4, EncodingType::Int4),
            "int8" => (8, EncodingType::Int8),
            "fp4_e2m1" => (4, EncodingType::Fp4E2M1),
            "fp8_e4m3" => (8, EncodingType::Fp8E4M3),
            "fp8_e5m2" => (8, EncodingType::Fp8E5M2),
            _ => return Err(PyValueError::new_err("unsupported parameter encoding")),
        };
        Ok(Self {
            inner: Parameter::encoded(offset, length, bits, encoding, scale, weight, radius)
                .map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }
    #[getter]
    fn length(&self) -> usize {
        self.inner.length
    }
    #[getter]
    fn bits(&self) -> u8 {
        self.inner.bits
    }
    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }
    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }
    #[getter]
    fn radius(&self) -> f32 {
        self.inner.radius
    }
    #[getter]
    fn encoding(&self) -> &'static str {
        match self.inner.encoding {
            EncodingType::Int4 => "int4",
            EncodingType::Int8 => "int8",
            EncodingType::Fp4E2M1 => "fp4_e2m1",
            EncodingType::Fp8E4M3 => "fp8_e4m3",
            EncodingType::Fp8E5M2 => "fp8_e5m2",
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "Parameter(offset={}, length={}, encoding={:?}, scale={}, weight={}, radius={})",
            self.offset(),
            self.length(),
            self.encoding(),
            self.scale(),
            self.weight(),
            self.radius()
        )
    }
}

pub(crate) fn parameters(values: Vec<PyRef<'_, PyParameter>>) -> Vec<Parameter> {
    values.iter().map(|value| value.inner).collect()
}
