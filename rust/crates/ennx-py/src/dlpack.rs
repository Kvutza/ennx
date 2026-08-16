use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type PyObject = Py<PyAny>;

const CUDA: i32 = 2;
const FLOAT: u8 = 2;
const BF16: u8 = 4;
const READ_ONLY: u64 = 1;
const LEGACY: &[u8] = b"dltensor\0";
const LEGACY_USED: &[u8] = b"used_dltensor\0";
const VERSIONED: &[u8] = b"dltensor_versioned\0";
const VERSIONED_USED: &[u8] = b"used_dltensor_versioned\0";

#[repr(C)]
#[derive(Clone, Copy)]
struct Version {
    major: u32,
    minor: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Device {
    kind: i32,
    index: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct Tensor {
    data: *mut c_void,
    device: Device,
    ndim: i32,
    dtype: DataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct Managed {
    tensor: Tensor,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Managed)>,
}

#[repr(C)]
struct ManagedV1 {
    version: Version,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut ManagedV1)>,
    flags: u64,
    tensor: Tensor,
}

const _: [(); 48] = [(); std::mem::size_of::<Tensor>()];
const _: [(); 64] = [(); std::mem::size_of::<Managed>()];
const _: [(); 80] = [(); std::mem::size_of::<ManagedV1>()];

enum InputKind {
    Legacy(*mut Managed),
    Versioned(*mut ManagedV1),
}

pub(crate) struct Input {
    capsule: PyObject,
    kind: InputKind,
    pub(crate) pointer: u64,
    pub(crate) len: usize,
}

impl Input {
    pub(crate) fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::read(source, BF16, 16, "BF16")
    }

    pub(crate) fn f32(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::read(source, FLOAT, 32, "FP32")
    }

    fn read(source: &Bound<'_, PyAny>, code: u8, bits: u8, name: &str) -> PyResult<Self> {
        let kwargs = PyDict::new(source.py());
        kwargs.set_item("stream", 1_u64)?;
        kwargs.set_item("max_version", (1_u32, 3_u32))?;
        let capsule = match source.call_method("__dlpack__", (), Some(&kwargs)) {
            Ok(capsule) => capsule,
            Err(error) if error.is_instance_of::<PyTypeError>(source.py()) => {
                let kwargs = PyDict::new(source.py());
                kwargs.set_item("stream", 1_u64)?;
                source.call_method("__dlpack__", (), Some(&kwargs))?
            }
            Err(error) => return Err(error),
        };
        let (kind, tensor) = unsafe {
            if ffi::PyCapsule_IsValid(capsule.as_ptr(), LEGACY.as_ptr().cast::<c_char>()) != 0 {
                let managed =
                    ffi::PyCapsule_GetPointer(capsule.as_ptr(), LEGACY.as_ptr().cast::<c_char>())
                        .cast::<Managed>();
                (InputKind::Legacy(managed), &(*managed).tensor)
            } else if ffi::PyCapsule_IsValid(capsule.as_ptr(), VERSIONED.as_ptr().cast::<c_char>())
                != 0
            {
                let managed = ffi::PyCapsule_GetPointer(
                    capsule.as_ptr(),
                    VERSIONED.as_ptr().cast::<c_char>(),
                )
                .cast::<ManagedV1>();
                if (*managed).version.major != 1 || (*managed).version.minor > 3 {
                    return Err(PyBufferError::new_err("unsupported DLPack major version"));
                }
                (InputKind::Versioned(managed), &(*managed).tensor)
            } else {
                return Err(PyBufferError::new_err("invalid DLPack capsule"));
            }
        };
        validate_tensor(tensor, code, bits, name)?;
        let dimensions = usize::try_from(tensor.ndim)
            .map_err(|_| PyValueError::new_err("invalid DLPack rank"))?;
        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, dimensions) };
        if !is_contiguous(tensor, shape) {
            return Err(PyBufferError::new_err(format!(
                "{name} input requires a contiguous DLPack tensor"
            )));
        }
        let len = shape.iter().try_fold(1usize, |size, &dimension| {
            let dimension = usize::try_from(dimension)
                .map_err(|_| PyValueError::new_err("invalid DLPack shape"))?;
            size.checked_mul(dimension)
                .ok_or_else(|| PyValueError::new_err("DLPack shape overflow"))
        })?;
        if len == 0 {
            return Err(PyValueError::new_err(format!(
                "{name} input requires a non-empty tensor"
            )));
        }
        let pointer = (tensor.data as u64)
            .checked_add(tensor.byte_offset)
            .ok_or_else(|| PyValueError::new_err("DLPack pointer overflow"))?;
        Ok(Self {
            capsule: capsule.unbind(),
            kind,
            pointer,
            len,
        })
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        Python::attach(|py| unsafe {
            let capsule = self.capsule.bind(py);
            let result = match self.kind {
                InputKind::Legacy(managed) => {
                    let result = ffi::PyCapsule_SetName(
                        capsule.as_ptr(),
                        LEGACY_USED.as_ptr().cast::<c_char>(),
                    );
                    if result == 0 {
                        if let Some(deleter) = (*managed).deleter {
                            deleter(managed);
                        }
                    }
                    result
                }
                InputKind::Versioned(managed) => {
                    let result = ffi::PyCapsule_SetName(
                        capsule.as_ptr(),
                        VERSIONED_USED.as_ptr().cast::<c_char>(),
                    );
                    if result == 0 {
                        if let Some(deleter) = (*managed).deleter {
                            deleter(managed);
                        }
                    }
                    result
                }
            };
            if result != 0 {
                ffi::PyErr_Clear();
            }
        });
    }
}

struct LegacyOwner {
    managed: Managed,
    shape: i64,
    _owner: PyObject,
    lease: Lease,
}

struct VersionOwner {
    managed: ManagedV1,
    shape: i64,
    stride: i64,
    _owner: PyObject,
    lease: Lease,
}

#[derive(Clone)]
enum Lease {
    Flag(Arc<AtomicBool>),
    Count(Arc<AtomicUsize>),
}

impl Lease {
    fn release(&self) {
        match self {
            Self::Flag(lease) => lease.store(false, Ordering::Release),
            Self::Count(lease) => {
                lease.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
}

unsafe extern "C" fn release_legacy(managed: *mut Managed) {
    if managed.is_null() {
        return;
    }
    let context = unsafe { (*managed).context.cast::<LegacyOwner>() };
    if !context.is_null() {
        Python::attach(|_| {
            let owner = unsafe { Box::from_raw(context) };
            owner.lease.release();
            drop(owner);
        });
    }
}

unsafe extern "C" fn release_versioned(managed: *mut ManagedV1) {
    if managed.is_null() {
        return;
    }
    let context = unsafe { (*managed).context.cast::<VersionOwner>() };
    if !context.is_null() {
        Python::attach(|_| {
            let owner = unsafe { Box::from_raw(context) };
            owner.lease.release();
            drop(owner);
        });
    }
}

unsafe extern "C" fn drop_legacy(capsule: *mut ffi::PyObject) {
    if unsafe { ffi::PyCapsule_IsValid(capsule, LEGACY.as_ptr().cast::<c_char>()) } == 0 {
        return;
    }
    let managed = unsafe {
        ffi::PyCapsule_GetPointer(capsule, LEGACY.as_ptr().cast::<c_char>()).cast::<Managed>()
    };
    if !managed.is_null() {
        if let Some(deleter) = unsafe { (*managed).deleter } {
            unsafe { deleter(managed) };
        }
    }
}

unsafe extern "C" fn drop_versioned(capsule: *mut ffi::PyObject) {
    if unsafe { ffi::PyCapsule_IsValid(capsule, VERSIONED.as_ptr().cast::<c_char>()) } == 0 {
        return;
    }
    let managed = unsafe {
        ffi::PyCapsule_GetPointer(capsule, VERSIONED.as_ptr().cast::<c_char>()).cast::<ManagedV1>()
    };
    if !managed.is_null() {
        if let Some(deleter) = unsafe { (*managed).deleter } {
            unsafe { deleter(managed) };
        }
    }
}

pub(crate) fn export(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicBool>,
    pointer: u64,
    len: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    export_with(py, owner, Lease::Flag(lease), pointer, len, max_version)
}

pub(crate) fn export_count(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicUsize>,
    pointer: u64,
    len: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    export_with(py, owner, Lease::Count(lease), pointer, len, max_version)
}

fn export_with(
    py: Python<'_>,
    owner: PyObject,
    lease: Lease,
    pointer: u64,
    len: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    if let Some((major, minor)) = max_version.filter(|version| version.0 >= 1) {
        let version = Version {
            major: 1,
            minor: if major == 1 { minor.min(3) } else { 3 },
        };
        export_versioned(py, owner, lease, pointer, len, version)
    } else {
        export_legacy(py, owner, lease, pointer, len)
    }
}

fn export_legacy(
    py: Python<'_>,
    owner: PyObject,
    lease: Lease,
    pointer: u64,
    len: usize,
) -> PyResult<PyObject> {
    let shape = i64::try_from(len).map_err(|_| PyValueError::new_err("BF16 length exceeds i64"))?;
    let mut state = Box::new(LegacyOwner {
        managed: Managed {
            tensor: tensor(pointer, std::ptr::null_mut(), std::ptr::null_mut()),
            context: std::ptr::null_mut(),
            deleter: Some(release_legacy),
        },
        shape,
        _owner: owner,
        lease,
    });
    state.managed.tensor.shape = &mut state.shape;
    state.managed.context = (&mut *state as *mut LegacyOwner).cast();
    let managed = &mut state.managed as *mut Managed;
    let _state = Box::into_raw(state);
    make_capsule(py, managed.cast(), LEGACY, drop_legacy, || unsafe {
        release_legacy(managed)
    })
}

fn export_versioned(
    py: Python<'_>,
    owner: PyObject,
    lease: Lease,
    pointer: u64,
    len: usize,
    version: Version,
) -> PyResult<PyObject> {
    let shape = i64::try_from(len).map_err(|_| PyValueError::new_err("BF16 length exceeds i64"))?;
    let mut state = Box::new(VersionOwner {
        managed: ManagedV1 {
            version,
            context: std::ptr::null_mut(),
            deleter: Some(release_versioned),
            flags: READ_ONLY,
            tensor: tensor(pointer, std::ptr::null_mut(), std::ptr::null_mut()),
        },
        shape,
        stride: 1,
        _owner: owner,
        lease,
    });
    state.managed.tensor.shape = &mut state.shape;
    state.managed.tensor.strides = &mut state.stride;
    state.managed.context = (&mut *state as *mut VersionOwner).cast();
    let managed = &mut state.managed as *mut ManagedV1;
    let _state = Box::into_raw(state);
    make_capsule(py, managed.cast(), VERSIONED, drop_versioned, || unsafe {
        release_versioned(managed)
    })
}

fn tensor(pointer: u64, shape: *mut i64, strides: *mut i64) -> Tensor {
    Tensor {
        data: pointer as *mut c_void,
        device: Device {
            kind: CUDA,
            index: 0,
        },
        ndim: 1,
        dtype: DataType {
            code: BF16,
            bits: 16,
            lanes: 1,
        },
        shape,
        strides,
        byte_offset: 0,
    }
}

fn make_capsule<F>(
    py: Python<'_>,
    pointer: *mut c_void,
    name: &'static [u8],
    destructor: ffi::PyCapsule_Destructor,
    failure: F,
) -> PyResult<PyObject>
where
    F: FnOnce(),
{
    let capsule =
        unsafe { ffi::PyCapsule_New(pointer, name.as_ptr().cast::<c_char>(), Some(destructor)) };
    if capsule.is_null() {
        failure();
        Err(PyErr::fetch(py))
    } else {
        Ok(unsafe { Bound::from_owned_ptr(py, capsule).unbind() })
    }
}

fn validate_tensor(tensor: &Tensor, code: u8, bits: u8, name: &str) -> PyResult<()> {
    if tensor.device.kind != CUDA || tensor.device.index != 0 {
        return Err(PyBufferError::new_err(format!(
            "{name} input requires a CUDA device-0 DLPack tensor"
        )));
    }
    if tensor.dtype.code != code || tensor.dtype.bits != bits || tensor.dtype.lanes != 1 {
        return Err(PyBufferError::new_err(format!(
            "{name} input requires {name} DLPack data"
        )));
    }
    if tensor.ndim <= 0 || tensor.shape.is_null() || tensor.data.is_null() {
        return Err(PyValueError::new_err(format!(
            "{name} input requires a non-empty tensor"
        )));
    }
    Ok(())
}

fn is_contiguous(tensor: &Tensor, shape: &[i64]) -> bool {
    if tensor.strides.is_null() {
        return true;
    }
    let strides = unsafe { std::slice::from_raw_parts(tensor.strides, shape.len()) };
    let mut expected = 1_i64;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        if dimension > 1 && stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dimension.max(1)) else {
            return false;
        };
        expected = next;
    }
    true
}
