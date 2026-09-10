use super::{Search, Trial};

/// Borrowed device buffer containing one trial's encoded parameters.
///
/// The handle is valid until the owning search is mutated or dropped. It is
/// intentionally opaque: callers should request the backend binding they can
/// actually consume instead of pattern-matching on search internals.
/// Complete any external GPU work using the binding before releasing the borrow:
/// submitting work does not extend the lifetime of the trial's buffer slot.
pub struct DeviceView<'a> {
    row_bytes: usize,
    #[allow(dead_code)]
    kind: ViewKind<'a>,
}

#[allow(dead_code)]
enum ViewKind<'a> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda { ptr: u64, stream: usize },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal {
        buffer: ::metal::Buffer,
        offset: usize,
    },
    #[cfg(feature = "opencl")]
    OpenCl {
        context: &'a opencl3::context::Context,
        queue: &'a opencl3::command_queue::CommandQueue,
        buffer: &'a opencl3::memory::Buffer<u8>,
        offset: usize,
    },
    #[doc(hidden)]
    Lifetime(std::marker::PhantomData<&'a ()>),
}

impl<'a> DeviceView<'a> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub(crate) fn cuda(ptr: u64, row_bytes: usize, stream: usize) -> Self {
        Self {
            row_bytes,
            kind: ViewKind::Cuda { ptr, stream },
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal(buffer: ::metal::Buffer, offset: usize, row_bytes: usize) -> Self {
        Self {
            row_bytes,
            kind: ViewKind::Metal { buffer, offset },
        }
    }

    #[cfg(feature = "opencl")]
    pub(crate) fn opencl(
        context: &'a opencl3::context::Context,
        queue: &'a opencl3::command_queue::CommandQueue,
        buffer: &'a opencl3::memory::Buffer<u8>,
        offset: usize,
        row_bytes: usize,
    ) -> Self {
        Self {
            row_bytes,
            kind: ViewKind::OpenCl {
                context,
                queue,
                buffer,
                offset,
            },
        }
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn as_cuda(&self) -> Option<(u64, usize)> {
        match self.kind {
            ViewKind::Cuda { ptr, stream } => Some((ptr, stream)),
            _ => None,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn as_metal(&self) -> Option<(&::metal::Buffer, usize)> {
        match &self.kind {
            ViewKind::Metal { buffer, offset } => Some((buffer, *offset)),
            _ => None,
        }
    }

    #[cfg(feature = "opencl")]
    pub fn as_opencl(
        &self,
    ) -> Option<(
        &'a opencl3::context::Context,
        &'a opencl3::command_queue::CommandQueue,
        &'a opencl3::memory::Buffer<u8>,
        usize,
    )> {
        match &self.kind {
            ViewKind::OpenCl {
                context,
                queue,
                buffer,
                offset,
            } => Some((context, queue, buffer, *offset)),
            _ => None,
        }
    }
}

pub fn device_views<'a>(
    search: &'a Search,
    trials: &[Trial],
) -> Result<Vec<DeviceView<'a>>, String> {
    trials
        .iter()
        .map(|&trial| search.device_view(trial))
        .collect()
}
