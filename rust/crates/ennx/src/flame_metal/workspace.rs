//! Host access to private tracked workspace buffers.

use super::*;

impl FlameEvaluator {
    pub(super) fn b(&self, w: W) -> &Buffer {
        &self.workspace[w as usize]
    }

    pub(super) fn write<T: Copy>(&self, w: W, values: &[T]) {
        assert!(std::mem::size_of_val(values) as u64 <= self.b(w).length());
        // Workspace is private, shared-storage, and every previous call waited.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                self.b(w).contents().cast(),
                std::mem::size_of_val(values),
            );
        }
    }

    pub(super) fn read<T: Copy>(&self, w: W, count: usize) -> Vec<T> {
        assert!(count * size_of::<T>() <= self.b(w).length() as usize);
        // Only called after waiting for GPU completion, never for borrowed weights.
        unsafe { std::slice::from_raw_parts(self.b(w).contents().cast::<T>(), count).to_vec() }
    }

    pub(super) fn shape(&self, rows: u32) -> Shape {
        Shape {
            rows,
            width: self.config.width,
            heads: self.config.heads,
            experts: self.config.experts,
            top_k: self.config.top_k,
            epsilon: self.config.epsilon,
            rope_base: self.config.rope_base,
            ..Shape::default()
        }
    }
}
