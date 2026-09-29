use super::*;

impl QwenEvaluator {
    pub(super) fn new_state(&self) -> Result<GenerationState> {
        self.batch_state(1)
    }

    pub(super) fn batch_state(&self, batch: u32) -> Result<GenerationState> {
        if batch == 0 || batch > self.max_tokens {
            return Err("Qwen generation batch exceeds the evaluator workspace".into());
        }
        let capacity = self
            .max_tokens
            .checked_add(255)
            .ok_or("Qwen generation cache capacity overflow")?
            / 256
            * 256;
        let per_cache = product(&[
            self.config.layers as usize,
            self.config.kv_heads as usize,
            capacity as usize,
            self.config.head_dim() as usize,
        ])?;
        let cache_elements = product(&[per_cache, batch as usize])?;
        let cache_bytes = (per_cache * size_of::<u16>()) as u64;
        let total_bytes = cache_bytes
            .checked_mul(u64::from(batch))
            .and_then(|value| value.checked_mul(2))
            .ok_or("Qwen generation cache size overflow")?;
        if total_bytes > self.runtime.device.max_buffer_length() {
            return Err("Qwen generation cache exceeds Metal maxBufferLength".into());
        }
        let limit = self.runtime.device.recommended_max_working_set_size();
        let allocated = self.runtime.device.current_allocated_size();
        let budget = limit.saturating_sub(limit / 10);
        if allocated
            .checked_add(total_bytes)
            .is_none_or(|value| value > budget)
        {
            return Err("Qwen generation cache exceeds the conservative working-set budget".into());
        }
        let key_cache = self.runtime.buffer::<u16>(cache_elements);
        let value_cache = self.runtime.buffer::<u16>(cache_elements);
        let position_buffer = self.runtime.buffer::<u32>(batch as usize);
        if key_cache.contents().is_null()
            || value_cache.contents().is_null()
            || position_buffer.contents().is_null()
        {
            return Err("Metal could not allocate the Qwen generation cache".into());
        }
        Ok(GenerationState {
            key_cache,
            value_cache,
            position_buffer,
            capacity,
            position: 0,
            batch,
            batch_index: 0,
            cache_stride: self.config.kv_heads as usize
                * capacity as usize
                * self.config.head_dim() as usize,
        })
    }
}
