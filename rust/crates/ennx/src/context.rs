//! Bounded query work over a persistent PISA context. No model weights live here.

pub const MAX_TOKENS: u32 = 1_048_576;
// A million new tokens plus a nonempty prompt need more than a million slots.
pub const MAX_CONTEXT: u32 = 2 * MAX_TOKENS;
pub const BLOCK: u32 = 64;
pub const WIDTH: u32 = 64;
pub const HEADS: u32 = 8;
pub const SELECTED: u32 = 8;
#[path = "context_probe.rs"]
pub mod probe;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub tokens: u32,
    pub queries: u32,
}

pub struct Output {
    pub blocks: Vec<u32>,
    pub values: Vec<u16>,
    pub device_ms: f64,
    pub wall_ms: f64,
}

impl Layout {
    pub fn new(tokens: u32, queries: u32) -> Result<Self, String> {
        if !(4096..=MAX_CONTEXT).contains(&tokens) || !tokens.is_power_of_two() {
            return Err("context must be a power of two from 4096 through 2097152".into());
        }
        if queries == 0 || queries > 4096 || queries % 4 != 0 {
            return Err("query capacity must be a multiple of four from 4 through 4096".into());
        }
        Ok(Self { tokens, queries })
    }

    pub fn range(self, start: u32, rows: u32) -> Result<(), String> {
        if rows == 0
            || rows > self.queries
            || rows % 4 != 0
            || start.checked_add(rows).is_none_or(|end| end > self.tokens)
        {
            return Err(
                "query range exceeds context or capacity, or is not a multiple of four".into(),
            );
        }
        Ok(())
    }

    pub fn leaves(self) -> u32 {
        self.tokens / BLOCK
    }
    pub fn nodes(self) -> u32 {
        2 * self.leaves() - 1
    }
    pub fn levels(self) -> u32 {
        self.leaves().trailing_zeros()
    }
    pub fn kv_bytes(self) -> u64 {
        u64::from(self.tokens) * u64::from(2 * WIDTH) * 2
    }
    pub fn tree_bytes(self) -> u64 {
        u64::from(self.nodes()) * u64::from(WIDTH) * 2
    }
    pub fn work_bytes(self) -> u64 {
        u64::from(self.queries) * u64::from(2 * HEADS * WIDTH * 2 + SELECTED * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds() {
        let layout = Layout::new(MAX_TOKENS, 128).unwrap();
        assert_eq!(layout.kv_bytes(), 268_435_456);
        assert_eq!(layout.tree_bytes(), 4_194_176);
        layout.range(MAX_TOKENS - 128, 128).unwrap();
        assert!(layout.range(MAX_TOKENS - 127, 128).is_err());
        assert!(layout.range(u32::MAX, 4).is_err());
        assert!(Layout::new(MAX_CONTEXT * 2, 128).is_err());
        assert!(Layout::new(65535, 128).is_err());
        assert!(Layout::new(MAX_TOKENS, 4097).is_err());
    }
}
