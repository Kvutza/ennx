//! Ported from yubo-research/enn, commit 506e98c506eeb849cffbf53d9ddf3a3a799c6830.
//! NumPy `Generator(PCG64(seed))` for `integers(0, high)`.

use crate::metric_seed::seed_words;

const MULT_HI: u128 = 2549297995355413924;
const MULT_LO: u128 = 4865540595714422341;
const MULT: u128 = (MULT_HI << 64) | MULT_LO;

pub struct MetricRng {
    state: u128,
    inc: u128,
    has_u32: bool,
    u32_buf: u32,
}

impl MetricRng {
    pub fn from_seed(seed: u64) -> Self {
        let words = seed_words(seed, 4);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let inc = (initseq << 1) | 1;
        let mut state = 0u128;
        state = state.wrapping_mul(MULT).wrapping_add(inc);
        state = state.wrapping_add(initstate);
        state = state.wrapping_mul(MULT).wrapping_add(inc);
        Self {
            state,
            inc,
            has_u32: false,
            u32_buf: 0,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((high >> 58) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.u32_buf;
        }
        let next = self.next_u64();
        self.has_u32 = true;
        self.u32_buf = (next >> 32) as u32;
        next as u32
    }

    /// `Generator.integers(0, high)` for `high >= 1` (exclusive upper bound).
    pub fn integers_high(&mut self, high: u64) -> u64 {
        debug_assert!(high >= 1);
        let rng = high - 1;
        if rng == 0 {
            return 0;
        }
        if rng <= u64::from(u32::MAX) {
            if rng == u64::from(u32::MAX) {
                return u64::from(self.next_u32());
            }
            return u64::from(self.lemire32(rng as u32));
        }
        self.lemire64(rng)
    }

    fn lemire32(&mut self, rng: u32) -> u32 {
        let rng_excl = u64::from(rng) + 1;
        let mut draw = || (u64::from(self.next_u32())) * rng_excl;
        let mut m = draw();
        let mut leftover = m as u32;
        if u64::from(leftover) < rng_excl {
            let threshold = (u32::MAX - rng) % (rng_excl as u32);
            while leftover < threshold {
                m = draw();
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    fn lemire64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng + 1;
        let mut draw = || (self.next_u64() as u128).wrapping_mul(u128::from(rng_excl));
        let mut m = draw();
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = draw();
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }
}
