//! Shared numerical fixture for Metal and CUDA cache kernels, not a trained model.
use super::{BLOCK, HEADS, Layout, Output, SELECTED, WIDTH};

pub trait Cache {
    fn write(&mut self, start: u32, values: &[u16]) -> Result<f64, String>;
    fn query(&mut self, start: u32, queries: &[u16]) -> Result<Output, String>;
    fn tree(&mut self) -> Result<Vec<u16>, String>;
}

pub struct Report {
    pub build_ms: f64,
    pub repair_ms: f64,
    pub device_ms: f64,
    pub wall_ms: f64,
    pub error: f32,
    pub ranges: Vec<u32>,
}

pub fn run(cache: &mut impl Cache, layout: Layout, repeats: u32) -> Result<Report, String> {
    if repeats == 0 || repeats > 100 || layout.queries > layout.tokens / 2 {
        return Err(
            "context repeats must be in 1..=100 and query count at most half the context".into(),
        );
    }
    let mut fixture = Fixture::new(layout);
    let prefix = layout.queries + 65;
    let split = (prefix * 2 * WIDTH) as usize;
    let initial_ms = cache.write(0, &fixture.kv[..split])?;
    let initial = cache.query(65, &fixture.queries)?;
    fixture.verify(65, &initial.blocks, &initial.values)?;
    if cache.query(prefix - 3, &fixture.queries).is_ok() {
        return Err("cache accepted queries beyond its initialized prefix".into());
    }
    let build_ms = initial_ms + cache.write(prefix, &fixture.kv[split..])?;
    if cache.tree()? != fixture.tree {
        return Err("context pyramid differs from FP16 reference".into());
    }
    let ranges = vec![
        0,
        1,
        60,
        64,
        layout.tokens / 2 - layout.queries / 2,
        layout.tokens - layout.queries,
    ];
    let mut error = 0.0_f32;
    for &start in &ranges {
        let result = cache.query(start, &fixture.queries)?;
        error = error.max(fixture.verify(start, &result.blocks, &result.values)?);
    }
    causality(cache, &fixture)?;
    let repair_ms = repair(cache, &mut fixture)?;
    let start = layout.tokens - layout.queries;
    let result = cache.query(start, &fixture.queries)?;
    error = error.max(fixture.verify(start, &result.blocks, &result.values)?);
    let (mut device, mut wall) = (Vec::new(), Vec::new());
    for _ in 0..repeats {
        let result = cache.query(start, &fixture.queries)?;
        device.push(result.device_ms);
        wall.push(result.wall_ms);
    }
    device.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    Ok(Report {
        build_ms,
        repair_ms,
        device_ms: device[device.len() / 2],
        wall_ms: wall[wall.len() / 2],
        error,
        ranges,
    })
}

fn causality(cache: &mut impl Cache, fixture: &Fixture) -> Result<(), String> {
    let prefix = fixture.layout.tokens / 2;
    let start = prefix - fixture.layout.queries;
    let before = cache.query(start, &fixture.queries)?;
    let poison = vec![0x4c00; ((fixture.layout.tokens - prefix) * 2 * WIDTH) as usize];
    cache.write(prefix, &poison)?;
    let after = cache.query(start, &fixture.queries)?;
    if before.blocks != after.blocks || before.values != after.values {
        return Err("future KV content changed a past query".into());
    }
    cache.write(prefix, &fixture.kv[(prefix * 2 * WIDTH) as usize..])?;
    Ok(())
}

fn repair(cache: &mut impl Cache, fixture: &mut Fixture) -> Result<f64, String> {
    // Cross a leaf boundary: both leaves and their entire ancestor closure must update.
    let start = fixture.layout.tokens - 129;
    let first = (start * 2 * WIDTH) as usize;
    let end = ((start + 4) * 2 * WIDTH) as usize;
    fixture.kv[first..end].fill(0xbc00);
    let elapsed = cache.write(start, &fixture.kv[first..end])?;
    fixture.tree = Fixture::pyramid(fixture.layout, &fixture.kv);
    if cache.tree()? != fixture.tree {
        return Err("incremental tree repair differs from full rebuild".into());
    }
    Ok(elapsed)
}

pub fn half(bits: u16) -> f32 {
    let exponent = (bits >> 10) & 31;
    let fraction = f32::from(bits & 1023);
    let magnitude = match exponent {
        0 => fraction * 2.0f32.powi(-24),
        31 => f32::NAN,
        _ => (1024.0 + fraction) * 2.0f32.powi(i32::from(exponent) - 25),
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let raw = value.to_bits();
    let sign = ((raw >> 16) & 0x8000) as u16;
    let magnitude = raw & 0x7fff_ffff;
    let exponent = ((raw >> 23) & 255) as i32 - 112;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = (raw & 0x7f_ffff) | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let rounded = mantissa + ((1 << (shift - 1)) - 1) + ((mantissa >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    let rounded = magnitude + 4095 + ((magnitude >> 13) & 1);
    sign | ((rounded >> 13) - (112 << 10)) as u16
}

pub struct Fixture {
    pub layout: Layout,
    pub kv: Vec<u16>,
    pub queries: Vec<u16>,
    pub tree: Vec<u16>,
}

impl Fixture {
    pub fn new(layout: Layout) -> Self {
        let mut kv = vec![0; (layout.tokens * WIDTH * 2) as usize];
        for row in 0..layout.tokens {
            let leaf = row / BLOCK;
            for dim in 0..WIDTH {
                let key = [0x3800, 0x3c00, 0xb800, 0xbc00][((leaf * 13 + dim * 7) % 4) as usize];
                let value = [0x3400, 0x3800, 0xb400, 0xb800][((row * 11 + dim * 3) % 4) as usize];
                kv[(row * WIDTH * 2 + dim) as usize] = key;
                kv[(row * WIDTH * 2 + WIDTH + dim) as usize] = value;
            }
        }
        // A remote block has a strong, distinctive key and a nonconstant value.
        let remote = layout.leaves() / 3;
        for row in remote * BLOCK..(remote + 1) * BLOCK {
            for dim in 0..WIDTH {
                kv[(row * WIDTH * 2 + dim) as usize] = 0x4800;
                kv[(row * WIDTH * 2 + WIDTH + dim) as usize] =
                    if dim % 2 == 0 { 0x3c00 } else { 0xbc00 };
            }
        }
        let queries = (0..layout.queries * HEADS * WIDTH)
            .map(|index| {
                let row = index / (HEADS * WIDTH);
                let head = index / WIDTH % HEADS;
                let dim = index % WIDTH;
                [0x3c00, 0x3800, 0xb800, 0x4000][((row + head * 3 + dim * 5) % 4) as usize]
            })
            .collect();
        let tree = Self::pyramid(layout, &kv);
        Self {
            layout,
            kv,
            queries,
            tree,
        }
    }

    pub fn pyramid(layout: Layout, kv: &[u16]) -> Vec<u16> {
        let mut tree = vec![0; (layout.nodes() * WIDTH) as usize];
        for leaf in 0..layout.leaves() {
            for dim in 0..WIDTH {
                let sum: f32 = (0..BLOCK)
                    .map(|token| half(kv[((leaf * BLOCK + token) * WIDTH * 2 + dim) as usize]))
                    .sum();
                tree[(leaf * WIDTH + dim) as usize] = bits(sum / BLOCK as f32);
            }
        }
        let (mut child, mut parent, mut count) = (0, layout.leaves(), layout.leaves());
        while count > 1 {
            for node in 0..count / 2 {
                for dim in 0..WIDTH {
                    let left = ((child + 2 * node) * WIDTH + dim) as usize;
                    tree[((parent + node) * WIDTH + dim) as usize] =
                        bits(0.5 * (half(tree[left]) + half(tree[left + WIDTH as usize])));
                }
            }
            child = parent;
            count /= 2;
            parent += count;
        }
        tree
    }

    pub fn blocks(&self, position: u32, query: &[u16]) -> [u32; SELECTED as usize] {
        let current = position / BLOCK;
        let forced = [0, current.saturating_sub(1), current];
        let route: Vec<f32> = (0..WIDTH)
            .map(|dim| {
                (0..HEADS)
                    .map(|head| half(query[(head * WIDTH + dim) as usize]))
                    .sum()
            })
            .collect();
        let mut candidates = std::array::from_fn::<_, 16, _>(|slot| slot as u32);
        let mut chosen = [u32::MAX; SELECTED as usize];
        for level in (0..=self.layout.levels() - 4).rev() {
            let offset = 2 * self.layout.leaves() - (2 * self.layout.leaves() >> level);
            let mut scores = [f32::NEG_INFINITY; 16];
            for (slot, &node) in candidates.iter().enumerate() {
                if node == u32::MAX {
                    continue;
                }
                let begin = node << level;
                let end = begin + (1 << level) - 1;
                if forced.iter().any(|&leaf| leaf >= begin && leaf <= end) {
                    scores[slot] = f32::INFINITY;
                } else if end < current {
                    scores[slot] = route
                        .iter()
                        .enumerate()
                        .map(|(dim, &q)| {
                            q * half(self.tree[((offset + node) * WIDTH) as usize + dim])
                        })
                        .sum();
                }
            }
            for output in &mut chosen {
                let best = (0..16)
                    .filter(|&slot| scores[slot] > f32::NEG_INFINITY)
                    .max_by(|&left, &right| {
                        scores[left]
                            .total_cmp(&scores[right])
                            .then_with(|| candidates[right].cmp(&candidates[left]))
                    });
                *output = best.map_or(u32::MAX, |slot| {
                    scores[slot] = f32::NEG_INFINITY;
                    candidates[slot]
                });
            }
            if level > 0 {
                for (slot, &node) in chosen.iter().enumerate() {
                    candidates[2 * slot] = if node == u32::MAX { node } else { 2 * node };
                    candidates[2 * slot + 1] = if node == u32::MAX { node } else { 2 * node + 1 };
                }
            }
        }
        chosen
    }

    pub fn output(&self, position: u32, query: &[u16], blocks: &[u32]) -> Vec<f32> {
        let tokens: Vec<u32> = blocks
            .iter()
            .filter(|&&block| block != u32::MAX)
            .flat_map(|&block| block * BLOCK..(block + 1) * BLOCK)
            .filter(|&row| row <= position)
            .collect();
        let mut output = Vec::with_capacity((HEADS * WIDTH) as usize);
        for head in 0..HEADS {
            let scores: Vec<f32> = tokens
                .iter()
                .map(|&row| {
                    (0..WIDTH)
                        .map(|dim| {
                            half(query[(head * WIDTH + dim) as usize])
                                * half(self.kv[(row * WIDTH * 2 + dim) as usize])
                        })
                        .sum::<f32>()
                        * 0.125
                })
                .collect();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores
                .iter()
                .map(|&score| (score - maximum).exp())
                .collect();
            let total: f32 = weights.iter().sum();
            for dim in 0..WIDTH {
                output.push(
                    tokens
                        .iter()
                        .zip(&weights)
                        .map(|(&row, &weight)| {
                            weight * half(self.kv[(row * WIDTH * 2 + WIDTH + dim) as usize])
                        })
                        .sum::<f32>()
                        / total,
                );
            }
        }
        output
    }

    pub fn verify(&self, start: u32, blocks: &[u32], output: &[u16]) -> Result<f32, String> {
        let mut error = 0.0_f32;
        for row in 0..self.layout.queries {
            let query =
                &self.queries[(row * HEADS * WIDTH) as usize..((row + 1) * HEADS * WIDTH) as usize];
            let expected = self.blocks(start + row, query);
            let actual = &blocks[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize];
            if actual != expected {
                return Err(format!(
                    "context selection mismatch at position {}: {actual:?} != {expected:?}",
                    start + row
                ));
            }
            let values = self.output(start + row, query, &expected);
            for (dim, &expected) in values.iter().enumerate() {
                let actual = half(output[(row * HEADS * WIDTH) as usize + dim]);
                if !actual.is_finite() {
                    return Err("non-finite context output".into());
                }
                error = error.max((actual - expected).abs());
            }
        }
        if error > 0.003 {
            return Err(format!("context attention error {error} exceeds 0.003"));
        }
        Ok(error)
    }
}
