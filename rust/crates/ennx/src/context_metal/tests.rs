use super::*;
use crate::config::IndexMode;
use crate::context::probe::{Fixture, half};

struct Probe {
    runtime: Arc<Runtime>,
    kernels: ContextKernels,
    fixture: Fixture,
    qkv: Buffer,
    kv: Buffer,
    tree: Buffer,
    blocks: Buffer,
    output: Buffer,
    weights: Buffer,
    start: u32,
}

impl Probe {
    fn new() -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let layout = Layout::new(65536, 128)?;
        let fixture = Fixture::new(layout);
        let start = 32768;
        let mut qkv = vec![0u16; 128 * 640];
        for row in 0..128usize {
            qkv[row * 640..row * 640 + 512]
                .copy_from_slice(&fixture.queries[row * 512..(row + 1) * 512]);
            qkv[row * 640 + 512..(row + 1) * 640].copy_from_slice(
                &fixture.kv[(start as usize + row) * 128..(start as usize + row + 1) * 128],
            );
        }
        let mut weights = vec![0u16; 4096];
        for dim in 0..64 {
            weights[dim * 64 + dim] = 0x3c00;
        }
        Ok(Self {
            kernels: ContextKernels::with_index(&runtime, layout, true)?,
            qkv: runtime.buffer_with(&qkv),
            kv: runtime.buffer_with(&fixture.kv),
            tree: runtime.buffer_with(&fixture.tree),
            blocks: runtime.buffer::<u32>(128 * 8),
            output: runtime.buffer::<u16>(128 * 512),
            weights: runtime.buffer_with(&weights),
            runtime,
            fixture,
            start,
        })
    }

    fn run(
        &self,
        mode: IndexMode,
        fresh: bool,
        attention: bool,
    ) -> Result<(Vec<u32>, Vec<u16>), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.kernels.encode_policy(
            &encoder,
            &self.qkv,
            &self.kv,
            &self.tree,
            &self.blocks,
            attention.then_some(&*self.output),
            self.start,
            128,
            Some(IndexPolicy {
                weights: &self.weights,
                offset: 0,
                block: 128,
                mode,
                layer: 1,
                fresh,
                reuse: 0.02,
            }),
        )?;
        encoder.end_encoding();
        complete(command)?;
        let blocks = unsafe {
            std::slice::from_raw_parts(self.blocks.contents().cast::<u32>(), 128 * 8).to_vec()
        };
        let output = unsafe {
            std::slice::from_raw_parts(self.output.contents().cast::<u16>(), 128 * 512).to_vec()
        };
        Ok((blocks, output))
    }

    fn verify(&self, mode: IndexMode, blocks: &[u32], values: &[u16]) -> Result<(), String> {
        let position = self.start + 127;
        for row in 0..128usize {
            let selected = &blocks[row * 8..row * 8 + 8];
            let query = &self.fixture.queries[row * 512..row * 512 + 512];
            if mode == IndexMode::Independent && selected != self.fixture.blocks(position, query) {
                return Err(format!(
                    "independent support differs from reference at query {row}"
                ));
            }
            let valid = selected
                .iter()
                .copied()
                .filter(|value| *value != u32::MAX)
                .collect::<Vec<_>>();
            if valid.iter().any(|block| *block > position / 64)
                || !valid.contains(&0)
                || !valid.contains(&(position / 64))
                || valid
                    .iter()
                    .enumerate()
                    .any(|(i, block)| valid[i + 1..].contains(block))
            {
                return Err(
                    "diffusion selected a future/duplicate block or lost a forced block".into(),
                );
            }
            let expected = self.fixture.output(position, query, selected);
            if values[row * 512..row * 512 + 512]
                .iter()
                .any(|value| !half(*value).is_finite())
            {
                return Err("selected attention produced a nonfinite value".into());
            }
            let error = expected
                .iter()
                .zip(&values[row * 512..row * 512 + 512])
                .map(|(a, b)| (*a - half(*b)).abs())
                .fold(0.0f32, f32::max);
            if error > 0.003 {
                return Err(format!("selected attention error {error}"));
            }
        }
        Ok(())
    }
}

#[test]
fn block_visibility() {
    metal::objc::rc::autoreleasepool(|| {
        let probe = Probe::new().unwrap();
        for mode in [
            IndexMode::Independent,
            IndexMode::Shared,
            IndexMode::Refined,
        ] {
            let first = probe.run(mode, true, true).unwrap();
            probe.verify(mode, &first.0, &first.1).unwrap();
            // Cache-only work must retain the selected support and leave the
            // unread attention output untouched. A later query must still match.
            unsafe {
                std::slice::from_raw_parts_mut(probe.output.contents().cast::<u16>(), 128 * 512)
                    .fill(0x7e00);
            }
            let cached = probe.run(mode, true, false).unwrap();
            assert_eq!(first.0, cached.0);
            assert!(cached.1.iter().all(|value| *value == 0x7e00));
            assert_eq!(first, probe.run(mode, false, true).unwrap());
            let counts = probe.kernels.index_counts();
            assert!(counts[0] >= 32);
            // Poison all future blocks, then force a new traversal. Past queries
            // must remain exactly invariant, including their selected support.
            unsafe {
                let kv = std::slice::from_raw_parts_mut(
                    probe.kv.contents().cast::<u16>(),
                    probe.fixture.kv.len(),
                );
                kv[((probe.start + 128) * 128) as usize..].fill(0x4800);
                let tree = Fixture::pyramid(probe.fixture.layout, kv);
                std::ptr::copy_nonoverlapping(
                    tree.as_ptr(),
                    probe.tree.contents().cast::<u16>(),
                    tree.len(),
                );
            }
            assert_eq!(first, probe.run(mode, true, true).unwrap());
            // Restore future content for the next arm.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    probe.fixture.kv.as_ptr(),
                    probe.kv.contents().cast::<u16>(),
                    probe.fixture.kv.len(),
                );
                std::ptr::copy_nonoverlapping(
                    probe.fixture.tree.as_ptr(),
                    probe.tree.contents().cast::<u16>(),
                    probe.fixture.tree.len(),
                );
            }
        }
        // Query drift must invalidate cached selections even at the same row.
        unsafe {
            let qkv = std::slice::from_raw_parts_mut(probe.qkv.contents().cast::<u16>(), 128 * 640);
            for row in qkv.chunks_exact_mut(640) {
                row[..512].fill(0xbc00);
            }
        }
        let before = probe.kernels.index_counts()[1];
        probe.run(IndexMode::Refined, false, true).unwrap();
        assert_eq!(probe.kernels.index_counts()[1] - before, 32);
    });
}
