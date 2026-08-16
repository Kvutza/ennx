use super::{decode_code, hash, perturb, score, Ask, Leaf, SparseEdit};
use crate::util::insert_neighbor;

pub(super) fn make_edits(
    seeds: &[u64],
    leaves: &[Leaf],
    num_pert: usize,
) -> Result<Vec<SparseEdit>, String> {
    let dimensions = leaves.iter().map(|leaf| leaf.length).sum::<usize>();
    if num_pert == 0 || num_pert > dimensions {
        return Err(format!(
            "num_pert must be between one and {dimensions}, got {num_pert}"
        ));
    }
    let mut edits = Vec::with_capacity(seeds.len() * num_pert);
    for &seed in seeds {
        let start = edits.len();
        for draw in 0..num_pert {
            let draw_key = (draw as u32).wrapping_mul(0x85eb_ca6b);
            let mut global = hash(seed ^ 0xd1b5_4a32_d192_ed03, draw_key) as usize % dimensions;
            while edits[start..]
                .iter()
                .any(|edit| global_index(*edit, leaves) == global)
            {
                global = (global + 1) % dimensions;
            }
            let (leaf, element) = locate(global, leaves);
            edits.push(SparseEdit {
                leaf: leaf as u32,
                element: element as u32,
            });
        }
    }
    Ok(edits)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select(
    base: &[u8],
    rows: &[&[u8]],
    history: &[(usize, f32)],
    seeds: &[u64],
    edits: &[SparseEdit],
    num_pert: usize,
    leaves: &[Leaf],
    config: Ask,
) -> (usize, f32) {
    let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
    let base_distances = rows
        .iter()
        .map(|row| row_distance(base, row, leaves))
        .collect::<Vec<_>>();
    let mut nearest = vec![(f32::INFINITY, 0usize); config.neighbors];
    let mut best = (0usize, f32::NEG_INFINITY);
    for (index, &seed) in seeds.iter().enumerate() {
        nearest.fill((f32::INFINITY, 0));
        let candidate_edits = &edits[index * num_pert..(index + 1) * num_pert];
        for (observation, row) in rows.iter().enumerate() {
            let distance = edit_distance(
                base_distances[observation],
                base,
                row,
                seed,
                candidate_edits,
                leaves,
                config.length,
            );
            insert_neighbor(&mut nearest, distance.max(0.0), observation);
        }
        let value = score(&nearest, history, draws[index], config);
        if value > best.1 || (value == best.1 && index < best.0) {
            best = (index, value);
        }
    }
    best
}

pub(super) fn materialize(
    base: &[u8],
    seed: u64,
    edits: &[SparseEdit],
    leaves: &[Leaf],
    length: f32,
) -> Vec<u8> {
    let mut row = base.to_vec();
    for &edit in edits {
        let leaf = leaves[edit.leaf as usize];
        let step = super::make_steps(&[Leaf { offset: 0, ..leaf }], length)[0];
        let byte_offset = byte_offset(leaves, edit.leaf as usize);
        let byte = byte_offset
            + if leaf.bits == 4 {
                edit.element as usize / 2
            } else {
                edit.element as usize
            };
        let shift = if leaf.bits == 4 {
            (edit.element & 1) * 4
        } else {
            0
        };
        let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
        let code = u32::from((row[byte] >> shift) & mask);
        let value = sparse_code(code, seed, leaf.offset as u32 + edit.element, step);
        if leaf.bits == 4 {
            row[byte] = (row[byte] & !(0x0f << shift)) | ((value as u8) << shift);
        } else {
            row[byte] = value as u8;
        }
    }
    row
}

fn edit_distance(
    mut distance: f32,
    base: &[u8],
    row: &[u8],
    seed: u64,
    edits: &[SparseEdit],
    leaves: &[Leaf],
    length: f32,
) -> f32 {
    let steps = super::make_steps(leaves, length);
    for &edit in edits {
        let leaf_index = edit.leaf as usize;
        let leaf = leaves[leaf_index];
        let byte = byte_offset(leaves, leaf_index)
            + if leaf.bits == 4 {
                edit.element as usize / 2
            } else {
                edit.element as usize
            };
        let shift = if leaf.bits == 4 {
            (edit.element & 1) * 4
        } else {
            0
        };
        let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
        let base_code = u32::from((base[byte] >> shift) & mask);
        let row_code = u32::from((row[byte] >> shift) & mask);
        let candidate = sparse_code(
            base_code,
            seed,
            leaf.offset as u32 + edit.element,
            steps[leaf_index],
        );
        let base_delta = decode_code(base_code, leaf.encoding, leaf.scale)
            - decode_code(row_code, leaf.encoding, leaf.scale);
        let candidate_delta = decode_code(candidate, leaf.encoding, leaf.scale)
            - decode_code(row_code, leaf.encoding, leaf.scale);
        distance += (candidate_delta * candidate_delta - base_delta * base_delta) * leaf.weight;
    }
    distance
}

fn row_distance(base: &[u8], row: &[u8], leaves: &[Leaf]) -> f32 {
    let steps = super::make_steps(leaves, 0.0);
    super::trial_distance(base, row, leaves, &steps, 0)
}

fn sparse_code(code: u32, seed: u64, element: u32, mut step: super::Step) -> u32 {
    if step.whole == 0 && step.threshold == 0 {
        return code;
    }
    step.whole = step.whole.max(1);
    step.threshold = 0;
    perturb(code, seed, element, step)
}

fn locate(global: usize, leaves: &[Leaf]) -> (usize, usize) {
    let mut start = 0usize;
    for (index, leaf) in leaves.iter().enumerate() {
        if global < start + leaf.length {
            return (index, global - start);
        }
        start += leaf.length;
    }
    unreachable!("global coordinate is bounded by total dimensions")
}

fn global_index(edit: SparseEdit, leaves: &[Leaf]) -> usize {
    leaves[..edit.leaf as usize]
        .iter()
        .map(|leaf| leaf.length)
        .sum::<usize>()
        + edit.element as usize
}

fn byte_offset(leaves: &[Leaf], leaf: usize) -> usize {
    leaves[..leaf].iter().map(|leaf| leaf.bytes()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_unique() {
        let leaves = [Leaf::new(0, 17, 4, 0.25, 1.0, 0.25).unwrap()];
        let edits = make_edits(&[7, 11], &leaves, 8).unwrap();
        for candidate in edits.chunks_exact(8) {
            let mut indices = candidate
                .iter()
                .map(|edit| edit.element)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            indices.dedup();
            assert_eq!(indices.len(), 8);
        }
    }

    #[test]
    fn changes_exact() {
        let leaves = [Leaf::new(0, 16, 4, 0.25, 1.0, 0.25).unwrap()];
        let base = [0x88; 8];
        let edits = make_edits(&[7], &leaves, 6).unwrap();
        let row = materialize(&base, 7, &edits, &leaves, 0.8);
        let changes = (0..16)
            .filter(|&element| {
                let byte = element / 2;
                let shift = (element & 1) * 4;
                ((base[byte] >> shift) & 0x0f) != ((row[byte] >> shift) & 0x0f)
            })
            .count();
        assert_eq!(changes, 6);
    }
}
