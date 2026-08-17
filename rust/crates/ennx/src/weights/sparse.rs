use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn sparse_union(rows: &[&[u32]]) -> Vec<u32> {
    let mut level: Vec<Vec<u32>> = rows.iter().map(|row| row.to_vec()).collect();
    if level.is_empty() {
        return Vec::new();
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut iter = level.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(merge_words(&left, &right));
            } else {
                next.push(left);
            }
        }
        level = next;
    }
    level.pop().unwrap_or_default()
}

fn merge_words(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() || j < right.len() {
        let value = if j == right.len() || (i < left.len() && left[i] < right[j]) {
            let value = left[i];
            i += 1;
            value
        } else if i == left.len() || right[j] < left[i] {
            let value = right[j];
            j += 1;
            value
        } else {
            let value = left[i];
            i += 1;
            j += 1;
            value
        };
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

pub fn sparse_xor(
    left_words: &[u32],
    left_masks: &[u32],
    right_words: &[u32],
    right_masks: &[u32],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    check_move(left_words, left_masks)?;
    check_move(right_words, right_masks)?;
    let mut words = Vec::with_capacity(left_words.len() + right_words.len());
    let mut masks = Vec::with_capacity(left_masks.len() + right_masks.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left_words.len() || j < right_words.len() {
        if j == right_words.len() || (i < left_words.len() && left_words[i] < right_words[j]) {
            words.push(left_words[i]);
            masks.push(left_masks[i]);
            i += 1;
        } else if i == left_words.len() || right_words[j] < left_words[i] {
            words.push(right_words[j]);
            masks.push(right_masks[j]);
            j += 1;
        } else {
            let mask = left_masks[i] ^ right_masks[j];
            if mask != 0 {
                words.push(left_words[i]);
                masks.push(mask);
            }
            i += 1;
            j += 1;
        }
    }
    Ok((words, masks))
}

pub fn check_move(words: &[u32], masks: &[u32]) -> Result<(), String> {
    if words.len() != masks.len() {
        return Err("move words and masks must have the same length".to_string());
    }
    for i in 0..words.len() {
        if masks[i] == 0 {
            return Err("move masks must be nonzero".to_string());
        }
        if i > 0 && words[i - 1] >= words[i] {
            return Err("move words must be strictly increasing".to_string());
        }
    }
    Ok(())
}

pub fn missing_words(cached: &[u32], query: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0usize;
    for &word in query {
        while i < cached.len() && cached[i] < word {
            i += 1;
        }
        if i == cached.len() || cached[i] != word {
            out.push(word);
        }
    }
    out
}

pub fn merge_values(
    words: &[u32],
    values: &[u32],
    extra_words: &[u32],
    extra_values: &[u32],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    if words.len() != values.len() || extra_words.len() != extra_values.len() {
        return Err("word and value arrays must have matching lengths".to_string());
    }
    let mut out_words = Vec::with_capacity(words.len() + extra_words.len());
    let mut out_values = Vec::with_capacity(values.len() + extra_values.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < words.len() || j < extra_words.len() {
        if j == extra_words.len() || (i < words.len() && words[i] < extra_words[j]) {
            out_words.push(words[i]);
            out_values.push(values[i]);
            i += 1;
        } else if i == words.len() || extra_words[j] < words[i] {
            out_words.push(extra_words[j]);
            out_values.push(extra_values[j]);
            j += 1;
        } else {
            out_words.push(words[i]);
            out_values.push(extra_values[j]);
            i += 1;
            j += 1;
        }
    }
    Ok((out_words, out_values))
}

pub fn take_words(words: &[u32], values: &[u32], query: &[u32]) -> Result<Vec<u32>, String> {
    if words.len() != values.len() {
        return Err("word and value arrays must have matching lengths".to_string());
    }
    let mut out = Vec::with_capacity(query.len());
    let mut i = 0usize;
    for &word in query {
        while i < words.len() && words[i] < word {
            i += 1;
        }
        if i == words.len() || words[i] != word {
            return Err(format!("word {word} is missing from cache"));
        }
        out.push(values[i]);
    }
    Ok(out)
}

pub fn apply_sparse(
    words: &[u32],
    values: &[u32],
    move_words: &[u32],
    move_masks: &[u32],
) -> Result<Vec<u32>, String> {
    if words.len() != values.len() {
        return Err("word and value arrays must have matching lengths".to_string());
    }
    check_move(move_words, move_masks)?;
    let mut out = values.to_vec();
    let mut j = 0usize;
    for (i, &word) in words.iter().enumerate() {
        while j < move_words.len() && move_words[j] < word {
            j += 1;
        }
        if j < move_words.len() && move_words[j] == word {
            out[i] ^= move_masks[j];
        }
    }
    Ok(out)
}

pub fn blocks_for_words(
    words: &[u32],
    word_ends: &[u32],
    widths: &[u8],
) -> Result<(Vec<(usize, usize, u8)>, usize), String> {
    if word_ends.len() != widths.len() {
        return Err("word_ends and widths must have the same length".to_string());
    }
    if words.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let mut bits_by_word = Vec::with_capacity(words.len());
    for &word in words {
        let spec = word_ends.partition_point(|&end| end <= word);
        if spec >= word_ends.len() {
            return Err(format!("word {word} is outside the weight layout"));
        }
        let bits = widths[spec];
        if bits != 4 && bits != 8 {
            return Err(format!("weight word width must be 4 or 8, got {bits}"));
        }
        bits_by_word.push(bits);
    }
    let mut blocks = Vec::new();
    let mut dimension = 0usize;
    let mut start = 0usize;
    while start < bits_by_word.len() {
        let bits = bits_by_word[start];
        let mut end = start + 1;
        while end < bits_by_word.len() && bits_by_word[end] == bits {
            end += 1;
        }
        let length = (end - start) * (32 / usize::from(bits));
        blocks.push((dimension, length, bits));
        dimension += length;
        start = end;
    }
    Ok((blocks, dimension))
}

pub fn draw_sparse(
    count: usize,
    size: usize,
    dimension: u64,
    parameter_starts: &[u64],
    parameter_ends: &[u64],
    word_offsets: &[u32],
    widths: &[u8],
    seed: u64,
) -> Result<Vec<(Vec<u32>, Vec<u32>)>, String> {
    if count == 0 || size == 0 || dimension == 0 {
        return Err("count, size, and dimension must be positive".to_string());
    }
    let n = parameter_ends.len();
    if parameter_starts.len() != n || word_offsets.len() != n || widths.len() != n || n == 0 {
        return Err("weight layout arrays must have the same positive length".to_string());
    }
    for i in 0..n {
        if parameter_starts[i] >= parameter_ends[i] {
            return Err("layout parameter ranges must be nonempty".to_string());
        }
        if i > 0 && parameter_starts[i] < parameter_ends[i - 1] {
            return Err("layout parameter ranges must be sorted and nonoverlapping".to_string());
        }
        if widths[i] != 4 && widths[i] != 8 {
            return Err(format!("weight width must be 4 or 8, got {}", widths[i]));
        }
    }
    if parameter_starts[0] != 0 || parameter_ends[n - 1] != dimension {
        return Err("layout parameter ranges must cover the full metric dimension".to_string());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let mut pairs = Vec::<(u32, u32)>::with_capacity(size);
        for _ in 0..size {
            let parameter = rng.gen_range(0..dimension);
            let spec = parameter_ends.partition_point(|&end| end <= parameter);
            let local = parameter - parameter_starts[spec];
            let width = u32::from(widths[spec]);
            let fields_per_word = 32 / width;
            let word = word_offsets[spec] + (local as u32 / fields_per_word);
            let bit = (local as u32 % fields_per_word) * width;
            let code = rng.gen_range(1..(1u32 << width));
            pairs.push((word, code << bit));
        }
        pairs.sort_unstable_by_key(|&(word, _)| word);
        let mut words = Vec::new();
        let mut masks = Vec::new();
        for (word, mask) in pairs {
            if words.last().copied() == Some(word) {
                let last = masks.last_mut().expect("last mask exists");
                *last ^= mask;
                if *last == 0 {
                    words.pop();
                    masks.pop();
                }
            } else {
                words.push(word);
                masks.push(mask);
            }
        }
        rows.push((words, masks));
    }
    Ok(rows)
}
