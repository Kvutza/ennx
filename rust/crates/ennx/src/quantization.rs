//! Quantization helpers for bit-packed weight encodings.

pub static FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub fn quantize_int4(values: impl IntoIterator<Item = f32>, scale: f32) -> Vec<u8> {
    pack_nibbles(values.into_iter().map(|value| {
        let q = (value / scale).round_ties_even();
        if q.is_nan() || q <= 0.0 {
            0
        } else if q >= 15.0 {
            15
        } else {
            q as u8
        }
    }))
}

pub fn quantize_fp4_e2m1(values: impl IntoIterator<Item = f32>, scale: f32) -> Vec<u8> {
    pack_nibbles(values.into_iter().map(|value| {
        let scaled = value / scale;
        let mut best_code = 0u8;
        let mut best_dist = f32::INFINITY;
        for (code, &candidate) in FP4_E2M1_LUT.iter().enumerate() {
            let dist = (scaled - candidate).abs();
            if dist < best_dist {
                best_dist = dist;
                best_code = code as u8;
            }
        }
        best_code
    }))
}

fn pack_nibbles(codes: impl IntoIterator<Item = u8>) -> Vec<u8> {
    let mut iter = codes.into_iter();
    let mut out = Vec::new();
    while let Some(low) = iter.next() {
        let high = iter.next().unwrap_or(0);
        out.push((low & 0x0f) | ((high & 0x0f) << 4));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{quantize_fp4_e2m1, quantize_int4};

    #[test]
    fn pack_odd() {
        assert_eq!(quantize_int4([0.0, 1.0, 2.0], 1.0), vec![0x10, 0x02]);
        assert_eq!(quantize_fp4_e2m1([0.0, 1.0, 2.0], 1.0), vec![0x20, 0x04]);
    }

    #[test]
    fn round_even() {
        assert_eq!(quantize_int4([0.5, 1.5, 2.5, 3.5], 1.0), vec![0x20, 0x42]);
    }

    #[test]
    fn clamp() {
        assert_eq!(quantize_int4([-3.0, 20.0], 1.0), vec![0xf0]);
    }
}
