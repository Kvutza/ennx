#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    Int4 = 0,
    Int8 = 1,
    Fp4E2M1 = 2,
    Fp8E4M3 = 3,
    Fp8E5M2 = 4,
}

impl EncodingType {
    pub fn parse(bits: u8, mode: Option<&str>) -> Result<Self, String> {
        match (bits, mode.map(|s| s.trim().to_ascii_lowercase()).as_deref()) {
            (4, None | Some("int") | Some("int4")) => Ok(Self::Int4),
            (8, None | Some("int") | Some("int8")) => Ok(Self::Int8),
            (4, Some("fp4") | Some("fp4_e2m1") | Some("e2m1")) => Ok(Self::Fp4E2M1),
            (8, Some("fp8") | Some("fp8_e4m3") | Some("e4m3")) => Ok(Self::Fp8E4M3),
            (8, Some("fp8_e5m2") | Some("e5m2")) => Ok(Self::Fp8E5M2),
            _ => Err(format!("unsupported encoding bits={bits}, mode={mode:?}")),
        }
    }
}

pub static FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub fn decode_code(code: u32, encoding: EncodingType, scale: f32) -> f32 {
    match encoding {
        EncodingType::Int4 | EncodingType::Int8 => code as f32 * scale,
        EncodingType::Fp4E2M1 => FP4_LUT[(code & 0x0f) as usize] * scale,
        EncodingType::Fp8E4M3 => decode_e4m3(code as u8) * scale,
        EncodingType::Fp8E5M2 => decode_e5m2(code as u8) * scale,
    }
}

pub fn decode_e4m3(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * (2.0f32).powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * (2.0f32).powi(exp as i32 - 7)
    }
}

pub fn decode_e5m2(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 2) & 0x1f;
    let mant = byte & 0x03;
    if exp == 0 {
        sign * (mant as f32 / 4.0) * (2.0f32).powi(-14)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + mant as f32 / 4.0) * (2.0f32).powi(exp as i32 - 15)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameter {
    pub offset: usize,
    pub length: usize,
    pub bits: u8,
    pub encoding: EncodingType,
    pub scale: f32,
    pub weight: f32,
    pub radius: f32,
}

impl Parameter {
    pub fn new(
        offset: usize,
        length: usize,
        bits: u8,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        Self::encoded(
            offset,
            length,
            bits,
            EncodingType::parse(bits, None)?,
            scale,
            weight,
            radius,
        )
    }

    pub fn encoded(
        offset: usize,
        length: usize,
        bits: u8,
        encoding: EncodingType,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        if length == 0 {
            return Err("leaf length must be positive".to_string());
        }
        if bits != 4 && bits != 8 {
            return Err(format!("leaf bits must be 4 or 8, got {bits}"));
        }
        let encoded_bits = match encoding {
            EncodingType::Int4 | EncodingType::Fp4E2M1 => 4,
            _ => 8,
        };
        if bits != encoded_bits {
            return Err("encoding does not match parameter bit width".into());
        }
        offset
            .checked_add(length)
            .ok_or("parameter range overflow")?;
        for (name, value) in [("scale", scale), ("weight", weight), ("radius", radius)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        Ok(Self {
            offset,
            length,
            bits,
            encoding,
            scale,
            weight,
            radius,
        })
    }

    pub(super) fn bytes(self) -> usize {
        match self.bits {
            4 => self.length.div_ceil(2),
            8 => self.length,
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
}
