//! Strongly-typed virtual register representations for PTX.

/// Physical target architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Sm75, // Turing (T4, RTX 2080)
    Sm80, // Ampere (A100)
    Sm90, // Hopper (H100)
}

impl TargetArch {
    pub fn ptx_target_str(&self) -> &'static str {
        match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm90 => "sm_90",
        }
    }
}

/// Strongly-typed 32-bit integer register (`%r<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg32(pub u32);

/// Strongly-typed 64-bit integer/pointer register (`%rd<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg64(pub u32);

/// Strongly-typed 32-bit single-precision float register (`%f<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegF32(pub u32);

/// Strongly-typed 1-bit predicate register (`%p<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegPred(pub u32);

/// A pair of 32-bit registers (for 64-bit vector or 2x 32-bit values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegPair(pub Reg32, pub Reg32);

/// A quad of 32-bit registers for 128-bit vector load/store (`ld.global.v4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegQuad(pub Reg32, pub Reg32, pub Reg32, pub Reg32);

/// Four 32-bit FP registers for Tensor Core accumulators (`{%f0, %f1, %f2, %f3}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmaAccumulator(pub RegF32, pub RegF32, pub RegF32, pub RegF32);
