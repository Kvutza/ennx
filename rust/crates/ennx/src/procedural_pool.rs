//! Procedural candidate identities, independent of accelerator pool storage.
//!
//! Arms namespace perturbation streams around a shared incumbent. They do not
//! imply independently materialized centers or independent trust regions.
use deser::{Deserialize, Serialize};

/// Program used to construct each dense procedural direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ProposalMethod {
    #[default]
    Independent,
    SpectralBasis,
    PolynomialThreshold,
}

impl ProposalMethod {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::SpectralBasis => "spectral-basis",
            Self::PolynomialThreshold => "polynomial-threshold",
        }
    }
}

/// Identity of a perturbation within an arm, not an index into model storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CandidateIdentity {
    pub arm: u32,
    pub slot: u32,
}

/// Requested logical pool. Validation does not allocate model-sized buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ProceduralPoolConfig {
    #[deser(default = default_arms())]
    pub arms: u32,
    #[deser(default = default_slots())]
    pub candidates_per_arm: u32,
}

const fn default_arms() -> u32 {
    1
}

const fn default_slots() -> u32 {
    4
}

impl Default for ProceduralPoolConfig {
    fn default() -> Self {
        Self {
            arms: default_arms(),
            candidates_per_arm: default_slots(),
        }
    }
}

/// A validated logical layout. Flattened IDs are metadata, never resident rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProceduralPool {
    arms: u32,
    slots: u32,
    count: u32,
}

impl ProceduralPoolConfig {
    pub fn resolve(self) -> Result<ProceduralPool, String> {
        if self.arms == 0 || self.candidates_per_arm == 0 {
            return Err("procedural pool arms and candidates_per_arm must be positive".into());
        }
        let count = self
            .arms
            .checked_mul(self.candidates_per_arm)
            .ok_or("procedural candidate count exceeds u32")?;
        Ok(ProceduralPool {
            arms: self.arms,
            slots: self.candidates_per_arm,
            count,
        })
    }

    /// Current resident kernels have a fixed four-slot GPU ABI. Do not silently
    /// interpret a larger logical pool as the legacy pool.
    pub fn resolve_resident(self) -> Result<ProceduralPool, String> {
        let pool = self.resolve()?;
        if pool.count != 4 {
            return Err("resident procedural pool requires exactly four total candidates; wider pools require a separately measured GPU pool/selection ABI".into());
        }
        Ok(pool)
    }
}

impl ProceduralPool {
    pub const fn legacy() -> Self {
        Self {
            arms: 1,
            slots: 4,
            count: 4,
        }
    }

    pub const fn arms(self) -> u32 {
        self.arms
    }

    pub const fn slots(self) -> u32 {
        self.slots
    }

    pub const fn count(self) -> u32 {
        self.count
    }

    pub fn identity(self, index: u32) -> Result<CandidateIdentity, String> {
        if index >= self.count {
            return Err("procedural candidate index is outside its pool".into());
        }
        Ok(CandidateIdentity {
            arm: index / self.slots,
            slot: index % self.slots,
        })
    }

    pub fn index(self, identity: CandidateIdentity) -> Result<u32, String> {
        if identity.arm >= self.arms || identity.slot >= self.slots {
            return Err("procedural candidate identity is outside its pool".into());
        }
        Ok(identity.arm * self.slots + identity.slot)
    }

    pub fn seed(self, root: u64, identity: CandidateIdentity) -> Result<u64, String> {
        self.index(identity)?;
        Ok(legacy_seed(arm_seed(root, identity.arm), identity.slot))
    }
}

/// Arm zero is exactly the historical root; other arms use domain-separated
/// streams. The caller still owns the experiment seed.
pub fn arm_seed(root: u64, arm: u32) -> u64 {
    if arm == 0 {
        root
    } else {
        crate::hash::splitmix64(root ^ 0x7072_6f63_6172_6d31 ^ u64::from(arm))
    }
}

/// Adjacent slots share noise and differ in radius, as in the legacy kernel.
pub fn legacy_seed(root: u64, slot: u32) -> u64 {
    let stream = slot / 2;
    let (low, high) = (root as u32, (root >> 32) as u32);
    u64::from(stream_word(low, high, stream))
        | (u64::from(stream_word(high, low, stream ^ 0x9e37_79b9)) << 32)
}

pub fn stream_word(low: u32, high: u32, element: u32) -> u32 {
    let mut value = low ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= high;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

/// Borrowed acquisition metadata. No candidate model is allocated or retained.
#[derive(Clone, Copy, Debug)]
pub struct ProceduralCandidate<'a> {
    pub identity: CandidateIdentity,
    pub seed: u64,
    pub radius: f32,
    pub reference_correlation: f32,
    pub history_distances: &'a [(i64, f32)],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_streams() {
        for (arms, slots) in [(2, 2), (4, 1)] {
            let pool = ProceduralPoolConfig {
                arms,
                candidates_per_arm: slots,
            }
            .resolve_resident()
            .unwrap();
            let seeds: Vec<_> = (0..4)
                .map(|i| pool.seed(73, pool.identity(i).unwrap()).unwrap())
                .collect();
            if slots == 2 {
                assert_eq!(seeds[0], seeds[1]);
                assert_eq!(seeds[2], seeds[3]);
                assert_ne!(seeds[0], seeds[2]);
            } else {
                for i in 0..4 {
                    for j in 0..i {
                        assert_ne!(seeds[i], seeds[j]);
                    }
                }
            }
        }
    }

    #[test]
    fn logical_ids() {
        let pool = ProceduralPoolConfig {
            arms: 3,
            candidates_per_arm: 4,
        }
        .resolve()
        .unwrap();
        assert_eq!(pool.count(), 12);
        for index in 0..pool.count() {
            let identity = pool.identity(index).unwrap();
            assert_eq!(pool.index(identity).unwrap(), index);
            assert_eq!(identity.arm, index / 4);
            assert_eq!(identity.slot, index % 4);
        }
        assert!(pool.identity(12).is_err());
        assert!(pool.index(CandidateIdentity { arm: 3, slot: 0 }).is_err());
        assert!(pool.index(CandidateIdentity { arm: 0, slot: 4 }).is_err());
    }

    #[test]
    fn checked_counts() {
        for config in [
            ProceduralPoolConfig {
                arms: 0,
                candidates_per_arm: 4,
            },
            ProceduralPoolConfig {
                arms: 1,
                candidates_per_arm: 0,
            },
            ProceduralPoolConfig {
                arms: u32::MAX,
                candidates_per_arm: 2,
            },
        ] {
            assert!(config.resolve().is_err());
        }
    }

    #[test]
    fn legacy_streams() {
        let pool = ProceduralPoolConfig {
            arms: 4,
            candidates_per_arm: 4,
        }
        .resolve()
        .unwrap();
        for root in [0, 1, u64::MAX, 0x0123_4567_89ab_cdef] {
            for slot in 0..4 {
                let identity = CandidateIdentity { arm: 0, slot };
                assert_eq!(pool.seed(root, identity).unwrap(), legacy_seed(root, slot));
            }
            assert_eq!(arm_seed(root, 0), root);
            assert_ne!(arm_seed(root, 1), arm_seed(root, 2));
            assert_ne!(
                pool.seed(root, CandidateIdentity { arm: 0, slot: 0 })
                    .unwrap(),
                pool.seed(root, CandidateIdentity { arm: 1, slot: 0 })
                    .unwrap()
            );
        }
        assert_eq!(
            pool.seed(7, CandidateIdentity { arm: 0, slot: 0 }).unwrap(),
            pool.seed(7, CandidateIdentity { arm: 0, slot: 1 }).unwrap()
        );
    }

    #[test]
    fn resident_requests() {
        assert_eq!(
            ProceduralPoolConfig::default().resolve_resident().unwrap(),
            ProceduralPool::legacy()
        );
        let config = ProceduralPoolConfig {
            arms: 4,
            candidates_per_arm: 4,
        };
        assert!(
            config
                .resolve_resident()
                .unwrap_err()
                .contains("GPU pool/selection ABI")
        );
        assert!(
            ProceduralPoolConfig {
                arms: 1,
                candidates_per_arm: 8
            }
            .resolve_resident()
            .is_err()
        );
    }

    #[test]
    fn legacy_vectors() {
        for (root, first, second) in [
            (0, 0x64be_0fdc_0000_0000, 0xe95d_acfa_acf4_c579),
            (1, 0xe053_e0b2_9c2c_3535, 0x6dc9_ff5f_4922_3844),
            (u64::MAX, 0xdbc6_255b_8903_4b71, 0x7299_3e00_a5bf_f79e),
            (
                0x0123_4567_89ab_cdef,
                0x127f_b55f_7644_bf59,
                0xd3de_f4e1_9631_e136,
            ),
        ] {
            assert_eq!(legacy_seed(root, 0), first);
            assert_eq!(legacy_seed(root, 1), first);
            assert_eq!(legacy_seed(root, 2), second);
            assert_eq!(legacy_seed(root, 3), second);
        }
    }
}
