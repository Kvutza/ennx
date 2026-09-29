//! Validate logical pool requests before a backend can silently reinterpret them.
use super::ConfigOverrides;
use crate::procedural_pool::ProceduralPool;

impl ConfigOverrides {
    pub fn resident_pool(&self) -> Result<ProceduralPool, String> {
        self.proposal_pool.unwrap_or_default().resolve_resident()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_tune;
    use crate::procedural_pool::ProceduralPoolConfig;

    #[test]
    fn default_pool() {
        let text = "version=2\nexperiment='end-to-end'\n[run]\nrounds=1";
        let implicit = parse_tune(text).unwrap();
        let explicit = parse_tune(&format!("{text}\n[proposal]\narms=1\ncandidates=4")).unwrap();
        assert_eq!(implicit.resident_pool().unwrap(), ProceduralPool::legacy());
        assert_eq!(
            explicit.resident_pool().unwrap(),
            implicit.resident_pool().unwrap()
        );
        assert_eq!(
            explicit.resident_enn(17).unwrap().ask.seed,
            implicit.resident_enn(17).unwrap().ask.seed
        );
    }

    #[test]
    fn unsupported_pool() {
        let text = "version=2\nexperiment='end-to-end'\n[proposal]\narms=4\ncandidates=16";
        assert!(
            parse_tune(text)
                .unwrap_err()
                .contains("GPU pool/selection ABI")
        );
        let overrides = ConfigOverrides {
            proposal_pool: Some(ProceduralPoolConfig {
                arms: 4,
                candidates_per_arm: 4,
            }),
            ..Default::default()
        };
        assert!(overrides.resident_enn(0).is_err());
    }

    #[test]
    fn invalid_pool() {
        let text = "version=2\nexperiment='end-to-end'\n[proposal]\n";
        for fields in ["arms=0", "arms=1\ncandidates=0", "arms=1\ncandidate=4"] {
            assert!(parse_tune(&format!("{text}{fields}")).is_err());
        }
    }

    #[test]
    fn fixed_layouts() {
        for (arms, slots) in [(1, 4), (2, 2), (4, 1)] {
            let config = parse_tune(&format!(
                "version=2\nexperiment='end-to-end'\n[proposal]\narms={arms}\ncandidates={}",
                arms * slots
            ))
            .unwrap();
            let pool = config.resident_enn(0).unwrap().pool;
            assert_eq!(pool.arms(), arms);
            assert_eq!(pool.slots(), slots);
            assert_eq!(pool.count(), 4);
        }
    }
}
