//! Immutable fixed-shape token streams for the 4K pretraining experiment.

use std::path::Path;

const MAGIC: &[u8; 8] = b"ENNXPTN1";
const HEADER_BYTES: usize = 24;
pub(crate) const VOCAB: u32 = 8192;
pub(crate) const CONTEXT: u32 = 4096;

#[derive(Debug)]
pub(crate) struct PretrainDataset {
    sequences: u32,
    tokens: Vec<u16>,
}

impl PretrainDataset {
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("pretraining dataset {}: {error}", path.display()))?;
        if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
            return Err("pretraining dataset has an invalid ENNXPTN1 header".into());
        }
        let field = |offset| {
            u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("four-byte field"),
            )
        };
        let vocab = field(8);
        let context = field(12);
        let sequences = field(16);
        let reserved = field(20);
        if vocab != VOCAB || context != CONTEXT || sequences < 2 || sequences % 2 != 0 {
            return Err(format!(
                "pretraining dataset requires vocab={VOCAB}, context={CONTEXT}, and a positive even sequence count; found vocab={vocab}, context={context}, sequences={sequences}"
            ));
        }
        if reserved != 0 {
            return Err("pretraining dataset reserved header field must be zero".into());
        }
        let elements = usize::try_from(u64::from(sequences) * u64::from(context))
            .map_err(|_| "pretraining dataset size does not fit this machine")?;
        let expected = HEADER_BYTES
            .checked_add(
                elements
                    .checked_mul(2)
                    .ok_or("pretraining token byte overflow")?,
            )
            .ok_or("pretraining dataset byte size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "pretraining dataset has {} bytes, expected {expected}",
                bytes.len()
            ));
        }
        let tokens = bytes[HEADER_BYTES..]
            .chunks_exact(2)
            .map(|value| u16::from_le_bytes([value[0], value[1]]))
            .collect::<Vec<_>>();
        if let Some((index, token)) = tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| u32::from(*token) >= vocab)
        {
            return Err(format!(
                "pretraining token {token} at offset {index} exceeds the vocabulary"
            ));
        }
        Ok(Self { sequences, tokens })
    }

    pub(crate) fn batches(&self) -> u32 {
        self.sequences / 2
    }

    pub(crate) fn sequences(&self) -> u32 {
        self.sequences
    }

    pub(crate) fn prefix(&self, count: usize) -> Result<&[u16], String> {
        self.tokens.get(..count).ok_or_else(|| {
            format!(
                "context workload needs {count} distinct corpus positions, dataset contains {}",
                self.tokens.len()
            )
        })
    }

    pub(crate) fn sequence(&self, index: u32) -> Result<&[u16], String> {
        if index >= self.sequences {
            return Err(format!(
                "pretraining sequence {index} exceeds the {} available sequences",
                self.sequences
            ));
        }
        let start = index as usize * CONTEXT as usize;
        Ok(&self.tokens[start..start + CONTEXT as usize])
    }

    pub(crate) fn batch(&self, index: u32) -> Result<&[u16], String> {
        if index >= self.batches() {
            return Err(format!(
                "pretraining batch {index} exceeds the {} available batches",
                self.batches()
            ));
        }
        let elements = 2 * CONTEXT as usize;
        let start = index as usize * elements;
        Ok(&self.tokens[start..start + elements])
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT, HEADER_BYTES, MAGIC, PretrainDataset, VOCAB};

    fn document(sequences: u32) -> Vec<u8> {
        let elements = sequences as usize * CONTEXT as usize;
        let mut bytes = Vec::with_capacity(HEADER_BYTES + elements * 2);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VOCAB.to_le_bytes());
        bytes.extend_from_slice(&CONTEXT.to_le_bytes());
        bytes.extend_from_slice(&sequences.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for index in 0..elements {
            bytes.extend_from_slice(&((index as u16) % VOCAB as u16).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn fixed_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(4)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert_eq!(data.batches(), 2);
        assert_eq!(data.sequences(), 4);
        assert_eq!(data.sequence(3).unwrap().len(), CONTEXT as usize);
        assert!(data.sequence(4).is_err());
        assert_eq!(data.batch(1).unwrap().len(), 2 * CONTEXT as usize);
        assert!(data.batch(2).is_err());
    }

    #[test]
    fn invalid_tokens() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        let mut bytes = document(2);
        bytes[HEADER_BYTES..HEADER_BYTES + 2].copy_from_slice(&(VOCAB as u16).to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(
            PretrainDataset::load(&path)
                .unwrap_err()
                .contains("exceeds the vocabulary")
        );
    }
}
