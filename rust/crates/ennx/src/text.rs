//! Exact decoder for the immutable byte-level BPE corpus tokenizer.

use ennx_wire::json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct ByteDecoder {
    vocabulary: Vec<Option<String>>,
    special: HashMap<u32, String>,
    bytes: HashMap<char, u8>,
}

impl ByteDecoder {
    pub fn load(path: &Path) -> Result<Self, String> {
        let document: Value = ennx_wire::json::from_reader(
            std::fs::File::open(path)
                .map_err(|error| format!("open tokenizer {}: {error}", path.display()))?,
        )
        .map_err(|error| format!("parse tokenizer {}: {error}", path.display()))?;
        Self::from_document(&document)
    }

    fn from_document(document: &Value) -> Result<Self, String> {
        let entries = ennx_wire::json::pointer(document, "/model/vocab")
            .and_then(|value| value.as_map())
            .ok_or("tokenizer model.vocab must be an object")?;
        let maximum = entries
            .values()
            .filter_map(|value| value.as_u64())
            .max()
            .ok_or("tokenizer vocabulary is empty")? as usize;
        let mut vocabulary = vec![None; maximum + 1];
        for (piece, id) in entries {
            let piece = piece
                .as_str()
                .ok_or("tokenizer vocabulary pieces must be strings")?;
            let id = id
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                .ok_or("tokenizer vocabulary ID must be a nonnegative integer")?;
            if id >= vocabulary.len() || vocabulary[id].replace(piece.to_owned()).is_some() {
                return Err("tokenizer vocabulary IDs must be unique".into());
            }
        }
        let mut special = HashMap::new();
        for entry in document
            .get("added_tokens")
            .and_then(|value| value.as_seq())
            .into_iter()
            .flatten()
        {
            if entry.get("special").and_then(|value| value.as_bool()) != Some(true) {
                continue;
            }
            let id = entry
                .get("id")
                .and_then(|value| value.as_u64())
                .and_then(|id| u32::try_from(id).ok())
                .ok_or("special token ID must fit u32")?;
            let content = entry
                .get("content")
                .and_then(|value| value.as_str())
                .ok_or("special token content must be a string")?;
            special.insert(id, content.to_owned());
        }
        Ok(Self {
            vocabulary,
            special,
            bytes: byte_decoder(),
        })
    }

    #[cfg(test)]
    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(tokens)?).into_owned())
    }

    /// Preserve arbitrary byte sequences; UTF-8 replacement is a display policy.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, String> {
        let mut decoded = Vec::new();
        for &token in tokens {
            if let Some(content) = self.special.get(&token) {
                decoded.extend_from_slice(content.as_bytes());
                continue;
            }
            let piece = self
                .vocabulary
                .get(token as usize)
                .and_then(Option::as_deref)
                .ok_or_else(|| format!("tokenizer has no vocabulary entry for ID {token}"))?;
            for character in piece.chars() {
                decoded.push(*self.bytes.get(&character).ok_or_else(|| {
                    format!("invalid byte-level tokenizer character {character:?}")
                })?);
            }
        }
        Ok(decoded)
    }
}

fn byte_decoder() -> HashMap<char, u8> {
    let mut direct = [false; 256];
    for byte in (b'!'..=b'~').chain(0xa1..=0xac).chain(0xae..=0xff) {
        direct[byte as usize] = true;
    }
    let mut next = 256u32;
    let mut result = HashMap::with_capacity(256);
    for byte in 0u16..=255 {
        let codepoint = if direct[byte as usize] {
            u32::from(byte)
        } else {
            let codepoint = next;
            next += 1;
            codepoint
        };
        result.insert(
            char::from_u32(codepoint).expect("byte alphabet is valid Unicode"),
            byte as u8,
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_pieces() {
        let document = ennx_wire::json::json!({
            "model":{"vocab":{"<|unknown|>":0,"001":1,"¯":2,"manager":3}},
            "added_tokens":[{"id":0,"content":"<|unknown|>","special":true}]
        });
        let tokenizer = ByteDecoder::from_document(&document).unwrap();
        assert_eq!(
            tokenizer.decode(&[1, 3, 0]).unwrap(),
            "001manager<|unknown|>"
        );
        assert_eq!(tokenizer.decode(&[2]).unwrap(), "�");
        assert_eq!(tokenizer.decode_bytes(&[2]).unwrap(), [0xaf]);
    }

    #[test]
    fn bpe_bytes() {
        let document = ennx_wire::json::json!({
            "model":{"vocab":{"return":0,"ret":1,"urn":2,"¯":3,"®":4}}
        });
        let tokenizer = ByteDecoder::from_document(&document).unwrap();
        let bytes = tokenizer.decode_bytes(&[0, 3]).unwrap();
        let reference = crate::reconstruction::Reference::new(&bytes);
        assert_eq!(
            reference.reward(&tokenizer.decode_bytes(&[1, 2, 3]).unwrap()),
            (0, 1.0)
        );
        assert_eq!(
            reference.distance(&tokenizer.decode_bytes(&[1, 2, 4]).unwrap()),
            1
        );
    }

    #[test]
    fn utf8_join() {
        let document = ennx_wire::json::json!({
            "model":{"vocab":{"Ã":0,"©":1}}
        });
        let tokenizer = ByteDecoder::from_document(&document).unwrap();
        assert_eq!(tokenizer.decode_bytes(&[0, 1]).unwrap(), [0xc3, 0xa9]);
        assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), "é");
        assert!(tokenizer.decode_bytes(&[2]).is_err());
    }
}
