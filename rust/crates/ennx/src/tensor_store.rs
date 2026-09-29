//! Framework-neutral tensor identity, versioning, and Safetensors interchange.
//!
//! Execution layouts are derived artifacts. Stable tensor IDs and logical
//! coordinates remain unchanged when a backend packs or transposes storage.

use deser::{Deserialize, Serialize};
use ennx_wire::json::Value;
use memmap2::{Mmap, MmapOptions};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;

const SAFETENSORS_PREFIX: usize = 8;
const SCHEMA: &str = "ennx.tensor-manifest.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[deser(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    Bf16,
    F32,
    F64,
}

const DTYPES: [DType; 13] = [
    DType::Bool,
    DType::U8,
    DType::I8,
    DType::I16,
    DType::U16,
    DType::I32,
    DType::U32,
    DType::I64,
    DType::U64,
    DType::F16,
    DType::Bf16,
    DType::F32,
    DType::F64,
];
const DTYPE_NAMES: [&str; 13] = [
    "BOOL", "U8", "I8", "I16", "U16", "I32", "U32", "I64", "U64", "F16", "BF16", "F32", "F64",
];

impl DType {
    pub const fn bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        DTYPE_NAMES
            .iter()
            .position(|name| *name == value)
            .map(|index| DTYPES[index])
            .ok_or_else(|| format!("unsupported Safetensors dtype {value:?}"))
    }

    const fn safetensors(self) -> &'static str {
        DTYPE_NAMES[self as usize]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub id: u64,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_len: u64,
}

impl TensorSpec {
    pub fn contiguous(
        name: impl Into<String>,
        dtype: DType,
        shape: Vec<usize>,
        byte_offset: u64,
    ) -> Result<Self, String> {
        let name = name.into();
        if name.is_empty() || shape.is_empty() || shape.contains(&0) {
            return Err("tensor name and dimensions must be nonempty".into());
        }
        let elements = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension).ok_or("tensor shape overflow")
        })?;
        let byte_len = elements
            .checked_mul(dtype.bytes())
            .ok_or("tensor byte length overflow")? as u64;
        Ok(Self {
            id: crate::hash::tensor_key(&name),
            name,
            dtype,
            shape,
            byte_offset,
            byte_len,
        })
    }

    pub fn coordinate_key(&self, logical_index: u64) -> Result<u64, String> {
        if logical_index >= self.byte_len / self.dtype.bytes() as u64 {
            return Err(format!("logical coordinate exceeds tensor {:?}", self.name));
        }
        Ok(crate::hash::splitmix64(
            self.id ^ crate::hash::splitmix64(logical_index),
        ))
    }
}

/// Stable identity for a logical interval. Backend byte offsets and packing
/// deliberately do not participate in this key.
pub fn block_key(tensor_id: u64, logical_start: u64, logical_len: u64) -> Result<u64, String> {
    if logical_len == 0 || logical_start.checked_add(logical_len).is_none() {
        return Err("logical tensor block must have a nonempty finite range".into());
    }
    Ok(crate::hash::splitmix64(
        tensor_id
            ^ crate::hash::splitmix64(logical_start)
            ^ crate::hash::splitmix64(logical_len ^ 0xa076_1d64_78bd_642f),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorManifest {
    pub schema: String,
    pub byte_len: u64,
    pub tensors: Vec<TensorSpec>,
}

impl TensorManifest {
    pub fn new(byte_len: u64, tensors: Vec<TensorSpec>) -> Result<Self, String> {
        let manifest = Self {
            schema: SCHEMA.into(),
            byte_len,
            tensors,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!(
                "unsupported tensor manifest schema {:?}",
                self.schema
            ));
        }
        let mut names = BTreeSet::new();
        let mut ids = BTreeSet::new();
        let mut ranges = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let rebuilt = TensorSpec::contiguous(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                tensor.byte_offset,
            )?;
            if rebuilt.id != tensor.id || rebuilt.byte_len != tensor.byte_len {
                return Err(format!(
                    "invalid identity or size for tensor {:?}",
                    tensor.name
                ));
            }
            let end = tensor
                .byte_offset
                .checked_add(tensor.byte_len)
                .ok_or("tensor range overflow")?;
            if end > self.byte_len {
                return Err(format!("tensor {:?} exceeds its arena", tensor.name));
            }
            if !names.insert(&tensor.name) || !ids.insert(tensor.id) {
                return Err(format!("duplicate tensor identity {:?}", tensor.name));
            }
            ranges.push((tensor.byte_offset, end, &tensor.name));
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "tensor storage overlaps between {:?} and {:?}",
                    pair[0].2, pair[1].2
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorBlock {
    pub key: u64,
    pub tensor_id: u64,
    pub logical_start: u64,
    pub logical_len: u64,
    pub scale: f32,
}

impl TensorBlock {
    pub fn new(
        tensor_id: u64,
        logical_start: u64,
        logical_len: u64,
        scale: f32,
    ) -> Result<Self, String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("tensor block scale must be positive and finite".into());
        }
        Ok(Self {
            key: block_key(tensor_id, logical_start, logical_len)?,
            tensor_id,
            logical_start,
            logical_len,
            scale,
        })
    }
}

/// An exact virtual candidate. Backends may resolve this transform lazily per
/// tile instead of allocating a complete candidate arena.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorVersion {
    pub parent: u64,
    pub id: u64,
    pub seed: u64,
    pub radius: f32,
    pub distribution: String,
    pub blocks: Vec<TensorBlock>,
    #[deser(default)]
    pub basis_seed: Option<u64>,
    #[deser(default)]
    pub threshold_words: Vec<u64>,
}

impl TensorVersion {
    pub fn perturb(
        parent: u64,
        seed: u64,
        radius: f32,
        distribution: impl Into<String>,
        mut blocks: Vec<TensorBlock>,
    ) -> Result<Self, String> {
        if !radius.is_finite() || radius < 0.0 || blocks.is_empty() {
            return Err(
                "tensor version requires a finite radius and positive finite scales".into(),
            );
        }
        let distribution = distribution.into();
        if distribution.is_empty() {
            return Err("tensor version distribution must be nonempty".into());
        }
        blocks.sort_unstable_by_key(|block| block.key);
        let mut id = crate::hash::splitmix64(parent ^ seed);
        id = crate::hash::splitmix64(id ^ u64::from(radius.to_bits()));
        let mut seen = BTreeSet::new();
        for block in &blocks {
            if block.key != block_key(block.tensor_id, block.logical_start, block.logical_len)?
                || !block.scale.is_finite()
                || block.scale <= 0.0
                || !seen.insert(block.key)
            {
                return Err("tensor version contains an invalid or duplicate logical block".into());
            }
            id = crate::hash::splitmix64(id ^ block.key ^ u64::from(block.scale.to_bits()));
        }
        for byte in distribution.bytes() {
            id = crate::hash::splitmix64(id ^ u64::from(byte));
        }
        Ok(Self {
            parent,
            id,
            seed,
            radius,
            distribution,
            blocks,
            basis_seed: None,
            threshold_words: Vec::new(),
        })
    }

    pub fn threshold(
        parent: u64,
        seed: u64,
        basis_seed: u64,
        radius: f32,
        blocks: Vec<TensorBlock>,
        words: &[u64; crate::threshold::TABLE_WORDS],
    ) -> Result<Self, String> {
        let mut version = Self::perturb(parent, seed, radius, "polynomial_threshold", blocks)?;
        version.basis_seed = Some(basis_seed);
        version.threshold_words.extend_from_slice(words);
        version.id = crate::hash::splitmix64(version.id ^ basis_seed);
        for &word in words {
            version.id = crate::hash::splitmix64(version.id ^ word);
        }
        Ok(version)
    }
}

pub struct TensorData<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

#[derive(Debug, Deserialize, Serialize)]
struct SafeTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct TensorView {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    range: Range<usize>,
}

pub struct SafeTensors {
    map: Mmap,
    tensors: Vec<TensorView>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
        // SAFETY: the map is read-only and retained for every returned view.
        let map = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| format!("map {}: {error}", path.display()))?;
        if map.len() < SAFETENSORS_PREFIX {
            return Err("Safetensors file is shorter than its header prefix".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(
            map[..SAFETENSORS_PREFIX].try_into().unwrap(),
        ))
        .map_err(|_| "Safetensors header length exceeds usize")?;
        let data_start = SAFETENSORS_PREFIX
            .checked_add(header_len)
            .ok_or("Safetensors header range overflow")?;
        if data_start > map.len() {
            return Err("Safetensors header exceeds the file".into());
        }
        let root: BTreeMap<String, Value> = ennx_wire::json::from_slice(&map[8..data_start])
            .map_err(|error| format!("invalid Safetensors header: {error}"))?;
        let mut tensors = Vec::new();
        let mut metadata = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in root {
            if name == "__metadata__" {
                metadata = ennx_wire::json::from_value(value)
                    .map_err(|error| format!("invalid Safetensors metadata: {error}"))?;
                continue;
            }
            let header: SafeTensorHeader = ennx_wire::json::from_value(value)
                .map_err(|error| format!("invalid tensor {name:?}: {error}"))?;
            let dtype = DType::parse(&header.dtype)?;
            let elements = header.shape.iter().try_fold(1usize, |count, &dimension| {
                count.checked_mul(dimension).ok_or("tensor shape overflow")
            })?;
            let expected = elements
                .checked_mul(dtype.bytes())
                .ok_or("tensor byte length overflow")?;
            let [start, end] = header.data_offsets;
            if start > end || end - start != expected {
                return Err(format!("invalid data range for tensor {name:?}"));
            }
            let absolute_start = data_start
                .checked_add(start)
                .ok_or("tensor offset overflow")?;
            let absolute_end = data_start
                .checked_add(end)
                .ok_or("tensor offset overflow")?;
            if absolute_end > map.len() {
                return Err(format!("tensor {name:?} exceeds the file"));
            }
            ranges.push((start, end, name.clone()));
            tensors.push(TensorView {
                name,
                dtype,
                shape: header.shape,
                range: absolute_start..absolute_end,
            });
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "Safetensors data overlap between {:?} and {:?}",
                    pair[0].2, pair[1].2
                ));
            }
        }
        tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        Ok(Self {
            map,
            tensors,
            metadata,
        })
    }

    pub fn tensors(&self) -> &[TensorView] {
        &self.tensors
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn bytes(&self, tensor: &TensorView) -> &[u8] {
        &self.map[tensor.range.clone()]
    }

    pub fn tensor(&self, name: &str) -> Option<(&TensorView, &[u8])> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| (&self.tensors[index], self.bytes(&self.tensors[index])))
    }
}

pub fn write_safetensors(
    path: &Path,
    metadata: BTreeMap<String, String>,
    tensors: &[TensorData<'_>],
) -> Result<(), String> {
    if tensors.is_empty() {
        return Err("cannot write an empty Safetensors file".into());
    }
    let mut names = BTreeSet::new();
    let mut sorted = tensors.iter().collect::<Vec<_>>();
    sorted.sort_unstable_by_key(|tensor| tensor.name);
    let mut root = BTreeMap::<String, Value>::new();
    root.insert(
        "__metadata__".into(),
        ennx_wire::json::to_value(metadata).map_err(|error| error.to_string())?,
    );
    let mut offset = 0usize;
    for tensor in &sorted {
        if tensor.name.is_empty() || !names.insert(tensor.name) {
            return Err(format!("empty or duplicate tensor name {:?}", tensor.name));
        }
        let elements = tensor.shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension).ok_or("tensor shape overflow")
        })?;
        let expected = elements
            .checked_mul(tensor.dtype.bytes())
            .ok_or("tensor byte length overflow")?;
        if expected != tensor.bytes.len() {
            return Err(format!(
                "tensor {:?} has {} bytes, expected {expected}",
                tensor.name,
                tensor.bytes.len()
            ));
        }
        let end = offset
            .checked_add(expected)
            .ok_or("Safetensors data range overflow")?;
        root.insert(
            tensor.name.into(),
            ennx_wire::json::to_value(SafeTensorHeader {
                dtype: tensor.dtype.safetensors().into(),
                shape: tensor.shape.to_vec(),
                data_offsets: [offset, end],
            })
            .map_err(|error| error.to_string())?,
        );
        offset = end;
    }
    let mut header = ennx_wire::json::to_vec(&root).map_err(|error| error.to_string())?;
    let padding = (8 - header.len() % 8) % 8;
    header.resize(header.len() + padding, b' ');
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(&(header.len() as u64).to_le_bytes())
        .and_then(|()| writer.write_all(&header))
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    for tensor in sorted {
        writer
            .write_all(tensor.bytes)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safetensors_bits() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("roundtrip.safetensors");
        let first = [0x00, 0x3c, 0x55, 0x35];
        let second = [1_u8, 2, 3];
        write_safetensors(
            &path,
            BTreeMap::from([("format".into(), "ennx".into())]),
            &[
                TensorData {
                    name: "weights",
                    dtype: DType::F16,
                    shape: &[2],
                    bytes: &first,
                },
                TensorData {
                    name: "tokens",
                    dtype: DType::U8,
                    shape: &[3],
                    bytes: &second,
                },
            ],
        )
        .unwrap();
        let mapped = SafeTensors::open(&path).unwrap();
        assert_eq!(mapped.tensor("weights").unwrap().1, first);
        assert_eq!(mapped.tensor("tokens").unwrap().1, second);
        assert_eq!(mapped.metadata()["format"], "ennx");
    }

    #[test]
    fn logical_identity() {
        let first = TensorSpec::contiguous("layer.0.weight", DType::F16, vec![2, 3], 0).unwrap();
        let moved = TensorSpec::contiguous("layer.0.weight", DType::F16, vec![2, 3], 4096).unwrap();
        assert_eq!(first.id, moved.id);
        assert_eq!(
            first.coordinate_key(5).unwrap(),
            moved.coordinate_key(5).unwrap()
        );
        assert_eq!(
            block_key(first.id, 0, 6).unwrap(),
            block_key(moved.id, 0, 6).unwrap()
        );
    }

    #[test]
    fn exact_versions() {
        let blocks = vec![
            TensorBlock::new(17, 0, 64, 1.0).unwrap(),
            TensorBlock::new(17, 64, 64, 0.5).unwrap(),
        ];
        let first = TensorVersion::perturb(7, 11, 0.25, "gaussian", blocks.clone()).unwrap();
        let same = TensorVersion::perturb(7, 11, 0.25, "gaussian", blocks.clone()).unwrap();
        let other = TensorVersion::perturb(7, 12, 0.25, "gaussian", blocks).unwrap();
        assert_eq!(first.id, same.id);
        assert_ne!(first.id, other.id);
    }
}
