//! Native quantized-weight package for resident experimental evaluators.
//!
//! ENNX reads this package itself.  Python/JAX is neither a loader nor a
//! buffer owner on the optimized path.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::forward_program::KdaPackedLinear;
use crate::trials::Parameter;

const MANIFEST: &str = "model.toml";

#[derive(Debug, Deserialize)]
struct Manifest {
    weights: String,
    scales: String,
    biases: String,
    #[serde(default)]
    linear: Vec<LinearManifest>,
}

#[derive(Debug, Deserialize)]
struct LinearManifest {
    name: String,
    byte_offset: usize,
    scale_offset: usize,
    bias_offset: usize,
    input_width: usize,
    output_width: usize,
    bits: u8,
    group_size: usize,
    #[serde(default)]
    element_offset: usize,
    #[serde(default)]
    perturb_whole: u32,
    #[serde(default)]
    perturb_threshold: u32,
}

/// Packed quantized weights and their named descriptors, loaded by ENNX.
#[derive(Debug)]
pub struct PackedModel {
    packed: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    linears: BTreeMap<String, KdaPackedLinear>,
}

impl PackedModel {
    /// Open an ENNX model package directory.
    ///
    /// The package contains `model.toml`, packed `u8` weights, and native
    /// little-endian `f32` scale and bias arenas.  These host arenas are copied
    /// into ENNX-owned Metal buffers exactly once by the resident executor.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, String> {
        let directory = directory.as_ref();
        let manifest_path = directory.join(MANIFEST);
        let manifest_text = fs::read_to_string(&manifest_path)
            .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?;
        let manifest: Manifest = toml::from_str(&manifest_text)
            .map_err(|error| format!("invalid {}: {error}", manifest_path.display()))?;
        let packed = read_file(directory, &manifest.weights)?;
        let scales = read_f32(directory, &manifest.scales)?;
        let biases = read_f32(directory, &manifest.biases)?;
        let mut linears = BTreeMap::new();
        for entry in manifest.linear {
            let descriptor = KdaPackedLinear {
                byte_offset: entry.byte_offset,
                scale_offset: entry.scale_offset,
                bias_offset: entry.bias_offset,
                input_width: entry.input_width,
                output_width: entry.output_width,
                bits: entry.bits,
                group_size: entry.group_size,
                element_offset: entry.element_offset,
                perturb_whole: entry.perturb_whole,
                perturb_threshold: entry.perturb_threshold,
            };
            descriptor.validate()?;
            if descriptor.byte_offset + descriptor.packed_bytes() > packed.len() {
                return Err(format!("linear {:?} exceeds weights arena", entry.name));
            }
            let groups = descriptor.row_groups() * descriptor.output_width;
            if descriptor.scale_offset + groups > scales.len()
                || descriptor.bias_offset + groups > biases.len()
            {
                return Err(format!(
                    "linear {:?} exceeds scale or bias arena",
                    entry.name
                ));
            }
            if linears.insert(entry.name.clone(), descriptor).is_some() {
                return Err(format!("duplicate linear name {:?}", entry.name));
            }
        }
        Ok(Self {
            packed,
            scales,
            biases,
            linears,
        })
    }

    pub fn packed(&self) -> &[u8] {
        &self.packed
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    pub fn trial_leaves(
        &self,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Vec<Parameter>, String> {
        let mut linears = self.linears.values().copied().collect::<Vec<_>>();
        linears.sort_by_key(|linear| linear.element_offset);
        let mut element_offset = 0usize;
        let mut byte_offset = 0usize;
        let mut leaves = Vec::with_capacity(linears.len());
        for linear in linears {
            if linear.element_offset != element_offset {
                return Err(format!(
                    "linear element offset {} does not continue parameter offset {element_offset}",
                    linear.element_offset
                ));
            }
            if linear.byte_offset != byte_offset {
                return Err(format!(
                    "linear byte offset {} does not continue packed byte offset {byte_offset}",
                    linear.byte_offset
                ));
            }
            let length = linear
                .input_width
                .checked_mul(linear.output_width)
                .ok_or("linear parameter count overflow")?;
            leaves.push(Parameter::new(
                linear.element_offset,
                length,
                linear.bits,
                scale,
                weight,
                radius,
            )?);
            element_offset = element_offset
                .checked_add(length)
                .ok_or("model parameter count overflow")?;
            byte_offset = byte_offset
                .checked_add(linear.packed_bytes())
                .ok_or("model packed byte count overflow")?;
        }
        Ok(leaves)
    }

    pub fn linear(&self, name: &str) -> Result<KdaPackedLinear, String> {
        self.linears
            .get(name)
            .copied()
            .ok_or_else(|| format!("model package has no linear {name:?}"))
    }

    /// Decode a one-row INT8 linear used as a KDA control vector.
    pub fn vector(&self, name: &str) -> Result<Vec<f32>, String> {
        let linear = self.linear(name)?;
        if linear.output_width != 1 || linear.bits != 8 {
            return Err(format!("{name:?} must be a one-row INT8 linear"));
        }
        let mut values = Vec::with_capacity(linear.input_width);
        for column in 0..linear.input_width {
            let code = self.packed[linear.byte_offset + column] as f32;
            let group = column / linear.group_size;
            values.push(
                code * self.scales[linear.scale_offset + group]
                    + self.biases[linear.bias_offset + group],
            );
        }
        Ok(values)
    }
}

fn read_file(directory: &Path, name: &str) -> Result<Vec<u8>, String> {
    let path = directory.join(name);
    fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn read_f32(directory: &Path, name: &str) -> Result<Vec<f32>, String> {
    let path = directory.join(name);
    let bytes = read_file(directory, name)?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!("{} is not a whole f32 arena", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact f32 chunk")))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::PackedModel;

    #[test]
    fn opens_python() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("model.toml"),
            "weights = 'weights.bin'\nscales = 'scales.bin'\nbiases = 'biases.bin'\n\n[[linear]]\nname = 'layer.decay'\nbyte_offset = 0\nscale_offset = 0\nbias_offset = 0\ninput_width = 4\noutput_width = 1\nbits = 8\ngroup_size = 2\n",
        )
        .unwrap();
        fs::write(directory.path().join("weights.bin"), [1_u8, 2, 3, 4]).unwrap();
        let arena = [2_f32, 3_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        fs::write(directory.path().join("scales.bin"), &arena).unwrap();
        fs::write(
            directory.path().join("biases.bin"),
            [0_f32, 1_f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let model = PackedModel::open(directory.path()).unwrap();
        assert_eq!(
            model.vector("layer.decay").unwrap(),
            vec![2.0, 4.0, 10.0, 13.0]
        );
        let leaves = model.trial_leaves(1.0, 1.0, 1.0).unwrap();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].length, 4);
    }
}
