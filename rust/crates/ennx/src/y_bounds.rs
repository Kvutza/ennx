//! Optional per-metric output bounds for ENNX.

use std::path::Path;

use ndarray::{Array2, Array3, ArrayView2};

use crate::error::ENNError;

pub fn unbounded_bounds(metrics: usize) -> Array2<f64> {
    Array2::from_shape_fn((metrics, 2), |(_, j)| {
        if j == 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    })
}

pub fn identity_bounds(bounds: &Array2<f64>) -> bool {
    bounds
        .rows()
        .into_iter()
        .all(|r| r[0] == f64::NEG_INFINITY && r[1] == f64::INFINITY)
}

pub fn validate_bounds(bounds: &Array2<f64>, metrics: usize) -> Result<(), ENNError> {
    if bounds.dim() != (metrics, 2) {
        return Err(ENNError::InvalidShape {
            expected: vec![metrics, 2],
            got: bounds.shape().to_vec(),
        });
    }
    for i in 0..metrics {
        let (a, b) = (bounds[[i, 0]], bounds[[i, 1]]);
        if a.is_nan() || b.is_nan() || (a.is_finite() && b.is_finite() && a >= b) {
            return Err(ENNError::InvalidParameter(format!(
                "invalid y_bounds row {i}: ({a}, {b})"
            )));
        }
        if (!a.is_finite() && a != f64::NEG_INFINITY) || (!b.is_finite() && b != f64::INFINITY) {
            return Err(ENNError::InvalidParameter(format!(
                "open y_bounds must use -inf/+inf at row {i}"
            )));
        }
    }
    Ok(())
}

fn warp_scalar(y: f64, a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (false, false) => y,
        (true, false) => (y - a).ln(),
        (false, true) => -(b - y).ln(),
        (true, true) => {
            let u = (y - a) / (b - a);
            (u / (1.0 - u)).ln()
        }
    }
}

fn inv_scalar(z: f64, a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (false, false) => z,
        (true, false) => a + z.exp(),
        (false, true) => b - (-z).exp(),
        (true, true) => {
            let s = 1.0 / (1.0 + (-z).exp());
            a + (b - a) * s
        }
    }
}

pub fn warp_y(y: ArrayView2<f64>, bounds: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
    validate_bounds(bounds, y.ncols())?;
    let mut out = Array2::zeros(y.raw_dim());
    for ((i, j), value) in y.indexed_iter() {
        let (a, b) = (bounds[[j, 0]], bounds[[j, 1]]);
        if !value.is_finite() || (a.is_finite() && *value <= a) || (b.is_finite() && *value >= b) {
            return Err(ENNError::InvalidParameter(format!(
                "y[{i},{j}] is outside its open bounds"
            )));
        }
        out[[i, j]] = warp_scalar(*value, a, b);
    }
    Ok(out)
}

pub fn inv_y(z: ArrayView2<f64>, bounds: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(z.raw_dim(), |(i, j)| {
        inv_scalar(z[[i, j]], bounds[[j, 0]], bounds[[j, 1]])
    })
}

pub fn warp_yvar(
    y: ArrayView2<f64>,
    yvar: ArrayView2<f64>,
    bounds: &Array2<f64>,
) -> Result<Array2<f64>, ENNError> {
    if y.raw_dim() != yvar.raw_dim() {
        return Err(ENNError::InvalidShape {
            expected: y.shape().to_vec(),
            got: yvar.shape().to_vec(),
        });
    }
    validate_bounds(bounds, y.ncols())?;
    let mut out = Array2::zeros(y.raw_dim());
    for ((i, j), value) in y.indexed_iter() {
        let (a, b) = (bounds[[j, 0]], bounds[[j, 1]]);
        let jac = match (a.is_finite(), b.is_finite()) {
            (false, false) => 1.0,
            (true, false) => 1.0 / (*value - a),
            (false, true) => 1.0 / (b - *value),
            (true, true) => {
                let u = (*value - a) / (b - a);
                1.0 / ((b - a) * u * (1.0 - u))
            }
        };
        let v = jac * jac * yvar[[i, j]];
        if !v.is_finite() {
            return Err(ENNError::InvalidParameter(format!(
                "warped yvar[{i},{j}] is not finite"
            )));
        }
        out[[i, j]] = v;
    }
    Ok(out)
}

pub fn naturalize_yvar(
    z: ArrayView2<f64>,
    zvar: ArrayView2<f64>,
    bounds: &Array2<f64>,
) -> Array2<f64> {
    Array2::from_shape_fn(z.raw_dim(), |(i, j)| {
        let (a, b) = (bounds[[j, 0]], bounds[[j, 1]]);
        let value = z[[i, j]];
        let jac = match (a.is_finite(), b.is_finite()) {
            (false, false) => 1.0,
            (true, false) => value.exp(),
            (false, true) => (-value).exp(),
            (true, true) => {
                let s = 1.0 / (1.0 + (-value).exp());
                (b - a) * s * (1.0 - s)
            }
        };
        zvar[[i, j]] * jac * jac
    })
}

fn bounds_json(bounds: &Array2<f64>) -> String {
    let rows = bounds
        .rows()
        .into_iter()
        .map(|row| {
            let value = |v: f64| {
                if v.is_finite() {
                    v.to_string()
                } else {
                    "null".to_string()
                }
            };
            format!("[{},{}]", value(row[0]), value(row[1]))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rows}]")
}

pub fn persist_metadata(work_dir: &Path, bounds: &Array2<f64>) -> Result<(), ENNError> {
    let path = work_dir.join("metadata.json");
    if !path.exists() {
        return Ok(());
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let field = format!("\"y_bounds\":{}", bounds_json(bounds));
    let updated = if let Some(start) = text.find("\"y_bounds\":") {
        let array_start = text[start..]
            .find('[')
            .map(|v| start + v)
            .ok_or_else(|| ENNError::InvalidParameter("malformed y_bounds metadata".to_string()))?;
        let mut depth = 0usize;
        let mut array_end = None;
        for (offset, ch) in text[array_start..].char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        array_end = Some(array_start + offset + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = array_end
            .ok_or_else(|| ENNError::InvalidParameter("malformed y_bounds metadata".to_string()))?;
        format!("{}{}{}", &text[..start], field, &text[end..])
    } else {
        let end = text
            .rfind('}')
            .ok_or_else(|| ENNError::InvalidParameter("malformed metadata.json".to_string()))?;
        let comma = if text[..end].trim_end().ends_with('{') {
            ""
        } else {
            ","
        };
        format!("{}{}{}{}", &text[..end], comma, field, &text[end..])
    };
    std::fs::write(path, updated).map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub fn load_metadata(work_dir: &Path, metrics: usize) -> Result<Option<Array2<f64>>, ENNError> {
    let path = work_dir.join("metadata.json");
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let Some(field_start) = text.find("\"y_bounds\":") else {
        return Ok(None);
    };
    let start = text[field_start..]
        .find('[')
        .map(|v| field_start + v)
        .ok_or_else(|| ENNError::InvalidParameter("malformed y_bounds metadata".to_string()))?;
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut depth = 0usize;
    for ch in text[start..].chars() {
        match ch {
            '[' => depth += 1,
            ']' => {
                if !token.trim().is_empty() {
                    tokens.push(token.trim().to_string());
                    token.clear();
                }
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' => {
                if !token.trim().is_empty() {
                    tokens.push(token.trim().to_string());
                    token.clear();
                }
            }
            c if !c.is_whitespace() => token.push(c),
            _ => {}
        }
    }
    if tokens.len() != metrics * 2 {
        return Err(ENNError::InvalidParameter(
            "persisted y_bounds shape mismatch".to_string(),
        ));
    }
    let mut bounds = Array2::zeros((metrics, 2));
    for i in 0..metrics {
        bounds[[i, 0]] = if tokens[2 * i] == "null" {
            f64::NEG_INFINITY
        } else {
            tokens[2 * i]
                .parse()
                .map_err(|_| ENNError::InvalidParameter("invalid y_bounds metadata".to_string()))?
        };
        bounds[[i, 1]] = if tokens[2 * i + 1] == "null" {
            f64::INFINITY
        } else {
            tokens[2 * i + 1]
                .parse()
                .map_err(|_| ENNError::InvalidParameter("invalid y_bounds metadata".to_string()))?
        };
    }
    validate_bounds(&bounds, metrics)?;
    Ok(Some(bounds))
}

pub fn naturalize(
    mu: &mut Array2<f64>,
    se: &mut Array2<f64>,
    se_epi: &mut Array2<f64>,
    se_ale: &mut Array2<f64>,
    bounds: &Array2<f64>,
) {
    if identity_bounds(bounds) {
        return;
    }
    for ((i, j), value) in mu.indexed_iter_mut() {
        let (a, b) = (bounds[[j, 0]], bounds[[j, 1]]);
        let z = *value;
        let jac = match (a.is_finite(), b.is_finite()) {
            (false, false) => 1.0,
            (true, false) => z.exp(),
            (false, true) => (-z).exp(),
            (true, true) => {
                let s = 1.0 / (1.0 + (-z).exp());
                (b - a) * s * (1.0 - s)
            }
        }
        .abs();
        *value = inv_scalar(z, a, b);
        se[[i, j]] *= jac;
        se_epi[[i, j]] *= jac;
        se_ale[[i, j]] *= jac;
    }
}

pub fn naturalize_batch(
    mu: &mut Array3<f64>,
    se: &mut Array3<f64>,
    se_epi: &mut Array3<f64>,
    se_ale: &mut Array3<f64>,
    bounds: &Array2<f64>,
) {
    if identity_bounds(bounds) {
        return;
    }
    for ((p, i, j), value) in mu.indexed_iter_mut() {
        let (a, b) = (bounds[[j, 0]], bounds[[j, 1]]);
        let z = *value;
        let jac = match (a.is_finite(), b.is_finite()) {
            (false, false) => 1.0,
            (true, false) => z.exp(),
            (false, true) => (-z).exp(),
            (true, true) => {
                let s = 1.0 / (1.0 + (-z).exp());
                (b - a) * s * (1.0 - s)
            }
        }
        .abs();
        *value = inv_scalar(z, a, b);
        se[[p, i, j]] *= jac;
        se_epi[[p, i, j]] *= jac;
        se_ale[[p, i, j]] *= jac;
    }
}

pub fn inverse_draws(draws: &mut Array3<f64>, bounds: &Array2<f64>) {
    if identity_bounds(bounds) {
        return;
    }
    let metrics = draws.shape()[2];
    assert_eq!(metrics, bounds.nrows());
    for ((_, _, j), value) in draws.indexed_iter_mut() {
        *value = inv_scalar(*value, bounds[[j, 0]], bounds[[j, 1]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bounded_trip2() {
        let bounds = array![[0.0, 1.0], [0.0, f64::INFINITY]];
        let y = array![[0.25, 2.0], [0.75, 5.0]];
        let z = warp_y(y.view(), &bounds).unwrap();
        let restored = inv_y(z.view(), &bounds);
        assert!((restored - y).iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn rejects_interval() {
        let bounds = array![[0.0, 1.0]];
        assert!(warp_y(array![[0.0]].view(), &bounds).is_err());
        assert!(warp_y(array![[1.0]].view(), &bounds).is_err());
    }
}
