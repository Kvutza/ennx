//! Exact hypervolume calculation for bounded maximization problems.

use ndarray::{ArrayView1, ArrayView2};
use thiserror::Error;

/// Errors that can occur during hypervolume calculation.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HypervolumeError {
    /// Input array has wrong dimensionality.
    #[error("y must be 2-dimensional, got shape {0:?}")]
    InvalidDimension(Vec<usize>),
    /// Input array has wrong number of columns.
    #[error("y must have 2 columns, got shape {0:?}")]
    InvalidColumnCount(Vec<usize>),
    /// Reference point has wrong shape.
    #[error("ref_point must have 2 elements, got shape {0:?}")]
    InvalidRefPoint(usize),
    /// Input and reference dimensions disagree.
    #[error("y has {columns} columns but ref_point has {reference} elements")]
    DimensionMismatch { columns: usize, reference: usize },
}

/// Exact dominated hypervolume for maximization in one or more dimensions.
///
/// The resident optimizer has at most 128 observations and eight objectives,
/// so an exact recursive slicing algorithm is preferable to a sampled proxy.
/// Dominated rows are removed before slicing.
pub fn hypervolume_max(
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<f64, HypervolumeError> {
    if y.ndim() != 2 {
        return Err(HypervolumeError::InvalidDimension(y.shape().to_vec()));
    }
    if y.ncols() != ref_point.len() {
        return Err(HypervolumeError::DimensionMismatch {
            columns: y.ncols(),
            reference: ref_point.len(),
        });
    }
    if y.ncols() == 0 || y.nrows() == 0 {
        return Ok(0.0);
    }
    let mut points = y
        .rows()
        .into_iter()
        .filter(|row| {
            row.iter()
                .zip(ref_point)
                .all(|(value, reference)| value.is_finite() && value > reference)
        })
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    pareto_prune(&mut points);
    let reference = ref_point.iter().copied().collect::<Vec<_>>();
    Ok(slice_volume(&points, &reference))
}

fn dominates(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right).all(|(a, b)| a >= b) && left.iter().zip(right).any(|(a, b)| a > b)
}

fn pareto_prune(points: &mut Vec<Vec<f64>>) {
    let keep = (0..points.len())
        .map(|index| {
            !(0..points.len()).any(|other| {
                other != index
                    && (dominates(&points[other], &points[index])
                        || (points[other] == points[index] && other < index))
            })
        })
        .collect::<Vec<_>>();
    let mut cursor = 0;
    points.retain(|_| {
        let retain = keep[cursor];
        cursor += 1;
        retain
    });
}

fn slice_volume(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    if reference.len() == 1 {
        return points
            .iter()
            .map(|point| point[0])
            .fold(reference[0], f64::max)
            - reference[0];
    }
    let axis = reference.len() - 1;
    let mut edges = points.iter().map(|point| point[axis]).collect::<Vec<_>>();
    edges.sort_by(f64::total_cmp);
    edges.dedup_by(|left, right| left.total_cmp(right).is_eq());
    let mut previous = reference[axis];
    let mut volume = 0.0;
    for edge in edges {
        let projected = points
            .iter()
            .filter(|point| point[axis] >= edge)
            .map(|point| point[..axis].to_vec())
            .collect::<Vec<_>>();
        volume += (edge - previous) * slice_volume(&projected, &reference[..axis]);
        previous = edge;
    }
    volume
}

/// Calculate 2D hypervolume for maximization problems.
///
/// This implements the "walking frontier" algorithm for computing
/// the hypervolume indicator in 2D objective space.
///
/// # Arguments
///
/// * `y` - Array of shape (N, 2) containing objective values to maximize
/// * `ref_point` - Array of shape (2,) containing reference point for hypervolume
///
/// # Returns
///
/// The hypervolume as a float, or an error if inputs are invalid.
///
/// # Algorithm
///
/// 1. Filter points that dominate the reference point in both dimensions
/// 2. Sort remaining points by first objective in descending order (stable)
/// 3. Walk the frontier, accumulating rectangular areas
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ennx::hypervolume2d_max;
///
/// let y = array![[1.0, 0.5], [0.5, 1.0]];
/// let ref_point = array![0.0, 0.0];
/// let hv = hypervolume2d_max(&y.view(), &ref_point.view()).unwrap();
/// assert!((hv - 0.75).abs() < 1e-10);
/// ```
pub fn hypervolume2d_max(
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<f64, HypervolumeError> {
    // Validate dimensions
    if y.ndim() != 2 {
        return Err(HypervolumeError::InvalidDimension(y.shape().to_vec()));
    }
    if y.ncols() != 2 {
        return Err(HypervolumeError::InvalidColumnCount(y.shape().to_vec()));
    }
    if ref_point.len() != 2 {
        return Err(HypervolumeError::InvalidRefPoint(ref_point.len()));
    }

    // Handle empty input
    if y.nrows() == 0 {
        return Ok(0.0);
    }

    let ref0 = ref_point[0];
    let ref1 = ref_point[1];

    // Collect points that dominate reference point in both dimensions
    let mut dominating: Vec<(f64, f64)> = Vec::with_capacity(y.nrows());
    for row in y.rows() {
        let x0 = row[0];
        let x1 = row[1];
        if x0 > ref0 && x1 > ref1 {
            dominating.push((x0, x1));
        }
    }

    // If no points dominate, hypervolume is zero
    if dominating.is_empty() {
        return Ok(0.0);
    }

    // Sort by first objective in descending order (stable sort for parity).
    // total_cmp avoids panic if NaNs ever reach this point.
    dominating.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Walking frontier algorithm
    let mut hv = 0.0;
    let mut best_y1 = ref1;

    for i in 0..dominating.len() {
        let (x0, y1) = dominating[i];

        // Update best y1 seen so far
        if y1 > best_y1 {
            best_y1 = y1;
        }

        // Next x coordinate (reference if at end)
        let x_next = if i + 1 < dominating.len() {
            dominating[i + 1].0
        } else {
            ref0
        };

        // Accumulate rectangle area
        hv += (x0 - x_next) * (best_y1 - ref1);
    }

    Ok(hv)
}

/// Convenience function for 1D arrays.
///
/// # Panics
///
/// Panics if the conversion from 1D to 2D view fails.
pub fn hypervolume_1d(
    y: &ArrayView1<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<f64, HypervolumeError> {
    let y_2d = y.view().insert_axis(ndarray::Axis(0));
    hypervolume2d_max(&y_2d, ref_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayView2, array};

    fn hypervolume_origin(y: &ArrayView2<f64>) -> f64 {
        let ref_point = array![0.0, 0.0];
        hypervolume2d_max(y, &ref_point.view()).unwrap()
    }

    #[test]
    fn test_emptyarray() {
        let y = Array2::<f64>::zeros((0, 2));
        assert_eq!(hypervolume_origin(&y.view()), 0.0);
    }

    #[test]
    fn test_nodominatingpoints() {
        let y = array![[-1.0, -1.0], [-0.5, -0.5]];
        assert_eq!(hypervolume_origin(&y.view()), 0.0);
    }

    #[test]
    fn test_simplehypervolume() {
        let y = array![[1.0, 0.5], [0.5, 1.0]];
        let result = hypervolume_origin(&y.view());
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_threepoints() {
        let y = array![[1.0, 1.0], [0.2, 0.2], [0.5, 0.5]];
        let result = hypervolume_origin(&y.view());
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn volume_3d() {
        let y = array![[1.0, 1.0, 1.0], [2.0, 0.5, 0.5]];
        let reference = array![0.0, 0.0, 0.0];
        let volume = hypervolume_max(&y.view(), &reference.view()).unwrap();
        assert!((volume - 1.25).abs() < 1e-12);
    }

    #[test]
    fn volume_parity() {
        let y = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.2]];
        let reference = array![0.0, 0.0];
        let exact = hypervolume_max(&y.view(), &reference.view()).unwrap();
        let specialized = hypervolume2d_max(&y.view(), &reference.view()).unwrap();
        assert!((exact - specialized).abs() < 1e-12);
    }

    #[test]
    fn test_withnandoesnotpanic() {
        let y = array![[1.0, 0.5], [f64::NAN, 2.0], [0.8, 0.3]];
        let ref_point = array![0.0, 0.0];
        let result = hypervolume2d_max(&y.view(), &ref_point.view()).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_invalidcolumncount() {
        let y = array![[1.0, 0.5, 0.3]]; // 3 columns instead of 2
        let ref_point = array![0.0, 0.0];
        let result = hypervolume2d_max(&y.view(), &ref_point.view());
        assert!(matches!(
            result,
            Err(HypervolumeError::InvalidColumnCount(_))
        ));
    }

    #[test]
    fn test_invalidrefpoint() {
        let y = array![[1.0, 0.5]];
        let ref_point = array![0.0]; // Wrong length
        let result = hypervolume2d_max(&y.view(), &ref_point.view());
        assert!(matches!(result, Err(HypervolumeError::InvalidRefPoint(1))));
    }

    #[test]
    fn test_hypervolume2dmax1d() {
        let y = array![1.0, 0.5];
        let ref_point = array![0.0, 0.0];
        let result = hypervolume_1d(&y.view(), &ref_point.view()).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }
}
