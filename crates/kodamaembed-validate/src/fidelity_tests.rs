use super::*;
use crate::vectors::Matrix;

#[test]
fn identical_matrices_have_perfect_fidelity() {
    let m = Matrix::from_rows(&[vec![0.6, 0.8], vec![1.0, 0.0]]).unwrap();
    let summary = compare(&m, &m).unwrap();
    assert_eq!(summary.rows, 2);
    assert_eq!(summary.min_cosine, 1.0);
    assert_eq!(summary.max_l2, 0.0);
    assert_eq!(summary.max_abs, 0.0);
    assert_eq!(summary.non_finite, 0);
}

#[test]
fn a_perturbed_row_reports_its_distance_and_cosine() {
    let reference = Matrix::from_rows(&[vec![1.0, 0.0]]).unwrap();
    let candidate = Matrix::from_rows(&[vec![0.6, 0.8]]).unwrap();
    let summary = compare(&reference, &candidate).unwrap();
    assert!((summary.min_cosine - 0.6).abs() < 1e-7);
    assert!((summary.max_l2 - (0.4f64 * 0.4 + 0.8 * 0.8).sqrt()).abs() < 1e-7);
    assert!((summary.max_abs - 0.8).abs() < 1e-7);
    assert!(summary.max_norm_error < 1e-7);
}

#[test]
fn non_finite_rows_on_either_side_are_counted_and_excluded() {
    let reference =
        Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![f32::INFINITY, 0.0]]).unwrap();
    let candidate =
        Matrix::from_rows(&[vec![f32::NAN, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    let summary = compare(&reference, &candidate).unwrap();
    assert_eq!(summary.non_finite, 2);
    assert_eq!(summary.min_cosine, 1.0);
}

#[test]
fn shapes_must_match() {
    let a = Matrix::from_rows(&[vec![1.0, 0.0]]).unwrap();
    let b = Matrix::from_rows(&[vec![1.0, 0.0, 0.0]]).unwrap();
    assert!(matches!(compare(&a, &b), Err(FidelityError::Shape { .. })));
}

#[test]
fn malformed_public_matrices_are_rejected_before_rows_are_read() {
    for matrix in [
        Matrix {
            rows: 2,
            dim: 2,
            data: vec![1.0],
        },
        Matrix {
            rows: usize::MAX,
            dim: 2,
            data: vec![],
        },
    ] {
        assert!(matches!(
            compare(&matrix, &matrix),
            Err(FidelityError::InvalidMatrix { .. })
        ));
    }
}
