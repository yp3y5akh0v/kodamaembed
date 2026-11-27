use super::*;

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("kodamaembed-vectors-{}-{name}", std::process::id()))
}

#[test]
fn a_matrix_round_trips_through_a_file() {
    let matrix =
        Matrix::from_rows(&[vec![1.0, -2.5, 3.25], vec![0.0, f32::MIN_POSITIVE, -0.0]]).unwrap();
    let path = temp_path("roundtrip");
    write(&path, &matrix).unwrap();
    let loaded = read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded, matrix);
    assert_eq!(loaded.row(1), &[0.0, f32::MIN_POSITIVE, -0.0]);
}

#[test]
fn ragged_rows_are_rejected() {
    let error = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
    assert!(matches!(
        error,
        VectorError::Ragged {
            row: 1,
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn ragged_late_rows_are_rejected_and_empty_matrices_remain_valid() {
    let rows = [vec![1.0; 256], vec![2.0; 256], vec![]];
    assert!(matches!(
        Matrix::from_rows(&rows),
        Err(VectorError::Ragged {
            row: 2,
            expected: 256,
            actual: 0
        })
    ));
    let empty = Matrix::from_rows(&[]).unwrap();
    assert!(empty.has_valid_shape());
    assert_eq!((empty.rows, empty.dim, empty.data.len()), (0, 0, 0));
}

#[test]
fn unaddressable_matrix_capacity_returns_an_error_without_allocation() {
    for (rows, dim) in [
        (usize::MAX, 2),
        (isize::MAX as usize / std::mem::size_of::<f32>() + 1, 1),
    ] {
        assert!(matches!(
            allocate_rows(rows, dim),
            Err(VectorError::Capacity { .. })
        ));
    }
}

#[test]
fn a_file_with_the_wrong_magic_is_rejected() {
    let path = temp_path("magic");
    std::fs::write(&path, [0u8; 32]).unwrap();
    let error = read(&path).unwrap_err();
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(error, VectorError::BadMagic));
}

#[test]
fn a_truncated_file_is_rejected() {
    let matrix = Matrix::from_rows(&[vec![1.0, 2.0]]).unwrap();
    let path = temp_path("truncated");
    write(&path, &matrix).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    let error = read(&path).unwrap_err();
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(
        error,
        VectorError::Length {
            rows: 1,
            dim: 2,
            ..
        }
    ));
}

#[test]
fn nonempty_zero_dimensional_matrices_are_rejected() {
    assert!(matches!(
        Matrix::from_rows(&[vec![]]),
        Err(VectorError::ZeroDimension)
    ));
    let path = temp_path("zero-dim");
    let mut header = Vec::from(*MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.extend_from_slice(&u64::MAX.to_le_bytes());
    header.extend_from_slice(&0_u64.to_le_bytes());
    std::fs::write(&path, header).unwrap();
    assert!(matches!(read(&path), Err(VectorError::ZeroDimension)));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn invalid_public_matrix_shapes_cannot_be_written() {
    let path = temp_path("invalid-shape");
    for matrix in [
        Matrix {
            rows: 2,
            dim: 2,
            data: vec![1.0],
        },
        Matrix {
            rows: usize::MAX,
            dim: 0,
            data: vec![],
        },
    ] {
        assert!(!matrix.has_valid_shape());
        assert_eq!(
            write(&path, &matrix).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
    assert!(!path.exists());
}
