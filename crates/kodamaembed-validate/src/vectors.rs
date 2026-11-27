//! Embedding matrices on disk: magic, version, rows, dim, then little-endian f32 values.

use std::{fmt, fs, io, path::Path};

const MAGIC: &[u8; 4] = b"KODV";
const VERSION: u32 = 1;
const HEADER: usize = 24;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum VectorError {
    Io(io::Error),
    BadMagic,
    BadVersion(u32),
    ZeroDimension,
    Capacity {
        rows: usize,
        dim: usize,
    },
    Allocation(std::collections::TryReserveError),
    Length {
        rows: u64,
        dim: u64,
        actual: usize,
    },
    Ragged {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "vector file I/O: {error}"),
            Self::BadMagic => write!(f, "not a KodamaEmbed vector file"),
            Self::BadVersion(version) => write!(f, "unsupported vector file version {version}"),
            Self::ZeroDimension => write!(
                f,
                "a nonempty embedding matrix must have a positive dimension"
            ),
            Self::Capacity { rows, dim } => {
                write!(
                    f,
                    "embedding matrix {rows} x {dim} exceeds addressable capacity"
                )
            }
            Self::Allocation(error) => write!(f, "embedding matrix allocation: {error}"),
            Self::Length { rows, dim, actual } => {
                write!(
                    f,
                    "vector file declares {rows} x {dim} values but has {actual} bytes"
                )
            }
            Self::Ragged {
                row,
                expected,
                actual,
            } => {
                write!(f, "row {row} has {actual} values, expected {expected}")
            }
        }
    }
}

impl std::error::Error for VectorError {}

impl From<io::Error> for VectorError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn allocate_rows(rows: usize, dim: usize) -> Result<Vec<f32>, VectorError> {
    let values = rows
        .checked_mul(dim)
        .filter(|&values| values <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or(VectorError::Capacity { rows, dim })?;
    let mut data = Vec::new();
    data.try_reserve_exact(values)
        .map_err(VectorError::Allocation)?;
    Ok(data)
}

impl Matrix {
    /// Public fields may have been constructed without `from_rows`.
    pub fn has_valid_shape(&self) -> bool {
        (self.rows == 0 || self.dim > 0) && self.rows.checked_mul(self.dim) == Some(self.data.len())
    }

    pub fn from_rows(rows: &[Vec<f32>]) -> Result<Self, VectorError> {
        let dim = rows.first().map_or(0, Vec::len);
        if !rows.is_empty() && dim == 0 {
            return Err(VectorError::ZeroDimension);
        }
        for (row, values) in rows.iter().enumerate() {
            if values.len() != dim {
                return Err(VectorError::Ragged {
                    row,
                    expected: dim,
                    actual: values.len(),
                });
            }
        }
        let mut data = allocate_rows(rows.len(), dim)?;
        for values in rows {
            data.extend_from_slice(values);
        }
        Ok(Self {
            rows: rows.len(),
            dim,
            data,
        })
    }

    pub fn row(&self, index: usize) -> &[f32] {
        assert!(index < self.rows, "row out of bounds");
        &self.data[index * self.dim..(index + 1) * self.dim]
    }
}

pub fn write(path: &Path, matrix: &Matrix) -> io::Result<()> {
    if !matrix.has_valid_shape() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid embedding matrix shape",
        ));
    }
    let mut bytes = Vec::with_capacity(HEADER + matrix.data.len() * 4);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&(matrix.rows as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.dim as u64).to_le_bytes());
    for value in &matrix.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)
}

pub fn read(path: &Path) -> Result<Matrix, VectorError> {
    let bytes = fs::read(path)?;
    if bytes.len() < 4 || &bytes[..4] != MAGIC {
        return Err(VectorError::BadMagic);
    }
    if bytes.len() < HEADER {
        return Err(VectorError::Length {
            rows: 0,
            dim: 0,
            actual: bytes.len(),
        });
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("four header bytes"));
    if version != VERSION {
        return Err(VectorError::BadVersion(version));
    }
    let rows = u64::from_le_bytes(bytes[8..16].try_into().expect("eight header bytes"));
    let dim = u64::from_le_bytes(bytes[16..24].try_into().expect("eight header bytes"));
    if rows > 0 && dim == 0 {
        return Err(VectorError::ZeroDimension);
    }
    let payload = rows
        .checked_mul(dim)
        .and_then(|values| values.checked_mul(4));
    if payload != Some((bytes.len() - HEADER) as u64) {
        return Err(VectorError::Length {
            rows,
            dim,
            actual: bytes.len(),
        });
    }
    let shape_error = || VectorError::Length {
        rows,
        dim,
        actual: bytes.len(),
    };
    let rows = usize::try_from(rows).map_err(|_| shape_error())?;
    let dim = usize::try_from(dim).map_err(|_| shape_error())?;
    let (values, _) = bytes[HEADER..].as_chunks::<4>();
    let data = values
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect();
    Ok(Matrix { rows, dim, data })
}

#[cfg(test)]
#[path = "vectors_tests.rs"]
mod tests;
