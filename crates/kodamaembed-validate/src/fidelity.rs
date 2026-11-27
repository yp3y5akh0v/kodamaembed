//! Numeric fidelity of candidate embeddings against a reference, computed in f64.

use crate::vectors::Matrix;
use std::fmt;

/// Rows with a non-finite value on either side are counted in `non_finite`
/// and excluded from every other statistic.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Summary {
    pub rows: usize,
    pub min_cosine: f64,
    pub mean_cosine: f64,
    pub max_l2: f64,
    pub max_abs: f64,
    pub max_norm_error: f64,
    pub non_finite: usize,
}

#[derive(Debug, PartialEq)]
pub enum FidelityError {
    InvalidMatrix {
        side: &'static str,
        rows: usize,
        dim: usize,
        values: usize,
    },
    Shape {
        reference: (usize, usize),
        candidate: (usize, usize),
    },
}

impl fmt::Display for FidelityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                reference,
                candidate,
            } => write!(
                f,
                "reference is {reference:?} but candidate is {candidate:?}"
            ),
            Self::InvalidMatrix {
                side,
                rows,
                dim,
                values,
            } => write!(
                f,
                "{side} matrix declares {rows} x {dim} values but contains {values}"
            ),
        }
    }
}

impl std::error::Error for FidelityError {}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

fn norm(values: &[f32]) -> f64 {
    dot(values, values).sqrt()
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector dimensions differ");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// `sqrt(|a|^2 |b|^2)` rather than `|a| |b|`: a correctly rounded square root
/// of a square is exact, so identical vectors score exactly 1.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector dimensions differ");
    let denominator = (dot(a, a) * dot(b, b)).sqrt();
    if denominator == 0.0 {
        return 0.0;
    }
    dot(a, b) / denominator
}

pub fn compare(reference: &Matrix, candidate: &Matrix) -> Result<Summary, FidelityError> {
    for (side, matrix) in [("reference", reference), ("candidate", candidate)] {
        if !matrix.has_valid_shape() {
            return Err(FidelityError::InvalidMatrix {
                side,
                rows: matrix.rows,
                dim: matrix.dim,
                values: matrix.data.len(),
            });
        }
    }
    if (reference.rows, reference.dim) != (candidate.rows, candidate.dim) {
        return Err(FidelityError::Shape {
            reference: (reference.rows, reference.dim),
            candidate: (candidate.rows, candidate.dim),
        });
    }
    let mut summary = Summary {
        rows: reference.rows,
        min_cosine: f64::INFINITY,
        mean_cosine: 0.0,
        max_l2: 0.0,
        max_abs: 0.0,
        max_norm_error: 0.0,
        non_finite: 0,
    };
    let mut finite_rows = 0usize;
    for index in 0..reference.rows {
        let (r, c) = (reference.row(index), candidate.row(index));
        if r.iter().chain(c).any(|v| !v.is_finite()) {
            summary.non_finite += 1;
            continue;
        }
        let cos = cosine(r, c);
        summary.min_cosine = summary.min_cosine.min(cos);
        summary.mean_cosine += cos;
        summary.max_l2 = summary.max_l2.max(l2_distance(r, c));
        let row_abs = r
            .iter()
            .zip(c)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0.0, f64::max);
        summary.max_abs = summary.max_abs.max(row_abs);
        summary.max_norm_error = summary.max_norm_error.max((norm(c) - norm(r)).abs());
        finite_rows += 1;
    }
    if finite_rows > 0 {
        summary.mean_cosine /= finite_rows as f64;
    } else {
        summary.min_cosine = f64::NAN;
        summary.mean_cosine = f64::NAN;
    }
    Ok(summary)
}

#[cfg(test)]
#[path = "fidelity_tests.rs"]
mod tests;
