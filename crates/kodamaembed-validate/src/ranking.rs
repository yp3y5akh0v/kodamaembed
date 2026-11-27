//! Per-query rankings and top-k comparison.

use crate::vectors::Matrix;
use rustc_hash::FxHashSet;

pub fn scores(passages: &Matrix, query: &[f32]) -> Vec<f64> {
    assert!(passages.has_valid_shape(), "invalid passage shape");
    assert_eq!(
        passages.dim,
        query.len(),
        "passage and query dimensions differ"
    );
    (0..passages.rows)
        .map(|index| {
            passages
                .row(index)
                .iter()
                .zip(query)
                .map(|(&p, &q)| f64::from(p) * f64::from(q))
                // Folding from -0.0 sums in input order and keeps a negative zero's sign.
                .fold(-0.0, |sum, product| sum + product)
        })
        .collect()
}

/// Indices by descending score; ties keep the lower index first.
pub fn ranked(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TopK {
    pub k: usize,
    pub overlap: usize,
    pub identical_set: bool,
    pub identical_order: bool,
}

pub fn compare_top_k(reference: &[f64], candidate: &[f64], k: usize) -> TopK {
    assert_eq!(reference.len(), candidate.len(), "score counts differ");
    let k = k.min(reference.len());
    let (r, c) = (ranked(reference), ranked(candidate));
    let wanted: FxHashSet<usize> = r[..k].iter().copied().collect();
    let overlap = c[..k].iter().filter(|index| wanted.contains(index)).count();
    TopK {
        k,
        overlap,
        identical_set: overlap == k,
        identical_order: r[..k] == c[..k],
    }
}

#[cfg(test)]
#[path = "ranking_tests.rs"]
mod tests;
