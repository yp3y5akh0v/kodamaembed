use super::*;
use crate::vectors::Matrix;

#[test]
fn scores_are_dot_products_with_the_query() {
    let passages = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.6, 0.8]]).unwrap();
    let s = scores(&passages, &[0.0, 1.0]);
    assert_eq!(s.len(), 2);
    assert!(s[0].abs() < 1e-12);
    assert!((s[1] - 0.8).abs() < 1e-7);
}

#[test]
fn ranking_breaks_ties_by_index() {
    assert_eq!(ranked(&[0.5, 0.9, 0.5, 0.1]), vec![1, 0, 2, 3]);
}

#[test]
fn top_k_comparison_distinguishes_sets_and_orders() {
    let reference = [0.9, 0.8, 0.7, 0.1];
    let swapped = [0.8, 0.9, 0.7, 0.1];
    let replaced = [0.9, 0.1, 0.7, 0.8];
    let same = compare_top_k(&reference, &swapped, 2);
    assert!(same.identical_set && !same.identical_order);
    let changed = compare_top_k(&reference, &replaced, 2);
    assert!(!changed.identical_set);
    assert_eq!(changed.overlap, 1);
}

#[test]
fn signed_zero_scores_match_a_sequential_sum_and_rank_by_total_order() {
    for dim in [1, 4, 5, 8, 9, 32, 384, 1024] {
        let query = vec![1.0; dim];
        let rows = [vec![-0.0; dim], vec![0.0; dim]];
        let actual = scores(&Matrix::from_rows(&rows).unwrap(), &query);
        assert_eq!(actual[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(actual[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(ranked(&actual), vec![1, 0]);
    }
}
