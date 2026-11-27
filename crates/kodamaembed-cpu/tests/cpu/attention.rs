use crate::support::{Rng, kernels};
use kodamaembed_cpu::attention;
use kodamaembed_model::relative::{BUCKETS, RelativeBias};

/// MiniLM's and e5-large's widths and head counts.
const WIDTHS: [(usize, usize); 2] = [(384, 12), (1024, 16)];

/// Around the 48-row query block, 16-key panel and 256-key K block; one token; an empty sequence.
const LENGTHS: [usize; 8] = [1, 5, 0, 47, 48, 49, 130, 300];

/// Softmax(q . k / sqrt(d) + bias) v per sequence and head in f64.
fn reference(
    qkv: &[f32],
    lengths: &[usize],
    hidden: usize,
    heads: usize,
    bias: Option<&RelativeBias>,
) -> Vec<f64> {
    let d = hidden / heads;
    let tokens: usize = lengths.iter().sum();
    let mut out = vec![0.0; tokens * hidden];
    let mut start = 0;
    for &len in lengths {
        for head in 0..heads {
            let at = |token: usize, part: usize, i: usize| {
                f64::from(qkv[(start + token) * 3 * hidden + part * hidden + head * d + i])
            };
            for q in 0..len {
                let scores: Vec<f64> = (0..len)
                    .map(|k| {
                        let score = (0..d).map(|i| at(q, 0, i) * at(k, 1, i)).sum::<f64>()
                            / (d as f64).sqrt();
                        match bias {
                            Some(bias) => score + f64::from(bias.keys(head, q, len)[k]),
                            None => score,
                        }
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for i in 0..d {
                    let value: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(k, w)| w * at(k, 2, i))
                        .sum();
                    out[(start + q) * hidden + head * d + i] = value / total;
                }
            }
        }
        start += len;
    }
    out
}

/// Q, K and V side by side per token, queries `query_scale` times wider.
fn inputs(tokens: usize, hidden: usize, query_scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut qkv = Vec::with_capacity(tokens * 3 * hidden);
    for _ in 0..tokens {
        qkv.extend(rng.fill(hidden, query_scale));
        qkv.extend(rng.fill(2 * hidden, 1.0));
    }
    qkv
}

#[test]
fn attention_matches_a_double_precision_reference() {
    for kernel in kernels() {
        for (hidden, heads) in WIDTHS {
            // Wide queries make peaked softmax rows; narrow ones flat rows.
            for (seed, query_scale) in [(1, 1.0f32), (2, 8.0)] {
                let tokens: usize = LENGTHS.iter().sum();
                let qkv = inputs(tokens, hidden, query_scale, seed);
                let expected = reference(&qkv, &LENGTHS, hidden, heads, None);
                let mut got = vec![0.0; tokens * hidden];
                attention(kernel, &qkv, &LENGTHS, hidden, heads, None, &mut got);
                let worst = got
                    .iter()
                    .zip(&expected)
                    .map(|(&g, &e)| (f64::from(g) - e).abs())
                    .max_by(f64::total_cmp)
                    .unwrap();
                // Scores round in proportion to the queries' scale.
                let tolerance = 6e-7 * f64::from(query_scale);
                assert!(
                    worst <= tolerance,
                    "{kernel:?} hidden {hidden}, query scale {query_scale}: worst error {worst:e}"
                );
            }
        }
    }
}

/// MPNet's relative bias joins every scaled score, across the query-block edges.
#[test]
fn relative_bias_attention_matches_the_reference() {
    const SPREAD: f32 = 2.0;
    let tokens: usize = LENGTHS.iter().sum();
    for kernel in kernels() {
        for (hidden, heads) in WIDTHS {
            let weights = Rng::new(4).fill(BUCKETS * heads, SPREAD);
            let bias = RelativeBias::new(&weights, heads, 300);
            let qkv = inputs(tokens, hidden, 1.0, 5);
            let expected = reference(&qkv, &LENGTHS, hidden, heads, Some(&bias));
            let mut got = vec![0.0; tokens * hidden];
            attention(kernel, &qkv, &LENGTHS, hidden, heads, Some(&bias), &mut got);
            let worst = got
                .iter()
                .zip(&expected)
                .map(|(&g, &e)| (f64::from(g) - e).abs())
                .max_by(f64::total_cmp)
                .unwrap();
            // Scores round in proportion to their size: the queries' scale plus the bias's.
            let tolerance = 6e-7 * f64::from(1.0 + SPREAD);
            assert!(worst <= tolerance, "{kernel:?} hidden {hidden}: {worst:e}");
        }
    }
}

/// A sequence's rows are the same bits alone or among others.
#[test]
fn attention_rows_do_not_depend_on_the_batch() {
    let (hidden, heads) = WIDTHS[0];
    let tokens: usize = LENGTHS.iter().sum();
    let qkv = inputs(tokens, hidden, 2.0, 3);
    for kernel in kernels() {
        let mut together = vec![0.0; tokens * hidden];
        attention(kernel, &qkv, &LENGTHS, hidden, heads, None, &mut together);
        let mut start = 0;
        for &len in &LENGTHS {
            let mut alone = vec![0.0; len * hidden];
            let own = &qkv[start * 3 * hidden..(start + len) * 3 * hidden];
            attention(kernel, own, &[len], hidden, heads, None, &mut alone);
            assert!(
                alone
                    .iter()
                    .zip(&together[start * hidden..(start + len) * hidden])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{kernel:?}: the sequence of {len} differs"
            );
            start += len;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn integer_attention_rejects_overflowing_head_depth_before_writes() {
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("fma")
        || !std::is_x86_feature_detected!("f16c")
        || !std::is_x86_feature_detected!("avxvnni")
    {
        return;
    }
    let hidden = 262_144;
    let mut qkv = vec![0.0; 2 * 3 * hidden];
    for (token, row) in qkv.chunks_exact_mut(3 * hidden).enumerate() {
        row[..hidden].fill(127.0);
        row[hidden..2 * hidden].fill(if token == 0 { 127.0 } else { -127.0 });
        row[2 * hidden..].fill(if token == 0 { 1.0 } else { 0.0 });
    }
    let centers = vec![0.0; hidden];
    let mut output = vec![42.0; 2 * hidden];
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kodamaembed_cpu::integer_attention(&qkv, &[2], hidden, 1, Some(&centers), &mut output);
    }))
    .expect_err("an INT32-overflowing head must be rejected");
    let reason = failure
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| failure.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(reason.contains("integer attention head width exceeds INT32 range"));
    assert!(output.iter().all(|&value| value == 42.0));
}
