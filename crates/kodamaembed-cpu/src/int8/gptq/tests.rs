use super::*;

fn rows(m: usize, k: usize, seed: usize) -> Vec<f32> {
    (0..m * k)
        .map(|at| {
            let x = ((at * 37 + seed * 101) % 251) as f32 - 125.0;
            x / 64.0 * (1.0 + (at % k) as f32 / k as f32)
        })
        .collect()
}

fn scales_of(rotated: &[f32], k: usize) -> Vec<f32> {
    rotated
        .chunks_exact(k)
        .map(|row| row.iter().fold(0.0_f32, |m, x| m.max(x.abs())) / 127.0)
        .collect()
}

fn exact_sums(rows: &[f32], k: usize, weights: Option<&[f32]>) -> Vec<f64> {
    let mut sums = vec![0.0_f64; k * k];
    for (t, row) in rows.chunks_exact(k).enumerate() {
        let weight = weights.map_or(1.0, |weights| f64::from(weights[t]));
        for i in 0..k {
            for j in 0..k {
                sums[i * k + j] += weight * f64::from(row[i]) * f64::from(row[j]);
            }
        }
    }
    sums
}

/// [`Hessian::add`]'s order in scalar code: f32 FMA chains, each added once in f64.
fn chunked_sums(rows: &[f32], k: usize, weights: Option<&[f32]>) -> Vec<f64> {
    let m = rows.len() / k;
    let mut sums = vec![0.0_f64; k * k];
    for first in (0..m).step_by(CHUNK) {
        let end = (first + CHUNK).min(m);
        for i in 0..k {
            for j in i..k {
                let mut total = 0.0_f32;
                for k0 in (first..end).step_by(gemm::KC) {
                    let mut acc = 0.0_f32;
                    for t in k0..(k0 + gemm::KC).min(end) {
                        let right = rows[t * k + j] * weights.map_or(1.0, |w| w[t]);
                        acc = rows[t * k + i].mul_add(right, acc);
                    }
                    total = if k0 == first { acc } else { total + acc };
                }
                sums[i * k + j] += f64::from(total);
            }
        }
    }
    sums
}

/// The unblocked algorithm, which the blocked factor must match bit for bit.
fn factor_reference(hessian: &Hessian, damp: f64) -> Vec<f64> {
    let (k, sums) = (hessian.k, &hessian.sums);
    let mut diagonal = 0.0;
    for i in 0..k {
        diagonal += sums[i * k + i];
    }
    let mean = diagonal / k as f64;
    let mut u = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in a..k {
            u[a * k + b] = sums[(k - 1 - b) * k + (k - 1 - a)];
        }
        u[a * k + a] += damp * mean;
    }
    for c in 0..k {
        let root = u[c * k + c].sqrt();
        u[c * k + c] = root;
        for x in &mut u[c * k + c + 1..(c + 1) * k] {
            *x /= root;
        }
        for r in c + 1..k {
            let a = -u[c * k + r];
            for col in r..k {
                u[r * k + col] = u[c * k + col].mul_add(a, u[r * k + col]);
            }
        }
    }
    u
}

/// [`dot`]'s lane order in scalar code.
fn dot_reference(x: &[f64], y: &[f64]) -> f64 {
    let mut lanes = [0.0_f64; 16];
    for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
        lanes[i % 16] = x.mul_add(y, lanes[i % 16]);
    }
    let mut width = 8;
    while width > 0 {
        for j in 0..width {
            lanes[j] += lanes[j + width];
        }
        width /= 2;
    }
    lanes[0]
}

/// GPTQ as Frantar et al. state it, through the inverse Hessian, in scalar f64.
fn gptq_reference(rotated: &[f32], scales: &[f32], h: &[f64], k: usize, damp: f64) -> Vec<i8> {
    let mut a = h.to_vec();
    let mean = (0..k).map(|i| a[i * k + i]).sum::<f64>() / k as f64;
    for i in 0..k {
        a[i * k + i] += damp * mean;
    }
    let mut inverse = vec![0.0_f64; k * k];
    for i in 0..k {
        inverse[i * k + i] = 1.0;
    }
    for c in 0..k {
        let pivot = a[c * k + c];
        for j in 0..k {
            a[c * k + j] /= pivot;
            inverse[c * k + j] /= pivot;
        }
        for r in 0..k {
            if r != c {
                let f = a[r * k + c];
                for j in 0..k {
                    a[r * k + j] -= f * a[c * k + j];
                    inverse[r * k + j] -= f * inverse[c * k + j];
                }
            }
        }
    }
    // Upper Cholesky U^T U = inverse.
    let mut u = vec![0.0_f64; k * k];
    for i in 0..k {
        let mut d = inverse[i * k + i];
        for l in 0..i {
            d -= u[l * k + i] * u[l * k + i];
        }
        u[i * k + i] = d.sqrt();
        for j in i + 1..k {
            let mut s = inverse[i * k + j];
            for l in 0..i {
                s -= u[l * k + i] * u[l * k + j];
            }
            u[i * k + j] = s / u[i * k + i];
        }
    }
    let mut out = Vec::with_capacity(rotated.len());
    for (row, &scale) in rotated.chunks_exact(k).zip(scales) {
        let mut w: Vec<f64> = row.iter().map(|&x| f64::from(x / scale)).collect();
        for i in 0..k {
            let q = w[i].round_ties_even().clamp(-127.0, 127.0);
            let e = (w[i] - q) / u[i * k + i];
            for j in i + 1..k {
                w[j] -= e * u[i * k + j];
            }
            out.push(q as i8);
        }
    }
    out
}

fn full(hessian: &Hessian) -> Vec<f64> {
    let k = hessian.k;
    let mut h = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            h[i * k + j] = hessian.sums[i * k + j];
            h[j * k + i] = hessian.sums[i * k + j];
        }
    }
    h
}

#[test]
fn hessian_sums_are_the_weighted_products() {
    if !super::super::is_supported() {
        return;
    }
    // Two chunks and a short one; a band past the last full NR panel.
    let (m, k) = (1100, 80);
    let operand = rows(m, k, 3);
    let weights: Vec<f32> = (0..m).map(|t| 0.25 + (t % 7) as f32 / 4.0).collect();
    for weights in [None, Some(&weights[..])] {
        let mut hessian = Hessian::new(k).unwrap();
        hessian
            .add(&operand[..600 * k], weights.map(|w| &w[..600]))
            .unwrap();
        hessian
            .add(&operand[600 * k..], weights.map(|w| &w[600..]))
            .unwrap();
        let exact = exact_sums(&operand, k, weights);
        let largest = exact.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        for i in 0..k {
            for j in 0..k {
                let got = if j >= i { hessian.sums[i * k + j] } else { 0.0 };
                let want = if j >= i { exact[i * k + j] } else { 0.0 };
                assert!(
                    (got - want).abs() <= 1e-5 * largest,
                    "({i}, {j}): {got} against {want}"
                );
            }
        }
    }
}

#[test]
fn hessian_sums_are_each_chunks_products_bit_for_bit() {
    if !super::super::is_supported() {
        return;
    }
    // A short last band, panel and chunk; chunks of two KC blocks.
    let (m, k) = (1100, 200);
    let operand = rows(m, k, 23);
    let weights: Vec<f32> = (0..m).map(|t| 0.5 + (t % 13) as f32 / 8.0).collect();
    for weights in [None, Some(&weights[..])] {
        let mut hessian = Hessian::new(k).unwrap();
        hessian.add(&operand, weights).unwrap();
        let want = chunked_sums(&operand, k, weights);
        for i in 0..k {
            for j in i..k {
                assert_eq!(
                    hessian.sums[i * k + j].to_bits(),
                    want[i * k + j].to_bits(),
                    "({i}, {j}), weighted {}",
                    weights.is_some()
                );
            }
        }
    }
}

#[test]
fn the_factor_is_the_unblocked_algorithms_bit_for_bit() {
    if !super::super::is_supported() {
        return;
    }
    // Seven steps, the last short; trailing rows in 21 tasks of two blocks.
    let k = 400;
    let mut hessian = Hessian::new(k).unwrap();
    hessian.add(&rows(900, k, 29), None).unwrap();
    let want = factor_reference(&hessian, 0.1);
    // SAFETY: the processor has AVX2 and FMA (is_supported).
    let got = unsafe { reversed_factor(&hessian, 0.1) }.unwrap();
    for c in 0..k {
        for a in c..k {
            assert_eq!(
                got[c * k + a].to_bits(),
                want[c * k + a].to_bits(),
                "U[{c}][{a}]"
            );
        }
    }
}

#[test]
fn dots_are_their_lanes_bit_for_bit() {
    if !super::super::is_supported() {
        return;
    }
    for len in [0, 1, 5, 15, 16, 17, 33, 100, 1023] {
        let x: Vec<f64> = (0..len)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 7.0)
            .collect();
        let y: Vec<f64> = (0..len)
            .map(|i| ((i * 53 % 97) as f64 - 48.0) / 3.0)
            .collect();
        // SAFETY: the processor has AVX2 and FMA (is_supported).
        let got = unsafe { dot(&x, &y) };
        assert_eq!(
            got.to_bits(),
            dot_reference(&x, &y).to_bits(),
            "length {len}"
        );
    }
}

#[test]
fn diagonal_hessians_round_each_digit_alone() {
    if !super::super::is_supported() {
        return;
    }
    let (n, k) = (19, 96);
    let rotated = rows(n, k, 5);
    let scales = scales_of(&rotated, k);
    // One feature a token: the Hessian is diagonal, nothing carries.
    let mut operand = vec![0.0_f32; 3 * k * k];
    for t in 0..3 * k {
        operand[t * k + t % k] = 1.0 + (t % 5) as f32;
    }
    let mut hessian = Hessian::new(k).unwrap();
    hessian.add(&operand, None).unwrap();
    let digits = digits(&rotated, &scales, &hessian, 0.1).unwrap();
    for (j, (row, &scale)) in rotated.chunks_exact(k).zip(&scales).enumerate() {
        for (t, &x) in row.iter().enumerate() {
            let rounded = (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
            assert_eq!(digits[j * k + t], rounded, "row {j} column {t}");
        }
    }
}

#[test]
fn digits_are_gptq_with_the_inverse_hessians_factor() {
    if !super::super::is_supported() {
        return;
    }
    // Past a task's 16 rows; a column whose carried errors push neighbors past 127.
    let (n, k, m) = (37, 80, 700);
    let mut rotated = rows(n, k, 7);
    for row in rotated.chunks_exact_mut(k) {
        row[7] *= 30.0;
    }
    let scales = scales_of(&rotated, k);
    let mut hessian = Hessian::new(k).unwrap();
    hessian.add(&rows(m, k, 11), None).unwrap();
    for damp in [0.01, 0.1] {
        let digits = digits(&rotated, &scales, &hessian, damp).unwrap();
        let expected = gptq_reference(&rotated, &scales, &full(&hessian), k, damp);
        assert_eq!(digits, expected, "damp {damp}");
        assert!(digits.iter().any(|&q| q.unsigned_abs() == 127));
    }
}

#[test]
fn the_same_bits_on_any_thread_count() {
    if !super::super::is_supported() {
        return;
    }
    let (n, k, m) = (70, 192, 1300);
    let rotated = rows(n, k, 13);
    let scales = scales_of(&rotated, k);
    let operand = rows(m, k, 17);
    let weights: Vec<f32> = (0..m).map(|t| 1.0 / (1.0 + (t % 11) as f32)).collect();
    let run = |threads| {
        crate::worker_pool("gptq-test", threads)
            .unwrap()
            .install(|| {
                let mut hessian = Hessian::new(k).unwrap();
                hessian.add(&operand, Some(&weights)).unwrap();
                let digits = digits(&rotated, &scales, &hessian, 0.1).unwrap();
                (hessian.sums, digits)
            })
    };
    let (sums, digits) = run(1);
    for threads in [3, 24] {
        let (other_sums, other_digits) = run(threads);
        assert!(
            sums.iter()
                .zip(&other_sums)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "sums on {threads} threads"
        );
        assert_eq!(digits, other_digits, "digits on {threads} threads");
    }
}

#[test]
fn malformed_inputs_are_refused() {
    if !super::super::is_supported() {
        return;
    }
    assert!(matches!(Hessian::new(0), Err(Error::Dimensions)));
    let k = 32;
    let mut hessian = Hessian::new(k).unwrap();
    assert!(matches!(
        hessian.add(&[1.0; 33], None),
        Err(Error::Dimensions)
    ));
    assert!(matches!(
        hessian.add(&[1.0; 64], Some(&[1.0])),
        Err(Error::Dimensions)
    ));
    let mut bad = vec![1.0_f32; 64];
    bad[5] = f32::NAN;
    assert!(matches!(hessian.add(&bad, None), Err(Error::FloatRange)));
    assert!(matches!(
        hessian.add(&[1.0; 64], Some(&[1.0, f32::INFINITY])),
        Err(Error::FloatRange)
    ));
    let rotated = rows(2, k, 1);
    let scales = scales_of(&rotated, k);
    // No rows taken yet: nothing to weigh the digits by.
    assert!(matches!(
        digits(&rotated, &scales, &hessian, 0.1),
        Err(Error::FloatRange)
    ));
    hessian.add(&rows(100, k, 2), None).unwrap();
    assert!(matches!(
        digits(&rotated[..k], &scales, &hessian, 0.1),
        Err(Error::Dimensions)
    ));
    assert!(matches!(
        digits(&rotated, &[1.0, 0.0], &hessian, 0.1),
        Err(Error::FloatRange)
    ));
    assert!(matches!(
        digits(&rotated, &scales, &hessian, -0.1),
        Err(Error::FloatRange)
    ));
    assert!(digits(&rotated, &scales, &hessian, 0.1).is_ok());
}
