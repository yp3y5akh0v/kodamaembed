use crate::support::{Rng, kernels};
use kodamaembed_cpu::{Epilogue, PackedWeight, gemm};

struct Case {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f32>,
    w: Vec<f32>,
    bias: Vec<f32>,
    residual: Vec<f32>,
}

impl Case {
    fn new(m: usize, n: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            m,
            n,
            k,
            a: rng.fill(m * k, 1.0),
            w: rng.fill(n * k, 0.1),
            bias: rng.fill(n, 0.5),
            residual: rng.fill(m * n, 1.0),
        }
    }

    /// A W^T + bias in f64, and the sum of the magnitudes of its terms.
    fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let mut linear = vec![0.0; self.m * self.n];
        let mut magnitude = vec![0.0; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                let (mut sum, mut size) = (f64::from(self.bias[j]), f64::from(self.bias[j].abs()));
                for p in 0..self.k {
                    let term =
                        f64::from(self.a[i * self.k + p]) * f64::from(self.w[j * self.k + p]);
                    sum += term;
                    size += term.abs();
                }
                linear[i * self.n + j] = sum;
                magnitude[i * self.n + j] = size;
            }
        }
        (linear, magnitude)
    }

    fn run(&self, kernel: kodamaembed_cpu::Kernel, epilogue: Epilogue) -> Vec<f32> {
        let w = PackedWeight::new(&self.w, self.n, self.k);
        let mut c = vec![0.0; self.m * self.n];
        gemm(kernel, &self.a, self.m, &w, epilogue, &mut c);
        c
    }

    fn epilogue(&self, kind: usize) -> Epilogue<'_> {
        match kind {
            0 => Epilogue::Bias(&self.bias),
            1 => Epilogue::BiasGelu(&self.bias),
            _ => Epilogue::BiasResidual(&self.bias, &self.residual),
        }
    }
}

fn gelu(x: f64) -> f64 {
    // erf by its Taylor series (converges fast for the |x| < 6 seen here).
    let z = x / std::f64::consts::SQRT_2;
    let (mut term, mut erf, mut j) = (z, z, 0.0);
    while term.abs() > 1e-17 * erf.abs().max(1e-300) {
        j += 1.0;
        term *= -z * z / j;
        erf += term / (2.0 * j + 1.0);
    }
    0.5 * x * (1.0 + erf * 2.0 / std::f64::consts::PI.sqrt())
}

/// Each value within `relative` of its terms' magnitude (plus the residual's).
fn assert_close(label: &str, got: &[f32], want: &[f64], magnitude: &[f64], relative: f64) {
    for (index, ((&got, &want), &size)) in got.iter().zip(want).zip(magnitude).enumerate() {
        let error = (f64::from(got) - want).abs();
        assert!(
            error <= relative * size.max(1.0),
            "{label} element {index}: {got} vs {want} (error {error:e})"
        );
    }
}

/// Rows and columns off the tile sizes (6 x 16), and K across its 256 blocks.
const SHAPES: [(usize, usize, usize); 4] =
    [(1, 16, 8), (7, 30, 300), (100, 384, 384), (61, 1536, 512)];

#[test]
fn fp32_gelu_extreme_inputs_follow_their_limits() {
    let magnitudes = [2.0_f32.powi(100), 2.0_f32.powi(127), f32::MAX, 1e20];
    // Full SIMD vectors and a scalar tail take both signs.
    let bias: Vec<f32> = magnitudes
        .into_iter()
        .flat_map(|x| [-x, x])
        .cycle()
        .take(19)
        .collect();
    let weight = PackedWeight::new(&vec![0.0; bias.len()], bias.len(), 1);
    for kernel in kernels() {
        let mut out = vec![f32::NAN; bias.len()];
        gemm(
            kernel,
            &[0.0],
            1,
            &weight,
            Epilogue::BiasGelu(&bias),
            &mut out,
        );
        for (&x, value) in bias.iter().zip(out) {
            let expected = if x < 0.0 { -0.0 } else { x };
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "{kernel:?}: gelu({x}) = {value}"
            );
        }
    }
}

#[test]
fn every_epilogue_matches_a_double_precision_reference() {
    for kernel in kernels() {
        for (seed, &(m, n, k)) in SHAPES.iter().enumerate() {
            let case = Case::new(m, n, k, seed as u64);
            let (linear, magnitude) = case.reference();
            let label = format!("{kernel:?} {m}x{n}x{k}");

            let got = case.run(kernel, Epilogue::Bias(&case.bias));
            assert_close(&format!("{label} bias"), &got, &linear, &magnitude, 1e-6);

            let expected: Vec<f64> = linear.iter().map(|&x| gelu(x)).collect();
            let got = case.run(kernel, Epilogue::BiasGelu(&case.bias));
            assert_close(&format!("{label} gelu"), &got, &expected, &magnitude, 1e-6);

            let expected: Vec<f64> = linear
                .iter()
                .zip(&case.residual)
                .map(|(&x, &r)| x + f64::from(r))
                .collect();
            let got = case.run(kernel, Epilogue::BiasResidual(&case.bias, &case.residual));
            assert_close(
                &format!("{label} residual"),
                &got,
                &expected,
                &magnitude,
                1e-6,
            );
        }
    }
}

/// A row has the same bits alone, first in a batch or beside an edge tile.
#[test]
fn rows_do_not_depend_on_the_batch() {
    let (n, k) = (100, 700);
    let big = Case::new(3000, n, k, 7);
    for kernel in kernels() {
        let all = big.run(kernel, Epilogue::BiasGelu(&big.bias));
        for (start, rows) in [(0, 1), (95, 13), (2990, 10), (3, 200)] {
            let part = Case {
                m: rows,
                n,
                k,
                a: big.a[start * k..(start + rows) * k].to_vec(),
                w: big.w.clone(),
                bias: big.bias.clone(),
                residual: Vec::new(),
            };
            let got = part.run(kernel, Epilogue::BiasGelu(&big.bias));
            assert!(
                got.iter()
                    .zip(&all[start * n..(start + rows) * n])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{kernel:?}: rows {start}..{} differ",
                start + rows
            );
        }
    }
}

/// Edge kernels accumulate and finish exactly as the full kernel does.
#[test]
fn every_short_row_kernel_matches_full_tiles_bit_for_bit() {
    for kernel in kernels() {
        for (n, k) in [(16, 255), (31, 257), (33, 513), (49, 385)] {
            let full = Case::new(12, n, k, 81);
            for kind in 0..3 {
                let together = full.run(kernel, full.epilogue(kind));
                for rows in 1..=5 {
                    let part = Case {
                        m: rows,
                        n,
                        k,
                        a: full.a[..rows * k].to_vec(),
                        w: full.w.clone(),
                        bias: full.bias.clone(),
                        residual: full.residual[..rows * n].to_vec(),
                    };
                    let alone = part.run(kernel, part.epilogue(kind));
                    assert!(
                        alone
                            .iter()
                            .zip(&together)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{kernel:?}: rows {rows}, n {n}, k {k}, epilogue {kind}"
                    );
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "dimensions overflow")]
fn overflowing_public_gemm_dimensions_fail_before_the_kernel() {
    let weight = PackedWeight::new(&[1.0; 32], 16, 2);
    gemm(
        kodamaembed_cpu::Kernel::Portable,
        &[],
        usize::MAX,
        &weight,
        Epilogue::Bias(&[0.0; 16]),
        &mut [],
    );
}
