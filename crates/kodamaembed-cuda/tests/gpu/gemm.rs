use crate::support::Rng;
use half::{f16, slice::HalfFloatSliceExt};
use kodamaembed_cuda::{
    Context, DeviceBuffer, Stream,
    kernels::{Gemm, Kernels, Low, PairedGemms, Turned, Turns},
};
use kodamaembed_model::{Geometry, Rotary};

/// The rotation base and the sequence length of the turned GEMMs' rows.
const THETA: f32 = 10_000.0;
const LONGEST: usize = 70;

/// Each of `rows` rows' place in sequences of [`LONGEST`] packed back to back.
fn positions(rows: usize) -> Vec<i32> {
    (0..rows).map(|row| (row % LONGEST) as i32).collect()
}

struct Case {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f16>,
    w: Vec<f16>,
    bias: Vec<f32>,
}

impl Case {
    fn new(m: usize, n: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let half = |values: Vec<f32>| values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        Self {
            m,
            n,
            k,
            a: half(rng.fill(m * k, 2.0)),
            w: half(rng.fill(n * k, 0.1)),
            bias: rng.fill(n, 0.5),
        }
    }

    /// Exact sums, and each element's sum of |a * w| for the tolerance.
    fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let mut exact = Vec::with_capacity(self.m * self.n);
        let mut magnitude = Vec::with_capacity(self.m * self.n);
        for row in self.a.chunks(self.k) {
            for (col, weights) in self.w.chunks(self.k).enumerate() {
                let (mut sum, mut size) = (f64::from(self.bias[col]), 0.0);
                for (&x, &y) in row.iter().zip(weights) {
                    let product = f64::from(x.to_f32()) * f64::from(y.to_f32());
                    sum += product;
                    size += product.abs();
                }
                exact.push(sum);
                magnitude.push(size);
            }
        }
        (exact, magnitude)
    }
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Taylor series below 2.5, the erfc continued fraction above: about 1e-14.
fn erf(x: f64) -> f64 {
    if x.abs() < 2.5 {
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        while term.abs() > 1e-17 * sum.abs().max(1e-300) {
            n += 1.0;
            term *= -x * x / n;
            sum += term / (2.0 * n + 1.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        let z = x.abs();
        let mut fraction = 0.0;
        for n in (1..60).rev() {
            fraction = f64::from(n) / 2.0 / (z + fraction);
        }
        let erfc = (-z * z).exp() / std::f64::consts::PI.sqrt() / (z + fraction);
        (1.0 - erfc).copysign(x)
    }
}

enum Output {
    Bias,
    Gelu,
    /// Half as many columns: one product a pair.
    Geglu,
    Residual(Vec<f32>),
}

fn run(case: &Case, output: &Output) -> Vec<f64> {
    let context = Context::new(0).unwrap();
    let kernels = Kernels::load(&context, Geometry::Large).unwrap();
    let paired = PairedGemms::load(&context).unwrap();
    let stream = Stream::new(&context).unwrap();
    let a = DeviceBuffer::from_slice(&context, &case.a).unwrap();
    let w = DeviceBuffer::from_slice(&context, &case.w).unwrap();
    let bias = DeviceBuffer::from_slice(&context, &case.bias).unwrap();
    let gemm = Gemm {
        a: &a,
        w: &w,
        bias: &bias,
        m: case.m,
        n: case.n,
        k: case.k,
    };
    let len = case.m * case.n;
    match output {
        Output::Bias | Output::Gelu | Output::Geglu => {
            let pairs = matches!(output, Output::Geglu);
            let out = DeviceBuffer::<f16>::zeroed(&context, if pairs { len / 2 } else { len });
            let out = out.unwrap();
            // SAFETY: every buffer lives until the stream is synchronized below.
            unsafe {
                match output {
                    Output::Bias => kernels.gemm_bias(&stream, &gemm, &out, None),
                    Output::Geglu => paired.gemm_bias_geglu(&stream, &gemm, &out),
                    _ => kernels.gemm_bias_gelu(&stream, &gemm, &out),
                }
                .unwrap();
            }
            stream.synchronize().unwrap();
            out.read()
                .unwrap()
                .iter()
                .map(|v| f64::from(v.to_f32()))
                .collect()
        }
        Output::Residual(residual) => {
            let residual = DeviceBuffer::from_slice(&context, residual).unwrap();
            let out = DeviceBuffer::<f32>::zeroed(&context, len).unwrap();
            // SAFETY: every buffer lives until the stream is synchronized below.
            unsafe {
                kernels
                    .gemm_bias_residual(&stream, &gemm, &residual, &out)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            out.read().unwrap().iter().map(|&v| f64::from(v)).collect()
        }
    }
}

/// FP32 accumulation stays far below 1e-5 of the summed magnitudes, an indexing error far above.
fn assert_close(got: &[f64], expected: &[f64], magnitude: &[f64], rounding: f64) {
    assert_eq!(got.len(), expected.len());
    for (index, ((&g, &e), &size)) in got.iter().zip(expected).zip(magnitude).enumerate() {
        let bound = 1e-5 * size + rounding * e.abs() + 1e-6;
        assert!((g - e).abs() <= bound, "element {index}: {g} vs {e}");
    }
}

#[test]
fn gemm_bias_matches_the_reference_with_partial_tiles() {
    for (m, n, k) in [(300, 256, 512), (17, 40, 64), (197, 160, 64)] {
        let case = Case::new(m, n, k, 1);
        let (expected, magnitude) = case.reference();
        let got = run(&case, &Output::Bias);
        assert_close(&got, &expected, &magnitude, f64::powi(2.0, -11));
    }
}

/// The FP16 outputs and, for the first `columns` columns, the low parts.
fn run_split(case: &Case, columns: usize) -> (Vec<f16>, Vec<f16>) {
    let context = Context::new(0).unwrap();
    let kernels = Kernels::load(&context, Geometry::Large).unwrap();
    let stream = Stream::new(&context).unwrap();
    let a = DeviceBuffer::from_slice(&context, &case.a).unwrap();
    let w = DeviceBuffer::from_slice(&context, &case.w).unwrap();
    let bias = DeviceBuffer::from_slice(&context, &case.bias).unwrap();
    let gemm = Gemm {
        a: &a,
        w: &w,
        bias: &bias,
        m: case.m,
        n: case.n,
        k: case.k,
    };
    let out = DeviceBuffer::<f16>::zeroed(&context, case.m * case.n).unwrap();
    let len = case.m * columns;
    let guard = f16::NEG_INFINITY;
    let low = DeviceBuffer::from_slice(&context, &vec![guard; len + case.n]).unwrap();
    let low_parts = Low { out: &low, columns };
    // SAFETY: every buffer lives until the stream is synchronized below.
    unsafe {
        kernels
            .gemm_bias(&stream, &gemm, &out, Some(&low_parts))
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut low = low.read().unwrap();
    assert!(low[len..].iter().all(|&value| value == guard));
    low.truncate(len);
    (out.read().unwrap(), low)
}

#[test]
fn low_parts_can_end_inside_a_warp() {
    let case = Case::new(17, 192, 64, 17);
    let (high, full) = run_split(&case, case.n);
    for columns in [0, 8, 24, 40, case.n - 8] {
        let (actual_high, partial) = run_split(&case, columns);
        assert_eq!(actual_high.reinterpret_cast(), high.reinterpret_cast());
        let expected: Vec<_> = full
            .reinterpret_cast()
            .chunks_exact(case.n)
            .flat_map(|row| &row[..columns])
            .copied()
            .collect();
        assert_eq!(partial.reinterpret_cast(), expected);
    }
}

#[test]
fn low_parts_recover_what_fp16_rounding_lost() {
    let case = Case::new(200, 384, 512, 5);
    let columns = 256;
    let (expected, magnitude) = case.reference();
    let (high, low) = run_split(&case, columns);
    for row in 0..case.m {
        for col in 0..case.n {
            let (e, size) = (expected[row * case.n + col], magnitude[row * case.n + col]);
            let h = f64::from(high[row * case.n + col].to_f32());
            assert!(
                (h - e).abs() <= 1e-5 * size + f64::powi(2.0, -11) * e.abs() + 1e-6,
                "high ({row}, {col}): {h} vs {e}"
            );
            if col < columns {
                let v = h + f64::from(low[row * columns + col].to_f32()) / 2048.0;
                assert!(
                    (v - e).abs() <= 1e-6 * size + 1e-6,
                    "high + low ({row}, {col}): {v} vs {e}"
                );
            }
        }
    }
}

#[test]
fn gemm_bias_gelu_matches_the_reference() {
    let case = Case::new(130, 384, 1024, 2);
    let (linear, magnitude) = case.reference();
    let expected: Vec<f64> = linear.iter().map(|&x| gelu(x)).collect();
    let got = run(&case, &Output::Gelu);
    assert_close(&got, &expected, &magnitude, f64::powi(2.0, -11));
}

/// Each pair of columns is an input and its gate: FP16 rounds their product once. The shapes
/// leave partial tiles, and the last is ModernBERT-base's.
#[test]
fn the_gated_gemm_is_gelu_of_each_input_times_its_gate() {
    for (m, n, k) in [
        (300, 256, 512),
        (17, 40, 64),
        (197, 160, 64),
        (65, 2304, 768),
    ] {
        let case = Case::new(m, n, k, 31);
        let (linear, magnitude) = case.reference();
        let got = run(&case, &Output::Geglu);
        assert_eq!(got.len(), m * n / 2);
        for (pair, &have) in got.iter().enumerate() {
            let (input, by) = (linear[2 * pair], linear[2 * pair + 1]);
            let want = gelu(input) * by;
            // The GELU's slope stays below 1.13.
            let sums =
                1.13 * magnitude[2 * pair] * by.abs() + magnitude[2 * pair + 1] * gelu(input).abs();
            assert!(
                (have - want).abs() <= 1e-5 * sums + f64::powi(2.0, -11) * want.abs() + 1e-6,
                "{n} x {k}, pair {pair}: {have} vs {want}"
            );
        }
    }
}

/// Each pair of the first columns turns by its row's position before FP16 rounds it; the other
/// columns stay. The shapes leave partial tiles, and the last is ModernBERT-base's.
#[test]
fn the_turned_gemm_turns_each_pair_of_its_first_columns_by_the_rows_position() {
    let context = Context::new(0).unwrap();
    let paired = PairedGemms::load(&context).unwrap();
    let stream = Stream::new(&context).unwrap();
    for (m, n, k, head, columns) in [
        (300, 256, 512, 64, 192),
        (17, 40, 64, 8, 24),
        (197, 160, 64, 32, 128),
        (65, 2304, 768, 64, 1536),
    ] {
        let case = Case::new(m, n, k, 37);
        let (linear, magnitude) = case.reference();
        let places = positions(m);
        let a = DeviceBuffer::from_slice(&context, &case.a).unwrap();
        let w = DeviceBuffer::from_slice(&context, &case.w).unwrap();
        let bias = DeviceBuffer::from_slice(&context, &case.bias).unwrap();
        let gemm = Gemm {
            a: &a,
            w: &w,
            bias: &bias,
            m,
            n,
            k,
        };
        let places_gpu = DeviceBuffer::from_slice(&context, &places).unwrap();
        let turns = Turns::upload(&context, head, THETA, LONGEST).unwrap();
        let turned = Turned {
            positions: &places_gpu,
            turns: &turns,
            columns,
            longest: LONGEST,
        };
        let out = DeviceBuffer::<f16>::zeroed(&context, m * n).unwrap();
        let low = DeviceBuffer::<f16>::zeroed(&context, m * columns).unwrap();
        let low_parts = Low { out: &low, columns };
        // SAFETY: every buffer lives until the stream is synchronized below.
        unsafe {
            paired
                .gemm_bias_turned(&stream, &gemm, &out, Some(&low_parts), &turned)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let (high, low) = (out.read().unwrap(), low.read().unwrap());
        for row in 0..m {
            for col in (0..n).step_by(2) {
                let at = row * n + col;
                let (x, y) = (linear[at], linear[at + 1]);
                let size = magnitude[at] + magnitude[at + 1];
                let wants = if col < columns {
                    let place = places[row] as usize;
                    let (cos, sin) = Rotary::turn(THETA, head, (col % head) / 2, place);
                    let (cos, sin) = (f64::from(cos), f64::from(sin));
                    [x * cos - y * sin, y * cos + x * sin]
                } else {
                    [x, y]
                };
                for (offset, want) in wants.into_iter().enumerate() {
                    let kept = f64::from(high[at + offset].to_f32());
                    assert!(
                        (kept - want).abs()
                            <= 1e-5 * size + f64::powi(2.0, -11) * want.abs() + 1e-6,
                        "{n} x {k}, high ({row}, {}): {kept} vs {want}",
                        col + offset
                    );
                    if col < columns {
                        let lost = f64::from(low[row * columns + col + offset].to_f32());
                        let value = kept + lost / 2048.0;
                        assert!(
                            (value - want).abs() <= 1e-6 * size + 1e-6,
                            "{n} x {k}, high + low ({row}, {}): {value} vs {want}",
                            col + offset
                        );
                    }
                }
            }
        }
    }
}

/// Every epilogue's output bits for the first `rows` rows.
fn prefix_bits(
    (context, kernels, paired, stream): (&Context, &Kernels, &PairedGemms, &Stream),
    case: &Case,
    rows: usize,
    residual: &[f32],
) -> Vec<u32> {
    let a = DeviceBuffer::from_slice(context, &case.a[..rows * case.k]).unwrap();
    let w = DeviceBuffer::from_slice(context, &case.w).unwrap();
    let bias = DeviceBuffer::from_slice(context, &case.bias).unwrap();
    let residual = DeviceBuffer::from_slice(context, &residual[..rows * case.n]).unwrap();
    let gemm = Gemm {
        a: &a,
        w: &w,
        bias: &bias,
        m: rows,
        n: case.n,
        k: case.k,
    };
    let halfs = |len: usize| DeviceBuffer::<f16>::zeroed(context, len).unwrap();
    let (high, gelu) = (halfs(rows * case.n), halfs(rows * case.n));
    let (low, geglu) = (halfs(rows * 128), halfs(rows * case.n / 2));
    let (turned_high, turned_low) = (halfs(rows * case.n), halfs(rows * 128));
    let out32 = DeviceBuffer::<f32>::zeroed(context, rows * case.n).unwrap();
    let places = DeviceBuffer::from_slice(context, &positions(rows)).unwrap();
    let turns = Turns::upload(context, 64, THETA, LONGEST).unwrap();
    let turned = Turned {
        positions: &places,
        turns: &turns,
        columns: 128,
        longest: LONGEST,
    };
    // SAFETY: every buffer lives until the stream is synchronized below.
    unsafe {
        let low = Low {
            out: &low,
            columns: 128,
        };
        kernels.gemm_bias(stream, &gemm, &high, Some(&low)).unwrap();
        kernels.gemm_bias_gelu(stream, &gemm, &gelu).unwrap();
        kernels
            .gemm_bias_residual(stream, &gemm, &residual, &out32)
            .unwrap();
        paired.gemm_bias_geglu(stream, &gemm, &geglu).unwrap();
        let low = Low {
            out: &turned_low,
            columns: 128,
        };
        paired
            .gemm_bias_turned(stream, &gemm, &turned_high, Some(&low), &turned)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let bits = |values: &DeviceBuffer<f16>| {
        let values = values.read().unwrap();
        values.into_iter().map(|v| u32::from(v.to_bits()))
    };
    bits(&high)
        .chain(bits(&low))
        .chain(bits(&gelu))
        .chain(out32.read().unwrap().into_iter().map(f32::to_bits))
        .chain(bits(&geglu))
        .chain(bits(&turned_high))
        .chain(bits(&turned_low))
        .collect()
}

/// e5-large's four projections, then all-MiniLM-L6-v2's and ModernBERT-base's contraction, at
/// row counts that reach every tiling.
#[test]
fn every_tiling_gives_the_bits_of_the_same_rows_in_a_batch() {
    let context = Context::new(0).unwrap();
    let kernels = Kernels::load(&context, Geometry::Large).unwrap();
    let paired = PairedGemms::load(&context).unwrap();
    let stream = Stream::new(&context).unwrap();
    let gpu = (&context, &kernels, &paired, &stream);
    let shapes: [(usize, usize, usize, &[usize]); 9] = [
        (
            512,
            3072,
            1024,
            &[
                1, 7, 16, 33, 64, 65, 96, 128, 129, 160, 161, 192, 193, 224, 225, 256, 257, 288,
                289, 320, 321, 384, 385,
            ],
        ),
        // 640 rows lie past the expansion's grid tilings, so the batch runs r128c176.
        (
            640,
            4096,
            1024,
            &[
                48, 64, 80, 128, 129, 192, 193, 240, 241, 256, 257, 288, 289, 320, 321, 384, 385,
                512,
            ],
        ),
        (
            512,
            1024,
            1024,
            &[33, 64, 96, 193, 224, 225, 288, 289, 320, 321, 383],
        ),
        (
            512,
            1024,
            4096,
            &[64, 80, 193, 224, 225, 240, 241, 256, 257, 288, 289, 383],
        ),
        (512, 1152, 384, &[33, 65, 256]),
        (512, 1536, 384, &[33, 65, 256]),
        (512, 384, 384, &[64, 128]),
        (512, 384, 1536, &[200]),
        (512, 768, 1152, &[33, 65, 200, 256]),
    ];
    for (seed, (m, n, k, row_counts)) in (6..).zip(shapes) {
        let case = Case::new(m, n, k, seed);
        let residual = Rng::new(seed + 100).fill(case.m * n, 5.0);
        let batch = prefix_bits(gpu, &case, case.m, &residual);
        for &rows in row_counts {
            let short = prefix_bits(gpu, &case, rows, &residual);
            // Each section holds m rows; take its first `rows`.
            let mut expected = Vec::with_capacity(short.len());
            let mut at = 0;
            for width in [n, 128, n, n, n / 2, n, 128] {
                expected.extend_from_slice(&batch[at..at + rows * width]);
                at += case.m * width;
            }
            assert_eq!(short.len(), expected.len());
            for (index, (s, e)) in short.iter().zip(&expected).enumerate() {
                assert_eq!(s, e, "{n} x {k}, {rows} rows, value {index}");
            }
        }
    }
}

#[test]
fn gemm_bias_residual_matches_the_reference_at_k_4096() {
    let case = Case::new(70, 1024, 4096, 3);
    let residual = Rng::new(4).fill(case.m * case.n, 5.0);
    let (linear, magnitude) = case.reference();
    let expected: Vec<f64> = linear
        .iter()
        .zip(&residual)
        .map(|(&x, &r)| x + f64::from(r))
        .collect();
    let got = run(&case, &Output::Residual(residual));
    assert_close(&got, &expected, &magnitude, 1e-7);
}
