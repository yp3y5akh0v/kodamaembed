use crate::support::{Rng, devices, matrix_devices};
use half::f16;
use kodamaembed_model::Rotary;
use kodamaembed_vulkan::{
    Binding, Buffer, Descriptors, Device, Kernel, KernelSpec, Push, Recorder, shaders,
};

struct Case {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f16>,
    w: Vec<f16>,
    bias: Vec<f32>,
    residual: Vec<f32>,
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
            residual: rng.fill(m * n, 5.0),
        }
    }

    /// Exact sums plus bias, and each element's sum of |a * w| for the tolerance.
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

struct Shader<'a> {
    spirv: &'a [u8],
    /// The build that takes its columns in pairs.
    paired: &'a [u8],
    subgroup: Option<u32>,
    /// Output rows and columns per workgroup.
    tile: (usize, usize),
    /// The row GEMM's K stage, its specialization constant 2.
    stage: Option<u32>,
}

/// Constants: the epilogue (0 bias, 1 GELU, 2 residual), then the low columns.
fn run(device: &Device, shader: &Shader, case: &Case, constants: &[u32]) -> (Vec<f16>, Vec<f32>) {
    let mut constants = vec![constants[0], constants.get(1).copied().unwrap_or(0)];
    constants.extend(shader.stage);
    execute(device, shader.spirv, shader, case, &constants, &[])
}

/// The rotation base and the sequence length of the paired builds' rows.
const THETA: f32 = 10_000.0;
const LONGEST: usize = 70;

/// Each of `rows` rows' place in sequences of [`LONGEST`] packed back to back.
fn positions(rows: usize) -> Vec<i32> {
    (0..rows).map(|row| (row % LONGEST) as i32).collect()
}

/// The paired build, its rows at [`positions`] and its heads `head` wide. Constants: the
/// epilogue (3 gated, 4 turned), then the low columns.
fn run_paired(
    device: &Device,
    shader: &Shader,
    case: &Case,
    constants: [u32; 2],
    head: usize,
) -> (Vec<f16>, Vec<f32>) {
    let places = Buffer::from_slice(device, &positions(case.m)).unwrap();
    let turns = Rotary::paired_turns(THETA, head, LONGEST);
    let turns = Buffer::from_slice(device, &turns).unwrap();
    // The head is constant 3; constant 2 is the row GEMM's stage alone.
    let [epilogue, low_columns] = constants;
    let constants = [
        epilogue,
        low_columns,
        shader.stage.unwrap_or(0),
        head as u32,
    ];
    let more = [places.binding(), turns.binding()];
    execute(device, shader.paired, shader, case, &constants, &more)
}

/// Runs `spirv` on the GEMM's six buffers, then `more`.
fn execute(
    device: &Device,
    spirv: &[u8],
    shader: &Shader,
    case: &Case,
    constants: &[u32],
    more: &[Binding],
) -> (Vec<f16>, Vec<f32>) {
    let bindings = 6 + more.len() as u32;
    let spec = KernelSpec {
        spirv,
        bindings,
        push_bytes: 12,
        constants,
        subgroup_size: shader.subgroup,
    };
    let kernel = unsafe { Kernel::new(device, &spec) }.unwrap();
    let descriptors = Descriptors::new(device, 1, bindings).unwrap();
    let len = case.m * case.n;
    let a = Buffer::from_slice(device, &case.a).unwrap();
    let w = Buffer::from_slice(device, &case.w).unwrap();
    let bias = Buffer::from_slice(device, &case.bias).unwrap();
    let residual = Buffer::from_slice(device, &case.residual).unwrap();
    let out16 = Buffer::<f16>::zeroed(device, len).unwrap();
    let out32 = Buffer::<f32>::zeroed(device, len).unwrap();
    let mut bound = vec![
        a.binding(),
        w.binding(),
        bias.binding(),
        residual.binding(),
        out16.binding(),
        out32.binding(),
    ];
    bound.extend_from_slice(more);
    let set = unsafe { descriptors.bind(&kernel, &bound) }.unwrap();
    let push = Push::new()
        .u32(case.m as u32)
        .u32(case.n as u32)
        .u32(case.k as u32);
    let groups = [
        (case.n / shader.tile.1) as u32,
        case.m.div_ceil(shader.tile.0) as u32,
        1,
    ];
    device
        .submit(|command| unsafe {
            Recorder::new(device, command).dispatch(&kernel, set, &push, groups);
            Ok(())
        })
        .unwrap();
    (out16.read(device).unwrap(), out32.read(device).unwrap())
}

fn widen16(values: &[f16]) -> Vec<f64> {
    values.iter().map(|v| f64::from(v.to_f32())).collect()
}

/// FP32 sums err far below 1e-5 of the summed magnitudes; an indexing mistake errs by whole values.
fn assert_close(label: &str, got: &[f64], expected: &[f64], magnitude: &[f64], rounding: f64) {
    for (index, ((&g, &e), &size)) in got.iter().zip(expected).zip(magnitude).enumerate() {
        let bound = 1e-5 * size + rounding * e.abs() + 1e-6;
        assert!(
            (g - e).abs() <= bound,
            "{label} element {index}: {g} vs {e}"
        );
    }
}

/// `depth` is the first case's K: an odd K-tile count, as the even ones end on the other buffer.
fn check_every_epilogue(device: &Device, label: &str, shader: &Shader, depth: usize) {
    let case = Case::new(300, 256, depth, 1);
    let (linear, magnitude) = case.reference();
    let half_rounding = f64::powi(2.0, -11);
    let (got, _) = run(device, shader, &case, &[0]);
    assert_close(label, &widen16(&got), &linear, &magnitude, half_rounding);

    // The bias epilogue's low parts: FP16 output plus low recovers the FP32 sum, in the leading
    // columns that hold queries and keys, and up to the last column.
    for columns in [128, case.n] {
        let (high, low) = run(device, shader, &case, &[0, columns as u32]);
        assert_close(label, &widen16(&high), &linear, &magnitude, half_rounding);
        let recovered: Vec<f64> = (0..case.m * columns)
            .map(|i| {
                let (row, col) = (i / columns, i % columns);
                f64::from(high[row * case.n + col].to_f32()) + f64::from(low[i])
            })
            .collect();
        let pick = |values: &[f64]| -> Vec<f64> {
            (0..case.m * columns)
                .map(|i| values[(i / columns) * case.n + i % columns])
                .collect()
        };
        assert_close(label, &recovered, &pick(&linear), &pick(&magnitude), 0.0);
    }

    let case = Case::new(130, 384, 1024, 2);
    let (linear, magnitude) = case.reference();
    let expected: Vec<f64> = linear.iter().map(|&x| gelu(x)).collect();
    let (got, _) = run(device, shader, &case, &[1]);
    assert_close(label, &widen16(&got), &expected, &magnitude, half_rounding);

    let case = Case::new(70, 1024, 4096, 3);
    let (linear, magnitude) = case.reference();
    let expected: Vec<f64> = linear
        .iter()
        .zip(&case.residual)
        .map(|(&x, &r)| x + f64::from(r))
        .collect();
    let (_, got) = run(device, shader, &case, &[2]);
    let got: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
    assert_close(label, &got, &expected, &magnitude, 1e-7);
}

/// The gated product of each pair of columns, an input and its gate, which FP16 rounds once.
fn check_gated(device: &Device, label: &str, shader: &Shader, depth: usize) {
    let case = Case::new(130, 384, depth, 31);
    let (linear, magnitude) = case.reference();
    let (got, _) = run_paired(device, shader, &case, [3, 0], 64);
    for pair in 0..case.m * case.n / 2 {
        let (input, by) = (linear[2 * pair], linear[2 * pair + 1]);
        let (have, want) = (f64::from(got[pair].to_f32()), gelu(input) * by);
        // The GELU's slope stays below 1.13; the shader's erf is within 1.5e-7, which the gate
        // scales by up to 8.
        let sums =
            1.13 * magnitude[2 * pair] * by.abs() + magnitude[2 * pair + 1] * gelu(input).abs();
        assert!(
            (have - want).abs() <= 1e-5 * sums + f64::powi(2.0, -11) * want.abs() + 4e-6,
            "{label} pair {pair}: {have} vs {want}"
        );
    }
}

/// Each pair of the low columns turned by its row's position, FP16 output plus low; the other
/// columns as the bias epilogue stores them.
fn check_turned(device: &Device, label: &str, shader: &Shader, depth: usize) {
    let (head, columns) = (64, 256);
    let case = Case::new(70, 384, depth, 37);
    let (linear, magnitude) = case.reference();
    let places = positions(case.m);
    let (high, low) = run_paired(device, shader, &case, [4, columns as u32], head);
    for row in 0..case.m {
        for col in (0..case.n).step_by(2) {
            let at = row * case.n + col;
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
                    (kept - want).abs() <= 1e-5 * size + f64::powi(2.0, -11) * want.abs() + 1e-6,
                    "{label} high ({row}, {}): {kept} vs {want}",
                    col + offset
                );
                if col < columns {
                    let value = kept + f64::from(low[row * columns + col + offset]);
                    assert!(
                        (value - want).abs() <= 1e-5 * size + 1e-6,
                        "{label} high + low ({row}, {}): {value} vs {want}",
                        col + offset
                    );
                }
            }
        }
    }
}

const PORTABLE: Shader = Shader {
    spirv: shaders::GEMM_PORTABLE,
    paired: shaders::GEMM_PORTABLE_PAIRED,
    subgroup: None,
    tile: (64, 64),
    stage: None,
};

const MATRIX: Shader = Shader {
    spirv: shaders::GEMM_MATRIX,
    paired: shaders::GEMM_MATRIX_PAIRED,
    subgroup: Some(32),
    tile: (128, 128),
    stage: None,
};

fn rows(stage: u32) -> Shader<'static> {
    Shader {
        spirv: shaders::GEMM_ROWS,
        paired: shaders::GEMM_ROWS_PAIRED,
        subgroup: Some(32),
        tile: (16, 64),
        stage: Some(stage),
    }
}

/// K depths with an odd count of each row GEMM's stages, as the even ones end on the other buffer.
const ROW_STAGES: [(u32, usize); 2] = [(64, 448), (128, 384)];

#[test]
fn the_portable_gemm_matches_the_reference_on_every_gpu() {
    for device in devices() {
        let label = format!("{} portable", device.name);
        check_every_epilogue(&device, &label, &PORTABLE, 480);
    }
}

#[test]
fn the_matrix_gemm_matches_the_reference_where_supported() {
    for device in matrix_devices() {
        let label = format!("{} matrix", device.name);
        check_every_epilogue(&device, &label, &MATRIX, 480);
    }
}

#[test]
fn the_row_gemm_matches_the_reference_where_supported() {
    for device in matrix_devices() {
        for (stage, depth) in ROW_STAGES {
            let label = format!("{} rows, K stage {stage}", device.name);
            check_every_epilogue(&device, &label, &rows(stage), depth);
        }
    }
}

/// `check` on the portable GEMM's paired build.
fn on_every_gpu(check: fn(&Device, &str, &Shader, usize)) {
    for device in devices() {
        let label = format!("{} portable", device.name);
        check(&device, &label, &PORTABLE, 480);
    }
}

/// `check` on the paired builds of the matrix GEMM and of the row GEMM at both K stages.
fn where_supported(check: fn(&Device, &str, &Shader, usize)) {
    for device in matrix_devices() {
        check(&device, &format!("{} matrix", device.name), &MATRIX, 480);
        for (stage, depth) in ROW_STAGES {
            let label = format!("{} rows, K stage {stage}", device.name);
            check(&device, &label, &rows(stage), depth);
        }
    }
}

#[test]
fn the_portable_gemm_gates_each_input_by_its_gate_on_every_gpu() {
    on_every_gpu(check_gated);
}

#[test]
fn the_matrix_and_row_gemms_gate_each_input_by_its_gate_where_supported() {
    where_supported(check_gated);
}

#[test]
fn the_portable_gemm_turns_the_pairs_of_its_low_columns_on_every_gpu() {
    on_every_gpu(check_turned);
}

#[test]
fn the_matrix_and_row_gemms_turn_the_pairs_of_their_low_columns_where_supported() {
    where_supported(check_turned);
}

/// The encoder picks either GEMM per batch, so a row's output must not depend on its batch's rows.
#[test]
fn the_row_gemm_writes_the_matrix_gemm_bits_where_supported() {
    let bits = |(half, single): (Vec<f16>, Vec<f32>)| {
        let half: Vec<u16> = half.iter().map(|value| value.to_bits()).collect();
        let single: Vec<u32> = single.iter().map(|value| value.to_bits()).collect();
        (half, single)
    };
    // Partial row tiles, both models' projection shapes, every epilogue.
    let shapes = [
        (1, 1152, 384),
        (15, 384, 384),
        (17, 384, 1536),
        (65, 3072, 1024),
        (300, 1024, 4096),
    ];
    for device in matrix_devices() {
        for (m, n, k) in shapes {
            let case = Case::new(m, n, k, m as u64);
            for constants in [[0, 128], [1, 0], [2, 0]] {
                let want = bits(run(&device, &MATRIX, &case, &constants));
                for stage in [64, 128] {
                    assert!(
                        bits(run(&device, &rows(stage), &case, &constants)) == want,
                        "{} M {m} N {n} K {k} epilogue {constants:?} stage {stage}",
                        device.name
                    );
                }
            }
            // The paired builds: the gated product, and two heads turned with their low parts.
            for constants in [[3, 0], [4, 128]] {
                let want = bits(run_paired(&device, &MATRIX, &case, constants, 64));
                for stage in [64, 128] {
                    assert!(
                        bits(run_paired(&device, &rows(stage), &case, constants, 64)) == want,
                        "{} M {m} N {n} K {k} paired epilogue {constants:?} stage {stage}",
                        device.name
                    );
                }
            }
        }
    }
}
