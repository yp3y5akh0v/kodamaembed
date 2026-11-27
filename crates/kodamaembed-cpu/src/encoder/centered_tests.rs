//! E5's centered INT8 recipe on two synthetic layers of its width.

use super::{Encoder, EncoderError, TokenSequence};
use crate::{CpuPrecision, Kernel};
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};

const HIDDEN: usize = 1024;

/// About 3.5 times the largest cosine distance from FP32 measured on this model.
const CENTERED_DISTANCE: f64 = 1.3e-4;

/// E5-large-v1's shape on two layers; the second defers its QKV norm.
fn shape() -> Shape {
    Shape {
        layers: 2,
        hidden: HIDDEN,
        heads: 16,
        intermediate: 4096,
        vocab: 30_522,
        max_positions: 512,
        type_vocab: 2,
        eps: 1e-12,
        positions: Positions::Sequential,
    }
}

fn supported() -> bool {
    Kernel::detect() == Kernel::Avx2
        && is_x86_feature_detected!("avxvnni")
        && is_x86_feature_detected!("f16c")
}

fn load(weights: &Synthetic, max_tokens: usize, precision: CpuPrecision) -> Encoder {
    let mut encoder = Encoder::new_without_shape_check(
        Kernel::Avx2,
        shape(),
        Output::Embedding(Pooling::Mean),
        max_tokens,
        weights,
        precision,
    )
    .unwrap();
    // Off the shared pool, which a worker-budget test parks.
    encoder.pool = Some(crate::worker_pool("centered-tests", 4).unwrap());
    encoder
}

/// Text-free calibration inputs: pseudo-random BERT words from a fixed seed.
fn word_calibration() -> Vec<TokenSequence> {
    let mut inputs = vec![(vec![101, 102], vec![0, 0]); 2];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..2 {
        for length in [16, 32, 48, 64, 96, 128, 160, 192] {
            let mut ids = vec![101];
            while ids.len() + 1 < length {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ids.push((1996 + (state >> 33) % (30_522 - 1996)) as i32);
            }
            ids.push(102);
            let types = vec![0; ids.len()];
            inputs.push((ids, types));
        }
    }
    inputs
}

fn tokenized(lengths: &[usize]) -> (Vec<i32>, Vec<i32>) {
    let mut ids = Vec::new();
    let mut word = 0usize;
    for &len in lengths {
        ids.push(101);
        for _ in 2..len {
            ids.push((1996 + word * 7919 % (30_522 - 1996)) as i32);
            word += 1;
        }
        ids.push(102);
    }
    let types = vec![0; ids.len()];
    (ids, types)
}

/// Interior tokens are needed: their mean calibrates that class.
fn calibration() -> Vec<TokenSequence> {
    vec![
        (vec![101, 102], vec![0, 0]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
        (
            [101].into_iter().chain(3000..3012).chain([102]).collect(),
            vec![0; 14],
        ),
    ]
}

#[test]
fn centered_encoder_keeps_batch_invariance_and_input_checks() {
    if !supported() {
        return;
    }
    let weights = Synthetic::new(&shape(), 73);
    let mut encoder = load(&weights, 128, CpuPrecision::E5CenteredInt8);
    assert_eq!(encoder.precision(), CpuPrecision::E5CenteredInt8);
    assert_ne!(encoder.numerical_identity(), "cpu-fp32-v1");
    encoder.calibrate(&word_calibration()).unwrap();
    let lengths = [3, 100, 1];
    let ids: Vec<_> = (0..104).map(|i| i % 29).collect();
    let together = encoder.encode(&ids, &[0; 104], &lengths).unwrap();
    assert!(together.iter().all(|x| x.is_finite()));
    let mut first = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let single = encoder
            .encode(&ids[first..first + len], &vec![0; len], &[len])
            .unwrap();
        assert!(
            single
                .iter()
                .zip(&together[i * HIDDEN..(i + 1) * HIDDEN])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let norm: f64 = single
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        first += len;
    }
    assert!(matches!(
        encoder.encode(&[-1], &[0], &[1]),
        Err(EncoderError::Input(_))
    ));
    assert!(matches!(
        encoder.encode(&[1], &[2], &[1]),
        Err(EncoderError::Input(_))
    ));
    assert!(matches!(
        encoder.encode(&[1], &[0], &[0]),
        Err(EncoderError::Input(_))
    ));
}

#[test]
fn centered_preset_is_batch_invariant_and_tracks_fp32() {
    if !supported() {
        return;
    }
    let weights = Synthetic::new(&shape(), 73);
    let mut centered = load(&weights, 512, CpuPrecision::E5CenteredInt8);
    centered.calibrate(&word_calibration()).unwrap();
    // Lengths as the tokenizer gives them; then ids with no special tokens,
    // one a lone token.
    let lengths = [3, 100, 2, 40, 64, 16, 128];
    let (ids, types) = tokenized(&lengths);
    let raw_lengths = [3, 100, 1, 40, 64];
    let raw: Vec<i32> = (0..raw_lengths.iter().sum::<usize>())
        .map(|i| (i * 7919 % 30_522) as i32)
        .collect();
    for (ids, lengths) in [(&ids[..], &lengths[..]), (&raw[..], &raw_lengths[..])] {
        let types = vec![0; ids.len()];
        let together = centered.encode(ids, &types, lengths).unwrap();
        let mut first = 0;
        for (i, &len) in lengths.iter().enumerate() {
            let single = centered
                .encode(&ids[first..first + len], &types[first..first + len], &[len])
                .unwrap();
            assert!(
                single
                    .iter()
                    .zip(&together[i * HIDDEN..(i + 1) * HIDDEN])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "sequence {i} of {len} tokens"
            );
            first += len;
        }
    }
    let reference = load(&weights, 512, CpuPrecision::Fp32)
        .encode(&ids, &types, &lengths)
        .unwrap();
    let ours = centered.encode(&ids, &types, &lengths).unwrap();
    for (i, &len) in lengths.iter().enumerate() {
        let distance = 1.0
            - ours[i * HIDDEN..(i + 1) * HIDDEN]
                .iter()
                .zip(&reference[i * HIDDEN..(i + 1) * HIDDEN])
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>();
        eprintln!("{len} tokens, 1 - cos to FP32: {distance:.3e}");
        assert!(distance < CENTERED_DISTANCE, "{len} tokens: {distance:.3e}");
    }
}

#[test]
fn a_failed_calibration_leaves_the_encoder_as_it_was() {
    if !supported() {
        return;
    }
    let weights = Synthetic::new(&shape(), 97);
    let mut encoder = load(&weights, 512, CpuPrecision::E5CenteredInt8);
    let lengths = [3, 40, 100];
    let (ids, types) = tokenized(&lengths);
    // No tokens between the special ones: the pass runs, then its means fail.
    assert!(matches!(
        encoder.calibrate(&[(vec![101, 102], vec![0, 0])]),
        Err(EncoderError::Input(_))
    ));
    // A sequence a type short and one a type long, the totals equal.
    assert!(matches!(
        encoder.calibrate(&[
            (vec![101, 7, 8, 102], vec![0, 0, 0]),
            (vec![101, 9, 10, 102], vec![0; 5]),
        ]),
        Err(EncoderError::Input(_))
    ));
    // Still uncalibrated: no static centers to encode on.
    assert!(matches!(
        encoder.encode(&ids, &types, &lengths),
        Err(EncoderError::Unsupported(_))
    ));
    encoder.calibrate(&word_calibration()).unwrap();
    let before = encoder.encode(&ids, &types, &lengths).unwrap();
    // Calibration quantizes the encoder once.
    assert!(matches!(
        encoder.calibrate(&word_calibration()),
        Err(EncoderError::Unsupported(_))
    ));
    let after = encoder.encode(&ids, &types, &lengths).unwrap();
    assert!(
        before
            .iter()
            .zip(&after)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
fn installed_calibration_tables_make_the_calibrated_encoder() {
    if !supported() {
        return;
    }
    let weights = Synthetic::new(&shape(), 83);
    let mut calibrated = load(&weights, 256, CpuPrecision::E5CenteredInt8);
    assert!(matches!(
        calibrated.calibration(),
        Err(EncoderError::Unsupported(_))
    ));
    calibrated.calibrate(&word_calibration()).unwrap();
    let tables = calibrated.calibration().unwrap();
    let mut installed = load(&weights, 256, CpuPrecision::E5CenteredInt8);
    // Cut short, run long, a digit of -128 (layer 0's first) and a NaN row
    // scale (layer 0's first, after QKV's [3072, 1024] digits): refused.
    let mut longer = tables.clone();
    longer.push(0);
    let mut digit = tables.clone();
    digit[0] = 0x80;
    let mut scale = tables.clone();
    let first_scale = 3 * HIDDEN * HIDDEN;
    scale[first_scale..first_scale + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    for bad in [&tables[..tables.len() - 1], &longer, &digit, &scale] {
        assert!(matches!(
            installed.install_calibration(bad),
            Err(EncoderError::Input(_))
        ));
    }
    let lengths = [3, 40, 100];
    let (ids, types) = tokenized(&lengths);
    assert!(matches!(
        installed.encode(&ids, &types, &lengths),
        Err(EncoderError::Unsupported(_))
    ));
    installed.install_calibration(&tables).unwrap();
    assert_eq!(installed.calibration().unwrap(), tables);
    for lengths in [&lengths[..], &[2][..], &[64, 64][..]] {
        let (ids, types) = tokenized(lengths);
        let want = calibrated.encode(&ids, &types, lengths).unwrap();
        let got = installed.encode(&ids, &types, lengths).unwrap();
        assert!(
            want.iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "lengths {lengths:?}"
        );
    }
    // Once only, and for a centered preset only.
    assert!(matches!(
        installed.install_calibration(&tables),
        Err(EncoderError::Unsupported(_))
    ));
    assert!(matches!(
        load(&weights, 256, CpuPrecision::Fp32).install_calibration(&tables),
        Err(EncoderError::Unsupported(_))
    ));
}

#[test]
fn calibration_takes_more_tokens_than_a_batch_with_the_same_centers() {
    if !supported() {
        return;
    }
    let weights = Synthetic::new(&shape(), 89);
    let inputs = word_calibration();
    assert!(inputs.iter().map(|(ids, _)| ids.len()).sum::<usize>() > 1024);
    let calibrated = |max_tokens| {
        let mut encoder = load(&weights, max_tokens, CpuPrecision::E5CenteredInt8);
        encoder.calibrate(&inputs).unwrap();
        encoder
    };
    let (mut batch, mut whole) = (calibrated(128), calibrated(2048));
    for lengths in [&[2, 17, 64][..], &[100][..]] {
        let (ids, types) = tokenized(lengths);
        let small = batch.encode(&ids, &types, lengths).unwrap();
        let large = whole.encode(&ids, &types, lengths).unwrap();
        assert!(
            small
                .iter()
                .zip(&large)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    let (ids, types) = tokenized(&[100, 100]);
    assert!(matches!(
        batch.encode(&ids, &types, &[100, 100]),
        Err(EncoderError::Input(_))
    ));
}

#[test]
fn calibrated_static_centers_stay_batch_invariant_and_center_first_and_last_tokens() {
    if !supported() {
        return;
    }
    let weights = Synthetic::new(&shape(), 79);
    assert!(matches!(
        load(&weights, 256, CpuPrecision::Fp32).calibrate(&calibration()),
        Err(EncoderError::Unsupported(_))
    ));
    let mut uncalibrated = load(&weights, 256, CpuPrecision::E5CenteredInt8);
    let mut calibrated = load(&weights, 256, CpuPrecision::E5CenteredInt8);
    for inputs in [
        vec![(vec![101], vec![0])],
        vec![(vec![101, 30_522], vec![0, 0])],
        // No tokens between first and last means no interior-token class.
        vec![(vec![101, 102], vec![0, 0])],
    ] {
        assert!(matches!(
            calibrated.calibrate(&inputs),
            Err(EncoderError::Input(_))
        ));
    }
    calibrated.calibrate(&calibration()).unwrap();
    let lengths = [3, 100, 1, 2, 40, 64];
    let tokens: usize = lengths.iter().sum();
    let ids: Vec<i32> = (0..tokens).map(|i| (i * 7919 % 30_522) as i32).collect();
    let types = vec![0; tokens];
    let together = calibrated.encode(&ids, &types, &lengths).unwrap();
    let mut first = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let (ids, types) = (&ids[first..first + len], &types[first..first + len]);
        let row = &together[i * HIDDEN..(i + 1) * HIDDEN];
        let single = calibrated.encode(ids, types, &[len]).unwrap();
        assert!(
            single
                .iter()
                .zip(row)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "sequence {i} alone"
        );
        // Every row centers on a static center: none before calibration.
        assert!(matches!(
            uncalibrated.encode(ids, types, &[len]),
            Err(EncoderError::Unsupported(_))
        ));
        first += len;
    }
}
