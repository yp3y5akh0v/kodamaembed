//! The GPU's centered INT8 encoder against the CPU's, every output bit for bit.

use kodamaembed_cpu::{CpuPrecision, Encoder, Kernel};
use kodamaembed_cuda::{Context, repro::ReproEncoder};
use kodamaembed_model::{
    Output, Pooling, Positions, Shape, reference::Synthetic, repro::EncoderExport,
};
use std::sync::OnceLock;

const E5: Shape = Shape {
    layers: 24,
    hidden: 1024,
    heads: 16,
    intermediate: 4096,
    vocab: 30_522,
    max_positions: 512,
    type_vocab: 2,
    eps: 1e-12,
    positions: Positions::Sequential,
};

/// Interior tokens are needed: their mean calibrates that class.
fn calibration() -> Vec<(Vec<i32>, Vec<i32>)> {
    vec![
        (vec![101, 102], vec![0, 0]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
    ]
}

fn centered_cpu() -> bool {
    Kernel::detect() == Kernel::Avx2
        && is_x86_feature_detected!("avxvnni")
        && is_x86_feature_detected!("f16c")
}

/// A calibrated export, built once for the tests that only read or corrupt it.
fn shared_export() -> &'static EncoderExport {
    static EXPORT: OnceLock<EncoderExport> = OnceLock::new();
    EXPORT.get_or_init(|| calibrated_cpu(1, 16).export_reproducible().unwrap())
}

/// A context whose L2 reads as `bytes`: `u32::MAX` holds every layer ahead, zero holds none.
fn with_l2(context: &Context, bytes: u32) -> Context {
    let mut context = context.clone();
    context.l2_cache_bytes = bytes;
    context
}

/// Without `stream` the GPU takes the per-layer launches; each runs with and without reads ahead.
fn same_bits_as_cpu(seed: u64, stream: bool) {
    if !centered_cpu() {
        return;
    }
    let mut cpu = calibrated_cpu(seed, 1024);
    let context = Context::new(0).unwrap();
    let export = cpu.export_reproducible().unwrap();
    let mut gpus = [0, u32::MAX].map(|l2| {
        let mut gpu = ReproEncoder::new(&with_l2(&context, l2), &export).unwrap();
        if !stream {
            gpu.set_aside_stream_engine();
        }
        (l2, gpu)
    });
    // Reordered sizes expose stale storage and graphs; adjacent row and
    // attention boundaries expose incomplete work assignments.
    let ten = [9, 3, 17, 2, 5, 8, 4, 6, 11, 3];
    let nine = [3, 4, 5, 6, 7, 3, 4, 5, 6];
    for (pass, lengths) in (0..3).flat_map(|pass| {
        [
            &[1][..],
            &[130, 65, 7],
            &[2],
            &[3, 17, 1, 2],
            &[64, 33, 5],
            &[100, 2],
            &ten,
            &[65; 9],
            &[16],
            &[17],
            &[24],
            &[25],
            &[31],
            &[32],
            &[33],
            &[40, 24],
            &[3, 3, 3, 3, 4],
            &[5, 9],
            &[9, 30, 12, 13],
            &nine,
        ]
        .map(|lengths| (pass, lengths))
    }) {
        let tokens: usize = lengths.iter().sum();
        let ids: Vec<i32> = (0..tokens)
            .map(|i| ((i as u64 * 7_919 + seed + 1_000 * pass) % 30_522) as i32)
            .collect();
        let types = vec![0; tokens];
        let expected = cpu.encode(&ids, &types, lengths).unwrap();
        for (l2, gpu) in &mut gpus {
            let actual = gpu.encode(&ids, &types, lengths).unwrap();
            assert_eq!(
                actual.len(),
                expected.len(),
                "pass {pass}, lengths {lengths:?}, L2 {l2}"
            );
            let differing = expected
                .iter()
                .zip(&actual)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let first = expected
                .iter()
                .zip(&actual)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(
                differing,
                0,
                "pass {pass}, lengths {lengths:?}, L2 {l2}: {differing} of {} values differ, first at {first:?}: cpu {:?} gpu {:?}",
                expected.len(),
                first.map(|i| expected[i]),
                first.map(|i| actual[i]),
            );
        }
    }
}

fn assert_same_bits(expected: &[f32], actual: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    let mismatch = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (e, a))| e.to_bits() != a.to_bits())
        .map(|(i, (e, a))| (i, e.to_bits(), a.to_bits()));
    assert_eq!(mismatch, None, "{label}: index, cpu, gpu");
}

fn sequence_input(len: usize, salt: usize) -> (Vec<i32>, Vec<i32>) {
    let ids = (0..len)
        .map(|i| (1 + (i * 7_919 + salt * 101) % 30_521) as i32)
        .collect();
    let types = (0..len).map(|i| ((i + salt) % 2) as i32).collect();
    (ids, types)
}

fn batch_input(lengths: &[usize], salt: usize) -> (Vec<i32>, Vec<i32>) {
    lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| sequence_input(len, salt + i))
        .fold((Vec::new(), Vec::new()), |(mut ids, mut types), (i, t)| {
            ids.extend(i);
            types.extend(t);
            (ids, types)
        })
}

fn calibrated_cpu(seed: u64, capacity: usize) -> Encoder {
    let weights = Synthetic::new(&E5, seed);
    let mut cpu = Encoder::new_with_precision(
        Kernel::Avx2,
        E5,
        Output::Embedding(Pooling::Mean),
        capacity,
        &weights,
        CpuPrecision::E5CenteredInt8,
    )
    .unwrap();
    cpu.calibrate(&calibration()).unwrap();
    cpu
}

#[test]
fn e5_boundaries_storage_growth_and_graph_eviction_match_the_cpu() {
    if !centered_cpu() {
        return;
    }
    let mut cpu = calibrated_cpu(113, 1088);
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &cpu.export_reproducible().unwrap()).unwrap();
    let lengths = [1, 65, 2, 512, 3, 64, 63];
    let (ids, types) = batch_input(&lengths, 11);
    let expected = cpu.encode(&ids, &types, &lengths).unwrap();
    assert_same_bits(
        &expected,
        &gpu.encode(&ids, &types, &lengths).unwrap(),
        "mixed boundary batch",
    );
    let mut at = 0;
    for (sequence, &len) in lengths.iter().enumerate() {
        let single = cpu
            .encode(&ids[at..at + len], &types[at..at + len], &[len])
            .unwrap();
        assert_same_bits(
            &single,
            &expected[sequence * E5.hidden..(sequence + 1) * E5.hidden],
            "CPU batch independence",
        );
        assert_same_bits(
            &single,
            &gpu.encode(&ids[at..at + len], &types[at..at + len], &[len])
                .unwrap(),
            "GPU single boundary sequence",
        );
        at += len;
    }
    // Repeated wide batches exercise attention publication counters on graph replay.
    for lengths in [vec![512, 512], vec![129, 257, 512]] {
        for pass in 0..2 {
            let (ids, types) = batch_input(&lengths, 113 + pass);
            let expected = cpu.encode(&ids, &types, &lengths).unwrap();
            assert_same_bits(
                &expected,
                &gpu.encode(&ids, &types, &lengths).unwrap(),
                &format!("wide batch pass {pass}, lengths {lengths:?}"),
            );
        }
    }
    // Grow sequence and token buffers independently to expose graphs that retain stale pointers.
    for growth in [vec![1; 129], vec![64; 17], vec![2], vec![1; 129]] {
        let (ids, types) = batch_input(&growth, 31);
        let expected = cpu.encode(&ids, &types, &growth).unwrap();
        assert_same_bits(
            &expected,
            &gpu.encode(&ids, &types, &growth).unwrap(),
            "capacity growth and reuse",
        );
    }
    assert_same_bits(
        &expected,
        &gpu.encode(&ids, &types, &lengths).unwrap(),
        "mixed batch after storage growth",
    );

    // Equal total sizes keep storage fixed while more than 64 shapes evict graphs.
    let shapes: Vec<Vec<usize>> = (1..15)
        .flat_map(|a| (1..16 - a).map(move |b| vec![a, b, 16 - a - b]))
        .take(70)
        .collect();
    assert_eq!(shapes.len(), 70);
    for pass in 0..2 {
        let bank: Vec<_> = (1..=14)
            .map(|len| {
                let (ids, types) = sequence_input(len, 71 + pass * 17 + len);
                let expected = cpu.encode(&ids, &types, &[len]).unwrap();
                (ids, types, expected)
            })
            .collect();
        let selected: Vec<_> = if pass == 0 {
            shapes.iter().collect()
        } else {
            // A surviving graph, an evicted one, and a different surviving graph.
            vec![&shapes[69], &shapes[0], &shapes[35]]
        };
        for lengths in selected {
            let mut ids = Vec::new();
            let mut types = Vec::new();
            let mut expected = Vec::new();
            for &len in lengths {
                let (i, t, e) = &bank[len - 1];
                ids.extend_from_slice(i);
                types.extend_from_slice(t);
                expected.extend_from_slice(e);
            }
            assert_same_bits(
                &expected,
                &gpu.encode(&ids, &types, lengths).unwrap(),
                &format!("graph cache pass {pass}, lengths {lengths:?}"),
            );
        }
    }
}

/// The longest sequence the persistent plan takes.
const STREAM_LENGTHS: usize = 16;

#[test]
#[ignore = "encodes every length and mixed batches on both devices"]
fn e5_every_length_and_random_batches_match_the_cpu() {
    if !centered_cpu() {
        return;
    }
    let mut cpu = calibrated_cpu(113, 4096);
    let context = Context::new(0).unwrap();
    let export = cpu.export_reproducible().unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    let mut layers = ReproEncoder::new(&context, &export).unwrap();
    layers.set_aside_stream_engine();
    let mut alone = |gpu: &mut ReproEncoder, len: usize, label: &str| {
        let (ids, types) = sequence_input(len, len);
        assert_same_bits(
            &cpu.encode(&ids, &types, &[len]).unwrap(),
            &gpu.encode(&ids, &types, &[len]).unwrap(),
            &format!("{label}, length {len}"),
        );
    };
    for len in 1..=512 {
        alone(&mut gpu, len, "alone");
    }
    for len in 1..=STREAM_LENGTHS {
        alone(&mut layers, len, "per-layer launches");
    }
    for lengths in [&[1, 129][..], &[15, 144, 1], &[32, 129]] {
        let (ids, types) = batch_input(lengths, 91);
        assert_same_bits(
            &cpu.encode(&ids, &types, lengths).unwrap(),
            &gpu.encode(&ids, &types, lengths).unwrap(),
            &format!("packed tile boundary, lengths {lengths:?}"),
        );
    }
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut below = |n: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % n
    };
    for batch in 0..64 {
        let count = 1 + below(8) as usize;
        let lengths: Vec<usize> = (0..count).map(|_| 1 + below(512) as usize).collect();
        let (ids, types) = batch_input(&lengths, batch);
        assert_same_bits(
            &cpu.encode(&ids, &types, &lengths).unwrap(),
            &gpu.encode(&ids, &types, &lengths).unwrap(),
            &format!("batch {batch}, lengths {lengths:?}"),
        );
    }
}

/// Synthetic SM counts exercise dispatch boundaries that one physical GPU cannot cover.
#[test]
fn e5_planned_grids_match_the_cpu() {
    if !centered_cpu() {
        return;
    }
    let mut cpu = calibrated_cpu(113, 128);
    let export = cpu.export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let cases: Vec<_> = [
        vec![1],
        vec![16],
        vec![5, 9],
        vec![3, 3, 3, 3, 4],
        vec![17],
        vec![9, 9],
        vec![40],
        vec![64],
        vec![70],
        vec![90],
        vec![64, 33],
        vec![128],
    ]
    .into_iter()
    .map(|lengths| {
        let (ids, types) = batch_input(&lengths, 91);
        let expected = cpu.encode(&ids, &types, &lengths).unwrap();
        (lengths, ids, types, expected)
    })
    .collect();
    for sms in [
        8, 24, 31, 32, 36, 40, 47, 48, 51, 52, 55, 56, 59, 60, 64, 76, 88, 104, 128,
    ] {
        let mut planned = context.clone();
        planned.sm_count = sms;
        let mut gpu = ReproEncoder::new(&planned, &export).unwrap();
        // A persistent plan's blocks must all be resident; other launches queue.
        if sms > context.sm_count {
            gpu.set_aside_stream_engine();
        }
        for (lengths, ids, types, expected) in &cases {
            for pass in 0..2 {
                assert_same_bits(
                    expected,
                    &gpu.encode(ids, types, lengths).unwrap(),
                    &format!("{sms} blocks, lengths {lengths:?}, pass {pass}"),
                );
            }
        }
    }
}

#[test]
fn valid_encodes_recover_after_validation_and_nonfinite_failures() {
    if !centered_cpu() {
        return;
    }
    let mut cpu = calibrated_cpu(113, 128);
    let mut export = cpu.export_reproducible().unwrap();
    // Token zero poisons only the deliberately failing calls.
    export.word[0] = f32::NAN;
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    for (ids, types, lengths) in [
        (vec![1], vec![0], vec![0]),
        (vec![1], vec![], vec![1]),
        (vec![-1], vec![0], vec![1]),
        (vec![30_522], vec![0], vec![1]),
        (vec![1], vec![2], vec![1]),
        (vec![1], vec![0], vec![usize::MAX, 2]),
    ] {
        assert!(gpu.encode(&ids, &types, &lengths).is_err());
        let (ids, types) = sequence_input(16, 113);
        let expected = cpu.encode(&ids, &types, &[16]).unwrap();
        assert_same_bits(
            &expected,
            &gpu.encode(&ids, &types, &[16]).unwrap(),
            "encode after validation error",
        );
    }
    for lengths in [vec![1], vec![16], vec![65], vec![3, 65]] {
        let (mut ids, types) = batch_input(&lengths, 127);
        let expected = cpu.encode(&ids, &types, &lengths).unwrap();
        let last = ids.len() - 1;
        let good = ids[last];
        ids[last] = 0;
        let error = gpu.encode(&ids, &types, &lengths).unwrap_err();
        assert!(error.to_string().contains("non-finite outputs"), "{error}");
        ids[last] = good;
        assert_same_bits(
            &expected,
            &gpu.encode(&ids, &types, &lengths).unwrap(),
            "encode after nonfinite output",
        );
    }
}

#[test]
fn two_encoders_keep_their_bits_when_moved_between_threads() {
    if !centered_cpu() {
        return;
    }
    let mut cpu = calibrated_cpu(113, 128);
    let export = cpu.export_reproducible().unwrap();
    let first_context = Context::new(0).unwrap();
    let first = ReproEncoder::new(&first_context, &export).unwrap();
    let second_context = Context::new(0).unwrap();
    let second = ReproEncoder::new(&second_context, &export).unwrap();
    let cases: Vec<_> = [vec![5, 9], vec![65, 3], vec![1; 17]]
        .into_iter()
        .map(|lengths| {
            let (ids, types) = batch_input(&lengths, 139);
            let expected = cpu.encode(&ids, &types, &lengths).unwrap();
            (lengths, ids, types, expected)
        })
        .collect();
    let mut encoders = vec![first, second];
    for migration in 0..2 {
        let together = std::sync::Barrier::new(encoders.len());
        encoders = std::thread::scope(|scope| {
            let handles: Vec<_> = encoders
                .into_iter()
                .enumerate()
                .map(|(index, mut gpu)| {
                    let cases = &cases;
                    let together = &together;
                    scope.spawn(move || {
                        together.wait();
                        for iteration in 0..6 {
                            let (lengths, ids, types, expected) =
                                &cases[(iteration + index) % cases.len()];
                            assert_same_bits(
                                expected,
                                &gpu.encode(ids, types, lengths).unwrap(),
                                &format!(
                                    "encoder {index}, migration {migration}, encode {iteration}"
                                ),
                            );
                        }
                        gpu
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
    }
    for mut gpu in encoders {
        let (lengths, ids, types, expected) = &cases[0];
        assert_same_bits(
            expected,
            &gpu.encode(ids, types, lengths).unwrap(),
            "encode after returning to the creating thread",
        );
    }
}

/// Values off FP16's grid near 2^-10, then +B and -B near 2^15, FP16's top binade.
fn fp16_range_column(len: usize, random: &mut impl FnMut() -> u64) -> Vec<f32> {
    let mut near = |scale: i32| {
        let r = random();
        let sign = if r >> 63 == 0 { 1.0 } else { -1.0 };
        sign * (1.0 + (r >> 40) as f32 / (1_u64 << 24) as f32) * 2_f32.powi(scale)
    };
    let mut big = 0.0;
    (0..len)
        .map(|j| match j % 8 {
            6 if j + 1 < len => {
                big = near(14);
                big
            }
            7 => -big,
            _ => near(-10),
        })
        .collect()
}

/// attend's scale_for(largest, 127), the sinks excluded.
fn value_step(largest: f32) -> f32 {
    let step = largest / 127.0;
    if step.is_subnormal() && f64::from(step) * 127.0 < f64::from(largest) {
        step.next_up()
    } else if step > 0.0 {
        step
    } else {
        1.0
    }
}

/// Constant keys cancel each +B, -B pair, leaving the sinks' share and its FP16 rounding.
#[test]
fn attention_matches_the_cpu_on_crafted_values() {
    if Kernel::detect() != Kernel::Avx2
        || !is_x86_feature_detected!("avxvnni")
        || !is_x86_feature_detected!("f16c")
    {
        return;
    }
    let export = calibrated_cpu(5, 512).export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    let h = E5.hidden;
    // Past one key chunk (the wide kernels), then within one (one launch).
    for lengths in [&[130, 7, 64, 1][..], &[40, 1, 7, 16]] {
        let tokens: usize = lengths.iter().sum();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        // Queries near one, keys a half, values cancelling.
        let mut qkv = vec![0.0_f32; tokens * 3 * h];
        let mut start = 0;
        for &len in lengths {
            for row in &mut qkv[start * 3 * h..(start + len) * 3 * h].chunks_exact_mut(3 * h) {
                for q in &mut row[..h] {
                    *q = 1.0 + (random() >> 40) as f32 / (1_u64 << 24) as f32;
                }
                row[h..2 * h].fill(0.5);
            }
            for column in 2 * h..3 * h {
                for (j, value) in fp16_range_column(len, &mut random).into_iter().enumerate() {
                    qkv[(start + j) * 3 * h + column] = value;
                }
            }
            start += len;
        }
        let mut expected = vec![0.0; tokens * h];
        kodamaembed_cpu::integer_attention(
            &qkv,
            lengths,
            h,
            E5.heads,
            Some(&export.layers[0].key_centers),
            &mut expected,
        );
        let (actual, steps) = gpu.crafted_attention(&qkv, lengths).unwrap();
        let differing = expected
            .iter()
            .zip(&actual)
            .filter(|(e, a)| half::f16::from_f32(**e).to_f32().to_bits() != a.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "lengths {lengths:?}: {differing} of {} context values differ",
            expected.len()
        );
        let mut start = 0;
        let mut differing = 0;
        for (s, &len) in lengths.iter().enumerate() {
            for c in 0..h {
                let largest = (start + 1..start + len - 1).fold(0.0_f32, |most, j| {
                    most.max(
                        half::f16::from_f32(qkv[j * 3 * h + 2 * h + c])
                            .to_f32()
                            .abs(),
                    )
                });
                let step = value_step(largest);
                differing += usize::from(step.to_bits() != steps[s * h + c].to_bits());
            }
            start += len;
        }
        assert_eq!(
            differing, 0,
            "lengths {lengths:?}: {differing} value steps differ"
        );
    }
}

/// A value near k + 1/2 steps whose quotient and reciprocal product round apart.
fn split_tie(step: f32, k: i32) -> Option<f32> {
    let reciprocal = 1.0 / step;
    let tie = ((f64::from(k) + 0.5) * f64::from(step)) as f32;
    let (mut below, mut above) = (tie, tie);
    for _ in 0..8 {
        for x in [below, above] {
            if (x / step).round_ties_even() != (x * reciprocal).round_ties_even() {
                return Some(x);
            }
        }
        below = below.next_down();
        above = above.next_up();
    }
    None
}

/// Queries on split ties, whose outputs a division in either device's packing would change.
#[test]
fn attention_packs_split_ties_alike() {
    if Kernel::detect() != Kernel::Avx2
        || !is_x86_feature_detected!("avxvnni")
        || !is_x86_feature_detected!("f16c")
    {
        return;
    }
    let export = calibrated_cpu(5, 512).export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    let (h, d) = (E5.hidden, E5.hidden / E5.heads);
    // Past one key chunk (the wide kernels), then within one (one launch).
    for lengths in [&[65, 33][..], &[64, 33]] {
        let tokens: usize = lengths.iter().sum();
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let mut unit = || (random() >> 40) as f32 / (1_u64 << 24) as f32;
        let mut qkv = vec![0.0_f32; tokens * 3 * h];
        let mut split = 0;
        for row in qkv.chunks_exact_mut(3 * h) {
            for head in row[..h].chunks_exact_mut(d) {
                let largest = 1.0 + unit();
                let step = largest / 127.0;
                head[0] = largest;
                for (t, x) in head.iter_mut().enumerate().skip(1) {
                    let k = ((t as i32 * 37) % 253) - 126;
                    *x = match split_tie(step, k) {
                        Some(x) => {
                            split += 1;
                            x
                        }
                        None => (k as f32) * step,
                    };
                }
            }
            for key in &mut row[h..2 * h] {
                *key = 2.0 * unit() - 1.0;
            }
            for value in &mut row[2 * h..] {
                *value = half::f16::from_f32(2.0 * unit() - 1.0).to_f32();
            }
        }
        assert!(split > tokens * h / 5, "{split} split ties");
        let mut expected = vec![0.0; tokens * h];
        kodamaembed_cpu::integer_attention(
            &qkv,
            lengths,
            h,
            E5.heads,
            Some(&export.layers[0].key_centers),
            &mut expected,
        );
        let (actual, _) = gpu.crafted_attention(&qkv, lengths).unwrap();
        let differing = expected
            .iter()
            .zip(&actual)
            .filter(|(e, a)| half::f16::from_f32(**e).to_f32().to_bits() != a.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "lengths {lengths:?}: {differing} of {} context values differ",
            expected.len()
        );
    }
}

/// Scores near 2^12, so the subtraction rounds across the table's first entries.
#[test]
fn attention_weighs_large_scores_alike() {
    if Kernel::detect() != Kernel::Avx2
        || !is_x86_feature_detected!("avxvnni")
        || !is_x86_feature_detected!("f16c")
    {
        return;
    }
    let export = calibrated_cpu(5, 512).export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    let h = E5.hidden;
    // Past one key chunk (the wide kernels), then within one (one launch).
    for lengths in [&[65, 33][..], &[64, 33]] {
        let tokens: usize = lengths.iter().sum();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let mut unit = || (random() >> 40) as f32 / (1_u64 << 24) as f32;
        let direction: Vec<f32> = (0..h).map(|_| 2.0 * unit() - 1.0).collect();
        let mut qkv = vec![0.0_f32; tokens * 3 * h];
        for row in qkv.chunks_exact_mut(3 * h) {
            let (queries, rest) = row.split_at_mut(h);
            let (keys, values) = rest.split_at_mut(h);
            for x in queries
                .iter_mut()
                .chain(keys.iter_mut())
                .zip(direction.iter().cycle())
            {
                let (x, &u) = x;
                *x = 48.0 * u + 0.5 * unit() - 0.25;
            }
            for value in values {
                *value = half::f16::from_f32(2.0 * unit() - 1.0).to_f32();
            }
        }
        let mut expected = vec![0.0; tokens * h];
        kodamaembed_cpu::integer_attention(
            &qkv,
            lengths,
            h,
            E5.heads,
            Some(&export.layers[0].key_centers),
            &mut expected,
        );
        let (actual, _) = gpu.crafted_attention(&qkv, lengths).unwrap();
        let differing = expected
            .iter()
            .zip(&actual)
            .filter(|(e, a)| half::f16::from_f32(**e).to_f32().to_bits() != a.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "lengths {lengths:?}: {differing} of {} context values differ",
            expected.len()
        );
    }
}

/// One token's keys near 1e37 overflow their scores: a row meeting them with a large positive dot is NaN.
#[test]
fn attention_loses_rows_whose_scores_overflow_alike() {
    if Kernel::detect() != Kernel::Avx2
        || !is_x86_feature_detected!("avxvnni")
        || !is_x86_feature_detected!("f16c")
    {
        return;
    }
    let export = calibrated_cpu(5, 512).export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    let h = E5.hidden;
    // Past one key chunk (the wide kernels), then within one (one launch).
    for lengths in [&[130, 65][..], &[40, 16]] {
        let tokens: usize = lengths.iter().sum();
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let mut unit = || (random() >> 40) as f32 / (1_u64 << 24) as f32;
        let mut qkv = vec![0.0_f32; tokens * 3 * h];
        let mut start = 0;
        for &len in lengths {
            for (j, row) in qkv[start * 3 * h..(start + len) * 3 * h]
                .chunks_exact_mut(3 * h)
                .enumerate()
            {
                let size = if j == 3 { 1e37 } else { 1.0 };
                for x in &mut row[..h] {
                    *x = 2.0 * unit() - 1.0;
                }
                for key in &mut row[h..2 * h] {
                    *key = size * (2.0 * unit() - 1.0);
                }
                for value in &mut row[2 * h..] {
                    *value = half::f16::from_f32(2.0 * unit() - 1.0).to_f32();
                }
            }
            start += len;
        }
        let mut expected = vec![0.0; tokens * h];
        kodamaembed_cpu::integer_attention(
            &qkv,
            lengths,
            h,
            E5.heads,
            Some(&export.layers[0].key_centers),
            &mut expected,
        );
        let lost = expected.iter().filter(|e| e.is_nan()).count();
        assert!(
            lost > expected.len() / 8 && lost < 7 * expected.len() / 8,
            "lengths {lengths:?}: {lost} of {} context values are NaN",
            expected.len()
        );
        let (actual, _) = gpu.crafted_attention(&qkv, lengths).unwrap();
        // A lost value is NaN on both devices, each with its own NaN's bits.
        let differing = expected
            .iter()
            .zip(&actual)
            .filter(|(e, a)| {
                let e = half::f16::from_f32(**e).to_f32();
                !(e.is_nan() && a.is_nan()) && e.to_bits() != a.to_bits()
            })
            .count();
        assert_eq!(
            differing,
            0,
            "lengths {lengths:?}: {differing} of {} context values differ",
            expected.len()
        );
    }
}

/// A row whose normalization depends on its sum of squares' order.
fn norm_order_sensitive_row(h: usize, random: &mut impl FnMut() -> u64) -> Vec<half::f16> {
    let outputs = |row: &[half::f16], reversed: bool| -> Vec<f32> {
        let squares: Vec<f64> = row.iter().map(|&x| f64::from(x.to_f32()).powi(2)).collect();
        let total = if reversed {
            squares.iter().rev().fold(-0.0, |sum, &s| sum + s)
        } else {
            squares.iter().fold(-0.0, |sum, &s| sum + s)
        };
        let norm = total.sqrt().max(1e-12);
        row.iter()
            .map(|&x| (f64::from(x.to_f32()) / norm) as f32)
            .collect()
    };
    for _ in 0..50_000_000 {
        let row: Vec<half::f16> = (0..h)
            .map(|_| {
                let r = random();
                let sign = if r >> 63 == 0 { 1.0 } else { -1.0 };
                let binade = (r >> 32) % 30;
                half::f16::from_f32(
                    sign * (1.0 + (r >> 40) as f32 / (1_u64 << 24) as f32)
                        * 2_f32.powi(binade as i32 - 14),
                )
            })
            .collect();
        if outputs(&row, false) != outputs(&row, true) {
            return row;
        }
    }
    panic!("no row with an order-sensitive norm in 50 million");
}

/// Columns exact in any order, plus an order-sensitive row for the norm.
#[test]
fn pooling_matches_the_cpu_on_fp16_rows() {
    if Kernel::detect() != Kernel::Avx2
        || !is_x86_feature_detected!("avxvnni")
        || !is_x86_feature_detected!("f16c")
    {
        return;
    }
    let export = calibrated_cpu(5, 512).export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, &export).unwrap();
    let h = E5.hidden;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut random = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let lengths = [130, 64, 7, 1];
    let tokens: usize = lengths.iter().sum();
    let mut hidden = vec![half::f16::ZERO; tokens * h];
    let mut start = 0;
    for &len in &lengths[..3] {
        for column in 0..h {
            for (j, value) in fp16_range_column(len, &mut random).into_iter().enumerate() {
                hidden[(start + j) * h + column] = half::f16::from_f32(value);
            }
        }
        start += len;
    }
    hidden[start * h..].copy_from_slice(&norm_order_sensitive_row(h, &mut random));
    let mut expected = vec![0.0; lengths.len() * h];
    let wide: Vec<f32> = hidden.iter().map(|x| x.to_f32()).collect();
    kodamaembed_cpu::pool_rows(&wide, h, &lengths, &mut expected);
    let actual = gpu.crafted_pool(&hidden, &lengths).unwrap();
    let mut differing: Vec<usize> = expected
        .iter()
        .zip(&actual)
        .enumerate()
        .filter(|(_, (e, a))| e.to_bits() != a.to_bits())
        .map(|(i, _)| i / h)
        .collect();
    differing.dedup();
    assert!(
        differing.is_empty(),
        "values differ in sequences {differing:?}"
    );
}

/// The scalar checks read only its constants.
fn constants_encoder(context: &Context) -> ReproEncoder {
    ReproEncoder::new(context, shared_export()).unwrap()
}

/// A change that leaves an export malformed.
type Corruption = fn(&mut EncoderExport);

/// A malformed export is an error, not a panic or out-of-bounds launches.
#[test]
fn malformed_exports_are_errors() {
    if !centered_cpu() {
        return;
    }
    let context = Context::new(0).unwrap();
    let corruptions: [(&str, Corruption); 15] = [
        ("no heads", |e| e.shape.heads = 0),
        ("an operand of no digit", |e| {
            e.layers[3].up.operand_digits = 0
        }),
        ("an operand of three digits", |e| {
            e.layers[23].qkv.operand_digits = 3
        }),
        ("two digits at an out projection", |e| {
            e.layers[5].out.operand_digits = 2
        }),
        ("two digits at a first QKV projection", |e| {
            e.layers[0].qkv.operand_digits = 2
        }),
        ("another model's widths", |e| {
            e.shape.hidden = 768;
            e.shape.intermediate = 3072;
        }),
        ("MPNet's relative attention bias", |e| {
            e.shape.positions = Positions::Mpnet;
        }),
        ("a layer short", |e| {
            e.layers.pop();
        }),
        ("a word table value short", |e| {
            e.word.pop();
        }),
        ("an up projection digit short", |e| {
            e.layers[0].up.q.pop();
        }),
        ("a down projection scale short", |e| {
            e.layers[1].down.scales.pop();
        }),
        ("an out projection bias short", |e| {
            e.layers[1].out.bias.pop();
        }),
        ("a LayerNorm gain short", |e| {
            e.layers[0].norm1.gamma.pop();
        }),
        ("a NaN epsilon", |e| e.layers[0].norm2.eps = f32::NAN),
        ("a weight at the end of the attention table's span", |e| {
            *e.attention_weights.last_mut().unwrap() = 1;
        }),
    ];
    for (what, corrupt) in corruptions {
        let mut export = shared_export().clone();
        corrupt(&mut export);
        assert!(ReproEncoder::new(&context, &export).is_err(), "{what}");
    }
}

/// Overflowing lengths and oversized batches fail before any storage or launch.
#[test]
fn encodes_past_the_sums_or_the_grids_are_errors() {
    if !centered_cpu() {
        return;
    }
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&context, shared_export()).unwrap();
    let before = gpu.encode(&[1], &[0], &[1]).unwrap();
    assert!(gpu.encode(&[1], &[0], &[usize::MAX, 2]).is_err());
    let mut mixed = vec![1; 65_536];
    mixed[0] = 129;
    for lengths in [vec![64; 65_536], vec![65; 65_536], mixed] {
        let tokens: usize = lengths.iter().sum();
        let error = gpu
            .encode(&vec![1; tokens], &vec![0; tokens], &lengths)
            .unwrap_err();
        assert!(error.to_string().contains("launch grids"), "{error}");
    }
    let after = gpu.encode(&[1], &[0], &[1]).unwrap();
    assert_same_bits(&before, &after, "encode after invalid batch dimensions");
}

/// All 2^32 floats: the GPU's GELU has the CPU's bits; NaN payloads may differ.
#[test]
#[ignore = "sweeps all 2^32 floats on both devices"]
fn gelu_matches_the_cpu_on_every_float() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    const CHUNK: u32 = 1 << 26;
    const PART: usize = 1 << 20;
    let mut x = vec![0.0_f32; CHUNK as usize];
    let mut expected = vec![0.0_f32; CHUNK as usize];
    for chunk in 0..(1_u64 << 32) / u64::from(CHUNK) {
        let first = u32::try_from(chunk * u64::from(CHUNK)).unwrap();
        let actual = gpu.gelus(first, CHUNK as usize).unwrap();
        // An inclusive range: the last chunk ends at u32::MAX.
        for (i, x) in (first..=first + (CHUNK - 1)).zip(&mut x) {
            *x = f32::from_bits(i);
        }
        std::thread::scope(|s| {
            for (x, out) in x.chunks(PART).zip(expected.chunks_mut(PART)) {
                s.spawn(move || kodamaembed_cpu::gelus(x, out));
            }
        });
        let first_differing = actual
            .iter()
            .zip(&expected)
            .position(|(a, e)| a.to_bits() != e.to_bits() && !(a.is_nan() && e.is_nan()));
        assert_eq!(
            first_differing.map(|i| (x[i], actual[i], expected[i])),
            None,
            "input, gpu, cpu"
        );
    }
}

/// All 2^32 floats as exponents: the GPU's table index is the CPU's, ties and specials included.
#[test]
fn attention_weight_indices_match_the_cpu_on_every_float() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    const CHUNK: u32 = 1 << 26;
    const PART: usize = 1 << 20;
    let mut x = vec![0.0_f32; CHUNK as usize];
    let mut expected = vec![0_u16; CHUNK as usize];
    for chunk in 0..(1_u64 << 32) / u64::from(CHUNK) {
        let first = u32::try_from(chunk * u64::from(CHUNK)).unwrap();
        let actual = gpu.weight_indices(first, CHUNK as usize).unwrap();
        // An inclusive range: the last chunk ends at u32::MAX.
        for (i, x) in (first..=first + (CHUNK - 1)).zip(&mut x) {
            *x = f32::from_bits(i);
        }
        std::thread::scope(|s| {
            for (x, out) in x.chunks(PART).zip(expected.chunks_mut(PART)) {
                s.spawn(move || kodamaembed_cpu::weight_indices(x, out));
            }
        });
        let first_differing = actual.iter().zip(&expected).position(|(a, e)| a != e);
        assert_eq!(
            first_differing.map(|i| (x[i], actual[i], expected[i])),
            None,
            "exponent, gpu, cpu"
        );
    }
}

#[test]
fn reciprocal_matches_the_division_on_every_normal_float_below_2_126() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    let (first, last) = (f32::MIN_POSITIVE.to_bits(), 2.0_f32.powi(126).to_bits() - 1);
    let mismatches = gpu
        .reciprocal_check(first, (last - first + 1) as usize)
        .unwrap();
    assert_eq!(mismatches, 0, "reciprocals differing");
}

/// encoder.rs lane_sum's order.
fn lane_order_square_sum(values: &[f64]) -> f64 {
    let mut lanes = [0.0_f64; 32];
    for (c, &m) in values.iter().enumerate() {
        lanes[c % 32] += m * m;
    }
    let mut width = 16;
    while width > 0 {
        for j in 0..width {
            lanes[j] += lanes[j + width];
        }
        width /= 2;
    }
    lanes[0]
}

#[test]
fn pooled_square_sums_take_the_cpu_lane_order() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    let h = 1024;
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    let values: Vec<f64> = (0..16 * h)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let unit = (x >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            unit * 2_f64.powi((x % 41) as i32 - 20)
        })
        .collect();
    let expected: Vec<f64> = values.chunks_exact(h).map(lane_order_square_sum).collect();
    let actual = gpu.square_sum_check(&values, h).unwrap();
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(e.to_bits(), a.to_bits(), "vector {i}: cpu {e:e} gpu {a:e}");
    }
    assert!(
        values.chunks_exact(h).zip(&expected).any(|(v, e)| v
            .iter()
            .map(|m| m * m)
            .sum::<f64>()
            .to_bits()
            != e.to_bits()),
        "the sequential order rounds every vector alike: no order would show"
    );
}

/// Means against `sum / len as f64`, with totals whose remainder decides the rounding.
#[test]
fn pool_means_take_the_cpu_quotients() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    let top = (1_i64 << 49) - 1;
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let (mut units, mut lens) = (Vec::new(), Vec::new());
    for len in 1..=512 {
        let mut totals = vec![
            0,
            1,
            -1,
            top,
            -top,
            i64::from(len),
            -12_345 * i64::from(len),
        ];
        for _ in 0..256 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let magnitude = ((x >> 8) & ((1 << (x % 49 + 1)) - 1)) as i64;
            totals.push(if x & 1 == 0 { magnitude } else { -magnitude });
        }
        lens.extend(std::iter::repeat_n(len, totals.len()));
        units.extend(totals);
    }
    let actual = gpu.pool_mean_check(&units, &lens).unwrap();
    for (i, (&u, &len)) in units.iter().zip(&lens).enumerate() {
        let expected = u as f64 * 2_f64.powi(-24) / f64::from(len);
        assert_eq!(
            expected.to_bits(),
            actual[i].to_bits(),
            "units {u} over {len}: cpu {expected:e} gpu {:e}",
            actual[i]
        );
    }
    let decided = units
        .iter()
        .zip(&lens)
        .filter(|&(&u, &len)| {
            if u == 0 {
                return false;
            }
            let (magnitude, rows) = (u.unsigned_abs(), u64::from(len.unsigned_abs()));
            let shifted = magnitude << magnitude.leading_zeros();
            let (q, r) = (shifted / rows, shifted % rows);
            r != 0 && (q as f64).to_bits() != ((q | 1) as f64).to_bits()
        })
        .count();
    assert!(
        decided >= 1000,
        "only {decided} totals whose rounding the remainder decides"
    );
}

/// Quotients against `(m / norm) as f32`, with midpoints the reciprocal alone misrounds.
#[test]
fn pool_quotients_take_the_cpu_divisions() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    let mut unit = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1_u64 << 53) as f64
    };
    let (mut means, mut norms) = (Vec::new(), Vec::new());
    for i in 0..200_000 {
        let norm = 1e-12 * 2_f64.powf(unit() * 61.0);
        let mean = if i % 2 == 0 {
            (2.0 * unit() - 1.0) * norm
        } else {
            let single = ((0.5 + 0.5 * unit()) as f32).to_bits();
            let midpoint =
                (f64::from(f32::from_bits(single)) + f64::from(f32::from_bits(single + 1))) / 2.0;
            let sign = if unit() < 0.5 { -1.0 } else { 1.0 };
            sign * midpoint * norm
        };
        means.push(mean);
        norms.push(norm);
    }
    for (mean, norm) in [
        (0.0, 1.0),
        (-0.0, 3.0),
        (1.5, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY),
    ] {
        means.push(mean);
        norms.push(norm);
    }
    means.push(f64::NAN);
    norms.push(2.0);
    let actual = gpu.pool_quotient_check(&means, &norms).unwrap();
    for (i, ((&m, &n), &a)) in means.iter().zip(&norms).zip(&actual).enumerate() {
        let expected = (m / n) as f32;
        assert!(
            expected.to_bits() == a.to_bits() || (expected.is_nan() && a.is_nan()),
            "pair {i}: {m:e} over {n:e}: cpu {expected:e} gpu {a:e}"
        );
    }
    let decided = means
        .iter()
        .zip(&norms)
        .filter(|&(&m, &n)| (m * (1.0 / n)) as f32 != (m / n) as f32)
        .count();
    assert!(
        decided >= 1000,
        "only {decided} quotients the reciprocal alone rounds otherwise"
    );
}

/// Every finite FP16 value: integer-pipe units equal the product by 2^24.
#[test]
fn pooling_units_match_the_product_on_every_fp16_value() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    assert_eq!(gpu.units_check(0, 1 << 16).unwrap(), 0);
}

/// Every nonnegative float: the certified quotient's steps equal scale_for's.
#[test]
fn query_and_key_steps_match_the_division_on_every_float() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    for first in [0_u32, 1 << 30] {
        let mismatches = gpu.scale_check(first, 1 << 30).unwrap();
        assert_eq!(mismatches, 0, "steps differing from {first:#x}");
    }
}

/// All 2^43 pairs a in [1, 2), b in [2^20, 2^21); power-of-two scaling covers smaller b.
#[test]
#[ignore = "2^43 divisions; run with --include-ignored"]
fn quotient_matches_the_division_on_every_mantissa_pair() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let context = Context::new(0).unwrap();
    let gpu = constants_encoder(&context);
    const CHUNK: i32 = 1024;
    let mismatches: u64 = (0..(1 << 20) / CHUNK)
        .map(|chunk| {
            gpu.quotient_check((1 << 20) + chunk * CHUNK, CHUNK as usize)
                .unwrap()
        })
        .sum();
    assert_eq!(mismatches, 0, "quotients differing");
}

#[test]
fn reads_ahead_ask_for_every_sector_of_a_layers_digits_once() {
    if !centered_cpu() {
        return;
    }
    let export = shared_export();
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&with_l2(&context, u32::MAX), export).unwrap();
    let (ids, types) = sequence_input(64, 0);
    gpu.encode(&ids, &types, &[64]).unwrap();
    // Prefetch wraps at the last layer, so the final checksum is for layer zero.
    let first = &export.layers[0];
    let expected = [&first.qkv, &first.out, &first.up, &first.down]
        .iter()
        .flat_map(|projection| projection.q.iter().step_by(32))
        .fold(0u32, |sum, &digit| sum.wrapping_add(u32::from(digit as u8)));
    let asked = gpu
        .read_ahead_sums()
        .unwrap()
        .iter()
        .fold(0u32, |sum, &thread| sum.wrapping_add(thread));
    assert_eq!(asked, expected);
}

#[test]
fn an_l2_too_small_for_two_layers_reads_nothing_ahead() {
    if !centered_cpu() {
        return;
    }
    let export = shared_export();
    let context = Context::new(0).unwrap();
    let first = &export.layers[0];
    let layer: usize = [&first.qkv, &first.out, &first.up, &first.down]
        .iter()
        .map(|projection| projection.q.len())
        .sum();
    let short = u32::try_from(2 * layer - 1).unwrap();
    let mut gpu = ReproEncoder::new(&with_l2(&context, short), export).unwrap();
    let (ids, types) = sequence_input(64, 0);
    gpu.encode(&ids, &types, &[64]).unwrap();
    assert!(gpu.read_ahead_sums().unwrap().iter().all(|&sum| sum == 0));
}

#[test]
fn a_batch_that_reads_ahead_records_graphs_of_one_root() {
    if !centered_cpu() {
        return;
    }
    let context = Context::new(0).unwrap();
    let mut gpu = ReproEncoder::new(&with_l2(&context, u32::MAX), shared_export()).unwrap();
    let (ids, types) = sequence_input(64, 0);
    gpu.encode(&ids, &types, &[64]).unwrap();
    // A second graph root delays the prefetch that should overlap the layer's first kernel.
    assert_eq!(gpu.graph_roots(), [1; 5]);
}

#[test]
fn e5_reproducible_encoder_matches_the_cpu_bit_for_bit() {
    same_bits_as_cpu(103, true);
}

#[test]
fn e5_layer_launches_match_the_cpu_bit_for_bit() {
    same_bits_as_cpu(109, false);
}
