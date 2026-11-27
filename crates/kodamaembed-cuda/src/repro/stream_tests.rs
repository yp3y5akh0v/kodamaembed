use super::{Context, ReproEncoder, STREAM_BYTES, STREAM_TILE_ROWS};
use kodamaembed_cpu::{CpuPrecision, Encoder, Kernel};
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};

fn equal(actual: &[f32], expected: &[f32], name: &str, lengths: &[usize], pass: usize) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}, pass {pass}, {lengths:?}"
    );
    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differing, 0, "{name}, pass {pass}, {lengths:?}");
}

#[test]
#[ignore = "needs a GPU of 40 or more multiprocessors and centered INT8 host support"]
fn e5_stream_engine_matches_cpu_and_reuse() {
    assert_eq!(Kernel::detect(), Kernel::Avx2);
    assert!(is_x86_feature_detected!("avxvnni"));
    assert!(is_x86_feature_detected!("f16c"));
    let shape = Shape {
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
    let weights = Synthetic::new(&shape, 183);
    let mut cpu = Encoder::new_with_precision(
        Kernel::Avx2,
        shape,
        Output::Embedding(Pooling::Mean),
        1024,
        &weights,
        CpuPrecision::E5CenteredInt8,
    )
    .unwrap();
    cpu.calibrate(&[
        (vec![101, 102], vec![0; 2]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
    ])
    .unwrap();
    let export = cpu.export_reproducible().unwrap();
    let context = Context::new(0).unwrap();
    let default = ReproEncoder::new(&context, &export).unwrap();
    let engine = default
        .stream_engine
        .as_ref()
        .unwrap_or_else(|| panic!("{} multiprocessors", context.sm_count));
    assert_eq!(engine.moments.len(), shape.layers * 2 * STREAM_TILE_ROWS);
    {
        // A plan past the shared memory takes the per-layer launches.
        let mut limited_context = context.clone();
        limited_context.max_shared_per_block = context.max_shared_per_block.min(STREAM_BYTES - 1);
        let mut limited = ReproEncoder::new(&limited_context, &export).unwrap();
        assert!(limited.stream_engine.is_none());
        for rows in [1, STREAM_TILE_ROWS] {
            assert!(limited.stream_engine_for(&[rows]).is_none());
            let ids: Vec<i32> = (0..rows).map(|i| 101 + 17 * i as i32).collect();
            let types: Vec<i32> = (0..rows).map(|i| (i % 2) as i32).collect();
            let expected = cpu.encode(&ids, &types, &[rows]).unwrap();
            equal(
                &limited.encode(&ids, &types, &[rows]).unwrap(),
                &expected,
                "logical shared-memory cap",
                &[rows],
                0,
            );
        }
    }
    let mut per_layer = ReproEncoder::new(&context, &export).unwrap();
    per_layer.set_aside_stream_engine();
    let mut models = [per_layer, default];
    let names = ["per-layer", "default"];
    let cases = [
        vec![8, 8],
        vec![3, 5, 4, 4],
        vec![1, 15],
        vec![2; 8],
        vec![17],
        vec![8, 9],
        vec![24],
        vec![25],
        vec![31],
        vec![32],
        vec![15, 16],
        vec![16, 1],
        vec![1, 16],
        vec![17, 14],
        vec![15, 17],
        vec![3, 13, 16],
        vec![1; 32],
        vec![33],
        vec![16],
        vec![64, 64],
        vec![65, 17],
        vec![32],
    ];
    for pass in 0..3 {
        for lengths in &cases {
            let tokens: usize = lengths.iter().sum();
            // Up to four sequences on one row tile take the stream.
            let streamed = tokens <= STREAM_TILE_ROWS && lengths.len() <= 4;
            assert_eq!(
                models[1].stream_engine_for(lengths).is_some(),
                streamed,
                "default route for {lengths:?}"
            );
            assert!(models[0].stream_engine_for(lengths).is_none());
            let ids: Vec<i32> = (0..tokens)
                .map(|i| ((i * 7919 + pass * 97 + 183) % shape.vocab) as i32)
                .collect();
            let types: Vec<i32> = (0..tokens).map(|i| ((i + pass) % 2) as i32).collect();
            let expected = cpu.encode(&ids, &types, lengths).unwrap();
            for (model, name) in models.iter_mut().zip(names) {
                equal(
                    &model.encode(&ids, &types, lengths).unwrap(),
                    &expected,
                    name,
                    lengths,
                    pass,
                );
            }
        }
    }
    // A shape the stream takes: its graph is captured, then set aside.
    let lengths = [7, 9];
    let ids: Vec<i32> = (0..16).map(|i| 101 + 17 * i).collect();
    let types: Vec<i32> = (0..16).map(|i| i % 2).collect();
    let expected = cpu.encode(&ids, &types, &lengths).unwrap();
    for (model, name) in models.iter_mut().zip(names) {
        assert!(model.encode(&[-1], &[0], &[1]).is_err());
        assert!(model.encode(&[101; 513], &[0; 513], &[513]).is_err());
        equal(
            &model.encode(&ids, &types, &lengths).unwrap(),
            &expected,
            name,
            &lengths,
            0,
        );
    }
    let model = &mut models[1];
    equal(
        &model.encode(&ids, &types, &lengths).unwrap(),
        &expected,
        "captured",
        &lengths,
        0,
    );
    assert!(!model.graphs.is_empty());
    model.set_aside_stream_engine();
    assert!(model.stream_engine.is_none());
    assert!(model.graphs.is_empty());
    for pass in 0..3 {
        equal(
            &model.encode(&ids, &types, &lengths).unwrap(),
            &expected,
            "set aside",
            &lengths,
            pass,
        );
    }
}
