use kodamaembed_cpu::{CpuPrecision, Encoder, EncoderError, Kernel};
use kodamaembed_model::{Head, Output, Pooling, Positions, Shape, Tensors, reference::Synthetic};

struct NoReads;
impl Tensors for NoReads {
    fn get(&self, _: &str, _: usize) -> Result<Vec<f32>, String> {
        panic!("unsupported preset must reject before reading weights")
    }
}

#[test]
fn existing_constructor_and_explicit_fp32_have_identical_bits() {
    let shape = Shape {
        layers: 1,
        hidden: 24,
        heads: 3,
        intermediate: 96,
        vocab: 32,
        max_positions: 32,
        type_vocab: 2,
        eps: 1e-12,
        positions: Positions::Sequential,
    };
    let weights = Synthetic::new(&shape, 71);
    let mut original = Encoder::new(
        Kernel::Portable,
        shape,
        Output::Embedding(Pooling::Mean),
        32,
        &weights,
    )
    .unwrap();
    let mut explicit = Encoder::new_with_precision(
        Kernel::Portable,
        shape,
        Output::Embedding(Pooling::Mean),
        32,
        &weights,
        CpuPrecision::Fp32,
    )
    .unwrap();
    assert_eq!(original.precision(), CpuPrecision::Fp32);
    assert_eq!(explicit.numerical_identity(), original.numerical_identity());
    let ids = [1, 2, 3, 4, 5, 6];
    let a = original.encode(&ids, &[0; 6], &[2, 4]).unwrap();
    let b = explicit.encode(&ids, &[0; 6], &[2, 4]).unwrap();
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn e5_preset_rejects_mismatched_shapes_before_weight_reads() {
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
    for other in [
        Shape {
            layers: 23,
            ..shape
        },
        Shape {
            hidden: 512,
            ..shape
        },
        Shape { heads: 8, ..shape },
        Shape {
            intermediate: 2048,
            ..shape
        },
        Shape {
            vocab: 30_521,
            ..shape
        },
        Shape {
            max_positions: 256,
            ..shape
        },
        Shape {
            type_vocab: 1,
            ..shape
        },
        Shape { eps: 1e-5, ..shape },
        Shape {
            positions: Positions::Roberta { pad: 1 },
            ..shape
        },
        Shape {
            positions: Positions::Mpnet,
            ..shape
        },
    ] {
        let error = Encoder::new_with_precision(
            Kernel::detect(),
            other,
            Output::Embedding(Pooling::Mean),
            512,
            &NoReads,
            CpuPrecision::E5CenteredInt8,
        )
        .err()
        .unwrap();
        assert!(matches!(error, EncoderError::Unsupported(reason) if reason.contains("evaluated")));
    }
    for (kernel, output) in [
        (Kernel::Portable, Output::Embedding(Pooling::Mean)),
        (Kernel::detect(), Output::Relevance(Head::Pooler)),
    ] {
        assert!(matches!(
            Encoder::new_with_precision(
                kernel,
                shape,
                output,
                512,
                &NoReads,
                CpuPrecision::E5CenteredInt8,
            ),
            Err(EncoderError::Unsupported(_))
        ));
    }
}
