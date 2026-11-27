use super::*;
use crate::test_support::{
    safetensors, tiny_bert_config, tiny_modernbert_checkpoint, tiny_modernbert_config,
    tiny_modernbert_cross_encoder,
};
use kodamaembed_model::{Activation, Head, Pooling, Rotary};

fn raw_checkpoint(header: serde_json::Value, data: &[u8]) -> SafeTensors<'static> {
    SafeTensors::from_bytes(safetensors(&header, data).into()).unwrap()
}

fn checkpoint(tensors: &[(&str, &[f32])]) -> SafeTensors<'static> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, values) in tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": [values.len()],
                "data_offsets": [start, data.len()],
            }),
        );
    }
    raw_checkpoint(header.into(), &data)
}

fn shape_of(config: &serde_json::Value) -> Result<Shape, String> {
    shape_from_bytes(&serde_json::to_vec(config).unwrap())
}

#[test]
fn bert_and_xlm_roberta_load_with_their_position_rules() {
    let mut config = tiny_bert_config(4);
    assert_eq!(shape_of(&config).unwrap().positions, Positions::Sequential);
    config["model_type"] = "xlm-roberta".into();
    assert!(shape_of(&config).is_err());
    config["pad_token_id"] = 1.into();
    assert_eq!(
        shape_of(&config).unwrap().positions,
        Positions::Roberta { pad: 1 }
    );
    config["model_type"] = "roberta".into();
    assert!(shape_of(&config).is_err());
}

#[test]
fn another_family_is_named_whatever_fields_its_config_has() {
    let config = serde_json::json!({"model_type": "distilbert", "dim": 768});
    assert_eq!(
        shape_of(&config).unwrap_err(),
        "a distilbert checkpoint, where KodamaEmbed runs bert, xlm-roberta, mpnet and modernbert"
    );
}

#[test]
fn modernbert_configs_load_with_their_rotary_positions() {
    let mut config = tiny_modernbert_config(4);
    // Two layers, every third from the first over the whole sequence: the second has a window.
    let rotary = Rotary {
        global_theta: 160_000.0,
        local_theta: 10_000.0,
        local_layers: 0b10,
        window: 4,
    };
    let shape = shape_of(&config).unwrap();
    assert_eq!(shape.positions, Positions::ModernBert(rotary));
    assert_eq!(
        (shape.layers, shape.hidden, shape.heads, shape.intermediate),
        (2, 8, 2, 6)
    );
    assert_eq!(
        (
            shape.vocab,
            shape.max_positions,
            shape.type_vocab,
            shape.eps
        ),
        (4, 16, 1, 1e-5)
    );
    config["global_rope_theta"] = 80_000.into();
    config["local_rope_theta"] = 40_000.into();
    assert_eq!(bases(&config), (80_000.0, 40_000.0));
    // What a file leaves out is transformers' default.
    let fields = config.as_object_mut().unwrap();
    for field in [
        "global_rope_theta",
        "local_rope_theta",
        "global_attn_every_n_layers",
    ] {
        fields.remove(field);
    }
    let Positions::ModernBert(defaults) = shape_of(&config).unwrap().positions else {
        panic!("a ModernBERT shape");
    };
    assert_eq!(defaults, rotary);
}

fn bases(config: &serde_json::Value) -> (f32, f32) {
    let Positions::ModernBert(rotary) = shape_of(config).unwrap().positions else {
        panic!("a ModernBERT shape");
    };
    (rotary.global_theta, rotary.local_theta)
}

/// As transformers 5 writes them: each layer's attention, and a rotation for each kind.
fn listed_modernbert_config() -> serde_json::Value {
    let mut config = tiny_modernbert_config(4);
    let fields = config.as_object_mut().unwrap();
    fields.remove("global_rope_theta");
    fields.remove("local_rope_theta");
    config["num_hidden_layers"] = 4.into();
    config["layer_types"] = serde_json::json!([
        "sliding_attention",
        "full_attention",
        "full_attention",
        "sliding_attention"
    ]);
    config["rope_parameters"] = serde_json::json!({
        "full_attention": {"rope_theta": 80000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 20000.0, "rope_type": "default"},
    });
    config
}

#[test]
fn modernbert_configs_that_list_their_layers_load_as_listed() {
    let mut config = listed_modernbert_config();
    let listed = Rotary {
        global_theta: 80_000.0,
        local_theta: 20_000.0,
        local_layers: 0b1001,
        window: 4,
    };
    // The list stands where the older period, which such files may keep, says otherwise.
    assert_eq!(config["global_attn_every_n_layers"], 3);
    assert_eq!(
        shape_of(&config).unwrap().positions,
        Positions::ModernBert(listed)
    );
    // A kind without a base of its own takes the older field's, then transformers' default.
    config["rope_parameters"]["sliding_attention"] = serde_json::json!({"rope_type": "default"});
    assert_eq!(bases(&config), (80_000.0, 10_000.0));
    config["local_rope_theta"] = 30_000.into();
    assert_eq!(bases(&config), (80_000.0, 30_000.0));
    config["rope_parameters"] = serde_json::Value::Null;
    assert_eq!(bases(&config), (160_000.0, 30_000.0));
}

#[test]
fn modernbert_layer_lists_and_rotations_kodamaembed_does_not_run_are_named() {
    let reason = |change: &dyn Fn(&mut serde_json::Value)| {
        let mut config = listed_modernbert_config();
        change(&mut config);
        shape_of(&config).unwrap_err()
    };
    let short = reason(&|config| {
        config["layer_types"].as_array_mut().unwrap().pop();
    });
    assert_eq!(short, "layer_types names 3 layers of 4");
    let unknown = reason(&|config| config["layer_types"][1] = "causal_attention".into());
    assert_eq!(unknown, "layer 1 with causal_attention");
    for kind in ["full_attention", "sliding_attention"] {
        let scaled = reason(&|config| config["rope_parameters"][kind]["rope_type"] = "yarn".into());
        assert_eq!(scaled, format!("{kind} with yarn rotary positions"));
    }
    let scaling = reason(&|config| {
        config["rope_scaling"] = serde_json::json!({"rope_type": "linear", "factor": 2.0});
    });
    assert_eq!(
        scaling,
        "rope_scaling, where KodamaEmbed runs unscaled rotary positions"
    );
    let mut periodic = tiny_modernbert_config(4);
    periodic["global_attn_every_n_layers"] = 0.into();
    assert_eq!(
        shape_of(&periodic).unwrap_err(),
        "global_attn_every_n_layers must be positive"
    );
}

#[test]
fn modernbert_configs_with_biases_or_another_activation_are_rejected() {
    for flag in ["attention_bias", "mlp_bias", "norm_bias"] {
        let mut config = tiny_modernbert_config(4);
        config[flag] = false.into();
        assert!(shape_of(&config).is_ok());
        config[flag] = true.into();
        let reason = shape_of(&config).unwrap_err();
        assert!(reason.contains(flag), "{reason}");
    }
    let mut config = tiny_modernbert_config(4);
    config["hidden_activation"] = "silu".into();
    assert_eq!(shape_of(&config).unwrap_err(), "activation silu");
    config.as_object_mut().unwrap().remove("local_attention");
    assert!(shape_of(&config).unwrap_err().contains("local_attention"));
}

#[test]
fn modernbert_checkpoints_are_read_bare_and_under_model() {
    let config = tiny_modernbert_config(4);
    let shape = shape_of(&config).unwrap();
    let output = Output::Embedding(Pooling::Mean);
    for prefix in ["", "model."] {
        let weights =
            SafeTensors::from_bytes(tiny_modernbert_checkpoint(&config, prefix).into()).unwrap();
        weights.validate_shapes(shape, output).unwrap();
        assert_eq!(weights.get("final_norm.weight", 8).unwrap(), [1.0; 8]);
        assert_eq!(weights.get("layers.1.mlp.Wi.weight", 96).unwrap().len(), 96);
        let head = Head::Prediction(Pooling::Cls, Activation::Gelu);
        assert_eq!(
            weights
                .validate_shapes(shape, Output::Relevance(head))
                .unwrap_err(),
            "head.dense.weight missing"
        );
    }
}

fn head_of(config: &serde_json::Value) -> Result<Head, String> {
    head_from_bytes(&serde_json::to_vec(config).unwrap())
}

#[test]
fn a_modernbert_head_is_as_its_config_names_it_or_as_transformers_defaults() {
    let mut config = tiny_modernbert_config(4);
    assert_eq!(
        head_of(&config).unwrap(),
        Head::Prediction(Pooling::Cls, Activation::Gelu)
    );
    config["classifier_pooling"] = "mean".into();
    config["classifier_activation"] = "silu".into();
    assert_eq!(
        head_of(&config).unwrap(),
        Head::Prediction(Pooling::Mean, Activation::Silu)
    );
    assert_eq!(head_of(&tiny_bert_config(4)).unwrap(), Head::Pooler);
}

#[test]
fn modernbert_heads_kodamaembed_does_not_run_are_named() {
    for (field, value, named) in [
        ("classifier_pooling", "max".into(), "classifier pooling max"),
        (
            "classifier_activation",
            "relu".into(),
            "classifier activation relu",
        ),
        ("classifier_bias", true.into(), "classifier_bias"),
    ] {
        let mut config = tiny_modernbert_config(4);
        config[field] = value;
        let reason = head_of(&config).unwrap_err();
        assert!(reason.contains(named), "{reason}");
    }
}

#[test]
fn a_modernbert_cross_encoder_keeps_its_encoder_under_model_and_its_head_at_the_root() {
    let config = tiny_modernbert_config(4);
    let shape = shape_of(&config).unwrap();
    let weights = SafeTensors::from_bytes(tiny_modernbert_cross_encoder(&config).into()).unwrap();
    let head = Head::Prediction(Pooling::Mean, Activation::Gelu);
    weights
        .validate_shapes(shape, Output::Relevance(head))
        .unwrap();
    assert_eq!(weights.get("final_norm.weight", 8).unwrap(), [1.0; 8]);
    assert_eq!(weights.get("head.norm.weight", 8).unwrap(), [1.0; 8]);
    assert_eq!(weights.get("classifier.bias", 1).unwrap().len(), 1);
    assert_eq!(
        weights
            .validate_shapes(shape, Output::Relevance(Head::Pooler))
            .unwrap_err(),
        "model.pooler.dense.weight missing"
    );
}

#[test]
fn modernbert_checkpoints_and_configs_only_load_together() {
    let modern = tiny_modernbert_config(4);
    let output = Output::Embedding(Pooling::Mean);
    let weights = SafeTensors::from_bytes(tiny_modernbert_checkpoint(&modern, "").into()).unwrap();
    assert_eq!(
        weights
            .validate_shapes(shape_of(&tiny_bert_config(4)).unwrap(), output)
            .unwrap_err(),
        "embeddings.word_embeddings.weight missing"
    );
    let mut deeper = modern.clone();
    deeper["num_hidden_layers"] = 3.into();
    assert_eq!(
        weights
            .validate_shapes(shape_of(&deeper).unwrap(), output)
            .unwrap_err(),
        "layers.2.attn_norm.weight missing"
    );
    let mut wider = modern;
    wider["intermediate_size"] = 8.into();
    assert_eq!(
        weights
            .validate_shapes(shape_of(&wider).unwrap(), output)
            .unwrap_err(),
        "layers.0.mlp.Wi.weight has shape [12, 8], expected [16, 8]"
    );
}

#[test]
fn offsets_and_shapes_past_the_address_space_are_errors() {
    let weights = raw_checkpoint(
        serde_json::json!({
            "far": {"dtype": "F32", "shape": [2], "data_offsets": [usize::MAX, usize::MAX]},
            "huge": {"dtype": "F32", "shape": [1u64 << 32, 1u64 << 32], "data_offsets": [0, 8]},
        }),
        &[0; 8],
    );
    assert!(weights.get("far", 2).is_err());
    assert!(weights.get("huge", 2).is_err());
}

#[test]
fn classification_checkpoints_nest_the_backbone_under_bert() {
    let weights = checkpoint(&[
        ("bert.embeddings.word_embeddings.weight", &[1.0, 2.0]),
        ("bert.pooler.dense.bias", &[3.0]),
        ("classifier.bias", &[4.0]),
    ]);
    assert_eq!(
        weights.get("embeddings.word_embeddings.weight", 2).unwrap(),
        [1.0, 2.0]
    );
    assert_eq!(weights.get("pooler.dense.bias", 1).unwrap(), [3.0]);
    assert_eq!(weights.get("classifier.bias", 1).unwrap(), [4.0]);
}

#[test]
fn xlm_roberta_classifiers_serve_their_head_as_the_pooler_and_classifier() {
    let weights = checkpoint(&[
        ("roberta.embeddings.word_embeddings.weight", &[1.0, 2.0]),
        ("classifier.dense.bias", &[3.0]),
        ("classifier.out_proj.bias", &[4.0]),
    ]);
    assert_eq!(
        weights.get("embeddings.word_embeddings.weight", 2).unwrap(),
        [1.0, 2.0]
    );
    assert_eq!(weights.get("pooler.dense.bias", 1).unwrap(), [3.0]);
    assert_eq!(weights.get("classifier.bias", 1).unwrap(), [4.0]);
}

#[test]
fn plain_checkpoints_are_read_by_their_own_names() {
    let weights = checkpoint(&[
        ("embeddings.word_embeddings.weight", &[1.0, 2.0]),
        ("pooler.dense.bias", &[3.0]),
    ]);
    assert_eq!(
        weights.get("embeddings.word_embeddings.weight", 2).unwrap(),
        [1.0, 2.0]
    );
    assert_eq!(weights.get("pooler.dense.bias", 1).unwrap(), [3.0]);
    assert!(weights.get("classifier.bias", 1).is_err());
}

#[test]
fn f16_tensors_widen_exactly_and_other_types_are_named_in_the_error() {
    // 1.0, -2.5, the smallest subnormal and the largest finite half.
    let halves: [u16; 4] = [0x3c00, 0xc100, 0x0001, 0x7bff];
    let data: Vec<u8> = halves.iter().flat_map(|half| half.to_le_bytes()).collect();
    let weights = raw_checkpoint(
        serde_json::json!({
            "half": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
            "double": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]},
            "short": {"dtype": "F16", "shape": [4], "data_offsets": [0, 6]},
        }),
        &data,
    );
    assert_eq!(
        weights.get("half", 4).unwrap(),
        [1.0, -2.5, 2f32.powi(-24), 65_504.0]
    );
    let reason = weights.get("double", 1).unwrap_err();
    assert!(reason.contains("F64"), "{reason}");
    assert!(weights.get("short", 4).is_err());
}

#[test]
fn bf16_tensors_widen_exactly() {
    // 1.0, -2.5, the smallest subnormal and the largest finite value: FP32's upper halves.
    let brains: [u16; 4] = [0x3f80, 0xc020, 0x0001, 0x7f7f];
    let data: Vec<u8> = brains.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    let weights = raw_checkpoint(
        serde_json::json!({
            "brain": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
            "short": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 6]},
        }),
        &data,
    );
    assert_eq!(
        weights.get("brain", 4).unwrap(),
        brains.map(|bits| f32::from_bits(u32::from(bits) << 16))
    );
    assert_eq!(weights.get("brain", 4).unwrap()[..2], [1.0, -2.5]);
    assert!(weights.get("short", 4).is_err());
}

#[test]
fn a_tensor_of_the_wrong_length_is_rejected() {
    let weights = checkpoint(&[("embeddings.word_embeddings.weight", &[1.0, 2.0])]);
    assert!(weights.get("embeddings.word_embeddings.weight", 3).is_err());
}

#[test]
fn transposed_checkpoint_matrices_are_rejected_even_with_the_same_count() {
    let weights = raw_checkpoint(
        serde_json::json!({
            "intermediate.dense.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        }),
        &[0; 24],
    );
    assert!(weights.get("intermediate.dense.weight", 6).is_ok());
    assert!(
        weights
            .check_shape("intermediate.dense.weight", &[3, 2])
            .is_err()
    );
}

#[test]
fn decoder_configs_are_rejected_before_loading_weights() {
    for (model_type, pad) in [("bert", 0), ("xlm-roberta", 1), ("mpnet", 1)] {
        for flag in ["is_decoder", "add_cross_attention"] {
            let mut config = tiny_bert_config(4);
            config["model_type"] = model_type.into();
            config["pad_token_id"] = pad.into();
            config[flag] = true.into();
            assert_eq!(
                shape_of(&config).unwrap_err(),
                "decoder and cross-attention configurations are unsupported"
            );
        }
    }
}

#[test]
fn mpnet_configs_load_with_mpnet_positions_and_one_segment() {
    let mut config = tiny_bert_config(4);
    config["model_type"] = "mpnet".into();
    config.as_object_mut().unwrap().remove("type_vocab_size");
    let shape = shape_of(&config).unwrap();
    assert_eq!((shape.positions, shape.type_vocab), (Positions::Mpnet, 1));
    config["relative_attention_num_buckets"] = 64.into();
    assert!(shape_of(&config).is_err());
    config["relative_attention_num_buckets"] = 32.into();
    assert!(shape_of(&config).is_ok());
    let mut bert = tiny_bert_config(4);
    bert.as_object_mut().unwrap().remove("type_vocab_size");
    assert!(shape_of(&bert).is_err());
}

#[test]
fn mpnet_configs_and_checkpoints_only_load_together() {
    let mut config = tiny_bert_config(4);
    let bert = shape_of(&config).unwrap();
    config["model_type"] = "mpnet".into();
    let mpnet = shape_of(&config).unwrap();
    let output = Output::Embedding(Pooling::Mean);
    let table = |shape: [usize; 2]| {
        raw_checkpoint(
            serde_json::json!({
                "encoder.relative_attention_bias.weight":
                    {"dtype": "F32", "shape": shape, "data_offsets": [0, 256]},
            }),
            &[0; 256],
        )
    };
    assert_eq!(
        table([32, 2]).validate_shapes(bert, output).unwrap_err(),
        "an MPNet checkpoint with another model's config.json"
    );
    assert_eq!(
        table([2, 32]).validate_shapes(mpnet, output).unwrap_err(),
        "encoder.relative_attention_bias.weight has shape [2, 32], expected [32, 2]"
    );
    assert_eq!(
        checkpoint(&[]).validate_shapes(mpnet, output).unwrap_err(),
        "encoder.relative_attention_bias.weight missing"
    );
}

#[test]
fn mpnet_checkpoints_serve_bert_names_bare_and_nested() {
    for prefix in ["", "mpnet."] {
        let at = |name: &str| format!("{prefix}{name}");
        let weights = checkpoint(&[
            (&at("embeddings.word_embeddings.weight"), &[1.0, 2.0]),
            (&at("encoder.relative_attention_bias.weight"), &[3.0]),
            (&at("encoder.layer.0.attention.attn.q.weight"), &[4.0]),
            (&at("encoder.layer.0.attention.attn.k.weight"), &[5.0]),
            (&at("encoder.layer.0.attention.attn.v.weight"), &[6.0]),
            (&at("encoder.layer.0.attention.attn.o.bias"), &[7.0]),
            (&at("encoder.layer.0.attention.LayerNorm.weight"), &[8.0]),
            (&at("encoder.layer.0.output.dense.bias"), &[9.0]),
            ("classifier.dense.bias", &[10.0]),
            ("classifier.out_proj.bias", &[11.0]),
        ]);
        for (name, value) in [
            ("encoder.relative_attention_bias.weight", 3.0),
            ("encoder.layer.0.attention.self.query.weight", 4.0),
            ("encoder.layer.0.attention.self.key.weight", 5.0),
            ("encoder.layer.0.attention.self.value.weight", 6.0),
            ("encoder.layer.0.attention.output.dense.bias", 7.0),
            ("encoder.layer.0.attention.output.LayerNorm.weight", 8.0),
            ("encoder.layer.0.output.dense.bias", 9.0),
        ] {
            assert_eq!(weights.get(name, 1).unwrap(), [value], "{name}");
        }
        assert_eq!(
            weights
                .get("embeddings.token_type_embeddings.weight", 2)
                .unwrap(),
            [0.0, 0.0]
        );
        if !prefix.is_empty() {
            assert_eq!(weights.get("pooler.dense.bias", 1).unwrap(), [10.0]);
            assert_eq!(weights.get("classifier.bias", 1).unwrap(), [11.0]);
        }
    }
}
