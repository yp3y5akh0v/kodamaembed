//! Fixtures the crate's unit tests share.

use std::path::{Path, PathBuf};

/// An isolated fixture directory that is removed even if the test panics.
pub(crate) struct TempDir(PathBuf);

impl TempDir {
    pub(crate) fn new(label: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "kodamaembed-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // A leftover directory would leak an earlier run's files into a test.
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Cleanup only: a panic here would hide the test's own failure.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn tiny_bert_config(vocab_size: usize) -> serde_json::Value {
    serde_json::json!({
        "model_type": "bert", "hidden_size": 8, "num_attention_heads": 2,
        "num_hidden_layers": 1, "intermediate_size": 16, "vocab_size": vocab_size,
        "max_position_embeddings": 16, "type_vocab_size": 2,
        "layer_norm_eps": 1e-5, "hidden_act": "gelu",
    })
}

/// A safetensors file: the header's length, the header, then the tensors' bytes.
pub(crate) fn safetensors(header: &serde_json::Value, data: &[u8]) -> Vec<u8> {
    let header = serde_json::to_vec(header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

/// A complete F32 checkpoint for a BERT `config`, with a pooler and a one-label
/// classifier when `relevance` asks for a cross-encoder's.
pub(crate) fn tiny_bert_checkpoint(config: &serde_json::Value, relevance: bool) -> Vec<u8> {
    let size = |field: &str| usize::try_from(config[field].as_u64().unwrap()).unwrap();
    let (hidden, intermediate) = (size("hidden_size"), size("intermediate_size"));
    let mut tensors = vec![
        (
            "embeddings.word_embeddings".to_owned(),
            vec![size("vocab_size"), hidden],
        ),
        (
            "embeddings.position_embeddings".to_owned(),
            vec![size("max_position_embeddings"), hidden],
        ),
        (
            "embeddings.token_type_embeddings".to_owned(),
            vec![size("type_vocab_size"), hidden],
        ),
    ];
    let mut biased = Vec::new();
    let mut norms = vec!["embeddings.LayerNorm".to_owned()];
    for layer in 0..size("num_hidden_layers") {
        let root = format!("encoder.layer.{layer}");
        for (part, rows, columns) in [
            ("attention.self.query", hidden, hidden),
            ("attention.self.key", hidden, hidden),
            ("attention.self.value", hidden, hidden),
            ("attention.output.dense", hidden, hidden),
            ("intermediate.dense", intermediate, hidden),
            ("output.dense", hidden, intermediate),
        ] {
            biased.push((format!("{root}.{part}"), rows, columns));
        }
        norms.push(format!("{root}.attention.output.LayerNorm"));
        norms.push(format!("{root}.output.LayerNorm"));
    }
    if relevance {
        biased.push(("pooler.dense".to_owned(), hidden, hidden));
        biased.push(("classifier".to_owned(), 1, hidden));
    }
    for (name, rows, columns) in biased {
        tensors.push((format!("{name}.weight"), vec![rows, columns]));
        tensors.push((format!("{name}.bias"), vec![rows]));
    }
    for name in norms {
        tensors.push((format!("{name}.weight"), vec![hidden]));
        tensors.push((format!("{name}.bias"), vec![hidden]));
    }
    for (name, _) in &mut tensors[..3] {
        name.push_str(".weight");
    }
    filled(tensors)
}

pub(crate) fn tiny_modernbert_config(vocab_size: usize) -> serde_json::Value {
    serde_json::json!({
        "model_type": "modernbert", "hidden_size": 8, "num_attention_heads": 2,
        "num_hidden_layers": 2, "intermediate_size": 6, "vocab_size": vocab_size,
        "max_position_embeddings": 16, "norm_eps": 1e-5, "hidden_activation": "gelu",
        "global_attn_every_n_layers": 3, "local_attention": 4,
        "global_rope_theta": 160000.0, "local_rope_theta": 10000.0,
    })
}

/// A complete F32 checkpoint for a ModernBERT `config`, its names under `prefix`.
pub(crate) fn tiny_modernbert_checkpoint(config: &serde_json::Value, prefix: &str) -> Vec<u8> {
    filled(modernbert_encoder(config, prefix))
}

/// A complete F32 cross-encoder for a ModernBERT `config`: the encoder under `model.`, the
/// prediction head and a one-label classifier at the root.
pub(crate) fn tiny_modernbert_cross_encoder(config: &serde_json::Value) -> Vec<u8> {
    let hidden = usize::try_from(config["hidden_size"].as_u64().unwrap()).unwrap();
    let mut tensors = modernbert_encoder(config, "model.");
    tensors.extend([
        ("head.dense.weight".to_owned(), vec![hidden, hidden]),
        ("head.norm.weight".to_owned(), vec![hidden]),
        ("classifier.weight".to_owned(), vec![1, hidden]),
        ("classifier.bias".to_owned(), vec![1]),
    ]);
    filled(tensors)
}

/// The names, under `prefix`, and shapes of a ModernBERT `config`'s encoder tensors.
fn modernbert_encoder(config: &serde_json::Value, prefix: &str) -> Vec<(String, Vec<usize>)> {
    let size = |field: &str| usize::try_from(config[field].as_u64().unwrap()).unwrap();
    let (hidden, intermediate) = (size("hidden_size"), size("intermediate_size"));
    let mut tensors = vec![
        (
            "embeddings.tok_embeddings.weight".to_owned(),
            vec![size("vocab_size"), hidden],
        ),
        ("embeddings.norm.weight".to_owned(), vec![hidden]),
        ("final_norm.weight".to_owned(), vec![hidden]),
    ];
    for layer in 0..size("num_hidden_layers") {
        let at = |part: &str| format!("layers.{layer}.{part}.weight");
        if layer > 0 {
            tensors.push((at("attn_norm"), vec![hidden]));
        }
        tensors.push((at("attn.Wqkv"), vec![3 * hidden, hidden]));
        tensors.push((at("attn.Wo"), vec![hidden, hidden]));
        tensors.push((at("mlp_norm"), vec![hidden]));
        tensors.push((at("mlp.Wi"), vec![2 * intermediate, hidden]));
        tensors.push((at("mlp.Wo"), vec![hidden, intermediate]));
    }
    for (name, _) in &mut tensors {
        name.insert_str(0, prefix);
    }
    tensors
}

/// A safetensors file of F32 tensors with the given names and shapes.
fn filled(tensors: Vec<(String, Vec<usize>)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let start = data.len();
        // Unit scales keep the layer norms' outputs from collapsing to their biases.
        let scale = name.ends_with("orm.weight");
        for index in 0..shape.iter().product::<usize>() {
            let value = if scale {
                1.0
            } else {
                ((index % 17) as f32 - 8.0) * 0.01
            };
            data.extend(value.to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({
                "dtype": "F32", "shape": shape, "data_offsets": [start, data.len()],
            }),
        );
    }
    safetensors(&header.into(), &data)
}
