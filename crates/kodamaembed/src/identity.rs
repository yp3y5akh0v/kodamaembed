//! Model and numerical identity; batching is absent as outputs are batch-invariant.

use kodamaembed_model::{Activation, Head, Output, Pooling};
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"kodamaembed/model-identity/v1";

/// The policy differs wherever output bits may; files enter as the SHA-256 the loader computed.
pub(crate) struct IdentityInputs<'a> {
    pub config_sha256: &'a [u8; 32],
    pub tokenizer_sha256: &'a [u8; 32],
    pub checkpoint_sha256: &'a [u8; 32],
    pub output: Output,
    pub passage_prefix: &'a str,
    pub query_prefix: &'a str,
    pub max_length: usize,
    pub numerical_policy: &'a [u8],
}

pub(crate) fn fingerprint(inputs: IdentityInputs<'_>) -> String {
    let mut hash = Sha256::new();
    field(&mut hash, b"domain", DOMAIN);
    field(&mut hash, b"config", inputs.config_sha256);
    field(&mut hash, b"tokenizer", inputs.tokenizer_sha256);
    field(&mut hash, b"checkpoint", inputs.checkpoint_sha256);
    field(
        &mut hash,
        b"output",
        match inputs.output {
            Output::Embedding(Pooling::Mean) => b"l2-normalized-mean-pooled",
            Output::Embedding(Pooling::Cls) => b"l2-normalized-cls-token",
            Output::Relevance(Head::Pooler) => b"first-token-relevance-logit",
            Output::Relevance(Head::Prediction(pooling, activation)) => {
                match (pooling, activation) {
                    (Pooling::Cls, Activation::Gelu) => b"cls-token-gelu-prediction-logit",
                    (Pooling::Cls, Activation::Silu) => b"cls-token-silu-prediction-logit",
                    (Pooling::Mean, Activation::Gelu) => b"mean-pooled-gelu-prediction-logit",
                    (Pooling::Mean, Activation::Silu) => b"mean-pooled-silu-prediction-logit",
                }
            }
        },
    );
    field(
        &mut hash,
        b"passage-prefix",
        inputs.passage_prefix.as_bytes(),
    );
    field(&mut hash, b"query-prefix", inputs.query_prefix.as_bytes());
    // u64, so 32-bit and 64-bit hosts give the same identity.
    field(
        &mut hash,
        b"max-length",
        &(inputs.max_length as u64).to_le_bytes(),
    );
    field(&mut hash, b"numerical-policy", inputs.numerical_policy);
    format!("sha256:{:x}", hash.finalize())
}

/// Length prefixes keep bytes from moving across field boundaries.
fn field(hash: &mut Sha256, name: &[u8], value: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name);
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

/// Lowercase hex, as the Hugging Face hub and `sha256sum` print digests.
pub(crate) fn hex(digest: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    digest
        .iter()
        .flat_map(|byte| {
            [
                DIGITS[usize::from(byte >> 4)],
                DIGITS[usize::from(byte & 0xf)],
            ]
        })
        .map(char::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: [u8; 32] = [1; 32];
    const TOKENIZER: [u8; 32] = [2; 32];
    const CHECKPOINT: [u8; 32] = [3; 32];
    const OTHER: [u8; 32] = [4; 32];

    #[test]
    fn digests_print_as_lowercase_hex() {
        let abc: [u8; 32] = Sha256::digest(b"abc").into();
        assert_eq!(
            hex(&abc),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(hex(&[0xab; 32]), "ab".repeat(32));
    }

    fn inputs() -> IdentityInputs<'static> {
        IdentityInputs {
            config_sha256: &CONFIG,
            tokenizer_sha256: &TOKENIZER,
            checkpoint_sha256: &CHECKPOINT,
            output: Output::Embedding(Pooling::Mean),
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            max_length: 512,
            numerical_policy: b"cpu-avx2:source-digest",
        }
    }

    #[test]
    fn identical_loaded_inputs_have_a_stable_sha256_identity() {
        let first = fingerprint(inputs());
        assert_eq!(first, fingerprint(inputs()));
        assert_eq!(first.len(), "sha256:".len() + 64);
        assert!(
            first["sha256:".len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
    }

    #[test]
    fn every_output_affecting_field_invalidates_the_identity() {
        let baseline = fingerprint(inputs());
        let alternatives = [
            IdentityInputs {
                config_sha256: &OTHER,
                ..inputs()
            },
            IdentityInputs {
                tokenizer_sha256: &OTHER,
                ..inputs()
            },
            IdentityInputs {
                checkpoint_sha256: &OTHER,
                ..inputs()
            },
            IdentityInputs {
                output: Output::Embedding(Pooling::Cls),
                ..inputs()
            },
            IdentityInputs {
                output: Output::Relevance(Head::Pooler),
                ..inputs()
            },
            IdentityInputs {
                passage_prefix: "document: ",
                ..inputs()
            },
            IdentityInputs {
                query_prefix: "question: ",
                ..inputs()
            },
            IdentityInputs {
                max_length: 256,
                ..inputs()
            },
            IdentityInputs {
                numerical_policy: b"cpu-avx2:changed-source-digest",
                ..inputs()
            },
        ];
        for alternative in alternatives {
            assert_ne!(baseline, fingerprint(alternative));
        }
    }

    #[test]
    fn every_pooling_and_head_has_its_own_identity() {
        let mut outputs = vec![
            Output::Embedding(Pooling::Mean),
            Output::Embedding(Pooling::Cls),
            Output::Relevance(Head::Pooler),
        ];
        for pooling in [Pooling::Cls, Pooling::Mean] {
            for activation in [Activation::Gelu, Activation::Silu] {
                outputs.push(Output::Relevance(Head::Prediction(pooling, activation)));
            }
        }
        let prints: Vec<String> = outputs
            .iter()
            .map(|&output| fingerprint(IdentityInputs { output, ..inputs() }))
            .collect();
        for (index, print) in prints.iter().enumerate() {
            assert!(!prints[..index].contains(print), "{:?}", outputs[index]);
        }
    }

    #[test]
    fn field_boundaries_are_unambiguous() {
        let first = IdentityInputs {
            passage_prefix: "ab",
            query_prefix: "c",
            ..inputs()
        };
        let second = IdentityInputs {
            passage_prefix: "a",
            query_prefix: "bc",
            ..inputs()
        };
        assert_ne!(fingerprint(first), fingerprint(second));
    }

    #[test]
    fn swapped_file_digests_are_another_identity() {
        let swapped = IdentityInputs {
            config_sha256: &TOKENIZER,
            tokenizer_sha256: &CONFIG,
            ..inputs()
        };
        assert_ne!(fingerprint(inputs()), fingerprint(swapped));
    }
}
