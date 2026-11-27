//! The quantized preset runs only its evaluated files and settings; checks hash the loaded bytes.

use crate::{CpuPrecision, EmbeddingModel, Error, ModelSpec};

/// The model the INT8 preset quantized, with exactly its registered settings.
const EVALUATED: EmbeddingModel = EmbeddingModel::E5Large;

/// `sha256`: the registry's pins before a download, the loaded files' digests after.
pub(super) fn check_artifacts(precision: CpuPrecision, sha256: [&str; 3]) -> Result<(), Error> {
    let model = match precision {
        CpuPrecision::Fp32 => return Ok(()),
        CpuPrecision::E5CenteredInt8 => EVALUATED,
    };
    for (pinned, actual) in model.entry().files.iter().zip(sha256) {
        if actual != pinned.sha256 {
            return Err(Error::Unsupported(format!(
                "quantized precision requires the evaluated {} files; {} SHA-256 {actual} differs from {}; float precision runs other checkpoints",
                model.repository(),
                pinned.name,
                pinned.sha256
            )));
        }
    }
    Ok(())
}

/// Checks the settings only; [`check_artifacts`] checks the files.
pub(super) fn centered_preset(spec: ModelSpec<'_>) -> Result<CpuPrecision, Error> {
    let evaluated = EVALUATED.spec();
    if spec == evaluated {
        return Ok(CpuPrecision::E5CenteredInt8);
    }
    Err(Error::Unsupported(format!(
        "quantized precision requires {} with {:?} pooling, max_length {}, passage prefix {:?} and query prefix {:?}; float precision runs other models",
        EVALUATED.repository(),
        evaluated.pooling,
        evaluated.max_length,
        evaluated.passage_prefix,
        evaluated.query_prefix
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pooling;

    /// The evaluated model's spec with one setting changed at a time.
    fn variations(model: EmbeddingModel) -> [ModelSpec<'static>; 4] {
        let spec = model.spec();
        [
            ModelSpec {
                pooling: Pooling::Cls,
                ..spec
            },
            ModelSpec {
                max_length: spec.max_length - 1,
                ..spec
            },
            ModelSpec {
                passage_prefix: "document: ",
                ..spec
            },
            ModelSpec {
                query_prefix: "question: ",
                ..spec
            },
        ]
    }

    #[test]
    fn fp32_does_not_inherit_the_approximate_model_allowlist() {
        let other = ["7".repeat(64), "8".repeat(64), "9".repeat(64)];
        let other = other.each_ref().map(String::as_str);
        assert!(check_artifacts(CpuPrecision::Fp32, other).is_ok());
        assert!(check_artifacts(CpuPrecision::E5CenteredInt8, other).is_err());
    }

    #[test]
    fn the_preset_accepts_exactly_its_evaluated_files() {
        let preset = CpuPrecision::E5CenteredInt8;
        let own = EVALUATED.entry().pins();
        assert!(check_artifacts(preset, own).is_ok());
        let minilm = EmbeddingModel::AllMiniLmL6V2.entry().pins();
        assert!(check_artifacts(preset, minilm).is_err());
        // One differing file is enough.
        let changed = "0".repeat(64);
        let mut tokenizer_changed = own;
        tokenizer_changed[1] = &changed;
        let Err(Error::Unsupported(reason)) = check_artifacts(preset, tokenizer_changed) else {
            panic!("a changed tokenizer must be rejected");
        };
        assert!(
            reason.contains("tokenizer.json") && reason.contains(EVALUATED.repository()),
            "{reason}"
        );
    }

    #[test]
    fn centered_int8_takes_only_the_evaluated_models_settings() {
        assert_eq!(
            centered_preset(EVALUATED.spec()).unwrap(),
            CpuPrecision::E5CenteredInt8
        );
        for changed in variations(EVALUATED) {
            assert!(centered_preset(changed).is_err(), "{changed:?}");
        }
    }

    /// e5-large-v2 shares e5-large's settings, so only its files tell them apart.
    #[test]
    fn int8_accepts_no_other_registered_model() {
        for &model in EmbeddingModel::ALL {
            let accepted = centered_preset(model.spec())
                .and_then(|preset| check_artifacts(preset, model.entry().pins()))
                .is_ok();
            assert_eq!(accepted, model == EVALUATED, "{model:?}");
        }
    }
}
