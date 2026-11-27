use std::num::NonZeroUsize;

/// Explicit worker counts for one CPU model; tokenization overlaps inference,
/// so the two pools run at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuWorkers {
    /// Workers used by CPU projection, attention, normalization and pooling.
    pub inference_workers: NonZeroUsize,
    /// Workers used by this instance's production tokenizer.
    pub tokenizer_workers: NonZeroUsize,
}

impl CpuWorkers {
    pub(crate) fn validate(self) -> Result<(), crate::Error> {
        let maximum = rayon::max_num_threads();
        if self.inference_workers.get() > maximum || self.tokenizer_workers.get() > maximum {
            return Err(crate::Error::Unsupported(format!(
                "CPU worker counts must not exceed Rayon's limit {maximum}"
            )));
        }
        Ok(())
    }
}
