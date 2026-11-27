//! The embedding engine interface and its KodamaEmbed implementation.

use crate::{benchmark::Task, vectors::Matrix};

/// Embedding model families and their input-format presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// `intfloat/e5-large` and `e5-large-v2`.
    E5,
    /// `sentence-transformers/all-MiniLM-L6-v2`.
    Minilm,
    /// `nomic-ai/modernbert-embed-base`.
    ModernbertEmbed,
}

impl Family {
    pub fn spec(self) -> kodamaembed::ModelSpec<'static> {
        match self {
            // e5-large-v2 has e5-large's prefixes, truncation and pooling.
            Self::E5 => kodamaembed::EmbeddingModel::E5Large.spec(),
            Self::Minilm => kodamaembed::EmbeddingModel::AllMiniLmL6V2.spec(),
            Self::ModernbertEmbed => kodamaembed::EmbeddingModel::ModernBertEmbedBase.spec(),
        }
    }
}

pub trait Engine {
    fn embed_passages(&mut self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>>;
    fn embed_queries(&mut self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>>;
    /// The device and numerical identity a run records.
    fn report(&self) -> Option<serde_json::Value>;
}

fn check_rows(kind: &str, expected: usize, actual: usize) -> crate::Result<()> {
    if actual != expected {
        return Err(format!("engine returned {actual} {kind} rows for {expected} inputs").into());
    }
    Ok(())
}

/// Rejects missing or extra rows before they can be saved under the task's identity.
pub fn embed_task(engine: &mut dyn Engine, task: &Task) -> crate::Result<(Matrix, Matrix)> {
    fn texts(rows: &[String]) -> Vec<&str> {
        rows.iter().map(String::as_str).collect()
    }
    let passages = engine.embed_passages(&texts(&task.passages))?;
    check_rows("passage", task.passages.len(), passages.len())?;
    let queries = engine.embed_queries(&texts(&task.queries))?;
    check_rows("query", task.queries.len(), queries.len())?;
    Ok((Matrix::from_rows(&passages)?, Matrix::from_rows(&queries)?))
}

pub struct Kodamaembed(pub kodamaembed::Embedder);

impl Engine for Kodamaembed {
    fn embed_passages(&mut self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        Ok(self.0.embed_passages(texts)?)
    }

    fn embed_queries(&mut self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        Ok(self.0.embed_queries(texts)?)
    }

    fn report(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "device": self.0.device_name(),
            "fingerprint": self.0.fingerprint(),
            "gpu_identity": self.0.gpu_identity().map_err(|error| error.to_string()),
        }))
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
