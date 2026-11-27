//! Retrieval-quality evaluation for KodamaEmbed.

pub mod benchmark;
pub mod engine;
pub mod fidelity;
pub mod power;
pub mod ranking;
pub mod vectors;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
