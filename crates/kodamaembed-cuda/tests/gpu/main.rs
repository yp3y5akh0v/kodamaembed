//! Kernel tests on the local GPU against double-precision CPU references.

mod attention;
mod driver;
mod encoder;
mod gemm;
mod norm;
mod pool;
#[cfg(target_arch = "x86_64")]
mod repro;
mod repro_portable;
mod support;
