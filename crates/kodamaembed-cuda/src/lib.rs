//! CUDA backend for KodamaEmbed, its kernels embedded as PTX.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/crest.svg",
    html_favicon_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/favicon.ico"
)]

pub mod driver;
pub mod encoder;
pub mod kernels;
pub mod repro;

pub use driver::{
    Args, Context, CudaError, DeviceBuffer, DeviceRepr, Event, Function, Module, Stream,
};
pub use encoder::{Encoder, EncoderError, MAX_SEQUENCES, MAX_TOKENS};

/// [`Encoder`]'s kernels and sources, hashed into fingerprints; the CPU's
/// sources identify [`repro::ReproEncoder`]'s outputs, which are its bits.
pub const NUMERICAL_SOURCES: &[&[u8]] = &[
    include_bytes!("../ptx/norm.ptx"),
    include_bytes!("../ptx/gemm.ptx"),
    include_bytes!("../ptx/attention.ptx"),
    include_bytes!("../ptx/pool.ptx"),
    include_bytes!("encoder.rs"),
    include_bytes!("kernels.rs"),
    include_bytes!("driver.rs"),
    include_bytes!("lib.rs"),
];

#[cfg(test)]
mod tests {
    /// The manifest stays out of the fingerprint: `cargo publish` rewrites it.
    #[test]
    fn fingerprint_inputs_exclude_the_manifest() {
        let manifest: &[u8] = include_bytes!("../Cargo.toml");
        assert!(!super::NUMERICAL_SOURCES.contains(&manifest));
    }
}
