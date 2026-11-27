//! E5-large-v1's profile: rotation signs from SHA-256 digest bits, and the operands of two digits.

use kodamaembed_model::repro::Kind;
const BITS: &[u8; 21504] = include_bytes!("e5_v1_signs.bin");
const PER_LAYER: usize = 7168;
/// The products whose operand outliers leave one digit's steps too coarse on short texts.
const TWO_DIGITS: [(usize, Kind); 2] = [(22, Kind::Ffn1), (23, Kind::Qkv)];

pub(crate) fn operand_digits(layer: usize, kind: Kind) -> usize {
    1 + usize::from(TWO_DIGITS.contains(&(layer, kind)))
}

pub(crate) fn signs(layer: usize, kind: Kind) -> Vec<f32> {
    assert!(layer < 24);
    let (offset, depth) = match kind {
        Kind::Qkv => (0, 1024),
        Kind::Out => (1024, 1024),
        Kind::Ffn1 => (2048, 1024),
        Kind::Ffn2 => (3072, 4096),
    };
    let begin = layer * PER_LAYER + offset;
    (begin..begin + depth)
        .map(|i| {
            if BITS[i / 8] & (1 << (i % 8)) != 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn every_sign_is_its_sha256_derivation() {
        for layer in 0..24_u32 {
            for (kind, name, depth) in [
                (Kind::Qkv, "qkv", 1024_u32),
                (Kind::Out, "out", 1024),
                (Kind::Ffn1, "ffn1", 1024),
                (Kind::Ffn2, "ffn2", 4096),
            ] {
                let derived: Vec<f32> = (0..depth / 256)
                    .flat_map(|counter| {
                        let digest = Sha256::new()
                            .chain_update(b"e5-int8-v1\0")
                            .chain_update(layer.to_le_bytes())
                            .chain_update(name)
                            .chain_update([0_u8])
                            .chain_update(depth.to_le_bytes())
                            .chain_update(counter.to_le_bytes())
                            .finalize();
                        (0..256).map(move |bit| {
                            if (digest[bit / 8] >> (bit % 8)) & 1 == 1 {
                                -1.0
                            } else {
                                1.0
                            }
                        })
                    })
                    .collect();
                assert_eq!(signs(layer as usize, kind), derived, "layer {layer} {name}");
            }
        }
    }

    #[test]
    fn two_digits_at_layer_22_up_and_layer_23_qkv_alone() {
        for layer in 0..24 {
            for kind in [Kind::Qkv, Kind::Out, Kind::Ffn1, Kind::Ffn2] {
                let two = [(22, Kind::Ffn1), (23, Kind::Qkv)].contains(&(layer, kind));
                assert_eq!(
                    operand_digits(layer, kind),
                    1 + usize::from(two),
                    "layer {layer} {}",
                    kind.name()
                );
            }
        }
    }
}
