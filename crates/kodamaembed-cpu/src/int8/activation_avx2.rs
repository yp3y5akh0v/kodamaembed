//! Exact byte-wise unsigned shifting and packing, with no floating-point arithmetic.
use super::Error;
use std::arch::x86_64::*;

pub(super) fn pack(values: &[i8], m: usize, k: usize, groups: usize) -> Result<Vec<u32>, Error> {
    let mut words = vec![0x8080_8080; super::len::<u32>(m, groups)?];
    pack_into(values, m, k, groups, &mut words)?;
    Ok(words)
}

/// Writes every byte of `words`, so a reused buffer needs no clearing.
pub(super) fn pack_into(
    values: &[i8],
    m: usize,
    k: usize,
    groups: usize,
    words: &mut [u32],
) -> Result<(), Error> {
    if !is_x86_feature_detected!("avx2") {
        return Err(Error::Unsupported);
    }
    if values.len() != super::len::<i8>(m, k)?
        || groups != k.div_ceil(4)
        || words.len() != super::len::<u32>(m, groups)?
    {
        return Err(Error::Dimensions);
    }
    if !k.is_multiple_of(4) {
        for row in words.chunks_exact_mut(groups) {
            row[groups - 1] = 0x8080_8080;
        }
    }
    // SAFETY: rows hold ceil(K/4)*4 destination bytes and loads read only K
    // source bytes; padding stays 128, signed qa = 0.
    unsafe { fill(values, m, k, groups, words) }
}

#[target_feature(enable = "avx2")]
unsafe fn fill(
    values: &[i8],
    m: usize,
    k: usize,
    groups: usize,
    words: &mut [u32],
) -> Result<(), Error> {
    unsafe {
        let shift = _mm256_set1_epi8(i8::MIN);
        let full = k / 32 * 32;
        for row in 0..m {
            let input = values.as_ptr().add(row * k);
            let output = words.as_mut_ptr().add(row * groups).cast::<u8>();
            for t in (0..full).step_by(32) {
                let x = _mm256_loadu_si256(input.add(t).cast());
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, shift)) != 0 {
                    return Err(Error::IntegerRange);
                }
                _mm256_storeu_si256(output.add(t).cast(), _mm256_xor_si256(x, shift));
            }
            for t in full..k {
                let q = *input.add(t);
                if q == i8::MIN {
                    return Err(Error::IntegerRange);
                }
                *output.add(t) = (q as u8) ^ 128;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_signed_values_and_depth_row_tails_match_scalar_words() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for k in [
            1_usize, 2, 3, 4, 5, 7, 15, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 63, 64, 65,
            127, 128, 129, 257, 384, 1536,
        ] {
            let m = 7;
            let q: Vec<_> = (0..m * k).map(|j| ((j % 255) as i16 - 127) as i8).collect();
            let groups = k.div_ceil(4);
            let scalar = super::super::pack_activation_scalar(&q, m, k, groups).unwrap();
            let vector = pack(&q, m, k, groups).unwrap();
            assert_eq!(scalar, vector, "m={m},k={k}");
            let actual: Vec<_> = vector.iter().flat_map(|word| word.to_le_bytes()).collect();
            let expected: Vec<_> = q
                .chunks_exact(k)
                .flat_map(|row| {
                    row.iter()
                        .map(|&v| (i16::from(v) + 128) as u8)
                        .chain(std::iter::repeat_n(128, groups * 4 - k))
                })
                .collect();
            assert_eq!(actual, expected, "layout k={k}");
        }
        let all: Vec<_> = (-127_i16..=127).map(|v| v as i8).collect();
        assert_eq!(
            pack(&all, 1, 255, 64).unwrap(),
            super::super::pack_activation_scalar(&all, 1, 255, 64).unwrap()
        );
    }

    #[test]
    fn minus128_rejected_in_every_vector_lane_and_scalar_tail() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for k in [1, 3, 31, 32, 33, 63, 64, 65, 129] {
            let m = 3;
            for index in 0..m * k {
                let mut q = vec![0; k * m];
                q[index] = i8::MIN;
                assert_eq!(
                    pack(&q, m, k, k.div_ceil(4)),
                    Err(Error::IntegerRange),
                    "k={k},bad={index}"
                );
            }
        }
    }
}
