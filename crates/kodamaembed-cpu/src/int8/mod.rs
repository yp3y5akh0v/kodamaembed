//! Exact INT32 projection dots. AVX-VNNI takes unsigned activations, so bytes
//! shift by 128 and each 64-deep part subtracts 128 times its column's digit sum.

use std::fmt;

const MR: usize = 6;
const NR: usize = 16;
const GROUP: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Dimensions,
    IntegerRange,
    DepthOverflow,
    Unsupported,
    FloatRange,
}

#[cfg(target_arch = "x86_64")]
mod fused_epilogue;
#[cfg(target_arch = "x86_64")]
mod gptq;
mod prepared;
pub(crate) use prepared::PreparedWeight;
pub(crate) use prepared::repro_constants;
#[cfg(target_arch = "x86_64")]
pub(crate) use prepared::{DeferredNorm, Operand, take_floats};
#[cfg(target_arch = "x86_64")]
mod activation_avx2;
pub(crate) mod e5_profile;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

/// Every partial sum and correction must fit INT32 in any accumulation order.
pub fn validate_depth(k: usize) -> Result<(), Error> {
    if k == 0 {
        return Err(Error::Dimensions);
    }
    let worst = k.checked_mul(383 * 127).ok_or(Error::DepthOverflow)?;
    if worst > i32::MAX as usize {
        return Err(Error::DepthOverflow);
    }
    Ok(())
}

fn len<T>(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize / size_of::<T>())
        .ok_or(Error::Dimensions)
}

fn check_values(values: &[i8]) -> Result<(), Error> {
    if values.contains(&i8::MIN) {
        Err(Error::IntegerRange)
    } else {
        Ok(())
    }
}

/// Packed W[n,k], zero-padded: [panel16][ceil(k/4)][column16][depth4].
pub struct Weight {
    n: usize,
    k: usize,
    groups: usize,
    panels: Vec<i8>,
}

/// The depth of a promoted dot's exact INT32 partial sums.
const HALF: usize = 64;

impl Weight {
    pub fn new(values: &[i8], n: usize, k: usize) -> Result<Self, Error> {
        validate_depth(k)?;
        if n == 0 || values.len() != len::<i8>(n, k)? {
            return Err(Error::Dimensions);
        }
        check_values(values)?;
        let groups = k.div_ceil(GROUP);
        let panel_count = n.div_ceil(NR);
        let panel_values = len::<i8>(groups, NR * GROUP)?;
        let mut panels = vec![0; len::<i8>(panel_count, panel_values)?];
        for j in 0..n {
            let (panel, col) = (j / NR, j % NR);
            for t in 0..k {
                let index =
                    panel * panel_values + (t / GROUP) * NR * GROUP + col * GROUP + t % GROUP;
                panels[index] = values[j * k + t];
            }
        }
        Ok(Self {
            n,
            k,
            groups,
            panels,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }

    pub(crate) fn panel_count(&self) -> usize {
        self.n.div_ceil(NR)
    }

    /// Requires `first < end <= panel_count()`.
    pub(crate) fn columns(&self, first: usize, end: usize) -> Columns<'_> {
        assert!(first < end && end <= self.panel_count());
        Columns {
            weight: self,
            first,
            end,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Columns<'a> {
    weight: &'a Weight,
    first: usize,
    end: usize,
}

impl<'a> From<&'a Weight> for Columns<'a> {
    fn from(weight: &'a Weight) -> Self {
        weight.columns(0, weight.panel_count())
    }
}

impl Columns<'_> {
    pub(crate) fn col(&self) -> usize {
        self.first * NR
    }

    pub(crate) fn n(&self) -> usize {
        (self.end * NR).min(self.weight.n) - self.col()
    }
}

/// Words of four qa + 128 bytes; padding is 128 (qa = 0) against zero weights.
pub struct Activation {
    m: usize,
    k: usize,
    groups: usize,
    words: Vec<u32>,
}

impl Activation {
    pub fn new(values: &[i8], m: usize, k: usize) -> Result<Self, Error> {
        validate_depth(k)?;
        if values.len() != len::<i8>(m, k)? {
            return Err(Error::Dimensions);
        }
        let groups = k.div_ceil(GROUP);
        #[cfg(target_arch = "x86_64")]
        let words = activation_avx2::pack(values, m, k, groups)?;
        #[cfg(not(target_arch = "x86_64"))]
        let words = {
            let _ = groups;
            return Err(Error::Unsupported);
        };
        Ok(Self {
            m,
            k,
            groups,
            words,
        })
    }

    pub(crate) fn concat<'a>(k: usize, parts: impl Iterator<Item = &'a Activation>) -> Self {
        let (mut m, mut words) = (0, Vec::new());
        for part in parts {
            assert_eq!(part.k, k);
            m += part.m;
            words.extend_from_slice(&part.words);
        }
        Self {
            m,
            k,
            groups: k.div_ceil(GROUP),
            words,
        }
    }

    /// Rows that workers packed into one buffer, so no part is copied.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn from_words(words: Vec<u32>, m: usize, k: usize) -> Self {
        let groups = k.div_ceil(GROUP);
        assert_eq!(
            words.len(),
            m * groups,
            "one row of words per activation row"
        );
        Self {
            m,
            k,
            groups,
            words,
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn into_words(self) -> Vec<u32> {
        self.words
    }
}

/// Rows of output cells `ldc` apart that one task writes: a whole output, or a group of
/// columns interleaved with other tasks' groups, which disjoint slices cannot express.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Target<'a> {
    first: *mut f32,
    rows: usize,
    cols: usize,
    ldc: usize,
    _cells: std::marker::PhantomData<&'a mut [f32]>,
}

#[cfg(target_arch = "x86_64")]
impl<'a> Target<'a> {
    /// All of `out`, `cols` values a row.
    pub(crate) fn whole(out: &'a mut [f32], cols: usize) -> Result<Self, Error> {
        if cols == 0 || !out.len().is_multiple_of(cols) {
            return Err(Error::Dimensions);
        }
        Ok(Self {
            first: out.as_mut_ptr(),
            rows: out.len() / cols,
            cols,
            ldc: cols,
            _cells: std::marker::PhantomData,
        })
    }

    /// # Safety
    /// `first` starts `rows` rows of `cols` cells `ldc` apart that nothing else touches while
    /// the target lives.
    pub(crate) unsafe fn shared(first: *mut f32, rows: usize, cols: usize, ldc: usize) -> Self {
        Self {
            first,
            rows,
            cols,
            ldc,
            _cells: std::marker::PhantomData,
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn row(&mut self, r: usize) -> &mut [f32] {
        assert!(r < self.rows, "row {r} of {}", self.rows);
        // SAFETY: the cells are this target's alone, and `&mut self` lends one row at a time.
        unsafe { std::slice::from_raw_parts_mut(self.first.add(r * self.ldc), self.cols) }
    }
}

#[cfg(test)]
fn pack_activation_scalar(
    values: &[i8],
    m: usize,
    k: usize,
    groups: usize,
) -> Result<Vec<u32>, Error> {
    check_values(values)?;
    let mut words = vec![0; len::<u32>(m, groups)?];
    for row in 0..m {
        for g in 0..groups {
            let mut bytes = [128_u8; GROUP];
            for (t, byte) in bytes.iter_mut().enumerate() {
                let depth = g * GROUP + t;
                if depth < k {
                    *byte = (i16::from(values[row * k + depth]) + 128) as u8;
                }
            }
            words[row * groups + g] = u32::from_le_bytes(bytes);
        }
    }
    Ok(words)
}

pub fn is_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avxvnni")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// An activation row holds the weight's depth once for each of its digits, one behind another.
fn check_dot(a: &Activation, w: &Weight) -> Result<(), Error> {
    if !a.k.is_multiple_of(w.k) || a.groups != a.k / w.k * w.groups {
        return Err(Error::Dimensions);
    }
    if !is_supported() {
        return Err(Error::Unsupported);
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64"))]
mod integer_tests;
