//! GPTQ INT8 digits (Frantar et al. 2022): Babai's nearest plane on a reversed Cholesky factor,
//! with no inverse formed; sums run in a fixed order, so the digits match on any thread count.

use super::Error;
use crate::gemm::{self, Kernel, NR, Panels};
use rayon::prelude::*;

/// A chunk's products run in f32, the sum of chunks in f64.
const CHUNK: usize = 512;
/// Rows of the Hessian's sums a task computes (a multiple of NR).
const BAND: usize = 64;
/// A multiple of four, for [`take_step`].
const STEP: usize = 64;
const _: () = assert!(STEP.is_multiple_of(4));
/// The step's TRAIL_WIDTH columns stay cached across these rows.
const TRAIL_ROWS: usize = 16;
const TRAIL_WIDTH: usize = 256;
/// Weight rows a rounding task carries through the columns together.
const ROWS: usize = 16;

/// sum_t w_t r_t r_t^T over the rotated operand rows; upper triangle, f64.
pub(crate) struct Hessian {
    k: usize,
    sums: Vec<f64>,
}

impl Hessian {
    pub(crate) fn new(k: usize) -> Result<Self, Error> {
        if k == 0 {
            return Err(Error::Dimensions);
        }
        Ok(Self {
            k,
            sums: vec![0.0; super::len::<f64>(k, k)?],
        })
    }

    /// Chunks add in token order, each once to the f64 sums: no result depends on the threads.
    pub(crate) fn add(&mut self, rows: &[f32], weights: Option<&[f32]>) -> Result<(), Error> {
        let k = self.k;
        if !rows.len().is_multiple_of(k) {
            return Err(Error::Dimensions);
        }
        let m = rows.len() / k;
        if weights.is_some_and(|weights| weights.len() != m) {
            return Err(Error::Dimensions);
        }
        if !rows
            .iter()
            .chain(weights.into_iter().flatten())
            .all(|x| x.is_finite())
        {
            return Err(Error::FloatRange);
        }
        if !super::is_supported() {
            return Err(Error::Unsupported);
        }
        let mut transposed = vec![0.0_f32; super::len::<f32>(k, CHUNK.min(m))?];
        let mut panels = vec![0.0_f32; gemm::panels_len(k, CHUNK.min(m).max(1))];
        for first in (0..m).step_by(CHUNK) {
            let t = CHUNK.min(m - first);
            let chunk = &rows[first * k..(first + t) * k];
            let chunk_weights = weights.map(|weights| &weights[first..first + t]);
            transposed[..k * t]
                .par_chunks_mut(BAND * t)
                .enumerate()
                .for_each(|(band, features)| {
                    let i0 = band * BAND;
                    for (r, row) in chunk.chunks_exact(k).enumerate() {
                        let row = &row[i0..i0 + features.len() / t];
                        for (i, &x) in row.iter().enumerate() {
                            features[i * t + r] = x;
                        }
                    }
                });
            panels[..gemm::panels_len(k, t)]
                .par_chunks_mut(t * NR)
                .enumerate()
                .for_each(|(p, panel)| {
                    let cols = NR.min(k - p * NR);
                    let tokens = panel.as_chunks_mut::<NR>().0.iter_mut();
                    for (r, (out, row)) in tokens.zip(chunk.chunks_exact(k)).enumerate() {
                        let values = &row[p * NR..][..cols];
                        match chunk_weights {
                            Some(weights) => {
                                for (out, &x) in out.iter_mut().zip(values) {
                                    *out = x * weights[r];
                                }
                            }
                            None => out[..cols].copy_from_slice(values),
                        }
                        out[cols..].fill(0.0);
                    }
                });
            let (transposed, panels) = (&transposed[..k * t], &panels[..]);
            self.sums
                .par_chunks_mut(BAND * k)
                .enumerate()
                .for_each_init(Vec::<f32>::new, |scratch, (band, sums)| {
                    // Upper triangle: row i holds columns i..k.
                    let (i0, rows) = (band * BAND, sums.len() / k);
                    let cols = k - i0;
                    scratch.resize(rows * cols, 0.0);
                    let right = Panels {
                        values: &panels[i0 / NR * t * NR..],
                        n: cols,
                        k: t,
                    };
                    gemm::product(
                        Kernel::Avx2,
                        &transposed[i0 * t..(i0 + rows) * t],
                        t,
                        rows,
                        right,
                        scratch,
                        cols,
                    );
                    for (r, (sums, products)) in sums
                        .chunks_exact_mut(k)
                        .zip(scratch.chunks_exact(cols))
                        .enumerate()
                    {
                        for (sum, &product) in sums[i0 + r..].iter_mut().zip(&products[r..]) {
                            *sum += f64::from(product);
                        }
                    }
                });
        }
        Ok(())
    }
}

/// Rows take steps as plain rounding does, x / scale in f32; `damp` scales the mean diagonal.
pub(crate) fn digits(
    rotated: &[f32],
    scales: &[f32],
    hessian: &Hessian,
    damp: f64,
) -> Result<Vec<i8>, Error> {
    let (n, k) = (scales.len(), hessian.k);
    if n == 0 || rotated.len() != super::len::<f32>(n, k)? {
        return Err(Error::Dimensions);
    }
    if !rotated.iter().chain(scales).all(|x| x.is_finite())
        || scales.iter().any(|&scale| scale <= 0.0)
        || !(damp.is_finite() && damp >= 0.0)
    {
        return Err(Error::FloatRange);
    }
    if !super::is_supported() {
        return Err(Error::Unsupported);
    }
    // SAFETY: the processor has AVX2 and FMA (is_supported).
    let factor = unsafe { reversed_factor(hessian, damp)? };
    let mut out = vec![0_i8; rotated.len()];
    out.par_chunks_mut(ROWS * k)
        .zip(rotated.par_chunks(ROWS * k).zip(scales.par_chunks(ROWS)))
        .try_for_each(|(out, (rows, scales))| {
            // SAFETY: as above.
            unsafe { round_rows(&factor, k, rows, scales, out) }
        })?;
    Ok(out)
}

/// U^T U = P (H + damp mean(diag H) I) P. Each entry's terms subtract in row
/// order, giving the unblocked algorithm's bits on any thread count.
#[target_feature(enable = "avx2,fma")]
unsafe fn reversed_factor(hessian: &Hessian, damp: f64) -> Result<Vec<f64>, Error> {
    let k = hessian.k;
    let sums = &hessian.sums;
    let mut diagonal = 0.0;
    for i in 0..k {
        diagonal += sums[i * k + i];
    }
    let mean = diagonal / k as f64;
    if !(mean.is_finite() && mean > 0.0) {
        return Err(Error::FloatRange);
    }
    // PHP's entry (a, b) is H's (k-1-b, k-1-a).
    let mut u = vec![0.0_f64; sums.len()];
    u.par_chunks_mut(STEP * k)
        .enumerate()
        .for_each(|(block, rows)| {
            let a0 = block * STEP;
            for b0 in (a0..k).step_by(STEP) {
                for (offset, row) in rows.chunks_exact_mut(k).enumerate() {
                    let a = a0 + offset;
                    for b in b0.max(a)..(b0 + STEP).min(k) {
                        row[b] = sums[(k - 1 - b) * k + (k - 1 - a)];
                    }
                }
            }
            for (offset, row) in rows.chunks_exact_mut(k).enumerate() {
                row[a0 + offset] += damp * mean;
            }
        });
    for j0 in (0..k).step_by(STEP) {
        let j1 = (j0 + STEP).min(k);
        for c in j0..j1 {
            let (done, rest) = u.split_at_mut((c + 1) * k);
            let row = &mut done[c * k..];
            let pivot = row[c];
            if !(pivot.is_finite() && pivot > 0.0) {
                return Err(Error::FloatRange);
            }
            let root = pivot.sqrt();
            row[c] = root;
            for x in &mut row[c + 1..] {
                *x /= root;
            }
            let row = &done[c * k..];
            for (r, below) in (c + 1..j1).zip(rest.chunks_exact_mut(k)) {
                // SAFETY: the caller's features.
                unsafe { axpy(-row[r], &row[r..], &mut below[r..]) };
            }
        }
        let (step, rest) = u.split_at_mut(j1 * k);
        let step = &step[j0 * k..];
        rest.par_chunks_mut(TRAIL_ROWS * k)
            .enumerate()
            .for_each(|(block, rows)| {
                let r0 = j1 + block * TRAIL_ROWS;
                for c0 in (r0..k).step_by(TRAIL_WIDTH) {
                    let c1 = (c0 + TRAIL_WIDTH).min(k);
                    for (offset, below) in rows.chunks_exact_mut(k).enumerate() {
                        let r = r0 + offset;
                        if r >= c1 {
                            break;
                        }
                        // SAFETY: the caller's features.
                        unsafe { take_step(step, k, r, &mut below[c0.max(r)..c1], c0.max(r)) };
                    }
                }
            });
    }
    Ok(u)
}

/// Four rows a pass with y in registers: the same FMAs in the same order as [`axpy`] row by row.
#[target_feature(enable = "avx2,fma")]
unsafe fn take_step(step: &[f64], k: usize, r: usize, y: &mut [f64], from: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(step.len(), STEP * k);
    let (len, to) = (y.len(), from + y.len());
    let whole = len - len % 4;
    for quad in step.chunks_exact(4 * k) {
        let x = [0, 1, 2, 3].map(|i| &quad[i * k + from..i * k + to]);
        let a = [0, 1, 2, 3].map(|i| -quad[i * k + r]);
        let av = [
            _mm256_set1_pd(a[0]),
            _mm256_set1_pd(a[1]),
            _mm256_set1_pd(a[2]),
            _mm256_set1_pd(a[3]),
        ];
        for at in (0..whole).step_by(4) {
            // SAFETY: four values at `at` in y and in each row of x.
            unsafe {
                let mut v = _mm256_loadu_pd(y.as_ptr().add(at));
                for (x, &a) in x.iter().zip(&av) {
                    v = _mm256_fmadd_pd(_mm256_loadu_pd(x.as_ptr().add(at)), a, v);
                }
                _mm256_storeu_pd(y.as_mut_ptr().add(at), v);
            }
        }
        for at in whole..len {
            let mut v = y[at];
            for (x, &a) in x.iter().zip(&a) {
                v = x[at].mul_add(a, v);
            }
            y[at] = v;
        }
    }
}

/// One fused multiply-add a value.
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(x.len(), y.len());
    let (xs, x_tail) = x.as_chunks::<4>();
    let (ys, y_tail) = y.as_chunks_mut::<4>();
    let av = _mm256_set1_pd(a);
    for (x, y) in xs.iter().zip(ys) {
        // SAFETY: four values each.
        unsafe {
            let v = _mm256_fmadd_pd(_mm256_loadu_pd(x.as_ptr()), av, _mm256_loadu_pd(y.as_ptr()));
            _mm256_storeu_pd(y.as_mut_ptr(), v);
        }
    }
    for (y, &x) in y_tail.iter_mut().zip(x_tail) {
        *y = x.mul_add(a, *y);
    }
}

/// A fixed order: 16 running FMA sums, then halved by partners.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot(x: &[f64], y: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_pd(); 4];
    let (xs, x_tail) = x.as_chunks::<16>();
    let (ys, y_tail) = y.as_chunks::<16>();
    for (x, y) in xs.iter().zip(ys) {
        for (j, sum) in sums.iter_mut().enumerate() {
            // SAFETY: lanes 4j to 4j + 3 of 16 values each.
            unsafe {
                let (x, y) = (x.as_ptr().add(4 * j), y.as_ptr().add(4 * j));
                *sum = _mm256_fmadd_pd(_mm256_loadu_pd(x), _mm256_loadu_pd(y), *sum);
            }
        }
    }
    let mut lanes = [0.0_f64; 16];
    for (lanes, sum) in lanes.as_chunks_mut::<4>().0.iter_mut().zip(sums) {
        // SAFETY: four values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum) };
    }
    for ((lane, &x), &y) in lanes.iter_mut().zip(x_tail).zip(y_tail) {
        *lane = x.mul_add(y, *lane);
    }
    let mut width = 8;
    while width > 0 {
        for j in 0..width {
            lanes[j] += lanes[j + width];
        }
        width /= 2;
    }
    lanes[0]
}

/// digit c = round(w_c + (sum_{a>c} (w_a - q_a) U[c][a]) / U[c][c]), which
/// is GPTQ's digit on the reversed factor.
#[target_feature(enable = "avx2,fma")]
unsafe fn round_rows(
    u: &[f64],
    k: usize,
    rows: &[f32],
    scales: &[f32],
    out: &mut [i8],
) -> Result<(), Error> {
    let count = scales.len();
    let mut w = vec![0.0_f64; count * k];
    for ((w, row), &scale) in w.chunks_exact_mut(k).zip(rows.chunks_exact(k)).zip(scales) {
        for (w, &x) in w.iter_mut().zip(row.iter().rev()) {
            *w = f64::from(x / scale);
        }
    }
    let mut rest = vec![0.0_f64; count * k];
    let mut digits = vec![0_i8; count * k];
    for c in (0..k).rev() {
        let row = &u[c * k..(c + 1) * k];
        for g in 0..count {
            let at = g * k;
            // SAFETY: the caller's features.
            let carried = unsafe { dot(&rest[at + c + 1..at + k], &row[c + 1..]) };
            let digit = (w[at + c] + carried / row[c]).round_ties_even();
            if !digit.is_finite() {
                return Err(Error::FloatRange);
            }
            let digit = digit.clamp(-127.0, 127.0);
            rest[at + c] = w[at + c] - digit;
            digits[at + c] = digit as i8;
        }
    }
    for (out, digits) in out.chunks_exact_mut(k).zip(digits.chunks_exact(k)) {
        for (out, &digit) in out.iter_mut().zip(digits.iter().rev()) {
            *out = digit;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
