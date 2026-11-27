//! INT32 tile epilogues with fixed conversion, scaling and addition order.

use super::{Activation, Columns, Error, GROUP, HALF, MR, NR, Target, Weight, check_dot, len};
use crate::{gemm::Epilogue, math};
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
struct TilePosition {
    rows: usize,
    row: usize,
    /// In the weight's columns (scales, bias, residual); `out_col` is the target's.
    col: usize,
    n: usize,
    out_col: usize,
}

/// f = fma(dot_p, step_p, f) per HALF-deep part in depth order, each dot converting exactly;
/// segments restart and add in order, and second digits walk the weights again in one chain.
#[allow(clippy::too_many_arguments)]
pub(super) fn dot_restored_into<'a>(
    a: &Activation,
    w: impl Into<Columns<'a>>,
    corrections: &[i32],
    steps: &[f32],
    col_scales: &[f32],
    segment: usize,
    epilogue: Epilogue<'_>,
    of_row: &[u32],
    out: &mut Target<'_>,
) -> Result<(), Error> {
    let columns = w.into();
    let w = columns.weight;
    check_dot(a, w)?;
    if !w.k.is_multiple_of(2 * HALF)
        || segment == 0
        || !segment.is_multiple_of(HALF)
        || !a.k.is_multiple_of(segment)
        || (a.k != w.k && segment != a.k)
        || steps.len() != len::<f32>(a.m, a.k / HALF)?
        || corrections.len() != len::<i32>(w.panel_count(), w.k / HALF * NR)?
    {
        return Err(Error::Dimensions);
    }
    let bias = match epilogue {
        Epilogue::Bias(b) | Epilogue::BiasGelu(b) | Epilogue::BiasResidual(b, _) => b,
    };
    if of_row.len() != a.m || !bias.len().is_multiple_of(w.n) {
        return Err(Error::Dimensions);
    }
    let rows_of_bias = bias.len() / w.n;
    let cells = len::<f32>(a.m, w.n)?;
    if of_row.iter().any(|&s| s as usize >= rows_of_bias)
        || out.rows() != a.m
        || out.cols() != columns.n()
        || col_scales.len() != w.n
        || matches!(epilogue, Epilogue::BiasResidual(_, r) if r.len() != cells)
    {
        return Err(Error::Dimensions);
    }
    let (parts, segment) = (w.k / HALF, segment / HALF);
    let position = |row: usize, rows: usize, panel: usize| TilePosition {
        rows,
        row,
        col: panel * NR,
        n: w.n,
        out_col: (panel - columns.first) * NR,
    };
    for row in (0..a.m).step_by(MR) {
        let rows = (a.m - row).min(MR);
        for panel in columns.first..columns.end {
            // Finished integers cross a noinline boundary before any float epilogue.
            let mut values = [[0.0_f32; NR]; MR];
            let tile = RestoredTile {
                a,
                w,
                corrections: &corrections[panel * parts * NR..][..parts * NR],
                row,
                panel,
                steps: &steps[row * (a.k / HALF)..],
                segment,
            };
            unsafe {
                match rows {
                    1 => restored_tile::<1>(tile, &mut values),
                    2 => restored_tile::<2>(tile, &mut values),
                    3 => restored_tile::<3>(tile, &mut values),
                    4 => restored_tile::<4>(tile, &mut values),
                    5 => restored_tile::<5>(tile, &mut values),
                    6 => restored_tile::<6>(tile, &mut values),
                    _ => unreachable!(),
                }
                finish_tile(
                    &values,
                    position(row, rows, panel),
                    col_scales,
                    epilogue,
                    of_row,
                    out,
                )?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct RestoredTile<'a> {
    a: &'a Activation,
    w: &'a Weight,
    corrections: &'a [i32],
    row: usize,
    panel: usize,
    steps: &'a [f32],
    segment: usize,
}

/// Exact INT32 part dots, each converted exactly and added by one FMA.
#[inline(never)]
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn restored_tile<const R: usize>(tile: RestoredTile<'_>, values: &mut [[f32; NR]; MR]) {
    let RestoredTile {
        a,
        w,
        corrections,
        row,
        panel,
        steps,
        segment,
    } = tile;
    let (parts, per_row) = (w.k / HALF, a.k / HALF);
    unsafe {
        let fourteen = _mm256_set1_ps(14.0);
        let magic = _mm256_set1_epi32(0x4B40_0000);
        let unbias = _mm256_set1_ps(-14.0 * 12_582_912.0);
        let mut f_lo = [_mm256_setzero_ps(); R];
        let mut f_hi = [_mm256_setzero_ps(); R];
        let mut total_lo = [_mm256_setzero_ps(); R];
        let mut total_hi = [_mm256_setzero_ps(); R];
        let activation = a.words.as_ptr().add(row * a.groups);
        let weight = w.panels.as_ptr().add(panel * w.groups * NR * GROUP);
        for at in 0..per_row {
            // A second digit's part meets the weights and correction of the first's.
            let part = at % parts;
            let inputs = activation.add(at * HALF / GROUP);
            let weights = weight.add(part * HALF / GROUP * NR * GROUP);
            let c0 = _mm256_loadu_si256(corrections.as_ptr().add(part * NR).cast());
            let c1 = _mm256_loadu_si256(corrections.as_ptr().add(part * NR + 8).cast());
            // From the bits of 1.5 * 2^23 less the correction, each dot ends as a float's bits.
            let mut lo = [_mm256_sub_epi32(magic, c0); R];
            let mut hi = [_mm256_sub_epi32(magic, c1); R];
            for g in 0..HALF / GROUP {
                let b0 = _mm256_loadu_si256(weights.add(g * NR * GROUP).cast());
                let b1 = _mm256_loadu_si256(weights.add(g * NR * GROUP + 32).cast());
                for r in 0..R {
                    let x = _mm256_set1_epi32(*inputs.add(r * a.groups + g) as i32);
                    lo[r] = _mm256_dpbusd_avx_epi32(lo[r], x, b0);
                    hi[r] = _mm256_dpbusd_avx_epi32(hi[r], x, b1);
                }
            }
            for r in 0..R {
                let step = _mm256_set1_ps(steps[r * per_row + at]);
                // |q . a| <= 64 * 127 * 127 < 2^24 / 14 < 2^22: the bits are 1.5 * 2^23 + q . a,
                // and the FMA rounds the exact 14 (q . a) to itself.
                let dot_lo = _mm256_fmadd_ps(_mm256_castsi256_ps(lo[r]), fourteen, unbias);
                let dot_hi = _mm256_fmadd_ps(_mm256_castsi256_ps(hi[r]), fourteen, unbias);
                f_lo[r] = _mm256_fmadd_ps(dot_lo, step, f_lo[r]);
                f_hi[r] = _mm256_fmadd_ps(dot_hi, step, f_hi[r]);
            }
            if (at + 1).is_multiple_of(segment) {
                for r in 0..R {
                    if at + 1 == segment {
                        total_lo[r] = f_lo[r];
                        total_hi[r] = f_hi[r];
                    } else {
                        total_lo[r] = _mm256_add_ps(total_lo[r], f_lo[r]);
                        total_hi[r] = _mm256_add_ps(total_hi[r], f_hi[r]);
                    }
                    f_lo[r] = _mm256_setzero_ps();
                    f_hi[r] = _mm256_setzero_ps();
                }
            }
        }
        for r in 0..R {
            _mm256_storeu_ps(values[r].as_mut_ptr(), total_lo[r]);
            _mm256_storeu_ps(values[r].as_mut_ptr().add(8), total_hi[r]);
        }
    }
}

/// fl(value * column scale), checked finite, then bias, residual and GELU.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn finish_tile(
    values: &[[f32; NR]; MR],
    position: TilePosition,
    col_scales: &[f32],
    epilogue: Epilogue<'_>,
    of_row: &[u32],
    out: &mut Target<'_>,
) -> Result<(), Error> {
    let TilePosition {
        rows,
        row,
        col,
        n,
        out_col,
    } = position;
    let cols = (n - col).min(NR);
    let full = cols / 8 * 8;
    let (biases, residual, gelu) = match epilogue {
        Epilogue::Bias(b) => (b, None, false),
        Epilogue::BiasGelu(b) => (b, None, true),
        Epilogue::BiasResidual(b, r) => (b, Some(r), false),
    };
    unsafe {
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
        let infinity = _mm256_set1_ps(f32::INFINITY);
        for r in 0..rows {
            let at = (row + r) * n + col;
            let to = &mut out.row(row + r)[out_col..out_col + cols];
            let bias = &biases[of_row[row + r] as usize * n..][..n];
            for j in (0..full).step_by(8) {
                let a = _mm256_loadu_ps(values[r].as_ptr().add(j));
                let mut value = _mm256_mul_ps(a, _mm256_loadu_ps(col_scales.as_ptr().add(col + j)));
                // Reject overflow before an epilogue can mask it.
                if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(
                    _mm256_and_ps(value, abs_mask),
                    infinity,
                )) != 255
                {
                    return Err(Error::FloatRange);
                }
                value = _mm256_add_ps(value, _mm256_loadu_ps(bias.as_ptr().add(col + j)));
                if let Some(residual) = residual {
                    value = _mm256_add_ps(value, _mm256_loadu_ps(residual.as_ptr().add(at + j)));
                }
                if gelu {
                    value = math::avx2::gelu(value);
                }
                _mm256_storeu_ps(to.as_mut_ptr().add(j), value);
            }
            for j in full..cols {
                let mut value = values[r][j] * col_scales[col + j];
                if !value.is_finite() {
                    return Err(Error::FloatRange);
                }
                value += bias[col + j];
                if let Some(residual) = residual {
                    value += residual[at + j];
                }
                if gelu {
                    value = math::gelu(value);
                }
                to[j] = value;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
