//! Prepared INT8 weights: FP16 rounding, signed H128 rotation and quantization.

use super::{Activation, Columns, Error, GROUP, HALF, NR, Weight, is_supported};
#[cfg(target_arch = "x86_64")]
use super::{Target, gptq};
use crate::gemm::SHORT_ROWS;
use half::f16;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
mod avx2;

const BLOCK: usize = 128;
const BAND: usize = 96;
/// The fastest part size measured for the shortest inputs.
const QUANTIZE_ROWS: usize = 2;
/// Larger parts allocate less; three per worker still balance the step.
const QUANTIZE_PARTS_PER_THREAD: usize = 3;
const _: () = assert!(BAND.is_multiple_of(QUANTIZE_ROWS));
const RECIP_SQRT_BLOCK: f32 = 0.088_388_346;
/// The restored dot's digits are 14 q, its steps each row's scale / 14.
const RESTORED_STEPS: f32 = 14.0;
/// GPTQ's damping as a fraction of the Hessian's mean diagonal.
#[cfg(target_arch = "x86_64")]
const GPTQ_DAMP: f64 = 0.1;
/// Operand rows rotated at a time for a Hessian.
#[cfg(target_arch = "x86_64")]
const HESSIAN_ROWS: usize = 512;

pub(crate) struct PreparedWeight {
    packed: Weight,
    scales: Vec<f32>,
    signs: Vec<f32>,
    /// The operand's digits a dot reads: with two it walks the weights twice in one chain.
    digits: usize,
    /// Calibrated weights only.
    restored: Option<Restored>,
    statics: Option<StaticCenters>,
    deferred: Option<Deferred>,
}

/// W (gamma (z - mu) rstd + beta) + b = rstd (W (gamma z) - mu W gamma) + W beta + b,
/// so the norm's statistics join only the epilogue.
struct Deferred {
    gains: Vec<f32>,
    /// W gamma and W beta + b: f64 sums in depth order, rounded once.
    sums: Vec<f32>,
    constants: Vec<f32>,
}

pub(crate) struct DeferredNorm<'a> {
    pub(crate) gains: &'a [f32],
    pub(crate) beta: &'a [f32],
    pub(crate) bias: &'a [f32],
}

/// A deferred LayerNorm's row weighs rstd^2 in the Hessian: the epilogue scales its dot by rstd.
#[derive(Clone, Copy)]
pub(crate) struct Operand<'a> {
    pub(crate) rows: &'a [f32],
    pub(crate) classes: &'a [u32],
    pub(crate) means: &'a [f64],
    pub(crate) weights: Option<&'a [f32]>,
}

/// Each token class's FP16 center and its product on the exact weights, computed once.
pub(crate) struct StaticCenters {
    centers: Vec<f32>,
    shared: Vec<f32>,
}

/// `corrections` cancel what each part's dot gains from activation bytes taken
/// as qa + 128; `scales` are each row's scale / 14.
struct Restored {
    corrections: Vec<i32>,
    scales: Vec<f32>,
}

#[cfg(target_arch = "x86_64")]
impl Deferred {
    /// Products of two f32 values are exact in f64; each sum rounds once.
    fn new(raw: &[f32], n: usize, k: usize, norm: DeferredNorm<'_>) -> Result<Self, Error> {
        let DeferredNorm { gains, beta, bias } = norm;
        if raw.len() != super::len::<f32>(n, k)?
            || gains.len() != k
            || beta.len() != k
            || bias.len() != n
        {
            return Err(Error::Dimensions);
        }
        let (sums, constants): (Vec<f32>, Vec<f32>) = raw
            .chunks_exact(k)
            .zip(bias)
            .map(|(row, &b)| {
                let (mut sum, mut constant) = (0.0_f64, 0.0_f64);
                for ((&w, &gain), &shift) in row.iter().zip(gains).zip(beta) {
                    sum += f64::from(w) * f64::from(gain);
                    constant += f64::from(w) * f64::from(shift);
                }
                (sum as f32, (constant + f64::from(b)) as f32)
            })
            .unzip();
        if sums
            .iter()
            .chain(&constants)
            .any(|value| !value.is_finite())
        {
            return Err(Error::FloatRange);
        }
        Ok(Self {
            gains: gains.to_vec(),
            sums,
            constants,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl StaticCenters {
    /// Each center rounds once to FP16 and each product once from an f64 sum.
    fn exact(raw: &[f32], n: usize, k: usize, means: &[f64]) -> Result<Self, Error> {
        if means.is_empty()
            || !means.len().is_multiple_of(k)
            || raw.len() != super::len::<f32>(n, k)?
        {
            return Err(Error::Dimensions);
        }
        let centers: Vec<f32> = means.iter().map(|&m| half_of(m)).collect();
        if centers.iter().any(|c| !c.is_finite()) {
            return Err(Error::FloatRange);
        }
        let classes = centers.len() / k;
        let shared: Vec<f32> = (0..super::len::<f32>(classes, n)?)
            .into_par_iter()
            .map(|at| {
                let (center, row) = (&centers[at / n * k..][..k], &raw[at % n * k..][..k]);
                let mut sum = 0.0_f64;
                for (&w, &c) in row.iter().zip(center) {
                    sum += f64::from(w) * f64::from(c);
                }
                sum as f32
            })
            .collect();
        if shared.iter().any(|value| !value.is_finite()) {
            return Err(Error::FloatRange);
        }
        Ok(Self { centers, shared })
    }

    /// In the digits' space: rows less their centers, rotated like the weights.
    fn hessian(&self, operand: Operand<'_>, signs: &[f32]) -> Result<gptq::Hessian, Error> {
        let Operand {
            rows,
            classes,
            weights,
            ..
        } = operand;
        let k = signs.len();
        if k == 0
            || rows.len() != super::len::<f32>(classes.len(), k)?
            || classes
                .iter()
                .any(|&c| (c as usize + 1) * k > self.centers.len())
            || weights.is_some_and(|weights| weights.len() != classes.len())
        {
            return Err(Error::Dimensions);
        }
        let mut hessian = gptq::Hessian::new(k)?;
        let mut rotated = vec![0.0_f32; super::len::<f32>(HESSIAN_ROWS.min(classes.len()), k)?];
        for first in (0..classes.len()).step_by(HESSIAN_ROWS) {
            let count = HESSIAN_ROWS.min(classes.len() - first);
            rotated[..count * k]
                .par_chunks_mut(k)
                .zip(rows[first * k..(first + count) * k].par_chunks(k))
                .zip(classes[first..first + count].par_iter())
                .for_each(|((out, row), &class)| {
                    let center = &self.centers[class as usize * k..][..k];
                    for (((out, &x), &c), &sign) in out.iter_mut().zip(row).zip(center).zip(signs) {
                        *out = (x - c) * sign;
                    }
                    hadamard(out);
                });
            hessian.add(
                &rotated[..count * k],
                weights.map(|weights| &weights[first..first + count]),
            )?;
        }
        Ok(hessian)
    }
}

impl PreparedWeight {
    /// GPTQ digits on the Hessian of the centered operand, which is what the digits multiply.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn calibrated(
        raw: &[f32],
        n: usize,
        k: usize,
        signs: &[f32],
        operand: Operand<'_>,
        norm: Option<DeferredNorm<'_>>,
        digits: usize,
    ) -> Result<Self, Error> {
        if !(1..=2).contains(&digits) {
            return Err(Error::Dimensions);
        }
        let rotated = rotated_weights(raw, n, k, signs)?;
        let scales = row_scales(&rotated, k)?;
        let statics = StaticCenters::exact(raw, n, k, operand.means)?;
        let hessian = statics.hessian(operand, signs)?;
        let values = gptq::digits(&rotated, &scales, &hessian, GPTQ_DAMP)?;
        Ok(Self {
            packed: Weight::new(&values, n, k)?,
            restored: Some(restored_terms(&values, &scales, n, k)?),
            statics: Some(statics),
            deferred: norm
                .map(|norm| Deferred::new(raw, n, k, norm))
                .transpose()?,
            scales,
            signs: signs.to_vec(),
            digits,
        })
    }

    /// Little-endian: digits, row scales, centers, their products, deferred sums.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn write_tables(&self, out: &mut Vec<u8>) -> Result<(), Error> {
        let Some(statics) = &self.statics else {
            return Err(Error::Unsupported);
        };
        out.extend(self.q_by_row()?.iter().map(|&q| q.to_le_bytes()[0]));
        let deferred = self.deferred.iter().flat_map(|d| [&d.sums, &d.constants]);
        for values in [&self.scales, &statics.centers, &statics.shared]
            .into_iter()
            .chain(deferred)
        {
            out.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        }
        Ok(())
    }

    /// Refuses digits past [-127, 127], non-positive scales and non-finite values.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn from_tables(
        n: usize,
        k: usize,
        signs: &[f32],
        gains: Option<&[f32]>,
        tables: &mut &[u8],
        digits: usize,
    ) -> Result<Self, Error> {
        use kodamaembed_model::repro::STATIC_CLASSES;
        if signs.len() != k
            || gains.is_some_and(|gains| gains.len() != k)
            || !(1..=2).contains(&digits)
        {
            return Err(Error::Dimensions);
        }
        let values: Vec<i8> = take(tables, super::len::<i8>(n, k)?)?
            .iter()
            .map(|&byte| i8::from_le_bytes([byte]))
            .collect();
        if values.contains(&i8::MIN) {
            return Err(Error::IntegerRange);
        }
        let scales = take_floats(tables, n)?;
        if scales.iter().any(|&scale| scale <= 0.0) {
            return Err(Error::FloatRange);
        }
        let statics = StaticCenters {
            centers: take_floats(tables, super::len::<f32>(STATIC_CLASSES, k)?)?,
            shared: take_floats(tables, super::len::<f32>(STATIC_CLASSES, n)?)?,
        };
        let deferred = gains
            .map(|gains| -> Result<Deferred, Error> {
                Ok(Deferred {
                    gains: gains.to_vec(),
                    sums: take_floats(tables, n)?,
                    constants: take_floats(tables, n)?,
                })
            })
            .transpose()?;
        Ok(Self {
            packed: Weight::new(&values, n, k)?,
            restored: Some(restored_terms(&values, &scales, n, k)?),
            statics: Some(statics),
            deferred,
            scales,
            signs: signs.to_vec(),
            digits,
        })
    }

    pub(crate) fn n(&self) -> usize {
        self.packed.n()
    }
    pub(crate) fn k(&self) -> usize {
        self.packed.k()
    }

    pub(crate) fn defers(&self) -> bool {
        self.deferred.is_some()
    }

    /// For another device's reproducible path; calibrated weights only.
    pub(crate) fn export(
        &self,
        bias: &[f32],
    ) -> Result<kodamaembed_model::repro::ProjectionExport, Error> {
        let (n, k) = (self.n(), self.k());
        let Some(statics) = &self.statics else {
            return Err(Error::Unsupported);
        };
        if bias.len() != n {
            return Err(Error::Dimensions);
        }
        Ok(kodamaembed_model::repro::ProjectionExport {
            n,
            k,
            q: self.q_by_row()?,
            scales: self.scales.clone(),
            signs: self.signs.clone(),
            centers: statics.centers.clone(),
            center_shared: statics.shared.clone(),
            bias: bias.to_vec(),
            operand_digits: self.digits,
            deferred: self
                .deferred
                .as_ref()
                .map(|d| kodamaembed_model::repro::DeferredExport {
                    sums: d.sums.clone(),
                    constants: d.constants.clone(),
                }),
        })
    }

    fn q_by_row(&self) -> Result<Vec<i8>, Error> {
        let (n, k) = (self.n(), self.k());
        let panel_values = super::len::<i8>(k.div_ceil(GROUP), NR * GROUP)?;
        let mut q = vec![0_i8; super::len::<i8>(n, k)?];
        for j in 0..n {
            for t in 0..k {
                let index =
                    j / NR * panel_values + t / GROUP * NR * GROUP + j % NR * GROUP + t % GROUP;
                q[j * k + t] = self.packed.panels[index];
            }
        }
        Ok(q)
    }

    /// Each row centers on its own class, so no row depends on another.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn project_restored_into(
        &self,
        input: &[f32],
        m: usize,
        row_classes: &[u32],
        segment: usize,
        epilogue: crate::gemm::Epilogue<'_>,
        out: &mut [f32],
    ) -> Result<(), Error> {
        use crate::gemm::Epilogue;
        let (k, n) = (self.k(), self.n());
        let (Some(restored), Some(statics), None) = (&self.restored, &self.statics, &self.deferred)
        else {
            return Err(Error::Unsupported);
        };
        let bias = epilogue_bias(epilogue);
        if input.len() != super::len::<f32>(m, k)?
            || out.len() != super::len::<f32>(m, n)?
            || row_classes.len() != m
            || row_classes
                .iter()
                .any(|&c| (c as usize + 1) * k > statics.centers.len())
            || bias.len() != n
            || matches!(epilogue, Epilogue::BiasResidual(_, r) if r.len() != out.len())
            || !self.restart_fits(segment)
        {
            return Err(Error::Dimensions);
        }
        if !is_supported() {
            return Err(Error::Unsupported);
        }
        // One f32 addition: the shared component plus the bias.
        let rows: Vec<f32> = statics
            .shared
            .chunks_exact(n)
            .flat_map(|shared| shared.iter().zip(bias).map(|(&value, &b)| value + b))
            .collect();
        let centering = avx2::Centering {
            means: &statics.centers,
            seq: row_classes,
            gains: None,
        };
        self.restored_bands(
            input,
            m,
            centering,
            restored,
            segment,
            with_bias(epilogue, &rows),
            None,
            out,
        )
    }

    /// fma(fma(-mean, W gamma, u), rstd, W beta + b) per row, then GELU if asked.
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_deferred_into(
        &self,
        input: &[f32],
        m: usize,
        row_classes: &[u32],
        mean: &[f32],
        rstd: &[f32],
        gelu: bool,
        segment: usize,
        out: &mut [f32],
    ) -> Result<(), Error> {
        use crate::gemm::Epilogue;
        let (k, n) = (self.k(), self.n());
        let (
            Some(restored),
            Some(statics),
            Some(Deferred {
                gains,
                sums,
                constants,
            }),
        ) = (&self.restored, &self.statics, &self.deferred)
        else {
            return Err(Error::Unsupported);
        };
        if input.len() != super::len::<f32>(m, k)?
            || out.len() != super::len::<f32>(m, n)?
            || row_classes.len() != m
            || row_classes
                .iter()
                .any(|&c| (c as usize + 1) * k > statics.centers.len())
            || mean.len() != m
            || rstd.len() != m
            || !self.restart_fits(segment)
        {
            return Err(Error::Dimensions);
        }
        if !is_supported() {
            return Err(Error::Unsupported);
        }
        let centering = avx2::Centering {
            means: &statics.centers,
            seq: row_classes,
            gains: Some(gains),
        };
        let finish = Finish {
            mean,
            rstd,
            sums,
            constants,
            gelu,
        };
        self.restored_bands(
            input,
            m,
            centering,
            restored,
            segment,
            Epilogue::Bias(&statics.shared),
            Some(finish),
            out,
        )
    }

    /// A segment is whole HALF-deep parts and divides the depth; two digits chain through one.
    #[cfg(target_arch = "x86_64")]
    fn restart_fits(&self, segment: usize) -> bool {
        let k = self.k();
        segment > 0
            && segment.is_multiple_of(HALF)
            && k.is_multiple_of(segment)
            && (self.digits == 1 || segment == k)
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    fn restored_bands(
        &self,
        input: &[f32],
        m: usize,
        centering: avx2::Centering<'_>,
        restored: &Restored,
        segment: usize,
        epilogue: crate::gemm::Epilogue<'_>,
        finish: Option<Finish<'_>>,
        out: &mut [f32],
    ) -> Result<(), Error> {
        let (n, row_classes) = (self.n(), centering.seq);
        // Second digits continue their segment's chain: the dot's one segment is its depth.
        let segment = self.digits * segment;
        self.bands(
            input,
            m,
            centering,
            out,
            |row, a, steps, columns, output| {
                super::fused_epilogue::dot_restored_into(
                    a,
                    columns,
                    &restored.corrections,
                    steps,
                    &restored.scales,
                    segment,
                    band_epilogue(epilogue, row, a.m, n),
                    &row_classes[row..row + a.m],
                    output,
                )?;
                if let Some(Finish {
                    mean,
                    rstd,
                    sums,
                    constants,
                    gelu,
                }) = finish
                {
                    let (col, cols) = (columns.col(), columns.n());
                    let (sums, constants) = (&sums[col..col + cols], &constants[col..col + cols]);
                    for r in 0..a.m {
                        let (mean, rstd) = (mean[row + r], rstd[row + r]);
                        // SAFETY: AVX2 and FMA are checked; the row and terms hold cols values.
                        unsafe {
                            finish_deferred(output.row(r), mean, rstd, sums, constants, gelu)
                        };
                    }
                }
                Ok(())
            },
        )
    }

    fn split(&self, m: usize) -> Split {
        let (bands, strips) = (m.div_ceil(BAND), m.div_ceil(super::MR));
        let threads = rayon::current_num_threads();
        if m <= SHORT_ROWS && threads > 1 {
            // Under two strips a worker, row bands leave workers idle; over it, column groups
            // would read every row from the cores that quantized it.
            if strips < 2 * threads {
                return Split::Short;
            }
            // Workers that each quantize and multiply their own rows share and copy nothing.
            return Split::Rows {
                height: strips.div_ceil(threads) * super::MR,
            };
        }
        let groups = if bands > threads {
            1
        } else {
            (GROUPS_PER_THREAD * threads / bands.max(1)).clamp(1, self.packed.panel_count())
        };
        if groups == 1 {
            Split::Rows { height: BAND }
        } else {
            Split::Columns { groups }
        }
    }

    /// A divisor of the band, because each band concatenates whole parts.
    fn quantize_rows(m: usize) -> usize {
        let wanted = m / (QUANTIZE_PARTS_PER_THREAD * rayon::current_num_threads());
        (QUANTIZE_ROWS..=wanted.max(QUANTIZE_ROWS))
            .rev()
            .find(|&rows| BAND.is_multiple_of(rows))
            .expect("band divisor")
    }

    /// Splits never change a row's arithmetic, so bits never depend on workers.
    #[cfg(target_arch = "x86_64")]
    fn bands<F>(
        &self,
        input: &[f32],
        m: usize,
        centering: avx2::Centering<'_>,
        out: &mut [f32],
        finish: F,
    ) -> Result<(), Error>
    where
        F: Fn(usize, &Activation, &[f32], Columns<'_>, &mut Target<'_>) -> Result<(), Error> + Sync,
    {
        let (k, n) = (self.k(), self.n());
        let depth = self.digits * k;
        let quantize = |row: usize, raw: &[f32]| {
            let rows = raw.len() / k;
            let centering = avx2::Centering {
                seq: &centering.seq[row..][..rows],
                ..centering
            };
            avx2::quantized_blocks(raw, rows, k, &self.signs, Some(centering), self.digits)
        };
        let activation = |row: usize, raw: &[f32]| {
            let (values, steps) = quantize(row, raw)?;
            Ok((Activation::new(&values, raw.len() / k, depth)?, steps))
        };
        let (bands, panels) = (m.div_ceil(BAND), self.packed.panel_count());
        let groups = match self.split(m) {
            Split::Short => return self.short(input, m, quantize, out, finish),
            Split::Rows { height } => {
                return input
                    .par_chunks(height * k)
                    .zip(out.par_chunks_mut(height * n))
                    .enumerate()
                    .try_for_each(|(band, (raw, output))| {
                        let (a, scales) = activation(band * height, raw)?;
                        let mut target = Target::whole(output, n)?;
                        finish(
                            band * height,
                            &a,
                            &scales,
                            (&self.packed).into(),
                            &mut target,
                        )
                    });
            }
            Split::Columns { groups } => groups,
        };
        let rows = Self::quantize_rows(m);
        let parts = input
            .par_chunks(rows * k)
            .enumerate()
            .map(|(part, raw)| activation(part * rows, raw))
            .collect::<Result<Vec<_>, Error>>()?;
        let quantized: Vec<(Activation, Vec<f32>)> = parts
            .chunks(BAND / rows)
            .map(|band| {
                let scales = band.iter().flat_map(|(_, s)| s.iter().copied()).collect();
                (
                    Activation::concat(depth, band.iter().map(|(a, _)| a)),
                    scales,
                )
            })
            .collect();
        let per = panels.div_ceil(groups);
        let groups = panels.div_ceil(per);
        let stripes = Stripes::new(out, n);
        (0..bands * groups).into_par_iter().try_for_each(|task| {
            let (band, group) = (task / groups, task % groups);
            let (a, scales) = &quantized[band];
            let columns = self
                .packed
                .columns(group * per, ((group + 1) * per).min(panels));
            // SAFETY: (band, group) and its cells are this task's alone.
            let mut target = unsafe { stripes.target(band * BAND, a.m, columns) };
            finish(band * BAND, a, scales, columns, &mut target)
        })
    }

    /// Rows quantize once into one activation; each worker writes a group of columns in place.
    #[cfg(target_arch = "x86_64")]
    fn short<Q, F>(
        &self,
        input: &[f32],
        m: usize,
        quantize: Q,
        out: &mut [f32],
        finish: F,
    ) -> Result<(), Error>
    where
        Q: Fn(usize, &[f32]) -> Result<(Vec<i8>, Vec<f32>), Error> + Sync,
        F: Fn(usize, &Activation, &[f32], Columns<'_>, &mut Target<'_>) -> Result<(), Error> + Sync,
    {
        let (k, n) = (self.k(), self.n());
        let (threads, panels) = (rayon::current_num_threads(), self.packed.panel_count());
        let depth = self.digits * k;
        let groups_k = depth.div_ceil(GROUP);
        let per_row = depth / HALF;
        let rows = m.div_ceil(threads).max(1);
        let (mut words, mut steps) = (SHORT_WORDS.take(), SHORT_STEPS.take());
        words.resize(super::len::<u32>(m, groups_k)?, 0);
        steps.resize(super::len::<f32>(m, per_row)?, 0.0);
        let quantized = words
            .par_chunks_mut(rows * groups_k)
            .zip(steps.par_chunks_mut(rows * per_row))
            .zip(input.par_chunks(rows * k))
            .enumerate()
            .try_for_each(|(part, ((words, steps), raw))| {
                let (values, part_steps) = quantize(part * rows, raw)?;
                steps.copy_from_slice(&part_steps);
                super::activation_avx2::pack_into(&values, raw.len() / k, depth, groups_k, words)
            });
        let a = Activation::from_words(words, m, depth);
        let multiplied = quantized.and_then(|()| {
            let groups = threads.min(panels);
            let stripes = Stripes::new(out, n);
            (0..groups).into_par_iter().try_for_each(|group| {
                let columns = self
                    .packed
                    .columns(group * panels / groups, (group + 1) * panels / groups);
                // SAFETY: each group writes only its own columns of every row.
                let mut target = unsafe { stripes.target(0, m, columns) };
                finish(0, &a, &steps, columns, &mut target)
            })
        });
        SHORT_WORDS.set(a.into_words());
        SHORT_STEPS.set(steps);
        multiplied
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Taken during use, so a re-entrant call on this thread fills its own.
    static SHORT_WORDS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    static SHORT_STEPS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// An output that parallel tasks write in disjoint cells.
#[cfg(target_arch = "x86_64")]
struct Stripes<'a> {
    out: *mut f32,
    n: usize,
    len: usize,
    _borrow: std::marker::PhantomData<&'a mut [f32]>,
}

// SAFETY: `out` stays borrowed while the stripes live; writers own their cells.
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for Stripes<'_> {}

#[cfg(target_arch = "x86_64")]
impl<'a> Stripes<'a> {
    fn new(out: &'a mut [f32], n: usize) -> Self {
        Self {
            out: out.as_mut_ptr(),
            n,
            len: out.len(),
            _borrow: std::marker::PhantomData,
        }
    }

    /// # Safety
    /// Nothing else touches these rows of these columns while the target lives.
    unsafe fn target(&self, row: usize, rows: usize, columns: Columns<'_>) -> Target<'_> {
        assert!(columns.col() + columns.n() <= self.n && (row + rows) * self.n <= self.len);
        // SAFETY: the cells lie inside `out` (asserted) and are the caller's alone.
        unsafe {
            Target::shared(
                self.out.add(row * self.n + columns.col()),
                rows,
                columns.n(),
                self.n,
            )
        }
    }
}

/// Smaller groups shorten the wait for a step's last group.
const GROUPS_PER_THREAD: usize = 16;

#[derive(Clone, Copy)]
enum Split {
    Short,
    Rows { height: usize },
    Columns { groups: usize },
}

/// In `kodamaembed_model::repro::EncoderExport`'s order.
pub(crate) fn repro_constants() -> Vec<f32> {
    use crate::math::{GELU_A, GELU_CAP};
    let mut f = vec![RECIP_SQRT_BLOCK];
    f.extend(GELU_A);
    f.push(GELU_CAP);
    f.push(std::f32::consts::FRAC_1_SQRT_2);
    f
}

fn epilogue_bias(epilogue: crate::gemm::Epilogue<'_>) -> &[f32] {
    use crate::gemm::Epilogue;
    match epilogue {
        Epilogue::Bias(b) | Epilogue::BiasGelu(b) | Epilogue::BiasResidual(b, _) => b,
    }
}

fn with_bias<'a>(
    epilogue: crate::gemm::Epilogue<'a>,
    bias: &'a [f32],
) -> crate::gemm::Epilogue<'a> {
    use crate::gemm::Epilogue;
    match epilogue {
        Epilogue::Bias(_) => Epilogue::Bias(bias),
        Epilogue::BiasGelu(_) => Epilogue::BiasGelu(bias),
        Epilogue::BiasResidual(_, residual) => Epilogue::BiasResidual(bias, residual),
    }
}

#[cfg(target_arch = "x86_64")]
fn band_epilogue(
    epilogue: crate::gemm::Epilogue<'_>,
    row: usize,
    rows: usize,
    n: usize,
) -> crate::gemm::Epilogue<'_> {
    use crate::gemm::Epilogue;
    match epilogue {
        Epilogue::BiasResidual(b, residual) => {
            Epilogue::BiasResidual(b, &residual[row * n..(row + rows) * n])
        }
        other => other,
    }
}

/// A deferred norm's terms, applied in each band's task while its rows are in that core's cache.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Finish<'a> {
    mean: &'a [f32],
    rstd: &'a [f32],
    sums: &'a [f32],
    constants: &'a [f32],
    gelu: bool,
}

/// fma(fma(-mean, sums[j], u), rstd, constants[j]): two roundings a value.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn finish_deferred(
    row: &mut [f32],
    mean: f32,
    rstd: f32,
    sums: &[f32],
    constants: &[f32],
    gelu: bool,
) {
    use std::arch::x86_64::*;
    let full = row.len() / 8 * 8;
    let (shift, scale) = (_mm256_set1_ps(-mean), _mm256_set1_ps(rstd));
    for j in (0..full).step_by(8) {
        // SAFETY: the caller's slices hold row.len() values.
        unsafe {
            let u = _mm256_loadu_ps(row.as_ptr().add(j));
            let v = _mm256_fmadd_ps(shift, _mm256_loadu_ps(sums.as_ptr().add(j)), u);
            let mut value = _mm256_fmadd_ps(v, scale, _mm256_loadu_ps(constants.as_ptr().add(j)));
            if gelu {
                value = crate::math::avx2::gelu(value);
            }
            _mm256_storeu_ps(row.as_mut_ptr().add(j), value);
        }
    }
    for j in full..row.len() {
        let value = (-mean).mul_add(sums[j], row[j]).mul_add(rstd, constants[j]);
        row[j] = if gelu {
            crate::math::gelu(value)
        } else {
            value
        };
    }
}

/// One rounding to FP16, nearest-even, out-of-range values to infinity.
fn half_of(value: f64) -> f32 {
    let exponent = (((value.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-14);
    let ulp = f64::from_bits(((exponent - 10 + 1023) as u64) << 52);
    let rounded = (value / ulp).round_ties_even() * ulp;
    if rounded.abs() > 65504.0 {
        f32::INFINITY.copysign(value as f32)
    } else {
        rounded as f32
    }
}

fn rotated_weights(raw: &[f32], n: usize, k: usize, signs: &[f32]) -> Result<Vec<f32>, Error> {
    super::validate_depth(k)?;
    if !k.is_multiple_of(BLOCK)
        || n == 0
        || signs.len() != k
        || raw.len() != super::len::<f32>(n, k)?
    {
        return Err(Error::Dimensions);
    }
    if signs.iter().any(|&s| s != 1.0 && s != -1.0) {
        return Err(Error::FloatRange);
    }
    rotated_rows_scalar(raw, n, k, signs)
}

/// Scalar: weights prepare once at load, and this path is the SIMD oracle.
fn rotated_rows_scalar(
    raw: &[f32],
    rows: usize,
    k: usize,
    signs: &[f32],
) -> Result<Vec<f32>, Error> {
    let mut rotated = vec![0.0_f32; super::len::<f32>(rows, k)?];
    for (source, row) in raw.chunks_exact(k).zip(rotated.chunks_exact_mut(k)) {
        for t in 0..k {
            if !source[t].is_finite() {
                return Err(Error::FloatRange);
            }
            row[t] = f16::from_f32(source[t]).to_f32() * signs[t];
            if !row[t].is_finite() {
                return Err(Error::FloatRange);
            }
        }
        hadamard(row);
    }
    Ok(rotated)
}

/// Each row's scale: its largest magnitude over 127, 1 for a zero row.
fn row_scales(rotated: &[f32], k: usize) -> Result<Vec<f32>, Error> {
    rotated
        .chunks_exact(k)
        .map(|row| {
            let mut largest = 0.0_f32;
            for &x in row {
                if !x.is_finite() {
                    return Err(Error::FloatRange);
                }
                largest = largest.max(x.abs());
            }
            let scale = if largest > 0.0 { largest / 127.0 } else { 1.0 };
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Error::FloatRange);
            }
            Ok(scale)
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn take<'a>(tables: &mut &'a [u8], count: usize) -> Result<&'a [u8], Error> {
    let Some((front, rest)) = tables.split_at_checked(count) else {
        return Err(Error::Dimensions);
    };
    *tables = rest;
    Ok(front)
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn take_floats(tables: &mut &[u8], count: usize) -> Result<Vec<f32>, Error> {
    let bytes = take(tables, super::len::<f32>(count, size_of::<f32>())?)?;
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| f32::from_le_bytes(bytes))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::FloatRange);
    }
    Ok(values)
}

#[cfg(target_arch = "x86_64")]
fn restored_terms(values: &[i8], scales: &[f32], n: usize, k: usize) -> Result<Restored, Error> {
    let parts = k / HALF;
    let mut corrections = vec![0_i32; super::len::<i32>(n.div_ceil(NR) * parts, NR)?];
    for j in 0..n {
        for t in 0..k {
            // |sum of a part's 64 digits| <= 64 * 127; times 128 fits INT32.
            corrections[(j / NR * parts + t / HALF) * NR + j % NR] +=
                128 * i32::from(values[j * k + t]);
        }
    }
    Ok(Restored {
        corrections,
        scales: scales.iter().map(|&scale| scale / RESTORED_STEPS).collect(),
    })
}

fn hadamard(values: &mut [f32]) {
    for block in values.as_chunks_mut::<BLOCK>().0 {
        for h in [1, 2, 4, 8, 16, 32, 64] {
            for group in block.chunks_exact_mut(2 * h) {
                for j in 0..h {
                    let a = group[j];
                    let b = group[h + j];
                    group[j] = a + b;
                    group[h + j] = a - b;
                }
            }
        }
        for x in block {
            *x *= RECIP_SQRT_BLOCK;
        }
    }
}

#[cfg(test)]
mod tests;
