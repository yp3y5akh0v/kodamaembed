use super::*;
use crate::int8::is_supported;

#[test]
fn restored_bias_rows_reject_bad_maps_before_writes() {
    if !is_supported() {
        return;
    }
    let (m, n, k) = (13, 33, 256);
    let qa: Vec<i8> = (0..m * k)
        .map(|at| ((at * 17 % 255) as i16 - 127) as i8)
        .collect();
    let qw: Vec<i8> = (0..n * k)
        .map(|at| ((at * 29 % 253) as i16 - 126) as i8)
        .collect();
    let a = Activation::new(&qa, m, k).unwrap();
    let w = Weight::new(&qw, n, k).unwrap();
    let corrections = vec![0; w.panel_count() * k / HALF * NR];
    let (steps, scales) = (vec![1.0; m * k / HALF], vec![1.0; n]);
    let rows: Vec<f32> = (0..3 * n).map(|at| (at % 11) as f32 * 0.25 - 1.0).collect();
    let of_row: Vec<u32> = (0..m as u32).map(|r| [2, 0, 1][r as usize % 3]).collect();
    let restored = |bias: &[f32], map: &[u32], out: &mut [f32]| {
        dot_restored_into(
            &a,
            &w,
            &corrections,
            &steps,
            &scales,
            k,
            Epilogue::Bias(bias),
            map,
            &mut Target::whole(out, n)?,
        )
    };
    let mut out = vec![42.0; m * n];
    let mut outside = of_row.clone();
    outside[m - 1] = 3;
    for map in [&outside[..], &of_row[1..]] {
        assert_eq!(restored(&rows, map, &mut out), Err(Error::Dimensions));
    }
    assert_eq!(
        restored(&rows[1..], &of_row, &mut out),
        Err(Error::Dimensions)
    );
    assert!(out.iter().all(|&v| v == 42.0));
    assert!(restored(&rows, &of_row, &mut out).is_ok());
}

#[test]
fn float_range_is_rejected_before_bias_residual_or_gelu_and_shapes_before_writes() {
    if !is_supported() {
        return;
    }
    let (n, k) = (17, 2 * HALF);
    let a = Activation::new(&vec![-127; k], 1, k).unwrap();
    let w = Weight::new(&vec![127; n * k], n, k).unwrap();
    let corrections = vec![0; w.panel_count() * k / HALF * NR];
    let bias = [f32::MAX; 17];
    let residual = [f32::MAX; 17];
    let restored = |step: f32, epilogue, out: &mut [f32]| {
        dot_restored_into(
            &a,
            &w,
            &corrections,
            &[step; 2],
            &[1.0; 17],
            k,
            epilogue,
            &[0],
            &mut Target::whole(out, n)?,
        )
    };
    for epi in [
        Epilogue::Bias(&bias),
        Epilogue::BiasGelu(&bias),
        Epilogue::BiasResidual(&bias, &residual),
    ] {
        let mut out = [42.0; 17];
        assert_eq!(restored(f32::MAX, epi, &mut out), Err(Error::FloatRange));
    }
    let mut out = [42.0; 17];
    assert_eq!(
        restored(1.0, Epilogue::Bias(&bias[..16]), &mut out),
        Err(Error::Dimensions)
    );
    assert_eq!(out, [42.0; 17]);
    assert_eq!(
        restored(
            1.0,
            Epilogue::BiasResidual(&bias, &residual[..16]),
            &mut out
        ),
        Err(Error::Dimensions)
    );
    assert_eq!(out, [42.0; 17]);
}

/// A row of two digits walks the weights twice in one chain; a restart between them is refused.
#[test]
fn second_digits_walk_the_weights_again_in_one_chain() {
    if !is_supported() {
        return;
    }
    let (m, n, k) = (7, 19, 2 * HALF);
    let (depth, parts) = (2 * k, 2 * k / HALF);
    let qa: Vec<i8> = (0..m * depth)
        .map(|at| ((at * 37 % 255) as i16 - 127) as i8)
        .collect();
    let qw: Vec<i8> = (0..n * k)
        .map(|at| ((at * 53 % 251) as i16 - 125) as i8)
        .collect();
    let a = Activation::new(&qa, m, depth).unwrap();
    let w = Weight::new(&qw, n, k).unwrap();
    let mut corrections = vec![0; w.panel_count() * k / HALF * NR];
    for j in 0..n {
        for t in 0..k {
            corrections[(j / NR * (k / HALF) + t / HALF) * NR + j % NR] +=
                128 * i32::from(qw[j * k + t]);
        }
    }
    // Steps unlike each other, so a part on another's step or out of order changes bits.
    let steps: Vec<f32> = (0..m * parts)
        .map(|at| 0.001 + (at * 7 % 13) as f32 / 3.0)
        .collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.5 + j as f32 / 16.0).collect();
    let bias: Vec<f32> = (0..n).map(|j| j as f32 / 8.0 - 1.0).collect();
    let restored = |segment: usize, out: &mut [f32]| {
        dot_restored_into(
            &a,
            &w,
            &corrections,
            &steps,
            &scales,
            segment,
            Epilogue::Bias(&bias),
            &vec![0_u32; m],
            &mut Target::whole(out, n)?,
        )
    };
    let mut out = vec![42.0; m * n];
    for segment in [k, HALF] {
        assert_eq!(restored(segment, &mut out), Err(Error::Dimensions));
    }
    assert!(out.iter().all(|&v| v == 42.0));
    restored(depth, &mut out).unwrap();
    for r in 0..m {
        for j in 0..n {
            let mut f = 0.0_f32;
            for at in 0..parts {
                let dot: i32 = (0..HALF)
                    .map(|t| {
                        let digit = qa[r * depth + at * HALF + t];
                        i32::from(digit) * 14 * i32::from(qw[j * k + at * HALF % k + t])
                    })
                    .sum();
                f = (dot as f32).mul_add(steps[r * parts + at], f);
            }
            let expected = f * scales[j] + bias[j];
            assert_eq!(out[r * n + j].to_bits(), expected.to_bits(), "r {r} j {j}");
        }
    }
}
