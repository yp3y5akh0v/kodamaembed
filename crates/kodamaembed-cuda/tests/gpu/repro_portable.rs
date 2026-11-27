use kodamaembed_cuda::{Args, Context, DeviceBuffer, Module, Stream};

#[test]
fn reciprocal_rounding_survives_a_legal_hardware_estimate() {
    let ptx = include_str!("../../ptx/repro.ptx");
    let entry = ptx
        .find(".visible .entry repro_reciprocal_check(")
        .expect("check kernel embedded");
    let start = entry + ptx[entry..].find('{').expect("kernel body");
    let mut depth = 0;
    let end = ptx[start..]
        .char_indices()
        .find_map(|(i, c)| {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            (depth == 0).then_some(start + i + c.len_utf8())
        })
        .expect("body closes");
    let mut reciprocals = 0;
    let mut references = 0;
    // Inject into compiled production arithmetic, preserving its refinement and oracle.
    let body: String = ptx[start..end]
        .lines()
        .map(|line| {
            let instruction = line.split("//").next().unwrap().trim();
            let (opcode, operands) = instruction
                .split_once(char::is_whitespace)
                .unwrap_or((instruction, ""));
            if opcode == "div.rn.f32" {
                references += 1;
            }
            let rewritten = if opcode.starts_with("rcp.") {
                reciprocals += 1;
                match opcode {
                    "rcp.approx.f32" | "rcp.approx.ftz.f32" => {
                        assert_eq!(operands.matches(';').count(), 1);
                        let (destination, _) = operands.split_once(',').expect("two operands");
                        format!("mov.f32 {}, 0f3F000000;", destination.trim())
                    }
                    "rcp.rn.f32" => line.to_owned(),
                    _ => panic!("unrecognized reciprocal instruction: {instruction}"),
                }
            } else {
                line.to_owned()
            };
            rewritten + "\n"
        })
        .collect();
    assert_eq!(reciprocals, 1, "reciprocal inlined");
    assert_eq!(references, 1, "oracle remains");
    let injected = format!("{}{}{}", &ptx[..start], body, &ptx[end..]);

    // For 2 - 2^-23, one half is within the instruction's one-ulp error bound.
    let denominator = f32::from_bits(0x3fff_ffff);
    let estimate = 0.5_f32;
    let ulp = f64::from(estimate.next_up()) - f64::from(estimate);
    assert!((1.0 / f64::from(denominator) - f64::from(estimate)).abs() < ulp);

    let context = Context::new(0).unwrap();
    let module = Module::load(&context, &injected).unwrap();
    let function = module.function("repro_reciprocal_check").unwrap();
    let stream = Stream::new(&context).unwrap();
    let mismatches = DeviceBuffer::<u64>::zeroed(&context, 1).unwrap();
    context.bind().unwrap();
    // SAFETY: the arguments match the check kernel; the count outlives its synchronized launch.
    unsafe {
        function
            .launch(
                &stream,
                [1, 1, 1],
                [32, 1, 1],
                0,
                &mut Args::new()
                    .i32(denominator.to_bits() as i32)
                    .i32(1)
                    .ptr(&mismatches),
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(
        mismatches.read().unwrap(),
        [0],
        "a permitted approximate reciprocal must not change the rounded result"
    );
}
