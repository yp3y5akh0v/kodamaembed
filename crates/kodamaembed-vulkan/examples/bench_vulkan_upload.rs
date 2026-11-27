//! Upload time of `Buffer::from_slice` against `Buffer::zeroed` plus `write`, in ABBA order.

use half::f16;
use kodamaembed_vulkan::{Buffer, Device, DeviceRepr, VulkanError};
use std::time::Instant;

const ITERATIONS: usize = 25;

fn measure<T: DeviceRepr>(
    device: &Device,
    values: &[T],
    zero_first: bool,
) -> Result<f64, VulkanError> {
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _buffer = if zero_first {
            let mut buffer = Buffer::zeroed(device, values.len())?;
            buffer.write(device, values)?;
            buffer
        } else {
            Buffer::from_slice(device, values)?
        };
    }
    Ok(started.elapsed().as_secs_f64())
}

fn compare<T: DeviceRepr>(device: &Device, name: &str, values: &[T]) -> Result<(), VulkanError> {
    let _ = Buffer::from_slice(device, values)?;
    let before = measure(device, values, true)?;
    let after = measure(device, values, false)?;
    let after = after + measure(device, values, false)?;
    let before = before + measure(device, values, true)?;
    let divisor = (2 * ITERATIONS) as f64;
    println!(
        "{name}: zero-then-copy {:.3} ms; direct upload {:.3} ms; {:.2}x",
        before * 1e3 / divisor,
        after * 1e3 / divisor,
        before / after
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0".into())
        .parse()?;
    let device = Device::new(index)?;
    println!("{}", device.name);
    compare(&device, "1024-element norm", &[1.0f32; 1024])?;
    compare(
        &device,
        "1024x1024 FP16 matrix",
        &vec![f16::ONE; 1024 * 1024],
    )?;
    Ok(())
}
