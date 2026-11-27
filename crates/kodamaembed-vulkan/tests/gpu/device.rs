use crate::support::devices;
use half::f16;
use kodamaembed_vulkan::{Buffer, Device, VulkanError};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

/// Clones share one pool and queue, so a second submission waits for the first.
#[test]
fn submissions_on_one_device_wait_for_each_other_across_threads() {
    for device in devices() {
        let first = device.submission();
        let started = AtomicBool::new(false);
        let other = device.clone();
        thread::scope(|scope| {
            let second = scope.spawn(|| {
                let _second = other.submission();
                started.store(true, Ordering::SeqCst);
            });
            thread::sleep(Duration::from_millis(200));
            assert!(
                !started.load(Ordering::SeqCst),
                "{}: a second submission started while the first was open",
                device.name
            );
            drop(first);
            second.join().unwrap();
        });
        assert!(started.load(Ordering::SeqCst), "{}", device.name);
    }
}

#[test]
fn odd_half_buffers_initialize_and_round_trip_every_element() {
    for device in devices() {
        let mut buffer = Buffer::<f16>::zeroed(&device, 3).unwrap();
        assert_eq!(buffer.read(&device).unwrap(), vec![f16::ZERO; 3]);
        let values = [
            f16::from_f32(1.25),
            f16::from_f32(-2.5),
            f16::from_f32(0.75),
        ];
        buffer.write(&device, &values).unwrap();
        assert_eq!(buffer.read(&device).unwrap(), values);
        let uploaded = Buffer::from_slice(&device, &values).unwrap();
        assert_eq!(uploaded.read(&device).unwrap(), values);
    }
}

#[test]
fn buffer_io_rejects_another_logical_device() {
    let device = Device::new(0).unwrap();
    let other = Device::new(0).unwrap();
    let mut buffer = Buffer::<f32>::zeroed(&device, 1).unwrap();
    assert!(buffer.write(&other, &[1.0]).is_err());
    assert!(buffer.read(&other).is_err());
    assert_eq!(buffer.read(&device).unwrap(), [0.0]);
}

#[test]
fn a_failed_device_open_keeps_the_loader_alive_through_cleanup() {
    assert!(Device::new(usize::MAX).is_err());
    // Vulkan must stay usable after the failed open.
    assert!(!Device::adapters().unwrap().is_empty());
}

/// NVIDIA's Linux driver cannot be reloaded more than a few dozen times in a process.
#[test]
fn opening_devices_repeatedly_keeps_the_driver_loadable() {
    for _ in 0..100 {
        assert!(!Device::adapters().unwrap().is_empty());
        drop(Device::new(0).unwrap());
    }
}

#[test]
fn failed_recording_is_discarded_before_a_later_submission() {
    for device in devices() {
        let mut submission = device.submission();
        assert!(
            submission
                .record(|_| Err(VulkanError("record failed".into())))
                .is_err()
        );
        submission.record(|_| Ok(())).unwrap();
        submission.run().unwrap();
    }
}
