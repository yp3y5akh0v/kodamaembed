use kodamaembed_cuda::{Context, DeviceBuffer, Event, Module, Stream};

#[test]
fn multiple_context_handles_retain_the_same_active_device() {
    let first = Context::new(0).unwrap();
    let buffer = DeviceBuffer::from_slice(&first, &[7i32, 11]).unwrap();
    let second = Context::new(0).unwrap();
    let other = DeviceBuffer::from_slice(&second, &[13i32, 17]).unwrap();
    drop(second);
    assert_eq!(buffer.read().unwrap(), [7, 11]);
    assert_eq!(other.read().unwrap(), [13, 17]);
    drop(first);
    assert_eq!(buffer.read().unwrap(), [7, 11]);
}

#[test]
fn safe_driver_operations_bind_their_context_after_moving_threads() {
    let context = Context::new(0).unwrap();
    let (mut buffer, stream, first, last, module) = std::thread::spawn(move || {
        let buffer = DeviceBuffer::from_slice(&context, &[1i32, 2, 3]).unwrap();
        let stream = Stream::new(&context).unwrap();
        let first = Event::new(&context).unwrap();
        let last = Event::new(&context).unwrap();
        let module = Module::load(&context, include_str!("../../ptx/norm.ptx")).unwrap();
        (buffer, stream, first, last, module)
    })
    .join()
    .unwrap();
    std::thread::spawn(move || {
        assert_eq!(buffer.read().unwrap(), [1, 2, 3]);
        buffer.write(&[4, 5, 6]).unwrap();
        assert_eq!(buffer.read().unwrap(), [4, 5, 6]);
        let function = module.function("layernorm_384").unwrap();
        function.set_max_dynamic_shared(0).unwrap();
        first.record(&stream).unwrap();
        stream.wait(&first).unwrap();
        last.record(&stream).unwrap();
        stream.synchronize().unwrap();
        assert!(first.elapsed_ms(&last).unwrap() >= 0.0);
    })
    .join()
    .unwrap();
}
