use super::{CudaError, STREAM_BYTES, load_optional_stream};
use std::cell::RefCell;

#[test]
fn insufficient_shared_memory_skips_the_query_and_optional_loader() {
    for limit in [0, 90_000, STREAM_BYTES - 1] {
        let result = load_optional_stream::<()>(
            STREAM_BYTES,
            limit,
            || panic!("a known resource failure needs no capability query"),
            || panic!("an unavailable plan must not configure a function"),
        );
        assert!(matches!(result, Ok(None)), "limit={limit}");
    }
}

#[test]
fn unsupported_cooperative_launch_skips_the_optional_loader() {
    let events = RefCell::new(Vec::new());
    let result = load_optional_stream::<()>(
        STREAM_BYTES,
        STREAM_BYTES,
        || {
            events.borrow_mut().push("query");
            Ok(false)
        },
        || panic!("an unavailable plan must not configure a function"),
    );
    assert!(matches!(result, Ok(None)));
    assert_eq!(*events.borrow(), ["query"]);
}

#[test]
fn supported_limits_load_once_after_the_capability_query() {
    for limit in [STREAM_BYTES, STREAM_BYTES + 1, u32::MAX] {
        let events = RefCell::new(Vec::new());
        let result = load_optional_stream(
            STREAM_BYTES,
            limit,
            || {
                events.borrow_mut().push("query");
                Ok(true)
            },
            || {
                events.borrow_mut().push("load");
                Ok(Some(17))
            },
        )
        .unwrap();
        assert_eq!(result, Some(17), "limit={limit}");
        assert_eq!(*events.borrow(), ["query", "load"]);
    }
}

fn error(call: &'static str, code: i32) -> CudaError {
    CudaError {
        call,
        code,
        name: format!("synthetic driver error {code}"),
    }
}

#[test]
fn capability_query_errors_propagate_before_loading() {
    let actual = load_optional_stream::<()>(
        STREAM_BYTES,
        STREAM_BYTES,
        || Err(error("cuDeviceGetAttribute", 999)),
        || panic!("a failed query must not configure a function"),
    )
    .unwrap_err();
    assert_eq!(actual.call, "cuDeviceGetAttribute");
    assert_eq!(actual.code, 999);
    assert_eq!(actual.name, "synthetic driver error 999");
}

#[test]
fn optional_loader_errors_are_not_reclassified_as_unavailable() {
    for (call, code) in [
        ("cuFuncSetAttribute", 1),
        ("cuOccupancyMaxActiveBlocksPerMultiprocessor", 999),
        ("cuMemAlloc_v2", 2),
    ] {
        let actual = load_optional_stream::<()>(
            STREAM_BYTES,
            STREAM_BYTES,
            || Ok(true),
            || Err(error(call, code)),
        )
        .unwrap_err();
        assert_eq!(actual.call, call);
        assert_eq!(actual.code, code);
        assert_eq!(actual.name, format!("synthetic driver error {code}"));
    }
}

#[test]
fn an_unresident_optional_kernel_keeps_the_existing_fallback() {
    let result = load_optional_stream::<()>(STREAM_BYTES, STREAM_BYTES, || Ok(true), || Ok(None));
    assert!(matches!(result, Ok(None)));
}
