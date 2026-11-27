//! The CUDA driver API, loaded at runtime from the installed driver.

#[cfg(not(target_family = "wasm"))]
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    fmt,
    marker::PhantomData,
    ptr,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

type CuResult = c_int;
type Handle = *mut c_void;
pub type DevicePtr = u64;

/// A value copied to and from device memory as raw bytes.
///
/// # Safety
/// Every bit pattern must be valid, with no padding, references or pointers.
///
/// Types with restricted bit patterns are refused:
///
/// ```compile_fail
/// use std::num::NonZeroU32;
/// use kodamaembed_cuda::{Context, DeviceBuffer};
/// let context = Context::new(0).unwrap();
/// let _ = DeviceBuffer::<NonZeroU32>::zeroed(&context, 1);
/// ```
pub unsafe trait DeviceRepr: Copy {}

macro_rules! device_repr {
    ($($ty:ty),* $(,)?) => {
        $(// SAFETY: these numeric types have no padding and accept every bit pattern.
        unsafe impl DeviceRepr for $ty {})*
    };
}

device_repr!(
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    half::f16,
    half::bf16
);
// SAFETY: arrays introduce no padding and inherit their elements' validity.
unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {}

const SUCCESS: CuResult = 0;
/// cuEventQuery's answer for work still running.
const NOT_READY: CuResult = 600;
const MULTIPROCESSOR_COUNT: c_int = 16;
const L2_CACHE_SIZE: c_int = 38;
const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const COMPUTE_CAPABILITY_MINOR: c_int = 76;
const COOPERATIVE_LAUNCH: c_int = 95;
const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
const FUNC_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
const FUNC_PREFERRED_SHARED_MEMORY_CARVEOUT: c_int = 9;
/// The largest share of a multiprocessor's memory for shared memory, in percent.
const CARVEOUT_MAX_SHARED: c_int = 100;
/// `DeviceBuffer`'s memsets and copies run on the legacy default stream and can
/// still be in flight when their calls return.
const STREAM_ORDERED_WITH_LEGACY: c_uint = 0;
const STREAM_NON_BLOCKING: c_uint = 1;
/// Avoids CPU spinning while waiting for GPU work.
const SYNC_POLICY_BLOCKING: c_int = 4;
const STREAM_ATTRIBUTE_SYNC_POLICY: c_int = 3;
/// A capture that only restricts the capturing thread's own calls.
const STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
/// Disables timestamps for events used only to poll completion.
const EVENT_DISABLE_TIMING: c_uint = 2;

#[cfg(windows)]
const LIBRARY: &str = "nvcuda.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "libcuda.so.1";

#[cfg(windows)]
const NVML_LIBRARY: &str = "nvml.dll";
#[cfg(not(windows))]
const NVML_LIBRARY: &str = "libnvidia-ml.so.1";

/// A wasm target opens no dynamic library: it is a machine without the driver.
#[cfg(target_family = "wasm")]
struct Library;

#[cfg(target_family = "wasm")]
impl Library {
    const ABSENT: &str = "this target opens no dynamic libraries";

    unsafe fn new(_name: &str) -> Result<Self, &'static str> {
        Err(Self::ABSENT)
    }

    unsafe fn get<T>(&self, _symbol: &[u8]) -> Result<&T, &'static str> {
        Err(Self::ABSENT)
    }
}

/// Optional vendor release metadata; inference depends only on the CUDA driver.
fn vendor_driver_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let read = || -> Option<String> {
            // SAFETY: NVIDIA's system library; its pointers stay within its lifetime.
            let library = unsafe { Library::new(NVML_LIBRARY) }.ok()?;
            // SAFETY: these are the exact published NVML C signatures.
            let (init, shutdown, get_version) = unsafe {
                (
                    symbol::<unsafe extern "C" fn() -> c_int>(&library, "nvmlInit_v2").ok()?,
                    symbol::<unsafe extern "C" fn() -> c_int>(&library, "nvmlShutdown").ok()?,
                    symbol::<unsafe extern "C" fn(*mut c_char, c_uint) -> c_int>(
                        &library,
                        "nvmlSystemGetDriverVersion",
                    )
                    .ok()?,
                )
            };
            // SAFETY: init/shutdown are reference counted; another caller's stays.
            if unsafe { init() } != SUCCESS {
                return None;
            }
            let mut version = [0u8; 256];
            // SAFETY: the destination has the stated byte capacity.
            let result =
                unsafe { get_version(version.as_mut_ptr().cast(), version.len() as c_uint) };
            // SAFETY: balances this call's successful NVML initialization.
            unsafe { shutdown() };
            if result != SUCCESS {
                return None;
            }
            CStr::from_bytes_until_nul(&version)
                .ok()
                .map(|version| version.to_string_lossy().into_owned())
        };
        read().unwrap_or_else(|| "unknown (NVML unavailable)".into())
    })
}

#[derive(Debug, Clone)]
pub struct CudaError {
    pub call: &'static str,
    pub code: i32,
    pub name: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} ({})", self.call, self.name, self.code)
    }
}

impl std::error::Error for CudaError {}

/// cuda.h CUlaunchAttribute: id, padding and a 64-byte value union aligned as its pointers.
#[repr(C, align(8))]
struct LaunchAttribute {
    id: c_int,
    pad: [u8; 4],
    value: AttributeValue,
}

/// The driver's CUlaunchAttributeValue union, shared by stream and launch attributes.
#[repr(C, align(8))]
struct AttributeValue([u8; 64]);

impl AttributeValue {
    fn integer(value: c_int) -> Self {
        let mut bytes = [0; 64];
        bytes[..4].copy_from_slice(&value.to_ne_bytes());
        Self(bytes)
    }
}

/// cuda.h CUlaunchConfig.
#[repr(C)]
struct LaunchConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_bytes: c_uint,
    stream: Handle,
    attributes: *mut LaunchAttribute,
    attribute_count: c_uint,
}

/// cuda.h CU_LAUNCH_ATTRIBUTE_COOPERATIVE.
const LAUNCH_ATTRIBUTE_COOPERATIVE: c_int = 2;

struct Api {
    _library: Library,
    init: unsafe extern "system" fn(c_uint) -> CuResult,
    driver_get_version: unsafe extern "system" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "system" fn(*mut c_int, c_int) -> CuResult,
    device_get_name: unsafe extern "system" fn(*mut c_char, c_int, c_int) -> CuResult,
    device_get_attribute: unsafe extern "system" fn(*mut c_int, c_int, c_int) -> CuResult,
    primary_ctx_retain: unsafe extern "system" fn(*mut Handle, c_int) -> CuResult,
    primary_ctx_release: unsafe extern "system" fn(c_int) -> CuResult,
    ctx_set_current: unsafe extern "system" fn(Handle) -> CuResult,
    ctx_get_current: unsafe extern "system" fn(*mut Handle) -> CuResult,
    ctx_push_current: unsafe extern "system" fn(Handle) -> CuResult,
    ctx_pop_current: unsafe extern "system" fn(*mut Handle) -> CuResult,
    module_load_data: unsafe extern "system" fn(*mut Handle, *const c_void) -> CuResult,
    module_unload: unsafe extern "system" fn(Handle) -> CuResult,
    module_get_function: unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> CuResult,
    func_set_attribute: unsafe extern "system" fn(Handle, c_int, c_int) -> CuResult,
    occupancy_max_active_blocks:
        unsafe extern "system" fn(*mut c_int, Handle, c_int, usize) -> CuResult,
    launch_kernel: unsafe extern "system" fn(
        Handle,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    launch_kernel_ex: unsafe extern "system" fn(
        *const LaunchConfig,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    mem_alloc: unsafe extern "system" fn(*mut DevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "system" fn(DevicePtr) -> CuResult,
    mem_get_info: unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult,
    mem_alloc_host: unsafe extern "system" fn(*mut *mut c_void, usize) -> CuResult,
    mem_free_host: unsafe extern "system" fn(*mut c_void) -> CuResult,
    mem_host_get_device_pointer:
        unsafe extern "system" fn(*mut DevicePtr, *mut c_void, c_uint) -> CuResult,
    memcpy_htod: unsafe extern "system" fn(DevicePtr, *const c_void, usize) -> CuResult,
    memcpy_htod_async:
        unsafe extern "system" fn(DevicePtr, *const c_void, usize, Handle) -> CuResult,
    memcpy_dtoh: unsafe extern "system" fn(*mut c_void, DevicePtr, usize) -> CuResult,
    memcpy_dtoh_async: unsafe extern "system" fn(*mut c_void, DevicePtr, usize, Handle) -> CuResult,
    memset_d8: unsafe extern "system" fn(DevicePtr, u8, usize) -> CuResult,
    stream_create: unsafe extern "system" fn(*mut Handle, c_uint) -> CuResult,
    stream_destroy: unsafe extern "system" fn(Handle) -> CuResult,
    stream_synchronize: unsafe extern "system" fn(Handle) -> CuResult,
    stream_set_attribute:
        unsafe extern "system" fn(Handle, c_int, *const AttributeValue) -> CuResult,
    stream_wait_event: unsafe extern "system" fn(Handle, Handle, c_uint) -> CuResult,
    stream_begin_capture: unsafe extern "system" fn(Handle, c_int) -> CuResult,
    stream_end_capture: unsafe extern "system" fn(Handle, *mut Handle) -> CuResult,
    graph_get_root_nodes: unsafe extern "system" fn(Handle, *mut Handle, *mut usize) -> CuResult,
    graph_instantiate: unsafe extern "system" fn(*mut Handle, Handle, u64) -> CuResult,
    graph_upload: unsafe extern "system" fn(Handle, Handle) -> CuResult,
    graph_launch: unsafe extern "system" fn(Handle, Handle) -> CuResult,
    graph_exec_destroy: unsafe extern "system" fn(Handle) -> CuResult,
    graph_destroy: unsafe extern "system" fn(Handle) -> CuResult,
    event_create: unsafe extern "system" fn(*mut Handle, c_uint) -> CuResult,
    event_destroy: unsafe extern "system" fn(Handle) -> CuResult,
    event_record: unsafe extern "system" fn(Handle, Handle) -> CuResult,
    event_query: unsafe extern "system" fn(Handle) -> CuResult,
    event_elapsed_time: unsafe extern "system" fn(*mut f32, Handle, Handle) -> CuResult,
    get_error_name: unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult,
}

/// # Safety
/// `T` must be the exact function pointer type of the exported symbol.
unsafe fn symbol<T: Copy>(library: &Library, name: &str) -> Result<T, String> {
    let symbol = format!("{name}\0");
    // SAFETY: the caller guarantees `T` matches the symbol's signature.
    unsafe { library.get::<T>(symbol.as_bytes()) }
        .map(|found| *found)
        .map_err(|error| format!("{name}: {error}"))
}

impl Api {
    fn load() -> Result<Self, CudaError> {
        // SAFETY: loading the vendor driver runs only its initializers.
        let library = unsafe { Library::new(LIBRARY) }
            .map_err(|error| invalid("open driver", &format!("{LIBRARY}: {error}")))?;
        Self::from_library(library).map_err(|reason| invalid("load driver", &reason))
    }

    fn from_library(library: Library) -> Result<Self, String> {
        // SAFETY: each signature is the published one for its versioned export.
        unsafe {
            Ok(Self {
                init: symbol(&library, "cuInit")?,
                driver_get_version: symbol(&library, "cuDriverGetVersion")?,
                device_get: symbol(&library, "cuDeviceGet")?,
                device_get_name: symbol(&library, "cuDeviceGetName")?,
                device_get_attribute: symbol(&library, "cuDeviceGetAttribute")?,
                primary_ctx_retain: symbol(&library, "cuDevicePrimaryCtxRetain")?,
                primary_ctx_release: symbol(&library, "cuDevicePrimaryCtxRelease_v2")?,
                ctx_set_current: symbol(&library, "cuCtxSetCurrent")?,
                ctx_get_current: symbol(&library, "cuCtxGetCurrent")?,
                ctx_push_current: symbol(&library, "cuCtxPushCurrent_v2")?,
                ctx_pop_current: symbol(&library, "cuCtxPopCurrent_v2")?,
                module_load_data: symbol(&library, "cuModuleLoadData")?,
                module_unload: symbol(&library, "cuModuleUnload")?,
                module_get_function: symbol(&library, "cuModuleGetFunction")?,
                func_set_attribute: symbol(&library, "cuFuncSetAttribute")?,
                occupancy_max_active_blocks: symbol(
                    &library,
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                )?,
                launch_kernel: symbol(&library, "cuLaunchKernel")?,
                launch_kernel_ex: symbol(&library, "cuLaunchKernelEx")?,
                mem_alloc: symbol(&library, "cuMemAlloc_v2")?,
                mem_free: symbol(&library, "cuMemFree_v2")?,
                mem_get_info: symbol(&library, "cuMemGetInfo_v2")?,
                mem_alloc_host: symbol(&library, "cuMemAllocHost_v2")?,
                mem_free_host: symbol(&library, "cuMemFreeHost")?,
                mem_host_get_device_pointer: symbol(&library, "cuMemHostGetDevicePointer_v2")?,
                memcpy_htod: symbol(&library, "cuMemcpyHtoD_v2")?,
                memcpy_htod_async: symbol(&library, "cuMemcpyHtoDAsync_v2")?,
                memcpy_dtoh: symbol(&library, "cuMemcpyDtoH_v2")?,
                memcpy_dtoh_async: symbol(&library, "cuMemcpyDtoHAsync_v2")?,
                memset_d8: symbol(&library, "cuMemsetD8_v2")?,
                stream_create: symbol(&library, "cuStreamCreate")?,
                stream_destroy: symbol(&library, "cuStreamDestroy_v2")?,
                stream_synchronize: symbol(&library, "cuStreamSynchronize")?,
                stream_set_attribute: symbol(&library, "cuStreamSetAttribute")?,
                stream_wait_event: symbol(&library, "cuStreamWaitEvent")?,
                stream_begin_capture: symbol(&library, "cuStreamBeginCapture_v2")?,
                stream_end_capture: symbol(&library, "cuStreamEndCapture")?,
                graph_get_root_nodes: symbol(&library, "cuGraphGetRootNodes")?,
                graph_instantiate: symbol(&library, "cuGraphInstantiateWithFlags")?,
                graph_upload: symbol(&library, "cuGraphUpload")?,
                graph_launch: symbol(&library, "cuGraphLaunch")?,
                graph_exec_destroy: symbol(&library, "cuGraphExecDestroy")?,
                graph_destroy: symbol(&library, "cuGraphDestroy")?,
                event_create: symbol(&library, "cuEventCreate")?,
                event_destroy: symbol(&library, "cuEventDestroy_v2")?,
                event_record: symbol(&library, "cuEventRecord")?,
                event_query: symbol(&library, "cuEventQuery")?,
                event_elapsed_time: symbol(&library, "cuEventElapsedTime")?,
                get_error_name: symbol(&library, "cuGetErrorName")?,
                _library: library,
            })
        }
    }
}

fn api() -> Result<&'static Api, CudaError> {
    static API: OnceLock<Result<Api, CudaError>> = OnceLock::new();
    API.get_or_init(Api::load).as_ref().map_err(Clone::clone)
}

fn check(call: &'static str, result: CuResult) -> Result<(), CudaError> {
    if result == SUCCESS {
        return Ok(());
    }
    let mut name: *const c_char = ptr::null();
    let name = match api() {
        // SAFETY: cuGetErrorName writes a pointer to a static string or fails.
        Ok(api) if unsafe { (api.get_error_name)(result, &mut name) } == SUCCESS => {
            // SAFETY: the driver returned a valid NUL-terminated static string.
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        }
        _ => "unknown error".into(),
    };
    Err(CudaError {
        call,
        code: result,
        name,
    })
}

struct ContextInner {
    device: c_int,
    handle: Handle,
}

// SAFETY: a primary context handle may be made current on any thread.
unsafe impl Send for ContextInner {}
// SAFETY: see above; the driver API is thread-safe.
unsafe impl Sync for ContextInner {}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            // SAFETY: releases the reference retained in `Context::new`.
            unsafe { (api.primary_ctx_release)(self.device) };
        }
    }
}

/// A device's primary context. Every resource keeps its context alive.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
    execution_identity: String,
    pub name: String,
    pub sm_count: u32,
    pub l2_cache_bytes: u32,
    pub compute_capability: (u32, u32),
    pub max_shared_per_block: u32,
}

pub(crate) struct CurrentContext {
    pop: Option<unsafe extern "system" fn(*mut Handle) -> CuResult>,
    // The context stack belongs to the thread that pushed it.
    _thread: PhantomData<*mut c_void>,
}

impl Drop for CurrentContext {
    fn drop(&mut self) {
        if let Some(pop) = self.pop {
            let mut popped = ptr::null_mut();
            // SAFETY: enter pushed one context on this thread and the driver stays loaded.
            unsafe { pop(&mut popped) };
        }
    }
}

impl Context {
    /// Compute capability without activating a primary context. A missing driver or device yields None.
    pub fn capability(ordinal: usize) -> Result<Option<(u32, u32)>, CudaError> {
        let device = match Self::device(ordinal) {
            Ok(device) => device,
            // A stub library, missing device, or driver without the required exports cannot run kernels.
            Err(error)
                if matches!(error.call, "open driver" | "load driver")
                    || matches!(error.code, 34 | 100 | 101) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        Ok(Some(Self::device_capability(device)?))
    }

    fn device(ordinal: usize) -> Result<c_int, CudaError> {
        let ordinal = c_int::try_from(ordinal)
            .map_err(|_| invalid("cuDeviceGet", "device ordinal exceeds i32"))?;
        let api = api()?;
        let mut device = 0;
        // SAFETY: device is a valid output pointer.
        unsafe {
            check("cuInit", (api.init)(0))?;
            check("cuDeviceGet", (api.device_get)(&mut device, ordinal))?;
        }
        Ok(device)
    }

    fn attribute(device: c_int, attribute: c_int) -> Result<u32, CudaError> {
        let mut value = 0;
        // SAFETY: value is a valid output pointer.
        check("cuDeviceGetAttribute", unsafe {
            (api()?.device_get_attribute)(&mut value, attribute, device)
        })?;
        Ok(value as u32)
    }

    fn device_capability(device: c_int) -> Result<(u32, u32), CudaError> {
        Ok((
            Self::attribute(device, COMPUTE_CAPABILITY_MAJOR)?,
            Self::attribute(device, COMPUTE_CAPABILITY_MINOR)?,
        ))
    }

    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let device = Self::device(ordinal)?;
        let api = api()?;
        let mut handle = ptr::null_mut();
        let mut name = [0 as c_char; 256];
        // SAFETY: handle is a valid output pointer.
        unsafe {
            check(
                "cuDevicePrimaryCtxRetain",
                (api.primary_ctx_retain)(&mut handle, device),
            )?;
        }
        let inner = Arc::new(ContextInner { device, handle });
        // SAFETY: the buffer holds the length passed and the driver NUL-terminates it.
        unsafe {
            check(
                "cuDeviceGetName",
                (api.device_get_name)(name.as_mut_ptr(), 256, device),
            )?;
        }
        // SAFETY: the driver wrote a NUL-terminated name.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let compute_capability = Self::device_capability(device)?;
        let mut driver_version = 0;
        // SAFETY: `driver_version` is a valid out-pointer.
        check("cuDriverGetVersion", unsafe {
            (api.driver_get_version)(&mut driver_version)
        })?;
        Ok(Self {
            execution_identity: format!(
                "cuda;device={name};cc={}.{};driver-api={driver_version};vendor-driver={}",
                compute_capability.0,
                compute_capability.1,
                vendor_driver_version()
            ),
            name,
            sm_count: Self::attribute(device, MULTIPROCESSOR_COUNT)?,
            l2_cache_bytes: Self::attribute(device, L2_CACHE_SIZE)?,
            compute_capability,
            max_shared_per_block: Self::attribute(device, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
            inner,
        })
    }

    pub(crate) fn enter(&self) -> Result<CurrentContext, CudaError> {
        let api = api()?;
        let mut current = ptr::null_mut();
        // SAFETY: current is a valid output pointer for this thread's context.
        check("cuCtxGetCurrent", unsafe {
            (api.ctx_get_current)(&mut current)
        })?;
        let pop = if current == self.inner.handle {
            None
        } else {
            // SAFETY: self retains the primary context.
            check("cuCtxPushCurrent", unsafe {
                (api.ctx_push_current)(self.inner.handle)
            })?;
            Some(api.ctx_pop_current)
        };
        Ok(CurrentContext {
            pop,
            _thread: PhantomData,
        })
    }

    pub(crate) fn cooperative_launch_supported(&self) -> Result<bool, CudaError> {
        let mut supported = 0;
        // SAFETY: `supported` is a valid out-pointer for the selected device.
        check("cuDeviceGetAttribute", unsafe {
            (api()?.device_get_attribute)(&mut supported, COOPERATIVE_LAUNCH, self.inner.device)
        })?;
        Ok(supported != 0)
    }

    /// This device's UUID; a MIG UUID names an instance, not a physical board.
    pub fn device_uuid(&self) -> Result<[u8; 16], CudaError> {
        #[repr(C)]
        struct Uuid {
            bytes: [u8; 16],
        }
        let api = api()?;
        // SAFETY: CUuuid is 16 bytes, and the signature is cuDeviceGetUuid_v2's.
        let get_uuid = unsafe {
            symbol::<unsafe extern "system" fn(*mut Uuid, c_int) -> CuResult>(
                &api._library,
                "cuDeviceGetUuid_v2",
            )
        }
        .map_err(|error| invalid("cuDeviceGetUuid_v2", &error))?;
        let mut uuid = Uuid { bytes: [0; 16] };
        // SAFETY: the device is this context's and the output outlives the call.
        check("cuDeviceGetUuid_v2", unsafe {
            get_uuid(&mut uuid, self.inner.device)
        })?;
        Ok(uuid.bytes)
    }

    /// Makes this context current on the calling thread.
    pub fn bind(&self) -> Result<(), CudaError> {
        // SAFETY: the handle is a retained primary context.
        check("cuCtxSetCurrent", unsafe {
            (api()?.ctx_set_current)(self.inner.handle)
        })
    }

    /// Free and total device memory in bytes, as the driver reports them.
    pub fn memory(&self) -> Result<(usize, usize), CudaError> {
        let _current = self.enter()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: both out-pointers are valid for the call.
        check("cuMemGetInfo", unsafe {
            (api()?.mem_get_info)(&mut free, &mut total)
        })?;
        Ok((free, total))
    }

    /// The driver API version and, through optional NVML, the vendor release.
    pub fn execution_identity(&self) -> String {
        self.execution_identity.clone()
    }

    fn same(&self, other: &Self) -> bool {
        self.inner.handle == other.inner.handle
    }
}

fn invalid(call: &'static str, message: &str) -> CudaError {
    CudaError {
        call,
        code: -1,
        name: message.into(),
    }
}

fn buffer_bytes<T>(len: usize) -> Result<usize, CudaError> {
    if size_of::<T>() == 0 {
        return Err(invalid(
            "device buffer",
            "zero-sized elements are unsupported",
        ));
    }
    len.checked_mul(size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            invalid(
                "device buffer",
                "element count exceeds the addressable byte size",
            )
        })
}

/// Device memory holding `len` values of `T`: its own allocation, or a view
/// into another buffer's (`alias`), which it never frees.
pub struct DeviceBuffer<T> {
    ptr: DevicePtr,
    len: usize,
    context: Context,
    owned: bool,
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Allocates device storage; callers initialize it before exposing it.
    fn allocate(context: &Context, len: usize) -> Result<Self, CudaError> {
        let bytes = buffer_bytes::<T>(len)?;
        let _current = context.enter()?;
        let api = api()?;
        let mut ptr = 0;
        if bytes > 0 {
            // SAFETY: `ptr` is a valid out-pointer; the allocation is `bytes` long.
            unsafe {
                check("cuMemAlloc", (api.mem_alloc)(&mut ptr, bytes))?;
            }
        }
        Ok(Self {
            ptr,
            len,
            context: context.clone(),
            owned: true,
            _marker: PhantomData,
        })
    }

    /// `len` values at `byte_offset` (a multiple of 256) of `arena`'s storage.
    ///
    /// # Safety
    /// `arena` outlives the view, and no concurrent work writes and reads
    /// overlapping bytes.
    pub unsafe fn alias<A: DeviceRepr>(
        arena: &DeviceBuffer<A>,
        byte_offset: usize,
        len: usize,
    ) -> Self {
        let bytes = buffer_bytes::<T>(len).expect("an addressable view");
        assert!(
            byte_offset.is_multiple_of(256) && byte_offset + bytes <= arena.len * size_of::<A>(),
            "a view inside its arena"
        );
        Self {
            ptr: arena.ptr + byte_offset as DevicePtr,
            len,
            context: arena.context.clone(),
            owned: false,
            _marker: PhantomData,
        }
    }

    pub fn zeroed(context: &Context, len: usize) -> Result<Self, CudaError> {
        let buffer = Self::allocate(context, len)?;
        let bytes = buffer_bytes::<T>(len)?;
        if bytes > 0 {
            let _current = context.enter()?;
            // SAFETY: `buffer` owns `bytes`, zero is valid for T, Drop frees on failure.
            check("cuMemsetD8", unsafe {
                (api()?.memset_d8)(buffer.ptr, 0, bytes)
            })?;
        }
        Ok(buffer)
    }

    pub fn from_slice(context: &Context, values: &[T]) -> Result<Self, CudaError> {
        let mut buffer = Self::allocate(context, values.len())?;
        buffer.write(values)?;
        Ok(buffer)
    }

    pub fn write(&mut self, values: &[T]) -> Result<(), CudaError> {
        assert!(
            values.len() <= self.len,
            "write past the end of a device buffer"
        );
        if values.is_empty() {
            return Ok(());
        }
        let _current = self.context.enter()?;
        // SAFETY: the source holds `values.len()` values and the destination at least that many.
        check("cuMemcpyHtoD", unsafe {
            (api()?.memcpy_htod)(self.ptr, values.as_ptr().cast(), size_of_val(values))
        })
    }

    /// Copies the first `count` values of `host` here in `stream` order, not waiting.
    ///
    /// # Safety
    /// `host` is neither written nor dropped until `stream` has run the copy.
    pub unsafe fn write_first_async(
        &self,
        count: usize,
        host: &HostBuffer<T>,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        assert!(
            count <= self.len && count <= host.len,
            "a copy within both buffers"
        );
        if !self.context.same(&stream.context) {
            return Err(invalid(
                "cuMemcpyHtoDAsync",
                "buffer and stream belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        // SAFETY: both hold `count` values; the caller keeps `host` until the copy.
        check("cuMemcpyHtoDAsync", unsafe {
            (api()?.memcpy_htod_async)(
                self.ptr,
                host.ptr.cast(),
                count * size_of::<T>(),
                stream.handle,
            )
        })
    }

    pub fn read(&self) -> Result<Vec<T>, CudaError> {
        self.read_first(self.len)
    }

    pub fn read_first(&self, count: usize) -> Result<Vec<T>, CudaError> {
        assert!(count <= self.len, "read past end");
        let mut values: Vec<T> = Vec::with_capacity(count);
        if count > 0 {
            let _current = self.context.enter()?;
            // SAFETY: the destination has capacity for `count` values, all written by the copy.
            unsafe {
                check(
                    "cuMemcpyDtoH",
                    (api()?.memcpy_dtoh)(
                        values.as_mut_ptr().cast(),
                        self.ptr,
                        count * size_of::<T>(),
                    ),
                )?;
                values.set_len(count);
            }
        }
        Ok(values)
    }

    /// Copies the first `count` values into `host` in `stream` order, not waiting.
    ///
    /// # Safety
    /// `host` is neither read nor dropped until `stream` has run the copy.
    pub unsafe fn read_first_async(
        &self,
        count: usize,
        host: &mut HostBuffer<T>,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        assert!(
            count <= self.len && count <= host.len,
            "a copy within both buffers"
        );
        if !self.context.same(&stream.context) {
            return Err(invalid(
                "cuMemcpyDtoHAsync",
                "buffer and stream belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        // SAFETY: both hold `count` values; the caller keeps `host` until the copy.
        check("cuMemcpyDtoHAsync", unsafe {
            (api()?.memcpy_dtoh_async)(
                host.ptr.cast(),
                self.ptr,
                count * size_of::<T>(),
                stream.handle,
            )
        })
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Page-locked host memory the device copies into without staging.
pub struct HostBuffer<T> {
    ptr: *mut T,
    len: usize,
    context: Context,
}

impl<T: DeviceRepr> HostBuffer<T> {
    /// `len` zeroed values.
    pub fn zeroed(context: &Context, len: usize) -> Result<Self, CudaError> {
        let bytes = buffer_bytes::<T>(len)?.max(1);
        let _current = context.enter()?;
        let mut ptr = ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer; the allocation is `bytes` long.
        check("cuMemAllocHost", unsafe {
            (api()?.mem_alloc_host)(&mut ptr, bytes)
        })?;
        let buffer = Self {
            ptr: ptr.cast(),
            len,
            context: context.clone(),
        };
        if !buffer.ptr.is_aligned() {
            return Err(invalid(
                "cuMemAllocHost",
                "host allocation does not satisfy element alignment",
            ));
        }
        // SAFETY: the allocation holds `bytes`, and zero is valid for T.
        unsafe { ptr::write_bytes(ptr.cast::<u8>(), 0, bytes) };
        Ok(buffer)
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the allocation holds `len` aligned, initialized values.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `len` aligned, initialized values, borrowed uniquely via `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// The device's address of the buffer, for kernels that read it in place.
    pub fn device_ptr(&self) -> Result<DevicePtr, CudaError> {
        let _current = self.context.enter()?;
        let mut ptr = 0;
        // SAFETY: a valid out-pointer to the context's page-locked memory.
        check("cuMemHostGetDevicePointer", unsafe {
            (api()?.mem_host_get_device_pointer)(&mut ptr, self.ptr.cast(), 0)
        })?;
        Ok(ptr)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// SAFETY: the buffer owns its memory, freeable from any thread with the context.
unsafe impl<T: Send> Send for HostBuffer<T> {}

/// Values a kernel stores with a mark in one 64-bit store while the host polls, so no
/// fence orders them before a flag and they arrive ahead of the stream's completion.
pub struct HostMarked {
    pairs: HostBuffer<u64>,
}

impl HostMarked {
    /// `len` cleared pairs.
    pub fn new(context: &Context, len: usize) -> Result<Self, CudaError> {
        Ok(Self {
            pairs: HostBuffer::zeroed(context, len)?,
        })
    }

    /// The device's address of the pairs.
    pub fn device_ptr(&self) -> Result<DevicePtr, CudaError> {
        self.pairs.device_ptr()
    }

    fn pair(&self, i: usize) -> &AtomicU64 {
        assert!(i < self.pairs.len, "pair {i} past {}", self.pairs.len);
        // SAFETY: the aligned pair outlives the borrow; the host uses only
        // atomics and the device stores only whole pairs.
        unsafe { AtomicU64::from_ptr(self.pairs.ptr.add(i)) }
    }

    /// Launching the marking work afterwards orders it after these stores.
    pub fn clear(&self, n: usize) {
        for i in 0..n {
            self.pair(i).store(0, Ordering::Relaxed);
        }
    }

    /// Whether the first `n` pairs are marked.
    pub fn all_marked(&self, n: usize) -> bool {
        (0..n).all(|i| self.pair(i).load(Ordering::Acquire) >> 32 != 0)
    }

    /// Valid once marked or once the marking work has run.
    pub fn values(&self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| f32::from_bits(self.pair(i).load(Ordering::Acquire) as u32))
            .collect()
    }
}

impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        if let Ok(api) = api()
            && let Ok(_current) = self.context.enter()
        {
            // SAFETY: frees the allocation made in `zeroed`.
            unsafe { (api.mem_free_host)(self.ptr.cast()) };
        }
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.owned
            && self.ptr != 0
            && let Ok(api) = api()
            && let Ok(_current) = self.context.enter()
        {
            // SAFETY: frees the allocation made in `zeroed`.
            unsafe { (api.mem_free)(self.ptr) };
        }
    }
}

pub struct Stream {
    handle: Handle,
    context: Context,
}

// SAFETY: the driver API is thread-safe; `drop` binds the context first.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new(context: &Context) -> Result<Self, CudaError> {
        Self::with_flags(context, STREAM_ORDERED_WITH_LEGACY)
    }

    /// Unordered with the legacy default stream, so other threads' copies
    /// cannot meet a capture.
    pub fn new_non_blocking(context: &Context) -> Result<Self, CudaError> {
        Self::with_flags(context, STREAM_NON_BLOCKING)
    }

    fn with_flags(context: &Context, flags: c_uint) -> Result<Self, CudaError> {
        let _current = context.enter()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a valid out-pointer.
        check("cuStreamCreate", unsafe {
            (api()?.stream_create)(&mut handle, flags)
        })?;
        let stream = Self {
            handle,
            context: context.clone(),
        };
        // Keep blocking waits local to this stream. The primary context may belong to another library.
        let policy = AttributeValue::integer(SYNC_POLICY_BLOCKING);
        // SAFETY: stream owns the handle and policy has the driver's union layout.
        check("cuStreamSetAttribute", unsafe {
            (api()?.stream_set_attribute)(handle, STREAM_ATTRIBUTE_SYNC_POLICY, &policy)
        })?;
        Ok(stream)
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        let _current = self.context.enter()?;
        // SAFETY: the handle is a live stream.
        check("cuStreamSynchronize", unsafe {
            (api()?.stream_synchronize)(self.handle)
        })
    }

    /// Orders this stream's later work after `event`'s record; inside a capture
    /// the record's stream and this one then record into one graph.
    pub fn wait(&self, event: &Event) -> Result<(), CudaError> {
        if !self.context.same(&event.context) {
            return Err(invalid(
                "cuStreamWaitEvent",
                "stream and event belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        // SAFETY: both handles are live.
        check("cuStreamWaitEvent", unsafe {
            (api()?.stream_wait_event)(self.handle, event.handle, 0)
        })
    }

    /// Records launches into a graph until [`Self::end_capture`]; only this
    /// thread is restricted.
    pub fn begin_capture(&self) -> Result<(), CudaError> {
        let _current = self.context.enter()?;
        // SAFETY: the handle is a live stream.
        check("cuStreamBeginCapture", unsafe {
            (api()?.stream_begin_capture)(self.handle, STREAM_CAPTURE_MODE_THREAD_LOCAL)
        })
    }

    /// Instantiates the recorded launches as a replayable graph.
    pub fn end_capture(&self) -> Result<Graph, CudaError> {
        let _current = self.context.enter()?;
        let api = api()?;
        let mut graph = ptr::null_mut();
        // SAFETY: the handle is a live stream; `graph` is a valid out-pointer.
        check("cuStreamEndCapture", unsafe {
            (api.stream_end_capture)(self.handle, &mut graph)
        })?;
        let mut roots = 0;
        // SAFETY: the captured graph; without an array the call only counts.
        let counted = check("cuGraphGetRootNodes", unsafe {
            (api.graph_get_root_nodes)(graph, ptr::null_mut(), &mut roots)
        });
        let mut exec = ptr::null_mut();
        let instantiated = counted.and_then(|()| {
            // SAFETY: the captured graph and a valid out-pointer; the instance keeps no reference.
            check("cuGraphInstantiateWithFlags", unsafe {
                (api.graph_instantiate)(&mut exec, graph, 0)
            })
        });
        // SAFETY: destroys the captured graph, which nothing else holds.
        unsafe { (api.graph_destroy)(graph) };
        instantiated?;
        Ok(Graph {
            handle: exec,
            context: self.context.clone(),
            roots,
        })
    }
}

/// A stream's recorded launches, replayed with one call.
pub struct Graph {
    handle: Handle,
    context: Context,
    roots: usize,
}

// SAFETY: as for `Stream`.
unsafe impl Send for Graph {}

impl Graph {
    /// The recorded launches that wait for no other. Each starts a chain of its own, and a
    /// second chain's later launches trail the first's.
    pub fn roots(&self) -> usize {
        self.roots
    }

    /// Readies the launches on the device so the first replay starts sooner.
    pub fn upload(&self, stream: &Stream) -> Result<(), CudaError> {
        if !self.context.same(&stream.context) {
            return Err(invalid(
                "cuGraphUpload",
                "graph and stream belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        // SAFETY: both handles are live; uploading runs no recorded launch.
        check("cuGraphUpload", unsafe {
            (api()?.graph_upload)(self.handle, stream.handle)
        })
    }

    /// Replays the recorded launches on `stream`, ordered as its own work.
    ///
    /// # Safety
    /// Every buffer the launches name must be live and hold what they expect.
    pub unsafe fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        if !self.context.same(&stream.context) {
            return Err(invalid(
                "cuGraphLaunch",
                "graph and stream belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        // SAFETY: both handles are live; the caller vouches for the buffers.
        check("cuGraphLaunch", unsafe {
            (api()?.graph_launch)(self.handle, stream.handle)
        })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let Ok(api) = api()
            && let Ok(_current) = self.context.enter()
        {
            // SAFETY: destroys the instance created in `end_capture`.
            unsafe { (api.graph_exec_destroy)(self.handle) };
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(api) = api()
            && let Ok(_current) = self.context.enter()
        {
            // SAFETY: destroys the stream created in `new`.
            unsafe { (api.stream_destroy)(self.handle) };
        }
    }
}

/// A point in a stream's work, to time between or to poll.
pub struct Event {
    handle: Handle,
    context: Context,
}

// SAFETY: as for `Stream`.
unsafe impl Send for Event {}

impl Event {
    pub fn new(context: &Context) -> Result<Self, CudaError> {
        // Flags 0 keep timing enabled.
        Self::with_flags(context, 0)
    }

    /// An event to poll ([`Self::query`]), not to time.
    pub fn untimed(context: &Context) -> Result<Self, CudaError> {
        Self::with_flags(context, EVENT_DISABLE_TIMING)
    }

    fn with_flags(context: &Context, flags: c_uint) -> Result<Self, CudaError> {
        let _current = context.enter()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a valid out-pointer.
        check("cuEventCreate", unsafe {
            (api()?.event_create)(&mut handle, flags)
        })?;
        Ok(Self {
            handle,
            context: context.clone(),
        })
    }

    /// Whether the work before the last record has run, without waiting.
    pub fn query(&self) -> Result<bool, CudaError> {
        let _current = self.context.enter()?;
        // SAFETY: the handle is a live event.
        match unsafe { (api()?.event_query)(self.handle) } {
            NOT_READY => Ok(false),
            result => check("cuEventQuery", result).map(|()| true),
        }
    }

    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        if !self.context.same(&stream.context) {
            return Err(invalid(
                "cuEventRecord",
                "event and stream belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        // SAFETY: both handles are live.
        check("cuEventRecord", unsafe {
            (api()?.event_record)(self.handle, stream.handle)
        })
    }

    /// Milliseconds of GPU time from `self` to `later`; both must have completed.
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32, CudaError> {
        if !self.context.same(&later.context) {
            return Err(invalid(
                "cuEventElapsedTime",
                "events belong to different contexts",
            ));
        }
        let _current = self.context.enter()?;
        let mut ms = 0.0;
        // SAFETY: `ms` is a valid out-pointer and both events are live.
        check("cuEventElapsedTime", unsafe {
            (api()?.event_elapsed_time)(&mut ms, self.handle, later.handle)
        })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(api) = api()
            && let Ok(_current) = self.context.enter()
        {
            // SAFETY: destroys the event created in `new`.
            unsafe { (api.event_destroy)(self.handle) };
        }
    }
}

struct ModuleInner {
    handle: Handle,
    context: Context,
    /// Whether every function asks for the largest shared-memory share.
    side_by_side: bool,
}

// SAFETY: module handles are usable from any thread with the context current.
unsafe impl Send for ModuleInner {}
// SAFETY: see above.
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Ok(api) = api()
            && let Ok(_current) = self.context.enter()
        {
            // SAFETY: unloads the module loaded in `Module::load`.
            unsafe { (api.module_unload)(self.handle) };
        }
    }
}

/// A PTX module, compiled for the device by the driver when loaded.
#[derive(Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

impl Module {
    pub fn load(context: &Context, ptx: &str) -> Result<Self, CudaError> {
        Self::load_with(context, ptx, false)
    }

    /// For kernels that run on two streams at once: every function asks for the largest
    /// shared-memory share, as kernels preferring different shares never share a multiprocessor.
    pub fn load_side_by_side(context: &Context, ptx: &str) -> Result<Self, CudaError> {
        Self::load_with(context, ptx, true)
    }

    fn load_with(context: &Context, ptx: &str, side_by_side: bool) -> Result<Self, CudaError> {
        let image = CString::new(ptx).map_err(|_| CudaError {
            call: "cuModuleLoadData",
            code: -1,
            name: "PTX contains a NUL byte".into(),
        })?;
        let _current = context.enter()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `image` is a NUL-terminated PTX string and `handle` a valid out-pointer.
        check("cuModuleLoadData", unsafe {
            (api()?.module_load_data)(&mut handle, image.as_ptr().cast())
        })?;
        Ok(Self {
            inner: Arc::new(ModuleInner {
                handle,
                context: context.clone(),
                side_by_side,
            }),
        })
    }

    pub fn function(&self, name: &str) -> Result<Function, CudaError> {
        let _current = self.inner.context.enter()?;
        let function = self.function_alone(name)?;
        if self.inner.side_by_side {
            // SAFETY: the function handle is live.
            check("cuFuncSetAttribute", unsafe {
                (api()?.func_set_attribute)(
                    function.handle,
                    FUNC_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    CARVEOUT_MAX_SHARED,
                )
            })?;
        }
        Ok(function)
    }

    /// A function that never runs beside another stream's kernels: it keeps the driver's
    /// choice of shared memory, and with it the L1 cache the largest share would take.
    pub fn function_alone(&self, name: &str) -> Result<Function, CudaError> {
        let symbol = CString::new(name)
            .map_err(|_| invalid("cuModuleGetFunction", "kernel name contains a NUL byte"))?;
        let _current = self.inner.context.enter()?;
        let mut handle = ptr::null_mut();
        // SAFETY: the module is loaded and `handle` is a valid out-pointer.
        check("cuModuleGetFunction", unsafe {
            (api()?.module_get_function)(&mut handle, self.inner.handle, symbol.as_ptr())
        })?;
        Ok(Function {
            handle,
            _module: self.clone(),
        })
    }
}

/// Kernel arguments, each stored in its own 8-byte slot. The driver reads each
/// parameter's size from the start of its slot (little-endian).
#[derive(Default)]
pub struct Args {
    slots: Vec<u64>,
}

impl Args {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ptr<T>(mut self, buffer: &DeviceBuffer<T>) -> Self {
        self.slots.push(buffer.ptr);
        self
    }

    pub fn raw(mut self, ptr: DevicePtr) -> Self {
        self.slots.push(ptr);
        self
    }

    pub fn i32(mut self, value: i32) -> Self {
        self.slots.push(u64::from(value as u32));
        self
    }

    pub fn f32(mut self, value: f32) -> Self {
        self.slots.push(u64::from(value.to_bits()));
        self
    }
}

pub struct Function {
    handle: Handle,
    _module: Module,
}

// SAFETY: a function launches from any thread with its context current.
unsafe impl Send for Function {}

impl Function {
    /// Allows launches with more than the default 48 KiB of dynamic shared memory.
    pub fn set_max_dynamic_shared(&self, bytes: u32) -> Result<(), CudaError> {
        let bytes = c_int::try_from(bytes)
            .map_err(|_| invalid("cuFuncSetAttribute", "shared memory size exceeds i32"))?;
        let _current = self._module.inner.context.enter()?;
        // SAFETY: the function handle is live.
        check("cuFuncSetAttribute", unsafe {
            (api()?.func_set_attribute)(self.handle, FUNC_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
        })
    }

    /// Blocks of this size one multiprocessor holds at once.
    pub fn resident_blocks(&self, threads: u32, shared_bytes: u32) -> Result<u32, CudaError> {
        let threads = c_int::try_from(threads).map_err(|_| {
            invalid(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                "block size exceeds i32",
            )
        })?;
        let _current = self._module.inner.context.enter()?;
        let mut blocks = 0;
        // SAFETY: `blocks` is a valid out-pointer and the function handle is live.
        check("cuOccupancyMaxActiveBlocksPerMultiprocessor", unsafe {
            (api()?.occupancy_max_active_blocks)(
                &mut blocks,
                self.handle,
                threads,
                shared_bytes as usize,
            )
        })?;
        Ok(blocks as u32)
    }

    /// # Safety
    /// `args` must match the kernel's parameters, its buffers must outlive the
    /// work, and the context, stream and buffers must be the function's.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut Args,
    ) -> Result<(), CudaError> {
        let _current = self._module.inner.context.enter()?;
        let mut params: Vec<*mut c_void> = args
            .slots
            .iter_mut()
            .map(|slot| ptr::from_mut(slot).cast())
            .collect();
        // SAFETY: the caller guarantees the arguments match the kernel.
        check("cuLaunchKernel", unsafe {
            (api()?.launch_kernel)(
                self.handle,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                stream.handle,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        })
    }

    /// Every block resident at once (or the launch fails), so blocks may wait
    /// on each other; a capture keeps the attribute.
    ///
    /// # Safety
    /// As [`Self::launch`].
    pub unsafe fn launch_cooperative(
        &self,
        stream: &Stream,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut Args,
    ) -> Result<(), CudaError> {
        let _current = self._module.inner.context.enter()?;
        let mut params: Vec<*mut c_void> = args
            .slots
            .iter_mut()
            .map(|slot| ptr::from_mut(slot).cast())
            .collect();
        let mut cooperative = LaunchAttribute {
            id: LAUNCH_ATTRIBUTE_COOPERATIVE,
            pad: [0; 4],
            value: AttributeValue::integer(1),
        };
        let config = LaunchConfig {
            grid,
            block,
            shared_bytes,
            stream: stream.handle,
            attributes: &mut cooperative,
            attribute_count: 1,
        };
        // SAFETY: the caller vouches for the arguments; the config outlives the call.
        check("cuLaunchKernelEx", unsafe {
            (api()?.launch_kernel_ex)(&config, self.handle, params.as_mut_ptr(), ptr::null_mut())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_sizes_reject_overflow_before_calling_the_driver() {
        assert_eq!(buffer_bytes::<f32>(17).unwrap(), 68);
        assert!(buffer_bytes::<f32>(usize::MAX / 4 + 1).is_err());
        assert!(buffer_bytes::<u8>(isize::MAX as usize + 1).is_err());
        assert!(buffer_bytes::<[u8; 0]>(1).is_err());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn driver_operations_restore_the_callers_context() {
        let api = api().unwrap();
        let device = Context::device(0).unwrap();
        let current = || {
            let mut handle = ptr::null_mut();
            // SAFETY: handle is a valid output pointer.
            check("cuCtxGetCurrent", unsafe {
                (api.ctx_get_current)(&mut handle)
            })
            .unwrap();
            handle
        };
        let previous = current();
        let mut foreign = ptr::null_mut();
        // SAFETY: signatures match cuda.h and the output pointer is valid.
        let destroy =
            unsafe {
                let create = symbol::<
                    unsafe extern "system" fn(*mut Handle, c_uint, c_int) -> CuResult,
                >(&api._library, "cuCtxCreate_v2")
                .unwrap();
                let destroy = symbol::<unsafe extern "system" fn(Handle) -> CuResult>(
                    &api._library,
                    "cuCtxDestroy_v2",
                )
                .unwrap();
                check("cuCtxCreate", create(&mut foreign, 0, device)).unwrap();
                destroy
            };
        let context = Context::new(0).unwrap();
        assert_eq!(current(), foreign);
        let buffer = DeviceBuffer::from_slice(&context, &[7_i32]).unwrap();
        assert_eq!(buffer.read().unwrap(), [7]);
        assert_eq!(current(), foreign);
        {
            let _current = context.enter().unwrap();
            assert_eq!(current(), context.inner.handle);
            assert!(context.memory().unwrap().1 > 0);
            assert_eq!(current(), context.inner.handle);
        }
        drop(buffer);
        drop(context);
        assert_eq!(current(), foreign);
        // SAFETY: this thread created the context and has finished using it.
        check("cuCtxDestroy", unsafe { destroy(foreign) }).unwrap();
        assert_eq!(current(), previous);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn host_buffer_rejects_misaligned_empty_allocation() {
        #[derive(Clone, Copy)]
        #[repr(C, align(536870912))]
        struct Aligned([u8; 536870912]);

        // SAFETY: all byte patterns are valid, with no padding or ownership.
        unsafe impl DeviceRepr for Aligned {}

        let context = Context::new(0).unwrap();
        match HostBuffer::<Aligned>::zeroed(&context, 0) {
            Ok(buffer) => assert!(
                buffer.ptr.is_aligned(),
                "safe HostBuffer accepted misaligned pointer {:p}",
                buffer.ptr
            ),
            Err(error) => assert_eq!(
                error.name,
                "host allocation does not satisfy element alignment"
            ),
        }
        assert!(
            HostBuffer::<f32>::zeroed(&context, 0)
                .unwrap()
                .as_slice()
                .is_empty()
        );
        assert_eq!(
            HostBuffer::<f32>::zeroed(&context, 4).unwrap().as_slice(),
            &[0.0; 4]
        );
    }
}
