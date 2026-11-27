//! A Vulkan device with one compute queue and blocking submission, alive while any handle is.

use crate::EncoderError;
use ash::{Entry, khr, vk};
use std::{
    ffi::CStr,
    fmt,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct VulkanError(pub String);

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vulkan: {}", self.0)
    }
}

impl std::error::Error for VulkanError {}

impl From<vk::Result> for VulkanError {
    fn from(result: vk::Result) -> Self {
        Self(format!("{result:?}"))
    }
}

/// The cooperative matrix shape the matrix shaders use: FP16 inputs, FP32 or
/// (attention's score corrections) FP16 accumulation, one subgroup per tile.
pub const MATRIX_M: u32 = 16;
pub const MATRIX_N: u32 = 16;
pub const MATRIX_K: u32 = 16;

fn matrix_limits(
    limits: &vk::PhysicalDeviceLimits,
    subgroups: &vk::PhysicalDeviceSubgroupSizeControlProperties<'_>,
) -> bool {
    // gemm_matrix's 8 subgroups and 48 KiB of shared tiles exceed Vulkan's minimum.
    limits.max_compute_work_group_invocations >= 256
        && limits.max_compute_work_group_size[0] >= 256
        && limits.max_compute_shared_memory_size >= 49_152
        && (subgroups.min_subgroup_size..=subgroups.max_subgroup_size).contains(&32)
        && subgroups.max_compute_workgroup_subgroups >= 8
        && subgroups
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
}

pub(crate) struct Shared {
    pub(crate) device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    /// Vulkan leaves pool and queue locking to the caller, and every `Device` clone shares them.
    submitting: Mutex<()>,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    subgroup_control: vk::PhysicalDeviceSubgroupSizeControlProperties<'static>,
    matrix_enabled: bool,
    execution_identity: String,
    quarantined: AtomicBool,
}

impl Shared {
    pub(crate) fn check_usable(&self) -> Result<(), VulkanError> {
        if self.quarantined.load(Ordering::Acquire) {
            Err(VulkanError(
                "device quarantined after an unsettled submission".into(),
            ))
        } else {
            Ok(())
        }
    }

    pub(crate) fn can_destroy(&self) -> bool {
        !self.quarantined.load(Ordering::Acquire)
    }

    fn quarantine(self: &Arc<Self>) {
        if !self.quarantined.swap(true, Ordering::AcqRel) {
            // Completion cannot be established, so even device teardown may
            // race the GPU. Keep the entire device alive until process exit.
            std::mem::forget(self.clone());
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        if !self.can_destroy() {
            return;
        }
        // SAFETY: every other handle holds this `Shared`, so none remains.
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

/// A GPU as the Vulkan loader reports it.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub index: usize,
    pub name: String,
    pub discrete: bool,
    /// A portability implementation (`VK_KHR_portability_subset`) such as MoltenVK, which is
    /// not fully conformant.
    pub portability: bool,
}

#[derive(Clone)]
pub struct Device {
    pub(crate) shared: Arc<Shared>,
    pub name: String,
    /// Whether 16x16x16 FP16 cooperative matrices (FP32 or FP16 sums) are enabled.
    pub cooperative_matrix: bool,
    pub subgroup_size: u32,
    vendor_id: u32,
    device_uuid: [u8; 16],
}

/// One loader and instance for the process, kept to exit: destroying the last instance unloads
/// the driver, and NVIDIA's Linux driver stops loading after a few dozen reloads (static TLS).
struct Runtime {
    entry: Entry,
    instance: ash::Instance,
}

impl Runtime {
    /// The process's runtime, created on first use; a failed creation is retried.
    fn get() -> Result<&'static Self, EncoderError> {
        static RUNTIME: Mutex<Option<&'static Runtime>> = Mutex::new(None);
        let mut slot = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(runtime) = *slot {
            return Ok(runtime);
        }
        Ok(*slot.insert(Box::leak(Box::new(Self::new()?))))
    }

    fn new() -> Result<Self, EncoderError> {
        let entry = loader()?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"kodamaembed")
            .api_version(vk::API_VERSION_1_3);
        // SAFETY: the loader is loaded and no layer is named.
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(instance_error)?;
        // The loader lists portability drivers such as MoltenVK only when asked to;
        // conformant drivers it lists either way.
        let portability = [khr::portability_enumeration::NAME.as_ptr()];
        let mut info = vk::InstanceCreateInfo::default().application_info(&application);
        if lists(&available, khr::portability_enumeration::NAME) {
            info = info
                .enabled_extension_names(&portability)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        // SAFETY: `info` and what it points to outlive the call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(instance_error)?;
        Ok(Self { entry, instance })
    }
}

/// The system's Vulkan loader.
#[cfg(not(target_family = "wasm"))]
fn loader() -> Result<Entry, EncoderError> {
    // SAFETY: loading the system Vulkan loader runs its initializers only.
    unsafe { Entry::load() }.map_err(|error| match error {
        ash::LoadingError::LibraryLoadFailure(_) => {
            EncoderError::Unsupported(format!("loader: {error}"))
        }
        _ => VulkanError(format!("loader: {error}")).into(),
    })
}

/// A wasm target opens no dynamic library: it is a machine without a loader.
#[cfg(target_family = "wasm")]
fn loader() -> Result<Entry, EncoderError> {
    Err(EncoderError::Unsupported(
        "loader: this target opens no dynamic libraries".into(),
    ))
}

fn instance_error(result: vk::Result) -> EncoderError {
    if result == vk::Result::ERROR_INCOMPATIBLE_DRIVER {
        EncoderError::Unsupported("no compatible Vulkan driver".into())
    } else {
        VulkanError::from(result).into()
    }
}

fn device_error(error: EncoderError) -> VulkanError {
    match error {
        EncoderError::Vulkan(error) => error,
        _ => VulkanError(error.to_string()),
    }
}

fn lists(extensions: &[vk::ExtensionProperties], wanted: &CStr) -> bool {
    extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .is_ok_and(|name| name == wanted)
    })
}

fn name(properties: &vk::PhysicalDeviceProperties) -> String {
    properties
        .device_name_as_c_str()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn execution_identity(
    properties: &vk::PhysicalDeviceProperties,
    subgroup_size: u32,
    matrix: bool,
) -> String {
    format!(
        "vulkan:vendor={:08x}:device={:08x}:driver={:08x}:api={:08x}:subgroup={subgroup_size}:matrix={matrix}",
        properties.vendor_id,
        properties.device_id,
        properties.driver_version,
        properties.api_version
    )
}

/// A GPU in [`adapter_order`].
#[derive(Clone, Copy)]
struct Physical {
    handle: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    portability: bool,
}

/// A stable order: the loader's own changes with the libraries a process has loaded, CUDA's too.
fn adapter_order(
    properties: &vk::PhysicalDeviceProperties,
    portability: bool,
    uuid: [u8; 16],
) -> (u8, bool, String, [u8; 16]) {
    let kind = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 2,
    };
    (kind, portability, name(properties), uuid)
}

/// # Safety
/// `instance` must be live.
unsafe fn physical_devices(instance: &ash::Instance) -> Result<Vec<Physical>, VulkanError> {
    // SAFETY: the caller guarantees a live instance.
    let handles = unsafe { instance.enumerate_physical_devices() }?;
    let mut keyed = Vec::with_capacity(handles.len());
    for handle in handles {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        // SAFETY: `handle` belongs to the live instance.
        let (properties, extensions) = unsafe {
            let mut both = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            instance.get_physical_device_properties2(handle, &mut both);
            (
                both.properties,
                instance.enumerate_device_extension_properties(handle)?,
            )
        };
        let portability = lists(&extensions, vk::KHR_PORTABILITY_SUBSET_NAME);
        let key = adapter_order(&properties, portability, id.device_uuid);
        keyed.push((
            key,
            Physical {
                handle,
                properties,
                portability,
            },
        ));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, physical)| physical).collect())
}

impl Device {
    /// The Vulkan GPUs as [`Device::new`] indexes them, stable across processes.
    pub fn adapters() -> Result<Vec<Adapter>, VulkanError> {
        Self::list(&Runtime::get().map_err(device_error)?.instance)
    }

    /// Adapters available for inference. A missing loader or incompatible driver yields an empty list.
    pub fn available_adapters() -> Result<Vec<Adapter>, VulkanError> {
        match Runtime::get() {
            Ok(runtime) => Self::list(&runtime.instance),
            Err(EncoderError::Unsupported(_)) => Ok(Vec::new()),
            Err(error) => Err(device_error(error)),
        }
    }

    fn list(instance: &ash::Instance) -> Result<Vec<Adapter>, VulkanError> {
        // SAFETY: the process's instance is never destroyed.
        let physicals = unsafe { physical_devices(instance) }?;
        Ok(physicals
            .iter()
            .enumerate()
            .map(|(index, physical)| Adapter {
                index,
                name: name(&physical.properties),
                discrete: physical.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                portability: physical.portability,
            })
            .collect())
    }

    pub fn new(index: usize) -> Result<Self, VulkanError> {
        Self::for_encoder(index).map_err(device_error)
    }

    /// Opens an adapter and distinguishes missing shader capabilities from driver failures.
    pub fn for_encoder(index: usize) -> Result<Self, EncoderError> {
        let runtime = Runtime::get()?;
        let instance = &runtime.instance;
        // SAFETY: the process's instance is never destroyed.
        let physicals = unsafe { physical_devices(instance) }?;
        let Physical {
            handle: physical,
            properties,
            portability,
        } = *physicals
            .get(index)
            .ok_or_else(|| VulkanError(format!("no GPU at index {index}")))?;
        // SAFETY: `physical` came from this instance.
        let (families, memory) = unsafe {
            (
                instance.get_physical_device_queue_family_properties(physical),
                instance.get_physical_device_memory_properties(physical),
            )
        };
        if properties.api_version < vk::API_VERSION_1_3 {
            return Err(EncoderError::Unsupported(format!(
                "{} requires Vulkan 1.3",
                name(&properties)
            )));
        }
        let family = families
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| EncoderError::Unsupported("no compute queue".into()))?
            as u32;

        let mut have11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut have12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut have13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut have_matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut subgroup_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        {
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut have11)
                .push_next(&mut have12)
                .push_next(&mut have13)
                .push_next(&mut have_matrix);
            let mut props = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut subgroup_control)
                .push_next(&mut id);
            // SAFETY: the chains point at live structs.
            unsafe {
                instance.get_physical_device_features2(physical, &mut query);
                instance.get_physical_device_properties2(physical, &mut props);
            }
        }
        if have11.storage_buffer16_bit_access == 0 || have12.shader_float16 == 0 {
            return Err(EncoderError::Unsupported(format!(
                "{} lacks 16-bit storage buffers or FP16 arithmetic",
                name(&properties)
            )));
        }
        if have13.subgroup_size_control == 0 || have13.compute_full_subgroups == 0 {
            return Err(EncoderError::Unsupported(format!(
                "{} lacks subgroup size control",
                name(&properties)
            )));
        }
        let operations = subgroup.supported_operations;
        let arithmetic = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
        if !subgroup
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            || !operations.contains(arithmetic)
            // Row shaders have 256 invocations and 64 subgroup partials.
            || subgroup.subgroup_size < 4
        {
            return Err(EncoderError::Unsupported(format!(
                "{} lacks subgroup arithmetic",
                name(&properties)
            )));
        }
        // SAFETY: `physical` came from this instance.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(VulkanError::from)?;
        // Matrix attention's two lanes a row share softmax state by shuffles.
        let cooperative_matrix = have_matrix.cooperative_matrix != 0
            && operations.contains(vk::SubgroupFeatureFlags::SHUFFLE)
            && matrix_limits(&properties.limits, &subgroup_control)
            && lists(&extensions, khr::cooperative_matrix::NAME)
            && Self::supports_f16_matrix(&runtime.entry, instance, physical)?;

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut want11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut want12 = vk::PhysicalDeviceVulkan12Features::default().shader_float16(true);
        let mut want13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let mut want_matrix =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut enabled_extensions = Vec::new();
        if cooperative_matrix {
            enabled_extensions.push(khr::cooperative_matrix::NAME.as_ptr());
        }
        // A device that lists the portability subset must enable it; compute uses none of
        // the features it may lack.
        if portability {
            enabled_extensions.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&enabled_extensions)
            .push_next(&mut want11)
            .push_next(&mut want12)
            .push_next(&mut want13);
        if cooperative_matrix {
            info = info.push_next(&mut want_matrix);
        }
        // SAFETY: `info` and its chain outlive the call.
        let device =
            unsafe { instance.create_device(physical, &info, None) }.map_err(VulkanError::from)?;
        // SAFETY: the queue family and index exist on this device.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the device is live.
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                // SAFETY: nothing else was created from the device.
                unsafe { device.destroy_device(None) };
                return Err(VulkanError::from(error).into());
            }
        };
        Ok(Self {
            name: name(&properties),
            cooperative_matrix,
            subgroup_size: subgroup.subgroup_size,
            vendor_id: properties.vendor_id,
            device_uuid: id.device_uuid,
            shared: Arc::new(Shared {
                device,
                queue,
                command_pool,
                submitting: Mutex::new(()),
                memory,
                limits: properties.limits,
                subgroup_control,
                matrix_enabled: cooperative_matrix && subgroup.subgroup_size == 32,
                execution_identity: execution_identity(
                    &properties,
                    subgroup.subgroup_size,
                    cooperative_matrix && subgroup.subgroup_size == 32,
                ),
                quarantined: AtomicBool::new(false),
            }),
        })
    }

    fn supports_f16_matrix(
        entry: &Entry,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
    ) -> Result<bool, VulkanError> {
        let matrix = khr::cooperative_matrix::Instance::new(entry, instance);
        // SAFETY: `physical` belongs to `instance`.
        let shapes = unsafe { matrix.get_physical_device_cooperative_matrix_properties(physical) }?;
        let accumulates_in = |accumulator: vk::ComponentTypeKHR| {
            shapes.iter().any(|shape| {
                shape.m_size == MATRIX_M
                    && shape.n_size == MATRIX_N
                    && shape.k_size == MATRIX_K
                    && shape.a_type == vk::ComponentTypeKHR::FLOAT16
                    && shape.b_type == vk::ComponentTypeKHR::FLOAT16
                    && shape.c_type == accumulator
                    && shape.result_type == accumulator
                    && shape.scope == vk::ScopeKHR::SUBGROUP
            })
        };
        Ok(accumulates_in(vk::ComponentTypeKHR::FLOAT32)
            && accumulates_in(vk::ComponentTypeKHR::FLOAT16))
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, VulkanError> {
        let memory = &self.shared.memory;
        (0..memory.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && memory.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or_else(|| VulkanError(format!("no memory type with {flags:?}")))
    }

    pub(crate) fn supports_subgroup_size(&self, size: u32) -> bool {
        let control = &self.shared.subgroup_control;
        size.is_power_of_two()
            && (control.min_subgroup_size..=control.max_subgroup_size).contains(&size)
            && control
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
    }

    pub(crate) fn matrix_enabled(&self) -> bool {
        // Callers can edit public adapter data; dispatch uses what was enabled.
        self.shared.matrix_enabled
    }

    /// Vendor and UUID of the selected device; an all-zero UUID means unknown.
    pub fn device_identity(&self) -> (u32, [u8; 16]) {
        (self.vendor_id, self.device_uuid)
    }

    /// Hardware, driver and shader-path identity for fingerprints; adapter
    /// ordinals and batch capacities are excluded.
    pub fn execution_identity(&self) -> String {
        self.shared.execution_identity.clone()
    }

    /// Records commands with `record`, submits them and waits for completion.
    pub fn submit(
        &self,
        record: impl FnOnce(vk::CommandBuffer) -> Result<(), VulkanError>,
    ) -> Result<(), VulkanError> {
        let mut submission = self.submission();
        submission.record(record)?;
        submission.run()
    }

    /// Command buffers to record one after another and submit together; other
    /// submissions on this device, from any thread, wait until it is dropped.
    pub fn submission(&self) -> Submission<'_> {
        self.submission_of(false)
    }

    /// As [`Device::submission`], recording command buffers to keep with
    /// [`Submission::run_kept`].
    pub fn reusable_submission(&self) -> Submission<'_> {
        self.submission_of(true)
    }

    fn submission_of(&self, reusable: bool) -> Submission<'_> {
        Submission {
            device: self,
            commands: Vec::new(),
            reusable,
            _lock: self.shared.lock(),
        }
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, ()> {
        // A panic while another submission held the lock left no command
        // buffer behind: its drop freed them during unwinding.
        self.submitting
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Waited for once, but each command buffer stays its own GPU submission, short
/// enough for the operating system's watchdog.
pub struct Submission<'d> {
    device: &'d Device,
    commands: Vec<vk::CommandBuffer>,
    reusable: bool,
    _lock: MutexGuard<'d, ()>,
}

/// Command buffers [`Submission::run_kept`] kept to submit again, freed on drop.
pub struct Commands {
    shared: Arc<Shared>,
    buffers: Vec<vk::CommandBuffer>,
}

impl Commands {
    /// Submits the command buffers again, in order, and waits for all of them.
    pub fn run(&self) -> Result<(), VulkanError> {
        let _lock = self.shared.lock();
        submit_and_wait(&self.shared, &self.buffers)
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _lock = self.shared.lock();
        if self.shared.can_destroy() {
            // SAFETY: on a device never quarantined every run settled, and the
            // submission lock is held.
            unsafe {
                self.shared
                    .device
                    .free_command_buffers(self.shared.command_pool, &self.buffers)
            };
        }
    }
}

fn submission_settled(
    submitted: Result<(), vk::Result>,
    waited: Result<(), vk::Result>,
    idle: impl FnOnce() -> Result<(), vk::Result>,
) -> bool {
    submitted.is_err()
        || matches!(waited, Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST))
        || matches!(idle(), Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST))
}

impl Submission<'_> {
    /// Records one more command buffer with `record`.
    pub fn record(
        &mut self,
        record: impl FnOnce(vk::CommandBuffer) -> Result<(), VulkanError>,
    ) -> Result<(), VulkanError> {
        let shared = &self.device.shared;
        shared.check_usable()?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(shared.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is live and this submission holds the device's lock.
        let command = unsafe { shared.device.allocate_command_buffers(&allocate) }?[0];
        self.commands.push(command);
        let begin = vk::CommandBufferBeginInfo::default().flags(if self.reusable {
            vk::CommandBufferUsageFlags::empty()
        } else {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        });
        let recorded = (|| {
            // SAFETY: `command` is a freshly allocated primary command buffer.
            unsafe { shared.device.begin_command_buffer(command, &begin) }?;
            record(command)?;
            // SAFETY: recording began above.
            Ok(unsafe { shared.device.end_command_buffer(command) }?)
        })();
        if recorded.is_err() {
            // A caller may recover and record more work. An unfinished
            // command buffer must never be included in a later `run`.
            self.commands.pop();
            // SAFETY: it has never been submitted and the pool lock is held.
            unsafe {
                shared
                    .device
                    .free_command_buffers(shared.command_pool, &[command])
            };
        }
        recorded
    }

    /// Submits the recorded command buffers in order and waits for all of them.
    pub fn run(self) -> Result<(), VulkanError> {
        submit_and_wait(&self.device.shared, &self.commands)
    }

    /// As [`Submission::run`], keeping the command buffers to submit again.
    pub fn run_kept(mut self) -> Result<Commands, VulkanError> {
        assert!(
            self.reusable,
            "only a reusable submission's buffers are kept"
        );
        submit_and_wait(&self.device.shared, &self.commands)?;
        Ok(Commands {
            shared: self.device.shared.clone(),
            buffers: std::mem::take(&mut self.commands),
        })
    }
}

/// Completion is polled this long before a blocking wait, which wakes late after short work.
const SPIN: Duration = Duration::from_millis(5);

/// Submits `commands` in order and waits for all of them; the caller holds the submission lock.
fn submit_and_wait(
    shared: &Arc<Shared>,
    commands: &[vk::CommandBuffer],
) -> Result<(), VulkanError> {
    shared.check_usable()?;
    let device = &shared.device;
    let submits: Vec<_> = commands
        .iter()
        .map(|command| vk::SubmitInfo::default().command_buffers(std::slice::from_ref(command)))
        .collect();
    // SAFETY: the device is live.
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
    // SAFETY: the queue, command buffers and fence are live.
    let submitted = unsafe { device.queue_submit(shared.queue, &submits, fence) };
    let waited = submitted.and_then(|()| {
        let started = Instant::now();
        loop {
            // SAFETY: the fence is live.
            match unsafe { device.get_fence_status(fence) } {
                Ok(true) => break Ok(()),
                Ok(false) if started.elapsed() < SPIN => std::hint::spin_loop(),
                // SAFETY: as above.
                Ok(false) => break unsafe { device.wait_for_fences(&[fence], true, u64::MAX) },
                Err(error) => break Err(error),
            }
        }
    });
    // A failed submission queued nothing and a lost device runs nothing;
    // after any other failed wait, only an idle device shows the work done.
    let settled = submission_settled(submitted, waited, || {
        // SAFETY: the device is live, and the caller holds the submission lock.
        unsafe { device.device_wait_idle() }
    });
    if settled {
        // SAFETY: no pending work refers to the fence.
        unsafe { device.destroy_fence(fence, None) };
    } else {
        // The GPU may still use the fence and the command buffers: the fence
        // is left alive, and quarantine keeps every owner from freeing them.
        shared.quarantine();
    }
    Ok(waited?)
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        let shared = &self.device.shared;
        if !shared.can_destroy() {
            return;
        }
        // SAFETY: no command buffer here is executing (never submitted, or
        // settled by `run`), and the submission lock is still held.
        unsafe {
            shared
                .device
                .free_command_buffers(shared.command_pool, &self.commands)
        };
    }
}

/// A buffer of `len` values of `T`, device-local or host-visible.
pub struct Buffer<T> {
    shared: Arc<Shared>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: usize,
    mapped: Option<*mut T>,
}

/// Values device kernels may overwrite byte for byte.
///
/// # Safety
/// Every bit pattern must be valid, with no padding, references or pointers.
///
/// ```compile_fail
/// use kodamaembed_vulkan::DeviceRepr;
/// fn requires_device_repr<T: DeviceRepr>() {}
/// requires_device_repr::<bool>();
/// ```
pub unsafe trait DeviceRepr: Copy {}

macro_rules! elements {
    ($($ty:ty),* $(,)?) => {$(
        // SAFETY: built-in integers and floats have no padding or invalid bits.
        unsafe impl DeviceRepr for $ty {}
    )*};
}

elements!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    usize,
    isize,
    f32,
    f64,
    half::f16,
    half::bf16
);

// SAFETY: arrays of valid representations have no extra padding.
unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {}

fn buffer_bytes<T: DeviceRepr>(len: usize) -> Result<u64, VulkanError> {
    if size_of::<T>() == 0 {
        return Err(VulkanError(
            "zero-sized buffer elements are unsupported".into(),
        ));
    }
    // FillBuffer rounds WHOLE_SIZE down to words: padding zeroes an odd last half.
    len.checked_mul(size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .and_then(|bytes| bytes.max(4).checked_add(3))
        .map(|bytes| (bytes & !3) as u64)
        .ok_or_else(|| VulkanError("buffer byte size overflows the address space".into()))
}

// SAFETY: the buffer exclusively owns its allocation.
unsafe impl<T: Send> Send for Buffer<T> {}

impl<T: DeviceRepr> Buffer<T> {
    fn create(
        device: &Device,
        len: usize,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<Self, VulkanError> {
        device.shared.check_usable()?;
        let vk_device = &device.shared.device;
        let size = buffer_bytes::<T>(len)?;
        if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
            && size > u64::from(device.shared.limits.max_storage_buffer_range)
        {
            return Err(VulkanError(format!(
                "storage buffer of {size} bytes exceeds this device's {} byte limit",
                device.shared.limits.max_storage_buffer_range
            )));
        }
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live.
        let buffer = unsafe { vk_device.create_buffer(&info, None) }?;
        // SAFETY: `buffer` was just created on this device.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };
        let bits = requirements.memory_type_bits;
        let memory_type = match device
            .memory_type(bits, flags | preferred)
            .or_else(|_| device.memory_type(bits, flags))
        {
            Ok(index) => index,
            Err(error) => {
                // SAFETY: the unbound buffer is live and has no users.
                unsafe { vk_device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: the allocation info is valid; failures destroy the buffer.
        let memory = match unsafe { vk_device.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { vk_device.destroy_buffer(buffer, None) };
                return Err(error.into());
            }
        };
        // Dropping `created` from here on frees both, whatever fails next.
        let mut created = Self {
            shared: device.shared.clone(),
            buffer,
            memory,
            len,
            mapped: None,
        };
        // SAFETY: the memory satisfies the buffer's requirements.
        unsafe { vk_device.bind_buffer_memory(buffer, memory, 0) }?;
        if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            // SAFETY: the memory is host-visible and mapped once for its lifetime.
            let mapped = unsafe {
                vk_device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }?;
            if !(mapped as usize).is_multiple_of(align_of::<T>()) {
                return Err(VulkanError(format!(
                    "host mapping does not satisfy element alignment {}",
                    align_of::<T>()
                )));
            }
            created.mapped = Some(mapped.cast());
        }
        Ok(created)
    }

    /// Device-local storage, zeroed.
    pub fn zeroed(device: &Device, len: usize) -> Result<Self, VulkanError> {
        let buffer = Self::create(
            device,
            len,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        // SAFETY: the command buffer is recording; the buffer outlives the wait.
        device.submit(|command| unsafe {
            device
                .shared
                .device
                .cmd_fill_buffer(command, buffer.buffer, 0, vk::WHOLE_SIZE, 0);
            crate::pipeline::Recorder::new(device, command).copies_to_shaders();
            Ok(())
        })?;
        Ok(buffer)
    }

    pub fn from_slice(device: &Device, values: &[T]) -> Result<Self, VulkanError> {
        // Every element is written below, so skip the memset and its round trip.
        let mut buffer = Self::create(
            device,
            values.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        buffer.write(device, values)?;
        Ok(buffer)
    }

    /// Memory the host reads is cached where offered; uncached reads skip the processor's caches.
    fn staging(device: &Device, len: usize, host_reads: bool) -> Result<Self, VulkanError> {
        Self::create(
            device,
            len,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            if host_reads {
                vk::MemoryPropertyFlags::HOST_CACHED
            } else {
                vk::MemoryPropertyFlags::empty()
            },
        )
    }

    fn zeroed_staging(device: &Device, len: usize, host_reads: bool) -> Result<Self, VulkanError> {
        let buffer = Self::staging(device, len, host_reads)?;
        let mapped = buffer.mapped.expect("mapped staging");
        // SAFETY: the mapping holds `len` values of `T`.
        unsafe { std::ptr::write_bytes(mapped, 0, len) };
        Ok(buffer)
    }

    /// Zeroed host-visible staging memory the host writes in place.
    pub fn host(device: &Device, len: usize) -> Result<Self, VulkanError> {
        Self::zeroed_staging(device, len, false)
    }

    /// Zeroed host-visible memory the device writes and the host reads in place.
    pub fn readback(device: &Device, len: usize) -> Result<Self, VulkanError> {
        Self::zeroed_staging(device, len, true)
    }

    /// A host buffer's values; submissions have completed by the time it looks.
    pub fn contents(&self) -> &[T] {
        assert!(
            self.shared.can_destroy(),
            "host access to a quarantined device"
        );
        let mapped = self.mapped.expect("host memory");
        // SAFETY: the mapping holds `len` values while `self` lives; nothing runs.
        unsafe { std::slice::from_raw_parts(mapped, self.len) }
    }

    pub fn contents_mut(&mut self) -> &mut [T] {
        assert!(
            self.shared.can_destroy(),
            "host access to a quarantined device"
        );
        let mapped = self.mapped.expect("host memory");
        // SAFETY: as for `contents`, with `self` borrowed exclusively.
        unsafe { std::slice::from_raw_parts_mut(mapped, self.len) }
    }

    /// Writes `values` to the start of the buffer and waits for the copy.
    pub fn write(&mut self, device: &Device, values: &[T]) -> Result<(), VulkanError> {
        self.check_device(device)?;
        assert!(values.len() <= self.len, "write past end");
        if values.is_empty() {
            return Ok(());
        }
        let staging = Self::staging(device, values.len(), false)?;
        let mapped = staging.mapped.expect("mapped staging");
        // SAFETY: the mapping holds `values.len()` values of `T`.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), mapped, values.len()) };
        let region = [vk::BufferCopy::default().size(size_of_val(values) as u64)];
        // SAFETY: the command buffer is recording; both buffers outlive the wait.
        device.submit(|command| unsafe {
            device
                .shared
                .device
                .cmd_copy_buffer(command, staging.buffer, self.buffer, &region);
            crate::pipeline::Recorder::new(device, command).copies_to_shaders();
            Ok(())
        })
    }

    pub fn read(&self, device: &Device) -> Result<Vec<T>, VulkanError> {
        self.read_first(device, self.len)
    }

    pub fn read_first(&self, device: &Device, count: usize) -> Result<Vec<T>, VulkanError> {
        self.check_device(device)?;
        assert!(count <= self.len, "read past end");
        if count == 0 {
            return Ok(Vec::new());
        }
        let staging = Self::staging(device, count, true)?;
        let region = [vk::BufferCopy::default().size((count * size_of::<T>()) as u64)];
        // SAFETY: the command buffer is recording; the checked regions are in bounds.
        device.submit(|command| unsafe {
            let recorder = crate::pipeline::Recorder::new(device, command);
            recorder.writes_to_copies();
            device
                .shared
                .device
                .cmd_copy_buffer(command, self.buffer, staging.buffer, &region);
            recorder.copies_to_host();
            Ok(())
        })?;
        let mapped = staging.mapped.expect("mapped staging");
        // SAFETY: the copy completed and the mapping holds `count` values.
        Ok(unsafe { std::slice::from_raw_parts(mapped, count) }.to_vec())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// This buffer as a kernel binding.
    pub fn binding(&self) -> Binding {
        Binding(self.buffer)
    }

    fn check_device(&self, device: &Device) -> Result<(), VulkanError> {
        self.shared.check_usable()?;
        if Arc::ptr_eq(&self.shared, &device.shared) {
            Ok(())
        } else {
            Err(VulkanError("buffer belongs to a different device".into()))
        }
    }
}

/// A buffer to bind to a kernel slot.
#[derive(Clone, Copy)]
pub struct Binding(pub(crate) vk::Buffer);

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        if !self.shared.can_destroy() {
            return;
        }
        // SAFETY: the buffer and memory are this device's and unused.
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_discovery_only_skips_incompatible_drivers() {
        assert!(matches!(
            instance_error(vk::Result::ERROR_INCOMPATIBLE_DRIVER),
            EncoderError::Unsupported(_)
        ));
        assert!(matches!(
            instance_error(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            EncoderError::Vulkan(_)
        ));
    }

    #[test]
    fn cleanup_requires_finished_work_or_device_loss() {
        let must_not_wait = || -> Result<(), vk::Result> { panic!("idle wait is unnecessary") };
        assert!(submission_settled(
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            must_not_wait
        ));
        assert!(submission_settled(Ok(()), Ok(()), must_not_wait));
        assert!(submission_settled(
            Ok(()),
            Err(vk::Result::ERROR_DEVICE_LOST),
            must_not_wait
        ));
        assert!(submission_settled(
            Ok(()),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            || Ok(())
        ));
        assert!(submission_settled(
            Ok(()),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            || Err(vk::Result::ERROR_DEVICE_LOST)
        ));
        assert!(!submission_settled(
            Ok(()),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            || Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        ));
    }

    #[test]
    fn buffer_sizes_cover_the_last_half_and_reject_overflow() {
        assert_eq!(buffer_bytes::<half::f16>(3).unwrap(), 8);
        assert_eq!(buffer_bytes::<u8>(5).unwrap(), 8);
        assert_eq!(buffer_bytes::<f32>(3).unwrap(), 12);
        assert_eq!(buffer_bytes::<u8>(0).unwrap(), 4);
        assert!(buffer_bytes::<f32>(usize::MAX).is_err());
        assert!(buffer_bytes::<[u8; 0]>(1).is_err());
    }

    #[test]
    fn execution_identity_is_stable_and_tracks_driver_and_shader_path() {
        let mut properties = vk::PhysicalDeviceProperties {
            vendor_id: 0x10de,
            device_id: 0x24dc,
            driver_version: 0x1234,
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };
        let baseline = execution_identity(&properties, 32, true);
        assert_eq!(baseline, execution_identity(&properties, 32, true));
        assert_ne!(baseline, execution_identity(&properties, 32, false));
        assert_ne!(baseline, execution_identity(&properties, 64, true));
        properties.driver_version += 1;
        assert_ne!(baseline, execution_identity(&properties, 32, true));
    }

    #[test]
    fn adapters_list_discrete_first_then_conformant_drivers_then_by_name() {
        let key = |kind, portability, name: &CStr| {
            let properties = vk::PhysicalDeviceProperties::default()
                .device_type(kind)
                .device_name(name)
                .unwrap();
            adapter_order(&properties, portability, [0; 16])
        };
        let discrete = vk::PhysicalDeviceType::DISCRETE_GPU;
        let integrated = vk::PhysicalDeviceType::INTEGRATED_GPU;
        assert!(key(discrete, false, c"A") < key(discrete, false, c"B"));
        assert!(key(discrete, false, c"B") < key(discrete, true, c"A"));
        assert!(key(discrete, true, c"A") < key(integrated, false, c"A"));
        assert!(key(integrated, true, c"A") < key(vk::PhysicalDeviceType::CPU, false, c"A"));
    }

    #[test]
    fn matrix_shaders_require_their_actual_shared_memory_and_subgroup_limits() {
        let mut limits = vk::PhysicalDeviceLimits {
            max_compute_work_group_invocations: 256,
            max_compute_work_group_size: [256, 256, 64],
            max_compute_shared_memory_size: 49_152,
            ..Default::default()
        };
        let mut subgroups = vk::PhysicalDeviceSubgroupSizeControlProperties::default()
            .min_subgroup_size(32)
            .max_subgroup_size(64)
            .max_compute_workgroup_subgroups(8)
            .required_subgroup_size_stages(vk::ShaderStageFlags::COMPUTE);
        assert!(matrix_limits(&limits, &subgroups));
        limits.max_compute_shared_memory_size = 32_768;
        assert!(!matrix_limits(&limits, &subgroups));
        limits.max_compute_shared_memory_size = 49_152;
        subgroups.required_subgroup_size_stages = vk::ShaderStageFlags::FRAGMENT;
        assert!(!matrix_limits(&limits, &subgroups));
        subgroups.required_subgroup_size_stages = vk::ShaderStageFlags::COMPUTE;
        subgroups.min_subgroup_size = 64;
        assert!(!matrix_limits(&limits, &subgroups));
    }
}
