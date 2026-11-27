//! Compute pipelines from embedded SPIR-V, their descriptor sets and dispatches.

use crate::device::{Binding, Buffer, Device, DeviceRepr, Shared, VulkanError};
use ash::vk;
use std::sync::Arc;

/// How to build a kernel: a compute shader bound to `bindings` storage buffers
/// (bindings 0..n) and a push-constant block of `push_bytes`.
#[derive(Clone, Copy)]
pub struct KernelSpec<'a> {
    pub spirv: &'a [u8],
    pub bindings: u32,
    pub push_bytes: u32,
    /// Values of specialization constants 0, 1, ...
    pub constants: &'a [u32],
    /// Pins the subgroup size and requires full subgroups, as
    /// cooperative-matrix shaders need.
    pub subgroup_size: Option<u32>,
}

pub struct Kernel {
    shared: Arc<Shared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    bindings: u32,
    push_bytes: u32,
}

impl Kernel {
    /// Builds a pipeline from a caller-provided shader module.
    ///
    /// # Safety
    /// The SPIR-V must be valid for this device and match `spec`'s bindings,
    /// push constants and specialization constants.
    pub unsafe fn new(device: &Device, spec: &KernelSpec) -> Result<Self, VulkanError> {
        device.shared.check_usable()?;
        let KernelSpec {
            spirv,
            bindings,
            push_bytes,
            constants,
            subgroup_size,
        } = *spec;
        if subgroup_size.is_some_and(|size| !device.supports_subgroup_size(size)) {
            return Err(VulkanError("unsupported compute subgroup size".into()));
        }
        if bindings
            > device
                .shared
                .limits
                .max_per_stage_descriptor_storage_buffers
            || bindings > device.shared.limits.max_descriptor_set_storage_buffers
        {
            return Err(VulkanError(
                "too many storage-buffer bindings for this device".into(),
            ));
        }
        if push_bytes > device.shared.limits.max_push_constants_size
            || !push_bytes.is_multiple_of(4)
        {
            return Err(VulkanError(
                "invalid push-constant size for this device".into(),
            ));
        }
        let vk_device = &device.shared.device;
        assert!(
            spirv.len().is_multiple_of(4),
            "SPIR-V is a sequence of words"
        );
        let words: Vec<u32> = spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let entries: Vec<_> = (0..constants.len() as u32)
            .map(|id| {
                vk::SpecializationMapEntry::default()
                    .constant_id(id)
                    .offset(id * 4)
                    .size(4)
            })
            .collect();
        let data: Vec<u8> = constants.iter().flat_map(|c| c.to_le_bytes()).collect();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let slots: Vec<_> = (0..bindings)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_bytes)];
        // SAFETY: create-infos outlive their calls; failures destroy what exists.
        unsafe {
            let module = vk_device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
            let set_layout = vk_device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&slots),
                None,
            );
            let set_layout = match set_layout {
                Ok(layout) => layout,
                Err(error) => {
                    vk_device.destroy_shader_module(module, None);
                    return Err(error.into());
                }
            };
            let set_layouts = [set_layout];
            let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if push_bytes > 0 {
                layout_info = layout_info.push_constant_ranges(&ranges);
            }
            let layout = match vk_device.create_pipeline_layout(&layout_info, None) {
                Ok(layout) => layout,
                Err(error) => {
                    vk_device.destroy_descriptor_set_layout(set_layout, None);
                    vk_device.destroy_shader_module(module, None);
                    return Err(error.into());
                }
            };
            let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(subgroup_size.unwrap_or(0));
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization);
            if subgroup_size.is_some() {
                stage = stage
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                    .push_next(&mut required);
            }
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline =
                vk_device.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            vk_device.destroy_shader_module(module, None);
            let pipeline = match pipeline {
                Ok(pipelines) => pipelines[0],
                Err((pipelines, error)) => {
                    for pipeline in pipelines {
                        vk_device.destroy_pipeline(pipeline, None);
                    }
                    vk_device.destroy_pipeline_layout(layout, None);
                    vk_device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(error.into());
                }
            };
            Ok(Self {
                shared: device.shared.clone(),
                pipeline,
                layout,
                set_layout,
                bindings,
                push_bytes,
            })
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if !self.shared.can_destroy() {
            return;
        }
        // SAFETY: the handles belong to this device and no submission is pending.
        unsafe {
            let device = &self.shared.device;
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// A pool of descriptor sets; each binds buffers to one kernel's slots.
pub struct Descriptors {
    shared: Arc<Shared>,
    pool: vk::DescriptorPool,
}

fn validate_descriptor_counts(sets: u32, buffers: u32) -> Result<(), VulkanError> {
    // These pool flags forbid zero counts; reject them before the driver sees them.
    if sets == 0 || buffers == 0 {
        return Err(VulkanError(
            "descriptor pool requires at least one set and storage buffer".into(),
        ));
    }
    Ok(())
}

impl Descriptors {
    pub fn new(device: &Device, sets: u32, buffers: u32) -> Result<Self, VulkanError> {
        validate_descriptor_counts(sets, buffers)?;
        device.shared.check_usable()?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(buffers)];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(sets)
            .pool_sizes(&sizes);
        // SAFETY: the device is live and `info` outlives the call.
        let pool = unsafe { device.shared.device.create_descriptor_pool(&info, None) }?;
        Ok(Self {
            shared: device.shared.clone(),
            pool,
        })
    }

    /// A set binding `buffers[i]` to slot `i` of `kernel`.
    ///
    /// # Safety
    /// The buffers must be this device's and outlive every submission using the
    /// set; allocation from this pool must be externally synchronized.
    pub unsafe fn bind(
        &self,
        kernel: &Kernel,
        buffers: &[Binding],
    ) -> Result<vk::DescriptorSet, VulkanError> {
        self.shared.check_usable()?;
        if !Arc::ptr_eq(&self.shared, &kernel.shared) {
            return Err(VulkanError("kernel belongs to a different device".into()));
        }
        assert_eq!(
            buffers.len(),
            kernel.bindings as usize,
            "one buffer per binding"
        );
        let layouts = [kernel.set_layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        let device = &self.shared.device;
        // SAFETY: the pool and layout are live.
        let set = unsafe { device.allocate_descriptor_sets(&allocate) }?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|binding| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(binding.0)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: the set and buffers are live and the writes match the layout.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::validate_descriptor_counts;

    #[test]
    fn descriptor_pool_rejects_zero_counts_before_driver_creation() {
        for (sets, buffers) in [(0, 0), (0, 1), (1, 0), (0, u32::MAX), (u32::MAX, 0)] {
            assert!(validate_descriptor_counts(sets, buffers).is_err());
        }
        assert!(validate_descriptor_counts(1, 1).is_ok());
        assert!(validate_descriptor_counts(u32::MAX, u32::MAX).is_ok());
    }
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        if !self.shared.can_destroy() {
            return;
        }
        // SAFETY: the pool belongs to this device; destroying it frees its sets.
        unsafe { self.shared.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// Push constants as little-endian 32-bit fields.
#[derive(Default)]
pub struct Push(Vec<u8>);

impl Push {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(mut self, value: u32) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn f32(mut self, value: f32) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }
}

/// Records dispatches into a command buffer being recorded.
pub struct Recorder<'a> {
    device: &'a ash::Device,
    command: vk::CommandBuffer,
}

impl<'a> Recorder<'a> {
    /// Wraps a raw command buffer.
    ///
    /// # Safety
    /// `command` must be `device`'s, recording, and exclusively this recorder's
    /// for every call through it.
    pub unsafe fn new(device: &'a Device, command: vk::CommandBuffer) -> Self {
        Self {
            device: &device.shared.device,
            command,
        }
    }

    /// Records a dispatch without validating shader buffer extents.
    ///
    /// # Safety
    /// Every object must be this device's and outlive the submission; the set,
    /// push constants and dimensions must make every shader access valid.
    pub unsafe fn dispatch(
        &self,
        kernel: &Kernel,
        set: vk::DescriptorSet,
        push: &Push,
        groups: [u32; 3],
    ) {
        assert_eq!(
            push.0.len() as u32,
            kernel.push_bytes,
            "push constants match the kernel"
        );
        // SAFETY: the command buffer is recording and every handle is live.
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                kernel.layout,
                0,
                &[set],
                &[],
            );
            if !push.0.is_empty() {
                self.device.cmd_push_constants(
                    self.command,
                    kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push.0,
                );
            }
            self.device
                .cmd_dispatch(self.command, groups[0], groups[1], groups[2]);
        }
    }

    /// Copies `count` values from `source[from..]` to `target[to..]`.
    ///
    /// # Safety
    /// Both buffers must be this device's and outlive the submission, and the
    /// regions must not overlap.
    pub unsafe fn copy<T: DeviceRepr>(
        &self,
        source: &Buffer<T>,
        from: usize,
        target: &Buffer<T>,
        to: usize,
        count: usize,
    ) {
        assert!(
            count > 0
                && from
                    .checked_add(count)
                    .is_some_and(|end| end <= source.len())
                && to.checked_add(count).is_some_and(|end| end <= target.len()),
            "a copy of at least one value inside both buffers"
        );
        let size = size_of::<T>() as u64;
        let region = [vk::BufferCopy::default()
            .src_offset(from as u64 * size)
            .dst_offset(to as u64 * size)
            .size(count as u64 * size)];
        // SAFETY: recording, both buffers live, and the region inside both.
        unsafe {
            self.device
                .cmd_copy_buffer(self.command, source.buffer, target.buffer, &region)
        };
    }

    /// Makes every earlier shader write visible to later shader reads and writes.
    pub fn barrier(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
    }

    /// Makes earlier copies visible to later shaders.
    pub fn copies_to_shaders(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
    }

    /// Makes earlier shader writes visible to later copies.
    pub fn shaders_to_copies(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        );
    }

    /// Buffer readback may follow either an upload/fill or a shader dispatch.
    pub(crate) fn writes_to_copies(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        );
    }

    /// Makes earlier copies into host memory visible once the submission ends.
    pub fn copies_to_host(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::HOST,
            vk::AccessFlags::HOST_READ,
        );
    }

    fn memory_barrier(
        &self,
        source_stage: vk::PipelineStageFlags,
        source_access: vk::AccessFlags,
        target_stage: vk::PipelineStageFlags,
        target_access: vk::AccessFlags,
    ) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(source_access)
            .dst_access_mask(target_access)];
        // SAFETY: the command buffer is recording.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                source_stage,
                target_stage,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }
}
