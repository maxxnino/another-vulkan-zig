const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");
const vma = @import("vma.zig");
const MyBuffer = @import("Buffer.zig");
pub const enable_safety = builtin.mode == .Debug or builtin.mode == .ReleaseSafe;

pub const BaseDispatch = vk.BaseWrapper(&.{
    .createInstance,
    .enumerateInstanceExtensionProperties,
    .enumerateInstanceLayerProperties,
});

const instance_vma = [_]vk.InstanceCommand{
    .getPhysicalDeviceProperties,
    .getPhysicalDeviceMemoryProperties,
};
const instance_debug = [_]vk.InstanceCommand{
    .createDebugUtilsMessengerEXT,
    .destroyDebugUtilsMessengerEXT,
};
const instance_command = [_]vk.InstanceCommand{
    .destroyInstance,                         .createDevice,
    .destroySurfaceKHR,                       .enumeratePhysicalDevices,
    .enumerateDeviceExtensionProperties,      .getPhysicalDeviceSurfaceFormatsKHR,
    .getPhysicalDeviceSurfacePresentModesKHR, .getPhysicalDeviceSurfaceCapabilitiesKHR,
    .getPhysicalDeviceQueueFamilyProperties,  .getPhysicalDeviceSurfaceSupportKHR,
    .getDeviceProcAddr,                       .getPhysicalDeviceFeatures,
    .getPhysicalDeviceFeatures2,
} ++ instance_vma ++ if (enable_safety) instance_debug else [_]vk.InstanceCommand{};
pub const InstanceDispatch = vk.InstanceWrapper(&instance_command);

const device_debug = [_]vk.DeviceCommand{
    .cmdBeginDebugUtilsLabelEXT,
    .cmdEndDebugUtilsLabelEXT,
    .cmdInsertDebugUtilsLabelEXT,
    .setDebugUtilsObjectNameEXT,
};
const device_vma = [_]vk.DeviceCommand{
    .createImage,
    .createBuffer,
    .destroyImage,
    .destroyBuffer,
    .cmdCopyBuffer,
    .bindImageMemory,
    .bindBufferMemory,
    .getImageMemoryRequirements2,
    .getBufferMemoryRequirements2,
    .mapMemory,
    .freeMemory,
    .unmapMemory,
    .allocateMemory,
    .flushMappedMemoryRanges,
};
const device_command = [_]vk.DeviceCommand{
    .destroyDevice,
    .getDeviceQueue,
    .createSemaphore,
    .createFence,
    .createImageView,
    .destroyImageView,
    .destroySemaphore,
    .destroyFence,
    .getSwapchainImagesKHR,
    .createSwapchainKHR,
    .destroySwapchainKHR,
    .acquireNextImageKHR,
    .deviceWaitIdle,
    .waitForFences,
    .resetFences,
    .queueSubmit,
    .queueSubmit2KHR,
    .queuePresentKHR,
    .createCommandPool,
    .destroyCommandPool,
    .allocateCommandBuffers,
    .freeCommandBuffers,
    .queueWaitIdle,
    .createShaderModule,
    .destroyShaderModule,
    .createPipelineLayout,
    .destroyPipelineLayout,
    .createRenderPass,
    .destroyRenderPass,
    .createGraphicsPipelines,
    .destroyPipeline,
    .createFramebuffer,
    .destroyFramebuffer,
    .createDescriptorSetLayout,
    .destroyDescriptorSetLayout,
    .createSampler,
    .destroySampler,
    .createDescriptorPool,
    .destroyDescriptorPool,
    .allocateDescriptorSets,
    .updateDescriptorSets,
    .beginCommandBuffer,
    .endCommandBuffer,
    .cmdDraw,
    .cmdBlitImage,
    .cmdSetScissor,
    .cmdSetViewport,
    .cmdDrawIndexed,
    .cmdBindPipeline,
    .cmdEndRenderPass,
    .cmdBeginRenderPass,
    .cmdPipelineBarrier,
    .cmdSetEvent2KHR,
    .cmdResetEvent2KHR,
    .cmdWaitEvents2KHR,
    .cmdPipelineBarrier2KHR,
    .cmdBindIndexBuffer,
    .cmdCopyBufferToImage,
    .cmdBindVertexBuffers,
    .cmdBindDescriptorSets,
} ++ device_vma ++ if (enable_safety) device_debug else [_]vk.InstanceCommand{};
pub const DeviceDispatch = vk.DeviceWrapper(&device_command);

pub fn getVmaVulkanFunction(vki: InstanceDispatch, vkd: DeviceDispatch) vma.VulkanFunctions {
    var vma_vulkan_func: vma.VulkanFunctions = undefined;

    // Instance Vma
    inline for (instance_vma) |cmd| {
        const name = comptime vk.InstanceCommand.symbol(cmd);
        if (!@hasField(vma.VulkanFunctions, name)) @compileError("No function " ++ name ++ " in Vma VulkanFunctions");
        @field(vma_vulkan_func, name) = @field(vki.dispatch, name);
    }

    // Device Vma
    inline for (device_vma) |cmd| {
        const name = comptime vk.DeviceCommand.symbol(cmd);
        if (!@hasField(vma.VulkanFunctions, name)) @compileError("No function " ++ name ++ " in Vma VulkanFunctions");
        @field(vma_vulkan_func, name) = @field(vkd.dispatch, name);
    }
    return vma_vulkan_func;
}

// Destroy Function
pub fn destroy(vkd: DeviceDispatch, dev: vk.Device, resource: anytype) void {
    const ResourceType = @TypeOf(resource);
    const name = @typeName(ResourceType);
    if (ResourceType == vk.Image or ResourceType == vk.Buffer or ResourceType == MyBuffer) {
        @compileError("Can not destroy single vk." ++ name ++ " call " ++ name ++ ".deinit(GraphicsContext) instead");
    }

    @field(DestroyLookupTable, name)(vkd, dev, resource, null);
}

pub fn create(vkd: DeviceDispatch, dev: vk.Device, create_info: anytype, object_name: ?[*:0]const u8) !CreateInfoToType(@TypeOf(create_info)) {
    _ = object_name;
    const CreateInfo = @TypeOf(create_info);
    const name = @typeName(CreateInfo);
    if (CreateInfo == vk.ImageCreateInfo or CreateInfo == vk.BufferCreateInfo or CreateInfo == MyBuffer.CreateInfo) {
        @compileError("Don't support " ++ @typeName(CreateInfo));
    }
    const tuple = @field(CreateLookupTable, name);
    const handle = blk: {
        if (tuple[1] == .pipeline) {
            var pipeline: vk.Pipeline = undefined;
            _ = try tuple[0](
                vkd,
                dev,
                .null_handle,
                1,
                @ptrCast([*]const CreateInfo, &create_info),
                null,
                @ptrCast([*]vk.Pipeline, &pipeline),
            );
            break :blk pipeline;
        }
        break :blk try tuple[0](vkd, dev, &create_info, null);
    };

    if (enable_safety) {
        if (object_name) |value| {
            try vkd.setDebugUtilsObjectNameEXT(dev, &.{
                .object_type = tuple[1],
                .object_handle = @enumToInt(handle),
                .p_object_name = value,
            });
        }
    }

    return handle;
}
const DestroyLookupTable = struct {
    const Event = DeviceDispatch.destroyEvent;
    const Fence = DeviceDispatch.destroyFence;
    const Buffer = DeviceDispatch.destroyBuffer;
    const Sampler = DeviceDispatch.destroySampler;
    const Pipeline = DeviceDispatch.destroyPipeline;
    const ImageView = DeviceDispatch.destroyImageView;
    const Semaphore = DeviceDispatch.destroySemaphore;
    const QueryPool = DeviceDispatch.destroyQueryPool;
    const BufferView = DeviceDispatch.destroyBufferView;
    const RenderPass = DeviceDispatch.destroyRenderPass;
    const CommandPool = DeviceDispatch.destroyCommandPool;
    const CuModuleNVX = DeviceDispatch.destroyCuModuleNVX;
    const Framebuffer = DeviceDispatch.destroyFramebuffer;
    const ShaderModule = DeviceDispatch.destroyShaderModule;
    const SwapchainKHR = DeviceDispatch.destroySwapchainKHR;
    const PipelineCache = DeviceDispatch.destroyPipelineCache;
    const CuFunctionNVX = DeviceDispatch.destroyCuFunctionNVX;
    const PipelineLayout = DeviceDispatch.destroyPipelineLayout;
    const DescriptorPool = DeviceDispatch.destroyDescriptorPool;
    const VideoSessionKHR = DeviceDispatch.destroyVideoSessionKHR;
    const ValidationCacheEXT = DeviceDispatch.destroyValidationCacheEXT;
    const PrivateDataSlotEXT = DeviceDispatch.destroyPrivateDataSlotEXT;
    const DescriptorSetLayout = DeviceDispatch.destroyDescriptorSetLayout;
    const DeferredOperationKHR = DeviceDispatch.destroyDeferredOperationKHR;
    const SamplerYcbcrConversion = DeviceDispatch.destroySamplerYcbcrConversion;
    const BufferCollectionFUCHSIA = DeviceDispatch.destroyBufferCollectionFUCHSIA;
    const AccelerationStructureNV = DeviceDispatch.destroyAccelerationStructureNV;
    const AccelerationStructureKHR = DeviceDispatch.destroyAccelerationStructureKHR;
    const DescriptorUpdateTemplate = DeviceDispatch.destroyDescriptorUpdateTemplate;
};

const CreateLookupTable = struct {
    const EventCreateInfo = .{ DeviceDispatch.createEvent, vk.ObjectType.event };
    const FenceCreateInfo = .{ DeviceDispatch.createFence, vk.ObjectType.fence };
    const SamplerCreateInfo = .{ DeviceDispatch.createSampler, vk.ObjectType.sampler };
    const ImageViewCreateInfo = .{ DeviceDispatch.createImageView, vk.ObjectType.image_view };
    const SemaphoreCreateInfo = .{ DeviceDispatch.createSemaphore, vk.ObjectType.semaphore };
    const QueryPoolCreateInfo = .{ DeviceDispatch.createQueryPool, vk.ObjectType.query_pool };
    const BufferViewCreateInfo = .{ DeviceDispatch.createBufferView, vk.ObjectType.buffer_view };
    const RenderPassCreateInfo = .{ DeviceDispatch.createRenderPass, vk.ObjectType.render_pass };
    const CommandPoolCreateInfo = .{ DeviceDispatch.createCommandPool, vk.ObjectType.command_pool };
    const RenderPassCreateInfo2 = .{ DeviceDispatch.createRenderPass2, vk.ObjectType.render_Pass };
    const CuModuleCreateInfoNVX = .{ DeviceDispatch.createCuModuleNVX, vk.ObjectType.cu_module_nvx };
    const FramebufferCreateInfo = .{ DeviceDispatch.createFramebuffer, vk.ObjectType.framebuffer };
    const ShaderModuleCreateInfo = .{ DeviceDispatch.createShaderModule, vk.ObjectType.shader_module };
    const SwapchainCreateInfoKHR = .{ DeviceDispatch.createSwapchainKHR, vk.ObjectType.swapchain_khr };
    const PipelineCacheCreateInfo = .{ DeviceDispatch.createPipelineCache, vk.ObjectType.pipeline_cache };
    const CuFunctionCreateInfoNVX = .{ DeviceDispatch.createCuFunctionNVX, vk.ObjectType.cu_function_nvx };
    const PipelineLayoutCreateInfo = .{ DeviceDispatch.createPipelineLayout, vk.ObjectType.pipeline_layout };
    const DescriptorPoolCreateInfo = .{ DeviceDispatch.createDescriptorPool, vk.ObjectType.descriptor_pool };
    const VideoSessionCreateInfoKHR = .{ DeviceDispatch.createVideoSessionKHR, vk.ObjectType.video_session_khr };
    const ComputePipelineCreateInfo = .{ DeviceDispatch.createComputePipelines, vk.ObjectType.pipeline };
    const GraphicsPipelineCreateInfo = .{ DeviceDispatch.createGraphicsPipelines, vk.ObjectType.pipeline };
    const ValidationCacheCreateInfoEXT = .{ DeviceDispatch.createValidationCacheEXT, vk.ObjectType.validation_cache_ext };
    const PrivateDataSlotCreateInfoEXT = .{ DeviceDispatch.createPrivateDataSlotEXT, vk.ObjectType.private_data_slot_ext };
    const DescriptorSetLayoutCreateInfo = .{ DeviceDispatch.createDescriptorSetLayout, vk.ObjectType.descriptor_set_layout };
    const RayTracingPipelineCreateInfoNV = .{ DeviceDispatch.createRayTracingPipelinesNV, vk.ObjectType.pipeline };
    const RayTracingPipelineCreateInfoKHR = .{ DeviceDispatch.createRayTracingPipelinesKHR, vk.ObjectType.pipeline };
    const SamplerYcbcrConversionCreateInfo = .{ DeviceDispatch.createSamplerYcbcrConversion, vk.ObjectType.sampler_ycbcr_conversion };
    const BufferCollectionCreateInfoFUCHSIA = .{ DeviceDispatch.createBufferCollectionFUCHSIA, vk.ObjectType.buffer_collection_fuchsia };
    const AccelerationStructureCreateInfoNV = .{ DeviceDispatch.createAccelerationStructureNV, vk.ObjectType.acceleration_structure_nv };
    const AccelerationStructureCreateInfoKHR = .{ DeviceDispatch.createAccelerationStructureKHR, vk.ObjectType.acceleration_structure_khr };
    const DescriptorUpdateTemplateCreateInfo = .{ DeviceDispatch.createDescriptorUpdateTemplate, vk.ObjectType.descriptor_update_template };
    const IndirectCommandsLayoutCreateInfoNV = .{ DeviceDispatch.createIndirectCommandsLayoutNV, vk.ObjectType.indirect_commands_layout_nv };
};

pub fn CreateInfoToType(comptime T: type) type {
    const des_type_name = @typeName(T);
    const resource_name = blk: {
        if (std.mem.indexOf(u8, des_type_name, "CreateInfo")) |index| {
            if (std.mem.indexOf(u8, des_type_name, "Pipeline") != null and
                std.mem.indexOf(u8, des_type_name, "Layout") == null)
            {
                break :blk "Pipeline";
            }

            const end = index + 10;
            if (end == des_type_name.len or des_type_name[end] == '2') {
                break :blk des_type_name[0..index];
            }

            break :blk des_type_name[0..index] ++ des_type_name[end..];
        }
        @compileError("Can't create resource with " ++ des_type_name);
    };
    return @field(vk, resource_name);
}
