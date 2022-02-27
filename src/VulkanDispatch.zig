const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");
const vma = @import("binding/vma.zig");
const MyBuffer = @import("Buffer.zig");
pub const enable_safety = std.debug.runtime_safety;
// pub const enable_safety = true;

pub const BaseDispatch = vk.BaseWrapper(.{
    .createInstance = true,
    .enumerateInstanceExtensionProperties = true,
    .enumerateInstanceLayerProperties = true,
});

pub const InstanceDispatch = vk.InstanceWrapper(.{
    //vma
    .getPhysicalDeviceProperties = true,
    .getPhysicalDeviceMemoryProperties = true,

    //debug
    .createDebugUtilsMessengerEXT = enable_safety,
    .destroyDebugUtilsMessengerEXT = enable_safety,

    //normal
    .destroyInstance = true,
    .createDevice = true,
    .destroySurfaceKHR = true,
    .enumeratePhysicalDevices = true,
    .enumerateDeviceExtensionProperties = true,
    .getPhysicalDeviceSurfaceFormatsKHR = true,
    .getPhysicalDeviceSurfacePresentModesKHR = true,
    .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
    .getPhysicalDeviceQueueFamilyProperties = true,
    .getPhysicalDeviceSurfaceSupportKHR = true,
    .getDeviceProcAddr = true,
    .getPhysicalDeviceFeatures = true,
    .getPhysicalDeviceFeatures2 = true,
    .getPhysicalDeviceFormatProperties = true,
});

pub const DeviceDispatch = vk.DeviceWrapper(.{
    //vma
    .createImage = true,
    .createBuffer = true,
    .destroyImage = true,
    .destroyBuffer = true,
    .cmdCopyBuffer = true,
    .bindImageMemory = true,
    .bindBufferMemory = true,
    .getImageMemoryRequirements2 = true,
    .getBufferMemoryRequirements2 = true,
    .mapMemory = true,
    .freeMemory = true,
    .unmapMemory = true,
    .allocateMemory = true,
    .flushMappedMemoryRanges = true,

    //debug
    .cmdBeginDebugUtilsLabelEXT = enable_safety,
    .cmdEndDebugUtilsLabelEXT = enable_safety,
    .cmdInsertDebugUtilsLabelEXT = enable_safety,
    .setDebugUtilsObjectNameEXT = enable_safety,

    //normal
    .destroyDevice = true,
    .getDeviceQueue = true,
    .createSemaphore = true,
    .createFence = true,
    .createImageView = true,
    .destroyImageView = true,
    .destroySemaphore = true,
    .destroyFence = true,
    .getSwapchainImagesKHR = true,
    .createSwapchainKHR = true,
    .destroySwapchainKHR = true,
    .acquireNextImageKHR = true,
    .deviceWaitIdle = true,
    .waitForFences = true,
    .resetFences = true,
    .queuePresentKHR = true,
    .createCommandPool = true,
    .destroyCommandPool = true,
    .allocateCommandBuffers = true,
    .freeCommandBuffers = true,
    .queueWaitIdle = true,
    .createShaderModule = true,
    .destroyShaderModule = true,
    .createPipelineLayout = true,
    .destroyPipelineLayout = true,
    .createRenderPass = true,
    .destroyRenderPass = true,
    .createGraphicsPipelines = true,
    .destroyPipeline = true,
    .createFramebuffer = true,
    .destroyFramebuffer = true,
    .createDescriptorSetLayout = true,
    .destroyDescriptorSetLayout = true,
    .createSampler = true,
    .destroySampler = true,
    .createDescriptorPool = true,
    .destroyDescriptorPool = true,
    .createDescriptorUpdateTemplate = true,
    .destroyDescriptorUpdateTemplate = true,
    .allocateDescriptorSets = true,
    .updateDescriptorSets = true,
    .beginCommandBuffer = true,
    .resetCommandPool = true,
    .endCommandBuffer = true,
    .cmdDraw = true,
    .cmdBlitImage = true,
    .cmdSetScissor = true,
    .cmdSetViewport = true,
    .cmdDrawIndexed = true,
    .cmdBindPipeline = true,
    .cmdPushConstants = true,
    .cmdEndRenderPass = true,
    .cmdBeginRenderPass = true,
    .queueSubmit2 = true,
    // .queueSubmit = true,
    // .cmdSetEvent2 = true,
    // .cmdResetEvent2 = true,
    // .cmdWaitEvents2 = true,
    .cmdBindIndexBuffer = true,
    .cmdCopyBufferToImage = true,
    .cmdBindVertexBuffers = true,
    .cmdBindDescriptorSets = true,
    .cmdPipelineBarrier2 = true,
    .cmdPushDescriptorSetWithTemplateKHR = true,

    //query
    .createQueryPool = true,
    .destroyQueryPool = true,
    .getQueryPoolResults = true,
    .cmdWriteTimestamp = true,
    .cmdResetQueryPool = true,
});

pub fn getVmaVulkanFunction(vki: InstanceDispatch, vkd: DeviceDispatch) vma.VulkanFunctions {
    return .{
        .getInstanceProcAddr = undefined,
        .getDeviceProcAddr = undefined,
        .getPhysicalDeviceProperties = vki.dispatch.vkGetPhysicalDeviceProperties,
        .getPhysicalDeviceMemoryProperties = vki.dispatch.vkGetPhysicalDeviceMemoryProperties,
        .allocateMemory = vkd.dispatch.vkAllocateMemory,
        .freeMemory = vkd.dispatch.vkFreeMemory,
        .mapMemory = vkd.dispatch.vkMapMemory,
        .unmapMemory = vkd.dispatch.vkUnmapMemory,
        .flushMappedMemoryRanges = vkd.dispatch.vkFlushMappedMemoryRanges,
        .invalidateMappedMemoryRanges = undefined, //vkd.dispatch.vkInvalidateMappedMemoryRanges,
        .bindBufferMemory = vkd.dispatch.vkBindBufferMemory,
        .bindImageMemory = vkd.dispatch.vkBindImageMemory,
        .getBufferMemoryRequirements = undefined, //vkd.dispatch.vkGetBufferMemoryRequirements,
        .getImageMemoryRequirements = undefined, //vkd.dispatch.vkGetImageMemoryRequirements,
        .createBuffer = vkd.dispatch.vkCreateBuffer,
        .destroyBuffer = vkd.dispatch.vkDestroyBuffer,
        .createImage = vkd.dispatch.vkCreateImage,
        .destroyImage = vkd.dispatch.vkDestroyImage,
        .cmdCopyBuffer = vkd.dispatch.vkCmdCopyBuffer,
        .getBufferMemoryRequirements2 = vkd.dispatch.vkGetBufferMemoryRequirements2,
        .getImageMemoryRequirements2 = vkd.dispatch.vkGetImageMemoryRequirements2,
        .bindBufferMemory2 = undefined, //vkd.dispatch.vkBindBufferMemory2,
        .bindImageMemory2 = undefined, //vkd.dispatch.vkBindImageMemory2,
        .getPhysicalDeviceMemoryProperties2 = undefined, //vkd.dispatch.vkGetPhysicalDeviceMemoryProperties2,
    };
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
