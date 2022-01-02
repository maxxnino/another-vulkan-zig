const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const vma = @import("vma.zig");
const Allocator = std.mem.Allocator;

const required_device_extensions = [_][]const u8{vk.extension_info.khr_swapchain.name};

const BaseDispatch = vk.BaseWrapper(&.{
    .createInstance,
});

const instance_vma = [_]vk.InstanceCommand{
    .getPhysicalDeviceProperties,
    .getPhysicalDeviceMemoryProperties,
};
const instance_command = [_]vk.InstanceCommand{
    .destroyInstance,                         .createDevice,
    .destroySurfaceKHR,                       .enumeratePhysicalDevices,
    .enumerateDeviceExtensionProperties,      .getPhysicalDeviceSurfaceFormatsKHR,
    .getPhysicalDeviceSurfacePresentModesKHR, .getPhysicalDeviceSurfaceCapabilitiesKHR,
    .getPhysicalDeviceQueueFamilyProperties,  .getPhysicalDeviceSurfaceSupportKHR,
    .getDeviceProcAddr,
} ++ instance_vma;
const InstanceDispatch = vk.InstanceWrapper(&instance_command);

const device_vma = [_]vk.DeviceCommand{
    .allocateMemory,               .bindBufferMemory,
    .bindImageMemory,              .createBuffer,
    .destroyBuffer,                .flushMappedMemoryRanges,
    .freeMemory,                   .getBufferMemoryRequirements,
    .getImageMemoryRequirements,   .mapMemory,
    .unmapMemory,                  .cmdCopyBuffer,
    .getBufferMemoryRequirements2,
};
const device_command = [_]vk.DeviceCommand{
    .destroyDevice,           .getDeviceQueue,        .createSemaphore,     .createFence,
    .createImageView,         .destroyImageView,      .destroySemaphore,    .destroyFence,
    .getSwapchainImagesKHR,   .createSwapchainKHR,    .destroySwapchainKHR, .acquireNextImageKHR,
    .deviceWaitIdle,          .waitForFences,         .resetFences,         .queueSubmit,
    .queuePresentKHR,         .createCommandPool,     .destroyCommandPool,  .allocateCommandBuffers,
    .freeCommandBuffers,      .queueWaitIdle,         .createShaderModule,  .destroyShaderModule,
    .createPipelineLayout,    .destroyPipelineLayout, .createRenderPass,    .destroyRenderPass,
    .createGraphicsPipelines, .destroyPipeline,       .createFramebuffer,   .destroyFramebuffer,
    .beginCommandBuffer,      .endCommandBuffer,      .cmdBeginRenderPass,  .cmdEndRenderPass,
    .cmdBindPipeline,         .cmdDraw,               .cmdSetViewport,      .cmdSetScissor,
    .cmdBindVertexBuffers,
} ++ device_vma;
const DeviceDispatch = vk.DeviceWrapper(&device_command);

pub const GraphicsContext = struct {
    vkb: BaseDispatch,
    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    instance: vk.Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,

    dev: vk.Device,
    graphics_queue: Queue,
    present_queue: Queue,
    allocator: vma.Allocator,

    pub fn init(allocator: Allocator, app_name: [*:0]const u8, window: glfw.Window) !GraphicsContext {
        var self: GraphicsContext = undefined;
        const vk_proc = @ptrCast(fn (instance: vk.Instance, procname: [*:0]const u8) callconv(.C) vk.PfnVoidFunction, glfw.getInstanceProcAddress);
        self.vkb = try BaseDispatch.load(vk_proc);

        const glfw_exts = try glfw.getRequiredInstanceExtensions();

        const app_info = vk.ApplicationInfo{
            .p_application_name = app_name,
            .application_version = vk.makeApiVersion(2, 0, 0, 0),
            .p_engine_name = app_name,
            .engine_version = vk.makeApiVersion(0, 0, 0, 0),
            .api_version = vk.API_VERSION_1_2,
        };

        self.instance = try self.vkb.createInstance(&.{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = @intCast(u32, glfw_exts.len),
            .pp_enabled_extension_names = @ptrCast([*]const [*:0]const u8, &glfw_exts[0]),
        }, null);

        self.vki = try InstanceDispatch.load(self.instance, vk_proc);
        errdefer self.vki.destroyInstance(self.instance, null);

        self.surface = try createSurface(self.instance, window);
        errdefer self.vki.destroySurfaceKHR(self.instance, self.surface, null);

        const candidate = try pickPhysicalDevice(self.vki, self.instance, allocator, self.surface);
        self.pdev = candidate.pdev;
        self.props = candidate.props;
        self.dev = try initializeCandidate(self.vki, candidate);
        self.vkd = try DeviceDispatch.load(self.dev, self.vki.dispatch.vkGetDeviceProcAddr);
        errdefer self.vkd.destroyDevice(self.dev, null);

        self.graphics_queue = Queue.init(self.vkd, self.dev, candidate.queues.graphics_family);
        self.present_queue = Queue.init(self.vkd, self.dev, candidate.queues.present_family);

        self.mem_props = self.vki.getPhysicalDeviceMemoryProperties(self.pdev);

        // const name = comptime vk.BaseCommand.symbol(.createInstance);
        // std.log.info("{}", .{@field(self.vkb.dispatch, name)});
        // std.log.info("{}", .{self.vki});
        // std.log.info("{}", .{self.vkd});
        const vma_fns = getVmaVulkanFunction(self.vki, self.vkd);

        self.allocator = try vma.Allocator.create(.{
            .flags = .{},
            .physicalDevice = self.pdev,
            .device = self.dev,
            .frameInUseCount = 0,
            .pVulkanFunctions = &vma_fns,
            .instance = self.instance,
            .vulkanApiVersion = vk.API_VERSION_1_2,
        });
        return self;
    }

    pub fn deinit(self: GraphicsContext) void {
        self.allocator.destroy();
        self.vkd.destroyDevice(self.dev, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);
    }

    pub fn deviceName(self: GraphicsContext) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.props.device_name, 0).?;
        return self.props.device_name[0..len];
    }

    pub fn destroy(self: GraphicsContext, resource: anytype) void {
        const ResourceType = @TypeOf(resource);
        const destroyFn = blk: {
            if (ResourceType == vk.Image or ResourceType == vk.Buffer) {
                const name = @typeName(ResourceType);
                @compileError("Can not destroy single vk." ++ name ++ " need vma.Create" ++ name ++ "Result");
            }
            if (ResourceType == vma.CreateBufferResult) return self.allocator.destroyBuffer(resource);
            if (ResourceType == vma.CreateImageResult) return self.allocator.destroyImage(resource);
            if (ResourceType == vk.Event) break :blk DeviceDispatch.destroyEvent;
            if (ResourceType == vk.Fence) break :blk DeviceDispatch.destroyFence;
            if (ResourceType == vk.Buffer) break :blk DeviceDispatch.destroyBuffer;
            if (ResourceType == vk.Sampler) break :blk DeviceDispatch.destroySampler;
            if (ResourceType == vk.Pipeline) break :blk DeviceDispatch.destroyPipeline;
            if (ResourceType == vk.ImageView) break :blk DeviceDispatch.destroyImageView;
            if (ResourceType == vk.Semaphore) break :blk DeviceDispatch.destroySemaphore;
            if (ResourceType == vk.QueryPool) break :blk DeviceDispatch.destroyQueryPool;
            if (ResourceType == vk.BufferView) break :blk DeviceDispatch.destroyBufferView;
            if (ResourceType == vk.RenderPass) break :blk DeviceDispatch.destroyRenderPass;
            if (ResourceType == vk.CommandPool) break :blk DeviceDispatch.destroyCommandPool;
            if (ResourceType == vk.CuModuleNVX) break :blk DeviceDispatch.destroyCuModuleNVX;
            if (ResourceType == vk.Framebuffer) break :blk DeviceDispatch.destroyFramebuffer;
            if (ResourceType == vk.ShaderModule) break :blk DeviceDispatch.destroyShaderModule;
            if (ResourceType == vk.SwapchainKHR) break :blk DeviceDispatch.destroySwapchainKHR;
            if (ResourceType == vk.PipelineCache) break :blk DeviceDispatch.destroyPipelineCache;
            if (ResourceType == vk.CuFunctionNVX) break :blk DeviceDispatch.destroyCuFunctionNVX;
            if (ResourceType == vk.PipelineLayout) break :blk DeviceDispatch.destroyPipelineLayout;
            if (ResourceType == vk.DescriptorPool) break :blk DeviceDispatch.destroyDescriptorPool;
            if (ResourceType == vk.VideoSessionKHR) break :blk DeviceDispatch.destroyVideoSessionKHR;
            if (ResourceType == vk.ValidationCacheEXT) break :blk DeviceDispatch.destroyValidationCacheEXT;
            if (ResourceType == vk.PrivateDataSlotEXT) break :blk DeviceDispatch.destroyPrivateDataSlotEXT;
            if (ResourceType == vk.DescriptorSetLayout) break :blk DeviceDispatch.destroyDescriptorSetLayout;
            if (ResourceType == vk.DeferredOperationKHR) break :blk DeviceDispatch.destroyDeferredOperationKHR;
            if (ResourceType == vk.RayTracingPipelinesNV) break :blk DeviceDispatch.destroyRayTracingPipelinesNV;
            if (ResourceType == vk.RayTracingPipelinesKHR) break :blk DeviceDispatch.destroyRayTracingPipelinesKHR;
            if (ResourceType == vk.SamplerYcbcrConversion) break :blk DeviceDispatch.destroySamplerYcbcrConversion;
            if (ResourceType == vk.BufferCollectionFUCHSIA) break :blk DeviceDispatch.destroyBufferCollectionFUCHSIA;
            if (ResourceType == vk.AccelerationStructureNV) break :blk DeviceDispatch.destroyAccelerationStructureNV;
            if (ResourceType == vk.AccelerationStructureKHR) break :blk DeviceDispatch.destroyAccelerationStructureKHR;
            if (ResourceType == vk.DescriptorUpdateTemplate) break :blk DeviceDispatch.destroyDescriptorUpdateTemplate;
            @compileError(@typeName(ResourceType) ++ " don't have drop function");
        };
        destroyFn(self.vkd, self.dev, resource, null);
    }

    pub fn create(self: GraphicsContext, create_info: anytype) !CreateInfoToType(@TypeOf(create_info)) {
        const CreateInfo = @TypeOf(create_info);
        const createFn = blk: {
            if (CreateInfo == vk.ImageCreateInfo or CreateInfo == vk.BufferCreateInfo) {
                @compileError("using createImage or createBuffer instead");
            }

            if (CreateInfo == vk.EventCreateInfo) break :blk DeviceDispatch.createEvent;
            if (CreateInfo == vk.FenceCreateInfo) break :blk DeviceDispatch.createFence;
            if (CreateInfo == vk.SamplerCreateInfo) break :blk DeviceDispatch.createSampler;
            if (CreateInfo == vk.ImageViewCreateInfo) break :blk DeviceDispatch.createImageView;
            if (CreateInfo == vk.SemaphoreCreateInfo) break :blk DeviceDispatch.createSemaphore;
            if (CreateInfo == vk.QueryPoolCreateInfo) break :blk DeviceDispatch.createQueryPool;
            if (CreateInfo == vk.BufferViewCreateInfo) break :blk DeviceDispatch.createBufferView;
            if (CreateInfo == vk.RenderPassCreateInfo) break :blk DeviceDispatch.createRenderPass;
            if (CreateInfo == vk.CommandPoolCreateInfo) break :blk DeviceDispatch.createCommandPool;
            if (CreateInfo == vk.RenderPassCreateInfo) break :blk DeviceDispatch.createRenderPass2;
            if (CreateInfo == vk.CuModuleCreateInfoNVX) break :blk DeviceDispatch.createCuModuleNVX;
            if (CreateInfo == vk.FramebufferCreateInfo) break :blk DeviceDispatch.createFramebuffer;
            if (CreateInfo == vk.ShaderModuleCreateInfo) break :blk DeviceDispatch.createShaderModule;
            if (CreateInfo == vk.SwapchainCreateInfoKHR) break :blk DeviceDispatch.createSwapchainKHR;
            if (CreateInfo == vk.PipelineCacheCreateInfo) break :blk DeviceDispatch.createPipelineCache;
            if (CreateInfo == vk.CuFunctionCreateInfoNVX) break :blk DeviceDispatch.createCuFunctionNVX;
            if (CreateInfo == vk.PipelineLayoutCreateInfo) break :blk DeviceDispatch.createPipelineLayout;
            if (CreateInfo == vk.DescriptorPoolCreateInfo) break :blk DeviceDispatch.createDescriptorPool;
            if (CreateInfo == vk.VideoSessionCreateInfoKHR) break :blk DeviceDispatch.createVideoSessionKHR;
            if (CreateInfo == vk.ValidationCacheCreateInfoEXT) break :blk DeviceDispatch.createValidationCacheEXT;
            if (CreateInfo == vk.PrivateDataSlotCreateInfoEXT) break :blk DeviceDispatch.createPrivateDataSlotEXT;
            if (CreateInfo == vk.DescriptorSetLayoutCreateInfo) break :blk DeviceDispatch.createDescriptorSetLayout;
            if (CreateInfo == vk.SamplerYcbcrConversionCreateInfo) break :blk DeviceDispatch.createSamplerYcbcrConversion;
            if (CreateInfo == vk.BufferCollectionCreateInfoFUCHSIA) break :blk DeviceDispatch.createBufferCollectionFUCHSIA;
            if (CreateInfo == vk.AccelerationStructureCreateInfoNV) break :blk DeviceDispatch.createAccelerationStructureNV;
            if (CreateInfo == vk.AccelerationStructureCreateInfoKHR) break :blk DeviceDispatch.createAccelerationStructureKHR;
            if (CreateInfo == vk.DescriptorUpdateTemplateCreateInfo) break :blk DeviceDispatch.createDescriptorUpdateTemplate;
            if (CreateInfo == vk.IndirectCommandsLayoutCreateInfoNV) break :blk DeviceDispatch.createIndirectCommandsLayoutNV;

            const pipelineCreateFn = blk2: {
                if (CreateInfo == vk.ComputePipelineCreateInfo) break :blk2 DeviceDispatch.createComputePipelines;
                if (CreateInfo == vk.GraphicsPipelineCreateInfo) break :blk2 DeviceDispatch.createGraphicsPipelines;
                if (CreateInfo == vk.RayTracingPipelineCreateInfoNV) break :blk2 DeviceDispatch.createRayTracingPipelinesNV;
                if (CreateInfo == vk.RayTracingPipelineCreateInfoKHR) break :blk2 DeviceDispatch.createRayTracingPipelinesKHR;
                @compileError(@typeName(CreateInfo) ++ " don't have create function");
            };
            var pipeline: vk.Pipeline = undefined;
            _ = try pipelineCreateFn(
                self.vkd,
                self.dev,
                .null_handle,
                1,
                @ptrCast([*]const CreateInfo, &create_info),
                null,
                @ptrCast([*]vk.Pipeline, &pipeline),
            );
            return pipeline;
        };
        return try createFn(self.vkd, self.dev, &create_info, null);
    }

    pub fn createBuffer(
        self: GraphicsContext,
        buffer_create_info: vk.BufferCreateInfo,
        allocation_create_info: vma.AllocatorCreateInfo,
    ) !vma.CreateBufferResult {
        return self.allocator.createBuffer(buffer_create_info, allocation_create_info);
    }
    pub fn createImage(
        self: GraphicsContext,
        image_create_info: vk.ImageCreateInfo,
        allocation_create_info: vma.AllocatorCreateInfo,
    ) !vma.CreateImageResult {
        return self.allocator.createImage(image_create_info, allocation_create_info);
    }
};


fn CreateInfoToType(comptime T: type) type {
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
pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    fn init(vkd: DeviceDispatch, dev: vk.Device, family: u32) Queue {
        return .{
            .handle = vkd.getDeviceQueue(dev, family, 0),
            .family = family,
        };
    }
};

fn createSurface(instance: vk.Instance, window: glfw.Window) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    if ((try glfw.createWindowSurface(instance, window, null, &surface)) != @enumToInt(vk.Result.success)) {
        return error.SurfaceInitFailed;
    }

    return surface;
}

fn initializeCandidate(vki: InstanceDispatch, candidate: DeviceCandidate) !vk.Device {
    const priority = [_]f32{1};
    const qci = [_]vk.DeviceQueueCreateInfo{
        .{
            .flags = .{},
            .queue_family_index = candidate.queues.graphics_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
        .{
            .flags = .{},
            .queue_family_index = candidate.queues.present_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
    };

    const queue_count: u32 = if (candidate.queues.graphics_family == candidate.queues.present_family)
        1
    else
        2;

    return try vki.createDevice(candidate.pdev, &.{
        .flags = .{},
        .queue_create_info_count = queue_count,
        .p_queue_create_infos = &qci,
        .enabled_layer_count = 0,
        .pp_enabled_layer_names = undefined,
        .enabled_extension_count = required_device_extensions.len,
        .pp_enabled_extension_names = @ptrCast([*]const [*:0]const u8, &required_device_extensions),
        .p_enabled_features = null,
    }, null);
}

const DeviceCandidate = struct {
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    queues: QueueAllocation,
};

const QueueAllocation = struct {
    graphics_family: u32,
    present_family: u32,
};

fn pickPhysicalDevice(
    vki: InstanceDispatch,
    instance: vk.Instance,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) !DeviceCandidate {
    var device_count: u32 = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &device_count, null);

    const pdevs = try allocator.alloc(vk.PhysicalDevice, device_count);
    defer allocator.free(pdevs);

    _ = try vki.enumeratePhysicalDevices(instance, &device_count, pdevs.ptr);

    for (pdevs) |pdev| {
        if (try checkSuitable(vki, pdev, allocator, surface)) |candidate| {
            return candidate;
        }
    }

    return error.NoSuitableDevice;
}

fn checkSuitable(
    vki: InstanceDispatch,
    pdev: vk.PhysicalDevice,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) !?DeviceCandidate {
    const props = vki.getPhysicalDeviceProperties(pdev);

    if (!try checkExtensionSupport(vki, pdev, allocator)) {
        return null;
    }

    if (!try checkSurfaceSupport(vki, pdev, surface)) {
        return null;
    }

    if (try allocateQueues(vki, pdev, allocator, surface)) |allocation| {
        return DeviceCandidate{
            .pdev = pdev,
            .props = props,
            .queues = allocation,
        };
    }

    return null;
}

fn allocateQueues(vki: InstanceDispatch, pdev: vk.PhysicalDevice, allocator: Allocator, surface: vk.SurfaceKHR) !?QueueAllocation {
    var family_count: u32 = undefined;
    vki.getPhysicalDeviceQueueFamilyProperties(pdev, &family_count, null);

    const families = try allocator.alloc(vk.QueueFamilyProperties, family_count);
    defer allocator.free(families);
    vki.getPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families.ptr);

    var graphics_family: ?u32 = null;
    var present_family: ?u32 = null;

    for (families) |properties, i| {
        const family = @intCast(u32, i);

        if (graphics_family == null and properties.queue_flags.graphics_bit) {
            graphics_family = family;
        }

        if (present_family == null and (try vki.getPhysicalDeviceSurfaceSupportKHR(pdev, family, surface)) == vk.TRUE) {
            present_family = family;
        }
    }

    if (graphics_family != null and present_family != null) {
        return QueueAllocation{
            .graphics_family = graphics_family.?,
            .present_family = present_family.?,
        };
    }

    return null;
}

fn checkSurfaceSupport(vki: InstanceDispatch, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
    var format_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);

    var present_mode_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);

    return format_count > 0 and present_mode_count > 0;
}

fn checkExtensionSupport(
    vki: InstanceDispatch,
    pdev: vk.PhysicalDevice,
    allocator: Allocator,
) !bool {
    var count: u32 = undefined;
    _ = try vki.enumerateDeviceExtensionProperties(pdev, null, &count, null);

    const propsv = try allocator.alloc(vk.ExtensionProperties, count);
    defer allocator.free(propsv);

    _ = try vki.enumerateDeviceExtensionProperties(pdev, null, &count, propsv.ptr);

    for (required_device_extensions) |ext| {
        for (propsv) |props| {
            const len = std.mem.indexOfScalar(u8, &props.extension_name, 0).?;
            const prop_ext_name = props.extension_name[0..len];
            if (std.mem.eql(u8, ext, prop_ext_name)) {
                break;
            }
        } else {
            return false;
        }
    }

    return true;
}

/// --------- Device Dispatch --------------
/// vma_vulkan_func.vkAllocateMemory                    = vkAllocateMemory
/// vma_vulkan_func.vkBindBufferMemory                  = vkBindBufferMemory
/// vma_vulkan_func.vkBindImageMemory                   = vkBindImageMemory
/// vma_vulkan_func.vkCreateBuffer                      = vkCreateBuffer
/// vma_vulkan_func.vkCreateImage                       = vkCreateImage
/// vma_vulkan_func.vkDestroyBuffer                     = vkDestroyBuffer
/// vma_vulkan_func.vkDestroyImage                      = vkDestroyImage
/// vma_vulkan_func.vkFlushMappedMemoryRanges           = vkFlushMappedMemoryRanges
/// vma_vulkan_func.vkFreeMemory                        = vkFreeMemory
/// vma_vulkan_func.vkGetBufferMemoryRequirements       = vkGetBufferMemoryRequirements
/// vma_vulkan_func.vkGetImageMemoryRequirements        = vkGetImageMemoryRequirements
/// vma_vulkan_func.vkInvalidateMappedMemoryRanges      = vkInvalidateMappedMemoryRanges
/// vma_vulkan_func.vkMapMemory                         = vkMapMemory
/// vma_vulkan_func.vkUnmapMemory                       = vkUnmapMemory
/// vma_vulkan_func.vkCmdCopyBuffer                     = vkCmdCopyBuffer
/// ---------- Instance Dispatch ---------------
/// vma_vulkan_func.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties
/// vma_vulkan_func.vkGetPhysicalDeviceProperties       = vkGetPhysicalDeviceProperties
fn getVmaVulkanFunction(vki: InstanceDispatch, vkd: DeviceDispatch) vma.VulkanFunctions {
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
