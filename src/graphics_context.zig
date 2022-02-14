const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");
const glfw = @import("glfw");
const vma = @import("binding/vma.zig");
const Buffer = @import("Buffer.zig");
const VulkanDispatch = @import("VulkanDispatch.zig");
const BaseDispatch = VulkanDispatch.BaseDispatch;
const InstanceDispatch = VulkanDispatch.InstanceDispatch;
const DeviceDispatch = VulkanDispatch.DeviceDispatch;
const Allocator = std.mem.Allocator;
const enable_safety = VulkanDispatch.enable_safety;
const srcToString = @import("util.zig").srcToString;

const required_device_extensions = [_][*:0]const u8{
    vk.extension_info.khr_swapchain.name,
    vk.extension_info.ext_descriptor_indexing.name,
    vk.extension_info.khr_synchronization_2.name,
    vk.extension_info.khr_push_descriptor.name,
};

const required_instance_extensions = [_][*:0]const u8{
    vk.extension_info.ext_debug_utils.name,
};

const required_device_feature = vk.PhysicalDeviceFeatures{
    .sampler_anisotropy = vk.TRUE,
    .sample_rate_shading = vk.TRUE,
    .texture_compression_bc = vk.TRUE,
};

const required_instance_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_synchronization2",
} ++ if (enable_safety) [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"} else [_][*:0]const u8{};

const required_validation_features = [_]vk.ValidationFeatureEnableEXT{
    .gpu_assisted_ext,
    .best_practices_ext,
    .synchronization_validation_ext,
};

pub const GraphicsContext = struct {
    vkb: BaseDispatch,
    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    instance: vk.Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    feature: vk.PhysicalDeviceFeatures,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    pool: vk.CommandPool,

    dev: vk.Device,
    graphics_queue: Queue,
    present_queue: Queue,
    allocator: vma.Allocator,
    support_format: vk.Format,
    debug_message: if (enable_safety) vk.DebugUtilsMessengerEXT else void,

    immutable_samplers: vk.Sampler,

    pub fn init(allocator: Allocator, app_name: [*:0]const u8, window: glfw.Window) !GraphicsContext {
        var self: GraphicsContext = undefined;
        const vk_proc = @ptrCast(
            fn (instance: vk.Instance, procname: [*:0]const u8) callconv(.C) vk.PfnVoidFunction,
            glfw.getInstanceProcAddress,
        );
        self.vkb = try BaseDispatch.load(vk_proc);

        const glfw_exts = try glfw.getRequiredInstanceExtensions();

        const app_info = vk.ApplicationInfo{
            .p_application_name = app_name,
            .application_version = vk.makeApiVersion(2, 0, 0, 0),
            .p_engine_name = app_name,
            .engine_version = vk.makeApiVersion(0, 0, 0, 0),
            .api_version = vk.API_VERSION_1_1,
        };
        var instance_exts = blk: {
            if (enable_safety) {
                var exts = try std.ArrayList([*:0]const u8).initCapacity(
                    allocator,
                    glfw_exts.len + required_instance_extensions.len,
                );
                {
                    try exts.appendSlice(glfw_exts);
                    for (required_instance_extensions) |e| {
                        try exts.append(e);
                    }
                }
                break :blk exts.toOwnedSlice();
            }

            break :blk glfw_exts;
        };
        defer if (enable_safety) {
            allocator.free(instance_exts);
        };

        const validation_features = blk: {
            if (enable_safety) {
                break :blk &vk.ValidationFeaturesEXT{
                    .enabled_validation_feature_count = @truncate(u32, required_validation_features.len),
                    .p_enabled_validation_features = @ptrCast(
                        [*]const vk.ValidationFeatureEnableEXT,
                        &required_validation_features,
                    ),
                    .disabled_validation_feature_count = 0,
                    .p_disabled_validation_features = undefined,
                };
            }

            break :blk null;
        };

        self.instance = try self.vkb.createInstance(&.{
            .flags = .{},
            .p_next = validation_features,
            .p_application_info = &app_info,
            .enabled_layer_count = if (enable_safety) 2 else 1,
            .pp_enabled_layer_names = &required_instance_layers,
            .enabled_extension_count = @intCast(u32, instance_exts.len),
            .pp_enabled_extension_names = @ptrCast([*]const [*:0]const u8, &instance_exts[0]),
        }, null);
        self.vki = try InstanceDispatch.load(self.instance, vk_proc);
        errdefer self.vki.destroyInstance(self.instance, null);

        //setup debug utils
        if (enable_safety) {
            self.debug_message = try self.vki.createDebugUtilsMessengerEXT(self.instance, &.{
                .flags = .{},
                .message_severity = .{
                    .verbose_bit_ext = true,
                    .info_bit_ext = true,
                    .warning_bit_ext = true,
                    .error_bit_ext = true,
                },
                .message_type = .{
                    .general_bit_ext = true,
                    .validation_bit_ext = true,
                    .performance_bit_ext = true,
                },
                .pfn_user_callback = debugCallback,
                .p_user_data = null,
            }, null);
        }
        self.surface = try createSurface(self.instance, window);
        errdefer self.vki.destroySurfaceKHR(self.instance, self.surface, null);

        const candidate = try pickPhysicalDevice(self.vki, self.instance, allocator, self.surface);
        self.pdev = candidate.pdev;
        self.props = candidate.props;
        self.feature = candidate.feature;
        // check if physical device suport BC7 compression for albedo image
        std.debug.assert(isFormatSupport(self.vki, self.pdev, .bc7_srgb_block));

        // check if physical device suport BC7 compression for normal, metallic, ao..
        std.debug.assert(isFormatSupport(self.vki, self.pdev, .bc7_unorm_block));

        self.dev = try initializeCandidate(self.vki, candidate);
        self.vkd = try DeviceDispatch.load(self.dev, self.vki.dispatch.vkGetDeviceProcAddr);
        try self.markHandle(self.dev, .device, srcToString(@src()));
        errdefer self.vkd.destroyDevice(self.dev, null);

        self.graphics_queue = Queue.init(self.vkd, self.dev, candidate.queues.graphics_family);
        self.present_queue = Queue.init(self.vkd, self.dev, candidate.queues.present_family);

        self.mem_props = self.vki.getPhysicalDeviceMemoryProperties(self.pdev);

        const vma_fns = VulkanDispatch.getVmaVulkanFunction(self.vki, self.vkd);

        self.allocator = try vma.Allocator.create(.{
            .flags = .{},
            .physicalDevice = self.pdev,
            .device = self.dev,
            .frameInUseCount = 0,
            .pVulkanFunctions = &vma_fns,
            .instance = self.instance,
            .vulkanApiVersion = vk.API_VERSION_1_1,
        });

        self.pool = try self.create(vk.CommandPoolCreateInfo{
            .flags = .{},
            .queue_family_index = self.graphics_queue.family,
        }, srcToString(@src()));

        self.immutable_samplers = try self.create(vk.SamplerCreateInfo{
            .flags = .{},
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .mip_lod_bias = 0,
            .anisotropy_enable = vk.TRUE,
            .max_anisotropy = self.props.limits.max_sampler_anisotropy,
            .compare_enable = vk.FALSE,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = vk.LOD_CLAMP_NONE,
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = vk.FALSE,
        }, "Graphic Context Immutable Sampler");

        return self;
    }

    pub fn deinit(self: GraphicsContext) void {
        self.destroy(self.immutable_samplers);
        self.allocator.destroy();
        self.destroy(self.pool);
        self.vkd.destroyDevice(self.dev, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        if (enable_safety) {
            self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_message, null);
        }

        self.vki.destroyInstance(self.instance, null);
    }

    pub fn deviceName(self: GraphicsContext) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.props.device_name, 0).?;
        return self.props.device_name[0..len];
    }

    pub fn destroy(self: GraphicsContext, resource: anytype) void {
        VulkanDispatch.destroy(self.vkd, self.dev, resource);
    }

    pub fn create(self: GraphicsContext, create_info: anytype, object_name: ?[*:0]const u8) !VulkanDispatch.CreateInfoToType(@TypeOf(create_info)) {
        return VulkanDispatch.create(self.vkd, self.dev, create_info, object_name);
    }

    pub fn beginOneTimeCommandBuffer(self: GraphicsContext) !vk.CommandBuffer {
        var cmdbuf: vk.CommandBuffer = undefined;
        try self.vkd.allocateCommandBuffers(self.dev, &.{
            .command_pool = self.pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast([*]vk.CommandBuffer, &cmdbuf));

        try self.vkd.beginCommandBuffer(cmdbuf, &.{
            .flags = .{ .one_time_submit_bit = true },
            .p_inheritance_info = null,
        });
        return cmdbuf;
    }

    pub fn endOneTimeCommandBuffer(self: GraphicsContext, cmdbuf: vk.CommandBuffer) !void {
        try self.vkd.endCommandBuffer(cmdbuf);
        // Create fence to ensure that the command buffer has finished executing
        const fence = try self.create(vk.FenceCreateInfo{ .flags = .{} }, srcToString(@src()));
        errdefer self.destroy(fence);

        // Submit to the queue
        try self.vkd.queueSubmit2KHR(self.graphics_queue.handle, 1, &[_]vk.SubmitInfo2KHR{.{
            .flags = .{},
            .wait_semaphore_info_count = 0,
            .p_wait_semaphore_infos = undefined,
            .command_buffer_info_count = 1,
            .p_command_buffer_infos = &[_]vk.CommandBufferSubmitInfoKHR{.{
                .command_buffer = cmdbuf,
                .device_mask = 0,
            }},
            .signal_semaphore_info_count = 0,
            .p_signal_semaphore_infos = undefined,
        }}, fence);

        // Wait for the fence to signal that command buffer has finished executing
        _ = try self.vkd.waitForFences(self.dev, 1, @ptrCast([*]const vk.Fence, &fence), vk.TRUE, std.math.maxInt(u64));

        self.destroy(fence);
        self.vkd.freeCommandBuffers(self.dev, self.pool, 1, @ptrCast([*]const vk.CommandBuffer, &cmdbuf));
    }

    pub fn printInstanceLayerAndExtension(self: GraphicsContext, allocator: Allocator) !void {
        {
            //layer
            std.log.info("====== Layer ======", .{});
            var layer_count: u32 = undefined;
            _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);
            var layers = try allocator.alloc(vk.LayerProperties, layer_count);
            _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, layers.ptr);
            for (layers) |l| {
                printLayer(l);
            }

            std.log.info("\n====== Extension ======", .{});
            //extention
            var instance_ext_count: u32 = undefined;
            _ = try self.vkb.enumerateInstanceExtensionProperties(null, &instance_ext_count, null);
            var instance_exts = try allocator.alloc(vk.ExtensionProperties, instance_ext_count);
            _ = try self.vkb.enumerateInstanceExtensionProperties(null, &instance_ext_count, instance_exts.ptr);
            for (instance_exts) |e| {
                printExtention(e);
            }
        }
    }

    pub fn markCommandBuffer(self: GraphicsContext, command_buffer: vk.CommandBuffer, label: [*:0]const u8) void {
        if (enable_safety) {
            self.vkd.cmdInsertDebugUtilsLabelEXT(command_buffer, &.{
                .p_label_name = label,
                .color = [4]f32{ 0, 0, 0, 0 },
            });
        }
    }

    pub fn markHandle(self: GraphicsContext, handle: anytype, object_type: vk.ObjectType, label: ?[*:0]const u8) !void {
        if (enable_safety) {
            if (label) |value| {
                try self.vkd.setDebugUtilsObjectNameEXT(self.dev, &.{
                    .object_type = object_type,
                    .object_handle = @enumToInt(handle),
                    .p_object_name = value,
                });
            }
        }
    }

    pub fn getSampleCount(self: GraphicsContext) vk.SampleCountFlags {
        _ = self;
        return .{ .@"4_bit" = true };
    }
};

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

    // enable khr_synchronization_2 feature
    const khr_synchronization_2 = vk.PhysicalDeviceSynchronization2FeaturesKHR{
        .synchronization_2 = vk.TRUE,
        // .p_next = null,
    };
    const descriptor_indexing = vk.PhysicalDeviceDescriptorIndexingFeatures{
        .p_next = @ptrCast(*const anyopaque, &khr_synchronization_2),
        // .shader_input_attachment_array_dynamic_indexing= Bool32 = FALSE,
        // .shader_uniform_texel_buffer_array_dynamic_indexing= Bool32 = FALSE,
        // .shader_storage_texel_buffer_array_dynamic_indexing= Bool32 = FALSE,
        // .shader_uniform_buffer_array_non_uniform_indexing= Bool32 = FALSE,
        .shader_sampled_image_array_non_uniform_indexing = vk.TRUE,
        // .shader_storage_buffer_array_non_uniform_indexing= Bool32 = FALSE,
        // .shader_storage_image_array_non_uniform_indexing= Bool32 = FALSE,
        // .shader_input_attachment_array_non_uniform_indexing= Bool32 = FALSE,
        // .shader_uniform_texel_buffer_array_non_uniform_indexing= Bool32 = FALSE,
        // .shader_storage_texel_buffer_array_non_uniform_indexing= Bool32 = FALSE,
        // .descriptor_binding_uniform_buffer_update_after_bind= Bool32 = FALSE,
        // .descriptor_binding_sampled_image_update_after_bind= Bool32 = FALSE,
        // .descriptor_binding_storage_image_update_after_bind= Bool32 = FALSE,
        // .descriptor_binding_storage_buffer_update_after_bind= Bool32 = FALSE,
        // .descriptor_binding_uniform_texel_buffer_update_after_bind= Bool32 = FALSE,
        // .descriptor_binding_storage_texel_buffer_update_after_bind= Bool32 = FALSE,
        .descriptor_binding_update_unused_while_pending = vk.TRUE,
        .descriptor_binding_partially_bound = vk.TRUE,
        .descriptor_binding_variable_descriptor_count = vk.TRUE,
        .runtime_descriptor_array = vk.TRUE,
    };

    return try vki.createDevice(candidate.pdev, &.{
        .flags = .{},
        .p_next = @ptrCast(*const anyopaque, &descriptor_indexing),
        .queue_create_info_count = queue_count,
        .p_queue_create_infos = &qci,
        .enabled_layer_count = 0,
        .pp_enabled_layer_names = undefined,
        .enabled_extension_count = required_device_extensions.len,
        .pp_enabled_extension_names = @ptrCast([*]const [*:0]const u8, &required_device_extensions),
        .p_enabled_features = &required_device_feature,
    }, null);
}

const DeviceCandidate = struct {
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    feature: vk.PhysicalDeviceFeatures,
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

    const feature = vki.getPhysicalDeviceFeatures(pdev);
    inline for (std.meta.fields(vk.PhysicalDeviceFeatures)) |field| {
        if (@field(required_device_feature, field.name) == vk.TRUE) {
            if (@field(feature, field.name) == vk.FALSE) return null;
        }
    }

    if (try allocateQueues(vki, pdev, allocator, surface)) |allocation| {
        return DeviceCandidate{
            .pdev = pdev,
            .props = props,
            .feature = feature,
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

fn isFormatSupport(vki: InstanceDispatch, p_dev: vk.PhysicalDevice, format: vk.Format) bool {
    const fp = vki.getPhysicalDeviceFormatProperties(p_dev, format);
    return fp.optimal_tiling_features.contains(.{
        .sampled_image_bit = true,
        .transfer_dst_bit = true,
    });
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

    // for (propsv) |props| {
    //     const len = std.mem.indexOfScalar(u8, &props.extension_name, 0).?;
    //     const prop_ext_name = props.extension_name[0..len];
    //     std.log.info("{s}", .{prop_ext_name});
    // }

    for (required_device_extensions) |ext| {
        for (propsv) |props| {
            const len = std.mem.indexOfScalar(u8, &props.extension_name, 0).?;
            const prop_ext_name = props.extension_name[0..len];
            if (std.mem.eql(u8, std.mem.span(ext), prop_ext_name)) {
                break;
            }
        } else {
            return false;
        }
    }

    return true;
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT.IntType,
    message_types: vk.DebugUtilsMessageTypeFlagsEXT.IntType,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    _ = message_types;
    _ = p_user_data;

    if (p_callback_data) |data| {
        const level = (vk.DebugUtilsMessageSeverityFlagsEXT{
            .warning_bit_ext = true,
        }).toInt();
        if (message_severity >= level) {
            std.log.info("{s}", .{data.p_message});

            if (data.object_count > 0) {
                std.log.info("----------Objects {}-----------\n", .{data.object_count});
                var i: u32 = 0;
                while (i < data.object_count) : (i += 1) {
                    const o: vk.DebugUtilsObjectNameInfoEXT = data.p_objects[i];
                    std.log.info("[{}-{s}]: {s}", .{
                        i,
                        @tagName(o.object_type),
                        o.p_object_name,
                    });
                }
                std.log.info("----------End Object-----------\n", .{});
            }
            if (data.cmd_buf_label_count > 0) {
                std.log.info("----------Labels {}------------\n", .{data.object_count});
                var i: u32 = 0;
                while (i < data.cmd_buf_label_count) : (i += 1) {
                    const o: vk.DebugUtilsLabelEXT = data.p_cmd_buf_labels[i];
                    std.log.info("[{}]: {s}", .{
                        i,
                        o.p_label_name,
                    });
                }
                std.log.info("----------End Label------------\n", .{});
            }
        }
    }

    return vk.FALSE;
}

fn printExtention(ext: vk.ExtensionProperties) void {
    printNullTerminateSlice(&ext.extension_name);
}

fn printLayer(layer: vk.LayerProperties) void {
    printNullTerminateSlice(&layer.layer_name);
    printNullTerminateSlice(&layer.description);
}

fn printNullTerminateSlice(str: []const u8) void {
    const len = std.mem.indexOfScalar(u8, str, 0).?;
    std.log.info("{s}", .{str[0..len]});
}
