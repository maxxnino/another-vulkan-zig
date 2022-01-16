const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");
const glfw = @import("glfw");
const vma = @import("vma.zig");
const Buffer = @import("Buffer.zig");
const VulkanDispatch = @import("VulkanDispatch.zig");
const BaseDispatch = VulkanDispatch.BaseDispatch;
const InstanceDispatch = VulkanDispatch.InstanceDispatch;
const DeviceDispatch = VulkanDispatch.DeviceDispatch;
const Allocator = std.mem.Allocator;
const enable_safety = VulkanDispatch.enable_safety;
const srcToString = @import("util.zig").srcToString;

const required_device_extensions = [_][:0]const u8{
    vk.extension_info.khr_swapchain.name,
    vk.extension_info.ext_descriptor_indexing.name,
};

const required_instance_extensions = [_][:0]const u8{
    vk.extension_info.ext_debug_utils.name,
};

const required_device_feature = vk.PhysicalDeviceFeatures{
    .sampler_anisotropy = vk.TRUE,
};

pub const GraphicsContext = struct {
    vkb: BaseDispatch,
    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    instance: vk.Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    pool: vk.CommandPool,

    dev: vk.Device,
    graphics_queue: Queue,
    present_queue: Queue,
    allocator: vma.Allocator,
    debug_message: if (enable_safety) vk.DebugUtilsMessengerEXT else void,

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

        const best_practices_validation = blk: {
            if (enable_safety) {
                break :blk &vk.ValidationFeaturesEXT{
                    .enabled_validation_feature_count = 1,
                    .p_enabled_validation_features = @ptrCast(
                        [*]const vk.ValidationFeatureEnableEXT,
                        &vk.ValidationFeatureEnableEXT.best_practices_ext,
                    ),
                    .disabled_validation_feature_count = 0,
                    .p_disabled_validation_features = undefined,
                };
            }

            break :blk null;
        };

        self.instance = try self.vkb.createInstance(&.{
            .flags = .{},
            .p_next = best_practices_validation,
            .p_application_info = &app_info,
            .enabled_layer_count = if (enable_safety) 1 else 0,
            .pp_enabled_layer_names = if (enable_safety) @ptrCast(
                [*]const [*:0]const u8,
                &"VK_LAYER_KHRONOS_validation",
            ) else undefined,
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
        self.dev = try initializeCandidate(self.vki, candidate, allocator);
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

        return self;
    }

    pub fn deinit(self: GraphicsContext) void {
        self.allocator.destroy();
        self.destroy(self.pool);
        self.vkd.destroyDevice(self.dev, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_message, null);
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
        defer self.vkd.freeCommandBuffers(self.dev, self.pool, 1, @ptrCast([*]const vk.CommandBuffer, &cmdbuf));
        try self.vkd.endCommandBuffer(cmdbuf);

        const si = vk.SubmitInfo{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = undefined,
            .p_wait_dst_stage_mask = undefined,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &cmdbuf),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = undefined,
        };
        try self.vkd.queueSubmit(self.graphics_queue.handle, 1, @ptrCast([*]const vk.SubmitInfo, &si), .null_handle);
        try self.vkd.queueWaitIdle(self.graphics_queue.handle);
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
            self.vkd.cmdInsertDebugUtilsLabelEXT(command_buffer, .{
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

fn initializeCandidate(vki: InstanceDispatch, candidate: DeviceCandidate, allocator: Allocator) !vk.Device {
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

    var device_exts = try std.ArrayList([*:0]const u8).initCapacity(
        allocator,
        required_device_extensions.len,
    );
    defer device_exts.deinit();
    for (required_device_extensions) |e| {
        try device_exts.append(e);
    }

    return try vki.createDevice(candidate.pdev, &.{
        .flags = .{},
        .queue_create_info_count = queue_count,
        .p_queue_create_infos = &qci,
        .enabled_layer_count = 0,
        .pp_enabled_layer_names = undefined,
        .enabled_extension_count = @truncate(u32, device_exts.items.len),
        .pp_enabled_extension_names = @ptrCast([*]const [*:0]const u8, &device_exts.items[0]),
        .p_enabled_features = &required_device_feature,
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

    // for (propsv) |props| {
    //     const len = std.mem.indexOfScalar(u8, &props.extension_name, 0).?;
    //     const prop_ext_name = props.extension_name[0..len];
    //     std.log.info("{s}", .{prop_ext_name});
    // }

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
