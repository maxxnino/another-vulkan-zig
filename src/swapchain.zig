const std = @import("std");
const vk = @import("vulkan");
const CommandBuffer = @import("command_pool.zig").CommandBuffer;
const srcToString = @import("util.zig").srcToString;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

const command_pool = @import("command_pool.zig");
const DrawPool = command_pool.DrawPool;
const Allocator = std.mem.Allocator;

pub const Swapchain = struct {
    pub const PresentState = enum {
        optimal,
        suboptimal,
    };

    gc: *const GraphicsContext,
    allocator: Allocator,

    surface_format: vk.SurfaceFormatKHR,
    present_mode: vk.PresentModeKHR,
    extent: vk.Extent2D,
    handle: vk.SwapchainKHR,

    swap_images: []SwapImage,
    image_index: u32,
    next_image_acquired: vk.Semaphore,

    pub fn init(gc: *const GraphicsContext, allocator: Allocator, extent: vk.Extent2D) !Swapchain {
        return try initRecycle(gc, allocator, extent, .null_handle);
    }

    pub fn initRecycle(gc: *const GraphicsContext, allocator: Allocator, extent: vk.Extent2D, old_handle: vk.SwapchainKHR) !Swapchain {
        const caps = try gc.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(gc.pdev, gc.surface);
        const actual_extent = findActualExtent(caps, extent);
        if (actual_extent.width == 0 or actual_extent.height == 0) {
            return error.InvalidSurfaceDimensions;
        }

        const surface_format = try findSurfaceFormat(gc, allocator);
        const present_mode = try findPresentMode(gc, allocator);

        var image_count = caps.min_image_count + 1;
        if (caps.max_image_count > 0) {
            image_count = std.math.min(image_count, caps.max_image_count);
        }

        const qfi = [_]u32{ gc.graphics_queue.family, gc.present_queue.family };
        const sharing_mode: vk.SharingMode = if (gc.graphics_queue.family != gc.present_queue.family)
            .concurrent
        else
            .exclusive;
        const qfis = if (sharing_mode == .exclusive)
            qfi[0..0]
        else
            qfi[0..1];

        const handle = try gc.vkd.createSwapchainKHR(gc.dev, &.{
            .flags = .{},
            .surface = gc.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = actual_extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true, .transfer_dst_bit = true },
            .image_sharing_mode = sharing_mode,
            .queue_family_index_count = @truncate(u32, qfis.len),
            .p_queue_family_indices = qfis.ptr,
            .pre_transform = caps.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = old_handle,
        }, null);
        errdefer gc.vkd.destroySwapchainKHR(gc.dev, handle, null);

        if (old_handle != .null_handle) {
            // Apparently, the old swapchain handle still needs to be destroyed after recreating.
            gc.vkd.destroySwapchainKHR(gc.dev, old_handle, null);
        }

        const swap_images = try initSwapchainImages(gc, handle, surface_format.format, allocator);
        errdefer for (swap_images) |si| si.deinit(gc);

        var next_image_acquired = try gc.create(vk.SemaphoreCreateInfo{ .flags = .{} }, srcToString(@src()));
        errdefer gc.destroy(next_image_acquired);

        const result = try gc.vkd.acquireNextImageKHR(gc.dev, handle, std.math.maxInt(u64), next_image_acquired, .null_handle);
        if (result.result != .success) {
            return error.ImageAcquireFailed;
        }

        std.mem.swap(vk.Semaphore, &swap_images[result.image_index].image_acquired, &next_image_acquired);
        return Swapchain{
            .gc = gc,
            .allocator = allocator,
            .surface_format = surface_format,
            .present_mode = present_mode,
            .extent = actual_extent,
            .handle = handle,
            .swap_images = swap_images,
            .image_index = result.image_index,
            .next_image_acquired = next_image_acquired,
        };
    }

    fn deinitExceptSwapchain(self: Swapchain) void {
        for (self.swap_images) |si| si.deinit(self.gc);
        self.gc.destroy(self.next_image_acquired);
    }

    pub fn waitForAllFences(self: Swapchain) !void {
        for (self.swap_images) |si| si.waitForFence(self.gc) catch {};
    }

    pub fn deinit(self: Swapchain) void {
        self.deinitExceptSwapchain();
        self.gc.vkd.destroySwapchainKHR(self.gc.dev, self.handle, null);
    }

    pub fn recreate(self: *Swapchain, new_extent: vk.Extent2D) !void {
        const gc = self.gc;
        const allocator = self.allocator;
        const old_handle = self.handle;
        self.deinitExceptSwapchain();
        self.* = try initRecycle(gc, allocator, new_extent, old_handle);
    }

    pub fn currentImage(self: Swapchain) vk.Image {
        return self.swap_images[self.image_index].image;
    }

    pub fn currentSwapImage(self: Swapchain) *const SwapImage {
        return &self.swap_images[self.image_index];
    }

    pub fn present(self: *Swapchain, cmdbuf: CommandBuffer, draw_pool: *DrawPool) !PresentState {
        // Simple method:
        // 1) Acquire next image
        // 2) Wait for and reset fence of the acquired image
        // 3) Submit command buffer with fence of acquired image,
        //    dependendent on the semaphore signalled by the first step.
        // 4) Present current frame, dependent on semaphore signalled by previous step
        // Problem: This way we can't reference the current image while rendering.
        // Better method: Shuffle the steps around such that acquire next image is the last step,
        // leaving the swapchain in a state with the current image.
        // 1) Wait for and reset fence of current image
        // 2) Submit command buffer, signalling fence of current image and dependent on
        //    the semaphore signalled by step 4.
        // 3) Present current frame, dependent on semaphore signalled by the submit
        // 4) Acquire next image, signalling its semaphore
        // One problem that arises is that we can't know beforehand which semaphore to signal,
        // so we keep an extra auxilery semaphore that is swapped around

        // Step 1: Make sure the current frame has finished rendering
        const current = self.currentSwapImage();
        try current.waitForFence(self.gc);
        if (current.cmdbuf) |cmd| {
            try draw_pool.done(self.gc.*, cmd);
        }

        try self.gc.vkd.resetFences(self.gc.dev, 1, @ptrCast([*]const vk.Fence, &current.frame_fence));

        // Step 2: Submit the command buffer
        self.swap_images[self.image_index].cmdbuf = cmdbuf;

        try self.gc.vkd.queueSubmit2KHR(self.gc.graphics_queue.handle, 1, &[_]vk.SubmitInfo2KHR{.{
            .flags = .{},
            .wait_semaphore_info_count = 1,
            .p_wait_semaphore_infos = &[_]vk.SemaphoreSubmitInfoKHR{.{
                .semaphore = current.image_acquired,
                .value = 1,
                .stage_mask = .{},
                .device_index = 0,
            }},
            .command_buffer_info_count = 1,
            .p_command_buffer_infos = &[_]vk.CommandBufferSubmitInfoKHR{.{
                .command_buffer = cmdbuf.cmd,
                .device_mask = 0,
            }},
            .signal_semaphore_info_count = 1,
            .p_signal_semaphore_infos = &[_]vk.SemaphoreSubmitInfoKHR{.{
                .semaphore = current.render_finished,
                .value = 1,
                .stage_mask = .{},
                .device_index = 0,
            }},
        }}, current.frame_fence);

        // Step 3: Present the current frame
        _ = try self.gc.vkd.queuePresentKHR(self.gc.present_queue.handle, &.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast([*]const vk.Semaphore, &current.render_finished),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast([*]const vk.SwapchainKHR, &self.handle),
            .p_image_indices = @ptrCast([*]const u32, &self.image_index),
            .p_results = null,
        });

        // Step 4: Acquire next frame
        const result = try self.gc.vkd.acquireNextImageKHR(
            self.gc.dev,
            self.handle,
            std.math.maxInt(u64),
            self.next_image_acquired,
            .null_handle,
        );

        std.mem.swap(vk.Semaphore, &self.swap_images[result.image_index].image_acquired, &self.next_image_acquired);
        self.image_index = result.image_index;

        return switch (result.result) {
            .success => .optimal,
            .suboptimal_khr => .suboptimal,
            else => unreachable,
        };
    }
};

const SwapImage = struct {
    image: vk.Image,
    view: vk.ImageView,
    image_acquired: vk.Semaphore,
    render_finished: vk.Semaphore,
    frame_fence: vk.Fence,
    cmdbuf: ?CommandBuffer = null,

    fn init(gc: *const GraphicsContext, image: vk.Image, format: vk.Format) !SwapImage {
        const view = try gc.create(vk.ImageViewCreateInfo{
            .flags = .{},
            .image = image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, srcToString(@src()));
        errdefer gc.destroy(view);

        const image_acquired = try gc.create(vk.SemaphoreCreateInfo{ .flags = .{} }, srcToString(@src()));
        errdefer gc.destroy(image_acquired);

        const render_finished = try gc.create(vk.SemaphoreCreateInfo{ .flags = .{} }, srcToString(@src()));
        errdefer gc.destroy(render_finished);

        const frame_fence = try gc.create(vk.FenceCreateInfo{ .flags = .{ .signaled_bit = true } }, srcToString(@src()));
        errdefer gc.destroy(frame_fence);

        return SwapImage{
            .image = image,
            .view = view,
            .image_acquired = image_acquired,
            .render_finished = render_finished,
            .frame_fence = frame_fence,
        };
    }

    fn deinit(self: SwapImage, gc: *const GraphicsContext) void {
        self.waitForFence(gc) catch return;
        gc.destroy(self.view);
        gc.destroy(self.image_acquired);
        gc.destroy(self.render_finished);
        gc.destroy(self.frame_fence);
    }

    fn waitForFence(self: SwapImage, gc: *const GraphicsContext) !void {
        _ = try gc.vkd.waitForFences(gc.dev, 1, @ptrCast([*]const vk.Fence, &self.frame_fence), vk.TRUE, std.math.maxInt(u64));
    }
};

fn initSwapchainImages(gc: *const GraphicsContext, swapchain: vk.SwapchainKHR, format: vk.Format, allocator: Allocator) ![]SwapImage {
    var count: u32 = undefined;
    _ = try gc.vkd.getSwapchainImagesKHR(gc.dev, swapchain, &count, null);
    const images = try allocator.alloc(vk.Image, count);
    defer allocator.free(images);
    _ = try gc.vkd.getSwapchainImagesKHR(gc.dev, swapchain, &count, images.ptr);

    const swap_images = try allocator.alloc(SwapImage, count);
    errdefer allocator.free(swap_images);

    var i: usize = 0;
    errdefer for (swap_images[0..i]) |si| si.deinit(gc);

    for (images) |image| {
        swap_images[i] = try SwapImage.init(gc, image, format);
        i += 1;
    }

    return swap_images;
}

fn findSurfaceFormat(gc: *const GraphicsContext, allocator: Allocator) !vk.SurfaceFormatKHR {
    const preferred = vk.SurfaceFormatKHR{
        .format = .b8g8r8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };

    var count: u32 = undefined;
    _ = try gc.vki.getPhysicalDeviceSurfaceFormatsKHR(gc.pdev, gc.surface, &count, null);
    const surface_formats = try allocator.alloc(vk.SurfaceFormatKHR, count);
    defer allocator.free(surface_formats);
    _ = try gc.vki.getPhysicalDeviceSurfaceFormatsKHR(gc.pdev, gc.surface, &count, surface_formats.ptr);

    for (surface_formats) |sfmt| {
        if (std.meta.eql(sfmt, preferred)) {
            return preferred;
        }
    }

    return surface_formats[0]; // There must always be at least one supported surface format
}

fn findPresentMode(gc: *const GraphicsContext, allocator: Allocator) !vk.PresentModeKHR {
    var count: u32 = undefined;
    _ = try gc.vki.getPhysicalDeviceSurfacePresentModesKHR(gc.pdev, gc.surface, &count, null);
    const present_modes = try allocator.alloc(vk.PresentModeKHR, count);
    defer allocator.free(present_modes);
    _ = try gc.vki.getPhysicalDeviceSurfacePresentModesKHR(gc.pdev, gc.surface, &count, present_modes.ptr);

    const preferred = [_]vk.PresentModeKHR{
        .fifo_khr,
        .mailbox_khr,
        .immediate_khr,
    };

    for (preferred) |mode| {
        if (std.mem.indexOfScalar(vk.PresentModeKHR, present_modes, mode) != null) {
            return mode;
        }
    }

    return .fifo_khr;
}

fn findActualExtent(caps: vk.SurfaceCapabilitiesKHR, extent: vk.Extent2D) vk.Extent2D {
    if (caps.current_extent.width != 0xFFFF_FFFF) {
        return caps.current_extent;
    } else {
        return .{
            .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
            .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
        };
    }
}
