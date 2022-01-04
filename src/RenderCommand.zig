const std = @import("std");
const vk = @import("vulkan");
const resources = @import("resources");
const vma = @import("vma.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Buffer = @import("Buffer.zig");
const Allocator = std.mem.Allocator;

const RenderCommand = @This();

command_buffers: []vk.CommandBuffer,
framebuffers: []vk.Framebuffer,
render_pass: vk.RenderPass,
pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
index: u32,
clear_value: vk.ClearValue,
viewport: vk.Viewport,
scissor: vk.Rect2D,
render_area: vk.Rect2D,

pub fn init(
    allocator: Allocator,
    gc: GraphicsContext,
    framebuffers: []vk.Framebuffer,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    extent: vk.Extent2D,
) !RenderCommand {
    var rc: RenderCommand = undefined;
    rc.viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @intToFloat(f32, extent.width),
        .height = @intToFloat(f32, extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };
    rc.clear_value = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
    };
    rc.scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };
    rc.render_area = rc.scissor;
    rc.index = 0;
    rc.framebuffers = framebuffers;
    rc.pipeline = pipeline;
    rc.pipeline_layout = pipeline_layout;
    rc.render_pass = render_pass;
    rc.command_buffers = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
    errdefer allocator.free(rc.command_buffers);

    try gc.vkd.allocateCommandBuffers(gc.dev, &.{
        .command_pool = gc.pool,
        .level = .primary,
        .command_buffer_count = @truncate(u32, rc.command_buffers.len),
    }, rc.command_buffers.ptr);

    return rc;
}

pub fn end(rc: *const RenderCommand, gc: GraphicsContext) !void {
    const cmdbuf = rc.command_buffers[rc.index - 1];
    gc.vkd.cmdEndRenderPass(cmdbuf);
    try gc.vkd.endCommandBuffer(cmdbuf);
}

pub fn deinit(rc: *RenderCommand, gc: GraphicsContext, allocator: Allocator) void {
    gc.vkd.freeCommandBuffers(gc.dev, gc.pool, @truncate(u32, rc.command_buffers.len), rc.command_buffers.ptr);
    allocator.free(rc.command_buffers);
}

pub fn begin(rc: *RenderCommand, gc: GraphicsContext) !?vk.CommandBuffer {
    if (rc.index >= rc.framebuffers.len) return null;
    const cmdbuf = rc.command_buffers[rc.index];
    try gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{},
        .p_inheritance_info = null,
    });

    gc.vkd.cmdSetViewport(cmdbuf, 0, 1, @ptrCast([*]const vk.Viewport, &rc.viewport));
    gc.vkd.cmdSetScissor(cmdbuf, 0, 1, @ptrCast([*]const vk.Rect2D, &rc.scissor));

    gc.vkd.cmdBeginRenderPass(cmdbuf, &.{
        .render_pass = rc.render_pass,
        .framebuffer = rc.framebuffers[rc.index],
        .render_area = rc.render_area,
        .clear_value_count = 1,
        .p_clear_values = @ptrCast([*]const vk.ClearValue, &rc.clear_value),
    }, .@"inline");

    gc.vkd.cmdBindPipeline(cmdbuf, .graphics, rc.pipeline);
    rc.index += 1;
    return cmdbuf;
}
