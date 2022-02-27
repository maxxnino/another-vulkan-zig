pool: vk.QueryPool,

const srcToString = @import("util.zig").srcToString;
const std = @import("std");
const vk = @import("vulkan");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Self = @This();

pub fn init(gc: GraphicsContext, count: usize) !Self {
    var self: Self = undefined;

    self.pool = try gc.create(vk.QueryPoolCreateInfo{
        .flags = .{},
        .query_type = .timestamp,
        .query_count = @truncate(u32, count) * 2,
        .pipeline_statistics = .{},
    }, "query");

    return self;
}

pub fn start(self: Self, gc: GraphicsContext, command_buffer: vk.CommandBuffer, query: u32) !u32 {
    var result = [_]u32{ 0, 0 };
    _ = try gc.vkd.getQueryPoolResults(
        gc.dev,
        self.pool,
        query * 2,
        2,
        @sizeOf(@TypeOf(result)),
        @ptrCast(*anyopaque, &result),
        @sizeOf(u32),
        .{},
    );

    gc.vkd.cmdResetQueryPool(command_buffer, self.pool, query * 2, 2);
    gc.vkd.cmdWriteTimestamp(command_buffer, .{ .bottom_of_pipe_bit = true }, self.pool, query * 2);
    return if(result[1] > result[0]) result[1] - result[0] else 0;
}

pub fn end(self: Self, gc: GraphicsContext, command_buffer: vk.CommandBuffer, query: u32) void {
    gc.vkd.cmdWriteTimestamp(command_buffer, .{ .bottom_of_pipe_bit = true }, self.pool, query * 2 + 1);
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.pool);
}
