const std = @import("std");
const assert = std.debug.assert;
const vk = @import("vulkan");
const vma = @import("binding/vma.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const srcToString = @import("util.zig").srcToString;

pub const CreateInfo = struct {
    size: vk.DeviceSize,
    buffer_usage: vk.BufferUsageFlags,
    memory_usage: vma.MemoryUsage,
    memory_flags: vma.AllocationCreateFlags,
};

const Buffer = @This();
buffer: vk.Buffer,
allocation: vma.Allocation,
info: vma.AllocationInfo,
create_info: CreateInfo,

pub fn init(gc: GraphicsContext, create_info: CreateInfo, object_name: ?[*:0]const u8) !Buffer {
    var allocation_info: vma.AllocationInfo = undefined;
    const result = try gc.allocator.createBufferAndGetInfo(
        .{
            .flags = .{},
            .size = create_info.size,
            .usage = create_info.buffer_usage,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        },
        .{
            .flags = create_info.memory_flags,
            .usage = create_info.memory_usage,
        },
        &allocation_info,
    );

    try gc.markHandle(result.buffer, .buffer, object_name);
    assert(allocation_info.pMappedData == null and !create_info.memory_flags.contains(.{ .create_mapped = true }));
    var buffer: Buffer = undefined;
    buffer.buffer = result.buffer;
    buffer.allocation = result.allocation;
    buffer.info = allocation_info;
    buffer.create_info = create_info;
    return buffer;
}

pub fn deinit(self: Buffer, gc: GraphicsContext) void {
    gc.allocator.destroyBuffer(self.buffer, self.allocation);
}

pub fn upload(self: Buffer, comptime T: type, gc: GraphicsContext, data: []const T) !void {
    switch (self.create_info.memory_usage) {
        .gpu_only => {
            const size = @sizeOf(T) * data.len;
            const stage_buffer = try Buffer.init(gc, .{
                .size = size,
                .buffer_usage = .{ .transfer_src_bit = true },
                .memory_usage = .cpu_to_gpu,
                .memory_flags = .{},
            }, srcToString(@src()));
            defer stage_buffer.deinit(gc);
            stage_buffer.upload(T, gc, data) catch unreachable;
            try stage_buffer.copyToBuffer(self, gc);
        },
        .cpu_to_gpu => {
            const gpu_mem = if (self.info.pMappedData) |mem|
                @intToPtr([*]T, @ptrToInt(mem))
            else
                try gc.allocator.mapMemory(self.allocation, T);

            for (data) |d, i| {
                gpu_mem[i] = d;
            }

            // Flush allocation
            try gc.allocator.flushAllocation(self.allocation, 0, self.info.size);
            if (self.info.pMappedData == null) {
                gc.allocator.unmapMemory(self.allocation);
            }
        },
        else => unreachable,
    }
}

fn copyToBuffer(src: Buffer, dst: Buffer, gc: GraphicsContext) !void {
    // TODO: because smallest buffer size is 256 byte.
    // if data size is < 256, group multiple data to one buffer
    // std.log.info("src: info size: {}, data size: {}", .{src.info.size, src.size});
    // std.log.info("dst: info size: {}, data size: {}", .{dst.info.size, dst.size});
    const cmdbuf = try gc.beginOneTimeCommandBuffer();

    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = src.create_info.size,
    };
    gc.vkd.cmdCopyBuffer(cmdbuf, src.buffer, dst.buffer, 1, @ptrCast([*]const vk.BufferCopy, &region));

    try gc.endOneTimeCommandBuffer(cmdbuf);
}
