const std = @import("std");
const assert = std.debug.assert;
const vk = @import("vulkan");
const vma = @import("vma.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Buffer = @import("Buffer.zig");
const srcToString = @import("util.zig").srcToString;

pub const CreateInfo = struct {
    flags: vk.ImageCreateFlags,
    image_type: vk.ImageType,
    format: vk.Format,
    extent: vk.Extent3D,
    mip_levels: u32,
    array_layers: u32,
    samples: vk.SampleCountFlags,
    tiling: vk.ImageTiling,
    usage: vk.ImageUsageFlags,
    memory_usage: vma.MemoryUsage,
    memory_flags: vma.AllocationCreateFlags,
};

const Image = @This();
image: vk.Image,
allocation: vma.Allocation,
layout: vk.ImageLayout,
format: vk.Format,

pub fn init(gc: GraphicsContext, create_info: CreateInfo, object_name: ?[*:0]const u8) !Image {
    const result = try gc.allocator.createImage(
        .{
            .flags = create_info.flags,
            .image_type = create_info.image_type,
            .format = create_info.format,
            .extent = create_info.extent,
            .mip_levels = create_info.mip_levels,
            .array_layers = create_info.array_layers,
            .samples = create_info.samples,
            .tiling = create_info.tiling,
            .usage = create_info.usage,
            .initial_layout = .@"undefined",
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        },
        .{
            .flags = create_info.memory_flags,
            .usage = create_info.memory_usage,
        },
    );

    try gc.markHandle(result.image, .image, object_name);
    var image: Image = undefined;
    image.image = result.image;
    image.allocation = result.allocation;
    image.format = create_info.format;
    image.layout = .@"undefined";
    return image;
}

pub fn deinit(self: Image, gc: GraphicsContext) void {
    gc.allocator.destroyImage(self.image, self.allocation);
}

pub fn changeLayout(
    self: *Image,
    gc: GraphicsContext,
    cmdbuf: vk.CommandBuffer,
    new_layout: vk.ImageLayout,
    src_access_mask: vk.AccessFlags,
    dst_access_mask: vk.AccessFlags,
    src_stage_mask: vk.PipelineStageFlags,
    dst_stage_mask: vk.PipelineStageFlags,
    subresource_range: vk.ImageSubresourceRange,
) void {
    const barrier = vk.ImageMemoryBarrier{
        .src_access_mask = src_access_mask,
        .dst_access_mask = dst_access_mask,
        .old_layout = self.layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = self.image,
        .subresource_range = subresource_range,
    };
    gc.vkd.cmdPipelineBarrier(
        cmdbuf,
        src_stage_mask,
        dst_stage_mask,
        .{},
        0,
        undefined,
        0,
        undefined,
        1,
        @ptrCast([*]const vk.ImageMemoryBarrier, &barrier),
    );

    self.layout = new_layout;
}

pub fn copyFromBuffer(self: Image, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, src: Buffer, width: u32, height: u32) void {
    const bic = vk.BufferImageCopy{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .image_offset = .{
            .x = 0,
            .y = 0,
            .z = 0,
        },
        .image_extent = .{
            .width = width,
            .height = height,
            .depth = 1,
        },
    };
    gc.vkd.cmdCopyBufferToImage(
        cmdbuf,
        src.buffer,
        self.image,
        .transfer_dst_optimal,
        1,
        @ptrCast([*]const vk.BufferImageCopy, &bic),
    );
}
// pub fn update(self: Buffer, comptime T: type, gc: GraphicsContext, in_data: []const T) !void {
//     const gpu_mem = if (self.info.pMappedData) |data|
//         @intToPtr([*]T, @ptrToInt(data))
//     else
//         try gc.allocator.mapMemory(self.allocation, T);

//     for (in_data) |d, i| {
//         gpu_mem[i] = d;
//     }
//     try self.flush(gc);
// }

// fn flush(self: Buffer, gc: GraphicsContext) !void {
//     try gc.allocator.flushAllocation(self.allocation, 0, self.info.size);
//     if (self.info.pMappedData == null) {
//         gc.allocator.unmapMemory(self.allocation);
//     }
// }

// pub fn copyToBuffer(src: Buffer, dst: Buffer, gc: GraphicsContext) !void {
//     // TODO: because smallest buffer size is 256 byte.
//     // if data size is < 256, group multiple data to one buffer
//     // std.log.info("src: info size: {}, data size: {}", .{src.info.size, src.size});
//     // std.log.info("dst: info size: {}, data size: {}", .{dst.info.size, dst.size});
//     const cmdbuf = try gc.beginOneTimeCommandBuffer();

//     const region = vk.BufferCopy{
//         .src_offset = 0,
//         .dst_offset = 0,
//         .size = src.size,
//     };
//     gc.vkd.cmdCopyBuffer(cmdbuf, src.buffer, dst.buffer, 1, @ptrCast([*]const vk.BufferCopy, &region));

//     try gc.endOneTimeCommandBuffer(cmdbuf);
// }

// pub fn copyFromBuffer() void {}
