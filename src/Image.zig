const std = @import("std");
const assert = std.debug.assert;
const vk = @import("vulkan");
const vma = @import("binding/vma.zig");
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
width: u32,
height: u32,

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
    image.layout = .@"undefined";
    image.format = create_info.format;
    image.width = create_info.extent.width;
    image.height = create_info.extent.height;
    return image;
}

pub fn deinit(self: Image, gc: GraphicsContext) void {
    gc.allocator.destroyImage(self.image, self.allocation);
}

pub fn changeLayout(
    self: *Image,
    gc: GraphicsContext,
    cmdbuf: vk.CommandBuffer,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    src_access_mask: vk.AccessFlags2KHR,
    dst_access_mask: vk.AccessFlags2KHR,
    src_stage_mask: vk.PipelineStageFlags2KHR,
    dst_stage_mask: vk.PipelineStageFlags2KHR,
    subresource_range: vk.ImageSubresourceRange,
) void {
    const barrier = vk.ImageMemoryBarrier2KHR{
        .src_stage_mask = src_stage_mask,
        .src_access_mask = src_access_mask,
        .dst_stage_mask = dst_stage_mask,
        .dst_access_mask = dst_access_mask,
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = self.image,
        .subresource_range = subresource_range,
    };
    const di = vk.DependencyInfoKHR{
        .dependency_flags = .{},
        .memory_barrier_count = 0,
        .p_memory_barriers = undefined,
        .buffer_memory_barrier_count = 0,
        .p_buffer_memory_barriers = undefined,
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = @ptrCast([*]const vk.ImageMemoryBarrier2KHR, &barrier),
    };
    gc.vkd.cmdPipelineBarrier2KHR(cmdbuf, &di);
    // gc.vkd.cmdPipelineBarrier(
    //     cmdbuf,
    //     src_stage_mask,
    //     dst_stage_mask,
    //     .{},
    //     0,
    //     undefined,
    //     0,
    //     undefined,
    //     1,
    //     @ptrCast([*]const vk.ImageMemoryBarrier, &barrier),
    // );
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

pub fn generateMipMap(
    self: *Image,
    gc: GraphicsContext,
    cmdbuf: vk.CommandBuffer,
    subresource_range: vk.ImageSubresourceRange,
) void {
    std.debug.assert(self.layout == .transfer_dst_optimal);
    var sr = subresource_range;
    sr.level_count = 1;
    sr.base_mip_level = 0;

    // Transition first mip level to transfer source for read during blit
    self.changeLayout(
        gc,
        cmdbuf,
        .transfer_dst_optimal,
        .transfer_src_optimal,
        .{ .transfer_write_bit_khr = true },
        .{ .transfer_read_bit_khr = true },
        .{ .all_transfer_bit_khr = true },
        .{ .all_transfer_bit_khr  = true },
        sr,
    );

    // generate mip level
    var i: u5 = 1;
    while (i < subresource_range.level_count) : (i += 1) {
        const ib = vk.ImageBlit{
            .src_subresource = .{
                .aspect_mask = sr.aspect_mask,
                .mip_level = i - 1,
                .base_array_layer = sr.base_array_layer,
                .layer_count = 1,
            },
            .src_offsets = .{
                .{ .x = 0, .y = 0, .z = 0 },
                .{
                    .x = @intCast(i32, self.width >> (i - 1)),
                    .y = @intCast(i32, self.height >> (i - 1)),
                    .z = 1,
                },
            },
            .dst_subresource = .{
                .aspect_mask = sr.aspect_mask,
                .mip_level = i,
                .base_array_layer = sr.base_array_layer,
                .layer_count = 1,
            },
            .dst_offsets = .{
                .{ .x = 0, .y = 0, .z = 0 },
                .{
                    .x = @intCast(i32, self.width >> i),
                    .y = @intCast(i32, self.height >> i),
                    .z = 1,
                },
            },
        };
        // Prepare current mip level as image blit destination
        sr.base_mip_level = i;
        self.changeLayout(
            gc,
            cmdbuf,
            .@"undefined",
            .transfer_dst_optimal,
            .{},
            .{ .transfer_write_bit_khr = true },
            .{ .all_transfer_bit_khr = true },
            .{ .all_transfer_bit_khr = true },
            sr,
        );
        // Blit from previous level
        gc.vkd.cmdBlitImage(
            cmdbuf,
            self.image,
            .transfer_src_optimal,
            self.image,
            .transfer_dst_optimal,
            1,
            @ptrCast([*]const vk.ImageBlit, &ib),
            .linear,
        );
        // Prepare current mip level as image blit source for next level
        self.changeLayout(
            gc,
            cmdbuf,
            .transfer_dst_optimal,
            .transfer_src_optimal,
            .{ .transfer_write_bit_khr = true },
            .{ .transfer_read_bit_khr = true },
            .{ .all_transfer_bit_khr = true },
            .{ .all_transfer_bit_khr = true },
            sr,
        );
    }
    // After the loop, all mip layers are in TRANSFER_SRC layout, so transition all to SHADER_READ
    self.changeLayout(
        gc,
        cmdbuf,
        .transfer_src_optimal,
        .shader_read_only_optimal,
        .{ .transfer_read_bit_khr = true },
        .{ .shader_sampled_read_bit_khr = true },
        .{ .all_transfer_bit_khr  = true },
        .{ .fragment_shader_bit_khr = true },
        subresource_range,
    );
}
