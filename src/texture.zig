const std = @import("std");
const vk = @import("vulkan");
const basisu = @import("binding/basisu.zig");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const StbImage = @import("binding/stb_image.zig").StbImage;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const TextureType = enum {
    /// photos/albedo textures
    srgb,

    /// for normal, metallic, ao roughness
    unorm,

    /// cube_map always in srgb space
    cube_map,
};

pub const Texture = struct {
    pub const Config = struct {
        /// true mean enable anisotropy
        anisotropy: bool = true,

        /// true mean enable mip map
        mip_map: bool = true,
    };
    image: Image,
    view: vk.ImageView,
    config: Config,

    pub fn loadFromMemory(
        gc: GraphicsContext,
        comptime @"type": TextureType,
        buffer: []const u8,
        in_width: u32,
        in_height: u32,
        channels: u32,
        config: Config,
        label: ?[*:0]const u8,
    ) !Texture {
        var texture: Texture = undefined;
        texture.config = config;
        const stage_buffer = try Buffer.init(gc, .{
            .size = buffer.len,
            .buffer_usage = .{ .transfer_src_bit = true },
            .memory_usage = .cpu_to_gpu,
            .memory_flags = .{},
        }, label);
        defer stage_buffer.deinit(gc);
        try stage_buffer.upload(u8, gc, buffer);
        const width = switch (@"type") {
            .srgb, .unorm => in_width,
            .cube_map => in_width / 6,
        };
        const height = in_height;

        const mip_levels = if (config.mip_map) calcMipLevel(width, height) else 1;
        texture.image = try Image.init(gc, .{
            .flags = if (@"type" == .cube_map) .{ .cube_compatible_bit = true } else .{},
            .image_type = .@"2d",
            .format = .r8g8b8a8_srgb,
            .extent = .{
                .width = width,
                .height = height,
                .depth = 1,
            },
            .mip_levels = mip_levels,
            .array_layers = if (@"type" == .cube_map) 6 else 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{
                .transfer_src_bit = true,
                .transfer_dst_bit = true,
                .sampled_bit = true,
            },
            .memory_usage = .gpu_only,
            .memory_flags = .{},
        }, label);
        var subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            // Only copy to the first mip map level,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = if (@"type" == .cube_map) 6 else 1,
        };

        const cmdbuf = try gc.beginOneTimeCommandBuffer();

        // Optimal image will be used as destination for the copy, so we must transfer from
        // our initial undefined image layout to the transfer destination layout
        texture.image.changeLayout(
            gc,
            cmdbuf,
            .@"undefined",
            .transfer_dst_optimal,
            comptime Image.accessMaskFrom(.@"undefined", .transfer_dst_optimal),
            if (config.mip_map) .{ .all_transfer_bit = true } else .{},
            .{ .all_transfer_bit = true },
            subresource_range,
        );
        // Copy the first mip of the chain, remaining mips will be generated if needed
        const bic = blk: {
            if (@"type" == .cube_map) {
                var temp: [6]vk.BufferImageCopy = undefined;
                const base_offset = channels * width;
                for (temp) |*t, index| {
                    const i = @truncate(u32, index);
                    t.* = vk.BufferImageCopy{
                        .buffer_offset = base_offset * i,
                        .buffer_row_length = in_width,
                        .buffer_image_height = height,
                        .image_subresource = .{
                            .aspect_mask = .{ .color_bit = true },
                            .mip_level = 0,
                            .base_array_layer = i,
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
                }
                break :blk temp;
            }
            break :blk [_]vk.BufferImageCopy{.{
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
            }};
        };
        gc.vkd.cmdCopyBufferToImage(
            cmdbuf,
            stage_buffer.buffer,
            texture.image.image,
            .transfer_dst_optimal,
            bic.len,
            &bic,
        );

        if (mip_levels > 1) {
            // pass total mip level to generate
            subresource_range.level_count = mip_levels;
            subresource_range.layer_count = 1;
            if (@"type" == .cube_map) {
                var i: u32 = 0;
                while (i < 6) : (i += 1) {
                    subresource_range.base_array_layer = i;
                    texture.image.generateMipMap(gc, cmdbuf, subresource_range);
                }
            } else {
                texture.image.generateMipMap(gc, cmdbuf, subresource_range);
            }
        } else {
            texture.image.changeLayout(
                gc,
                cmdbuf,
                .transfer_dst_optimal,
                .shader_read_only_optimal,
                comptime Image.accessMaskFrom(.transfer_dst_optimal, .shader_read_only_optimal),
                .{ .all_transfer_bit = true },
                .{ .fragment_shader_bit = true },
                subresource_range,
            );
        }

        try gc.endOneTimeCommandBuffer(cmdbuf);

        texture.view = try gc.create(vk.ImageViewCreateInfo{
            .flags = .{},
            .image = texture.image.image,
            .view_type = if (@"type" == .cube_map) .cube else .@"2d",
            .format = texture.image.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = mip_levels,
                .base_array_layer = 0,
                .layer_count = if (@"type" == .cube_map) 6 else 1,
            },
        }, label);

        return texture;
    }

    pub fn loadFromFile(gc: GraphicsContext, comptime @"type": TextureType, filename: [*:0]const u8, config: Config) !Texture {
        const image = try StbImage.loadFromFile(filename, .rgb_alpha);
        defer image.free();
        return loadFromMemory(gc, @"type", image.pixels, image.width, image.height, image.channels, config, filename);
    }

    pub fn loadCompressFromFile(
        gpa: std.mem.Allocator,
        gc: GraphicsContext,
        texture_type: TextureType,
        filename: [*:0]const u8,
    ) !Texture {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const allocator = arena.allocator();

        const format: vk.Format = switch (texture_type) {
            .srgb, .cube_map => .bc7_srgb_block,
            .unorm => .bc7_unorm_block,
        };

        var texture: Texture = undefined;
        const file = try std.fs.cwd().openFileZ(filename, .{});
        defer file.close();
        const data = try file.readToEndAlloc(allocator, std.math.maxInt(u32));
        const data_len = @truncate(u32, data.len);

        basisu.start(data.ptr, data_len);

        const bytes_per_block = basisu.bytesPerBlock(.bc7_rgba);

        const image_data = try calculateImageData(allocator, data.ptr, data_len, bytes_per_block);
        std.log.info("{s} is {}kb", .{ filename, image_data.total_bytes / 1024 });

        const image_info: struct {
            width: u32,
            height: u32,
            layer_count: u32,
            level_count: u32,
        } = blk: {
            if (texture_type == .cube_map) {
                std.debug.assert(image_data.images.len == 6);
                break :blk .{
                    .width = image_data.images[0].width,
                    .height = image_data.images[0].height,
                    .layer_count = 6,
                    .level_count = image_data.images[0].total_levels,
                };
            }
            std.debug.assert(image_data.images.len == 1);
            break :blk .{
                .width = image_data.images[0].width,
                .height = image_data.images[0].height,
                .layer_count = 1,
                .level_count = image_data.images[0].total_levels,
            };
        };
        const stage_buffer = try Buffer.init(gc, .{
            .size = image_data.total_bytes,
            .buffer_usage = .{ .transfer_src_bit = true },
            .memory_usage = .cpu_to_gpu,
            .memory_flags = .{},
        }, filename);
        defer stage_buffer.deinit(gc);

        var gpu_mem = try stage_buffer.mapMemory(gc, u8);
        for (image_data.levels) |*li| {
            std.debug.assert(basisu.transcodeImageLevel(
                data.ptr,
                data_len,
                li,
                // WARN: using num_blocks_x as placeholder for buffer offset
                @ptrCast(*anyopaque, &gpu_mem[li.num_blocks_x]),
                .bc7_rgba,
            ));
        }

        try stage_buffer.flushAllocation(gc);

        texture.image = try Image.init(gc, .{
            .flags = if (texture_type == .cube_map) .{ .cube_compatible_bit = true } else .{},
            .image_type = .@"2d",
            .format = format,
            .extent = .{
                .width = image_info.width,
                .height = image_info.height,
                .depth = 1,
            },
            .mip_levels = image_info.level_count,
            .array_layers = image_info.layer_count,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{
                .transfer_dst_bit = true,
                .sampled_bit = true,
            },
            .memory_usage = .gpu_only,
            .memory_flags = .{},
        }, filename);

        const cmdbuf = try gc.beginOneTimeCommandBuffer();
        const subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = image_info.level_count,
            .base_array_layer = 0,
            .layer_count = image_info.layer_count,
        };

        texture.image.changeLayout(
            gc,
            cmdbuf,
            .@"undefined",
            .transfer_dst_optimal,
            comptime Image.accessMaskFrom(.@"undefined", .transfer_dst_optimal),
            .{},
            .{ .all_transfer_bit = true },
            subresource_range,
        );

        gc.vkd.cmdCopyBufferToImage(
            cmdbuf,
            stage_buffer.buffer,
            texture.image.image,
            .transfer_dst_optimal,
            @truncate(u32, image_data.bics.len),
            image_data.bics.ptr,
        );
        texture.image.changeLayout(
            gc,
            cmdbuf,
            .transfer_dst_optimal,
            .shader_read_only_optimal,
            comptime Image.accessMaskFrom(.transfer_dst_optimal, .shader_read_only_optimal),
            .{ .all_transfer_bit = true },
            .{ .fragment_shader_bit = true },
            subresource_range,
        );

        try gc.endOneTimeCommandBuffer(cmdbuf);

        texture.view = try gc.create(vk.ImageViewCreateInfo{
            .flags = .{},
            .image = texture.image.image,
            .view_type = if (texture_type == .cube_map) .cube else .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = image_info.level_count,
                .base_array_layer = 0,
                .layer_count = image_info.layer_count,
            },
        }, filename);

        return texture;
    }

    pub fn createDepthStencilTexture(gc: GraphicsContext, width: u32, height: u32, label: ?[*:0]const u8) !Texture {
        var texture: Texture = undefined;
        texture.image = try Image.init(gc, .{
            .flags = .{},
            .image_type = .@"2d",
            .format = .d32_sfloat_s8_uint,
            .extent = .{
                .width = width,
                .height = height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = gc.getSampleCount(),
            .tiling = .optimal,
            .usage = .{
                .depth_stencil_attachment_bit = true,
            },
            .memory_usage = .gpu_only,
            .memory_flags = .{},
        }, label);
        const subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{ .depth_bit = true, .stencil_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        };
        const cmdbuf = try gc.beginOneTimeCommandBuffer();
        texture.image.changeLayout(
            gc,
            cmdbuf,
            .@"undefined",
            .depth_attachment_optimal,
            comptime Image.accessMaskFrom(.@"undefined", .depth_attachment_optimal),
            .{},
            .{ .early_fragment_tests_bit = true },
            subresource_range,
        );
        try gc.endOneTimeCommandBuffer(cmdbuf);

        texture.view = try gc.create(vk.ImageViewCreateInfo{
            .flags = .{},
            .image = texture.image.image,
            .view_type = .@"2d",
            .format = texture.image.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = subresource_range,
        }, label);
        return texture;
    }

    pub fn createRenderTexture(gc: GraphicsContext, width: u32, height: u32, format: vk.Format, label: ?[*:0]const u8) !Texture {
        var texture: Texture = undefined;
        texture.image = try Image.init(gc, .{
            .flags = .{},
            .image_type = .@"2d",
            .format = format,
            .extent = .{
                .width = width,
                .height = height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = gc.getSampleCount(),
            .tiling = .optimal,
            .usage = .{
                .transient_attachment_bit = true,
                .color_attachment_bit = true,
            },
            .memory_usage = .gpu_only,
            .memory_flags = .{},
        }, label);

        var subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        };

        texture.view = try gc.create(vk.ImageViewCreateInfo{
            .flags = .{},
            .image = texture.image.image,
            .view_type = .@"2d",
            .format = texture.image.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = subresource_range,
        }, label);

        return texture;
    }

    pub fn deinit(self: Texture, gc: GraphicsContext) void {
        self.image.deinit(gc);
        gc.destroy(self.view);
    }
};

fn calcMipLevel(width: u32, height: u32) u32 {
    const log2 = std.math.log2(std.math.max(width, height));
    return @floatToInt(u32, std.math.floor(@intToFloat(f32, log2))) + 1;
}

const ImageResult = struct {
    images: []basisu.ImageInfo,
    levels: []basisu.ImageLevelInfo,
    bics: []vk.BufferImageCopy,
    total_bytes: u32,
};

/// the caller owns the return slices in ImageResult
fn calculateImageData(allocator: std.mem.Allocator, data: [*]const u8, size: u32, bytes_per_block: u32) !ImageResult {
    var result: ImageResult = undefined;
    const total_image = basisu.totalImages(data, size);
    result.images = try allocator.alloc(basisu.ImageInfo, total_image);

    var total_image_levels: u32 = 0;
    {
        var i: u32 = 0;
        while (i < total_image) : (i += 1) {
            std.debug.assert(basisu.imageInfo(data, size, &result.images[i], i));
            total_image_levels += result.images[i].total_levels;
        }
    }

    result.levels = try allocator.alloc(basisu.ImageLevelInfo, total_image_levels);
    result.bics = try allocator.alloc(vk.BufferImageCopy, total_image_levels);
    result.total_bytes = 0;
    var level_index: u32 = 0;
    for (result.images) |ii, index| {
        var j: u32 = 0;
        const i = @truncate(u32, index);
        while (j < ii.total_levels) : (j += 1) {
            defer level_index += 1;

            std.debug.assert(basisu.imageLevelInfo(
                data,
                size,
                &result.levels[level_index],
                i,
                j,
            ));

            result.bics[level_index] = .{
                .buffer_offset = result.total_bytes,
                .buffer_row_length = 0,
                .buffer_image_height = 0,
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = j,
                    .base_array_layer = i,
                    .layer_count = 1,
                },
                .image_offset = .{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                .image_extent = .{
                    .width = result.levels[level_index].width,
                    .height = result.levels[level_index].height,
                    .depth = 1,
                },
            };
            // WARN: using num_blocks_x as placeholder for buffer offset
            result.levels[level_index].num_blocks_x = result.total_bytes;
            result.total_bytes += result.levels[level_index].total_blocks * bytes_per_block;
        }
    }

    return result;
}
