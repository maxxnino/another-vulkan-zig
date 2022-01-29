const std = @import("std");
const vk = @import("vulkan");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const StbImage = @import("binding/stb_image.zig").StbImage;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const TextureType = enum {
    cube_map,
    texture,
};

pub const Texture = struct {
    pub const Config = struct {
        /// true mean enable anisotropy
        anisotropy: bool = false,

        /// true mean enable mip map
        mip_map: bool = false,
    };
    image: Image,
    view: vk.ImageView,
    smapler: vk.Sampler,
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
        try stage_buffer.update(u8, gc, buffer);
        const width = switch (@"type") {
            .texture => in_width,
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
            if (config.mip_map) .{ .all_transfer_bit_khr = true } else .{},
            .{ .all_transfer_bit_khr = true },
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
                .{ .all_transfer_bit_khr = true },
                .{ .fragment_shader_bit_khr = true },
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

        texture.smapler = try gc.create(vk.SamplerCreateInfo{
            .flags = .{},
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .mip_lod_bias = 0,
            .anisotropy_enable = if (config.anisotropy == true) vk.TRUE else vk.FALSE,
            .max_anisotropy = if (config.anisotropy == true) gc.props.limits.max_sampler_anisotropy else undefined,
            .compare_enable = vk.FALSE,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = if (config.mip_map) @intToFloat(f32, mip_levels) else 0,
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = vk.FALSE,
        }, label);
        return texture;
    }

    pub fn loadFromFile(gc: GraphicsContext, comptime @"type": TextureType, filename: [*:0]const u8, config: Config) !Texture {
        const image = try StbImage.loadFromFile(filename, .rgb_alpha);
        defer image.free();
        return loadFromMemory(gc, @"type", image.pixels, image.width, image.height, image.channels, config, filename);
    }

    pub fn deinit(self: Texture, gc: GraphicsContext) void {
        self.image.deinit(gc);
        gc.destroy(self.view);
        gc.destroy(self.smapler);
    }
};

pub const DepthStencilTexture = struct {
    image: Image,
    view: vk.ImageView,

    pub fn init(gc: GraphicsContext, width: u32, height: u32, label: ?[*:0]const u8) !DepthStencilTexture {
        var texture: DepthStencilTexture = undefined;
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
            .{ .early_fragment_tests_bit_khr = true },
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

    pub fn deinit(self: DepthStencilTexture, gc: GraphicsContext) void {
        self.image.deinit(gc);
        gc.destroy(self.view);
    }
};

fn calcMipLevel(width: u32, height: u32) u32 {
    const log2 = std.math.log2(std.math.max(width, height));
    return @floatToInt(u32, std.math.floor(@intToFloat(f32, log2))) + 1;
}

pub const RenderTarget = struct {
    image: Image,
    view: vk.ImageView,

    pub fn init(gc: GraphicsContext, width: u32, height: u32, format: vk.Format, label: ?[*:0]const u8) !RenderTarget {
        var texture: RenderTarget = undefined;
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

    pub fn deinit(self: RenderTarget, gc: GraphicsContext) void {
        self.image.deinit(gc);
        gc.destroy(self.view);
    }
};
