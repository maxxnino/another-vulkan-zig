const std = @import("std");
const vk = @import("vulkan");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const StbImage = @import("stb_image.zig").StbImage;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const Option = struct {
    anisotropy: bool = false,
    mip_map: bool = false,
};

pub const Texture2D = struct {
    image: Image,
    view: vk.ImageView,
    smapler: vk.Sampler,
    format: vk.Format,

    pub fn loadFromFile(gc: GraphicsContext, filename: [*:0]const u8, opt: Option) !Texture2D {
        var texture: Texture2D = undefined;
        const data = try StbImage.loadFromFile(filename, .rgb_alpha);
        defer data.free();

        const stage_buffer = try Buffer.init(gc, .{
            .size = data.totalByte(),
            .buffer_usage = .{ .transfer_src_bit = true },
            .memory_usage = .cpu_to_gpu,
            .memory_flags = .{},
        }, filename);
        defer stage_buffer.deinit(gc);
        try stage_buffer.update(u8, gc, data.pixels);
        texture.image = try Image.init(gc, .{
            .flags = .{},
            .image_type = .@"2d",
            .format = .r8g8b8a8_srgb,
            .extent = .{
                .width = data.width,
                .height = data.height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{
                .transfer_dst_bit = true,
                .sampled_bit = true,
            },
            .initial_layout = .@"undefined",
            .memory_usage = .gpu_only,
            .memory_flags = .{},
        }, filename);

        const cmdbuf = try gc.beginOneTimeCommandBuffer();
        texture.image.changeLayout(gc, cmdbuf, .{
            .new_layout = .transfer_dst_optimal,
            .src_access_mask = .{},
            .dst_access_mask = .{ .transfer_write_bit = true },
            .src_stage_mask = .{ .top_of_pipe_bit = true },
            .dst_stage_mask = .{ .transfer_bit = true },
        });
        texture.image.copyFromBuffer(gc, cmdbuf, stage_buffer, data.width, data.height);
        texture.image.changeLayout(gc, cmdbuf, .{
            .new_layout = .shader_read_only_optimal,
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .src_stage_mask = .{ .transfer_bit = true },
            .dst_stage_mask = .{ .fragment_shader_bit = true },
        });
        try gc.endOneTimeCommandBuffer(cmdbuf);

        texture.view = try gc.create(vk.ImageViewCreateInfo{
            .flags = .{},
            .image = texture.image.image,
            .view_type = .@"2d",
            .format = texture.image.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, filename);

        texture.smapler = try gc.create(vk.SamplerCreateInfo{
            .flags = .{},
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .mip_lod_bias = 0,
            .anisotropy_enable = if (opt.anisotropy == true) vk.TRUE else vk.FALSE,
            .max_anisotropy = if (opt.anisotropy == true) gc.props.limits.max_sampler_anisotropy else undefined,
            .compare_enable = vk.FALSE,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = 0,
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = vk.FALSE,
        }, filename);
        return texture;
    }

    pub fn deinit(self: Texture2D, gc: GraphicsContext) void {
        self.image.deinit(gc);
        gc.destroy(self.view);
        gc.destroy(self.smapler);
    }
};
