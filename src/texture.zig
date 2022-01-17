const std = @import("std");
const vk = @import("vulkan");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const StbImage = @import("stb_image.zig").StbImage;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const Texture2D = struct {
    pub const Option = struct {
        /// true mean enable anisotropy
        anisotropy: bool = false,

        /// true mean enable mip map
        mip_map: bool = false,
    };
    image: Image,
    view: vk.ImageView,
    smapler: vk.Sampler,
    opts: Option,

    pub fn loadFromFile(gc: GraphicsContext, filename: [*:0]const u8, opts: Option) !Texture2D {
        var texture: Texture2D = undefined;
        texture.opts = opts;
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
            .memory_usage = .gpu_only,
            .memory_flags = .{},
        }, filename);
        const subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        };

        const cmdbuf = try gc.beginOneTimeCommandBuffer();
        texture.image.changeLayout(
            gc,
            cmdbuf,
            .transfer_dst_optimal,
            .{},
            .{ .transfer_write_bit = true },
            .{ .top_of_pipe_bit = true },
            .{ .transfer_bit = true },
            subresource_range,
        );
        texture.image.copyFromBuffer(gc, cmdbuf, stage_buffer, data.width, data.height);
        texture.image.changeLayout(
            gc,
            cmdbuf,
            .shader_read_only_optimal,
            .{ .transfer_write_bit = true },
            .{ .shader_read_bit = true },
            .{ .transfer_bit = true },
            .{ .fragment_shader_bit = true },
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
            .anisotropy_enable = if (opts.anisotropy == true) vk.TRUE else vk.FALSE,
            .max_anisotropy = if (opts.anisotropy == true) gc.props.limits.max_sampler_anisotropy else undefined,
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
            .samples = .{ .@"1_bit" = true },
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
            .depth_stencil_attachment_optimal,
            .{},
            .{ .depth_stencil_attachment_read_bit = true, .depth_stencil_attachment_write_bit = true },
            .{ .top_of_pipe_bit = true },
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

    pub fn deinit(self: DepthStencilTexture, gc: GraphicsContext) void {
        self.image.deinit(gc);
        gc.destroy(self.view);
    }
};
