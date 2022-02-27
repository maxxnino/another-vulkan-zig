const std = @import("std");
const vk = @import("vulkan");
const tex = @import("texture.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const DescriptorLayout = @import("DescriptorLayout.zig");
const Shader = @import("Shader.zig");
const Pipeline = @import("Pipeline.zig");
const PipelineLayout = @import("PipelineLayout.zig");
const QueryPool = @import("QueryPool.zig");
const Options = Pipeline.Options;

pipeline: Pipeline,
render_pass: vk.RenderPass,
depth: tex.Texture,
msaa: tex.Texture,
extent: vk.Extent2D,
format: vk.Format,
label: ?[*:0]const u8,
query_pool: QueryPool,

const Self = @This();

pub fn init(
    gc: GraphicsContext,
    extent: vk.Extent2D,
    shaders: []Shader,
    format: vk.Format,
    opts: Options,
    pipeline_layout: PipelineLayout,
    label: ?[*:0]const u8,
) !Self {
    var renderer: Self = undefined;
    renderer.label = label;
    renderer.extent = extent;
    renderer.format = format;

    renderer.query_pool = try QueryPool.init(gc, 3);
    renderer.render_pass = try createRenderPass(gc, format, opts, label);

    renderer.pipeline = try Pipeline.createBasicPipeline(
        gc,
        renderer.render_pass,
        shaders,
        opts,
        pipeline_layout,
        label,
    );
    renderer.depth = try tex.Texture.createDepthStencilTexture(
        gc,
        extent.width,
        extent.height,
        label,
    );

    renderer.msaa = try tex.Texture.createRenderTexture(
        gc,
        extent.width,
        extent.height,
        format,
        label,
    );
    return renderer;
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    self.pipeline.deinit(gc);
    gc.destroy(self.render_pass);
    self.depth.deinit(gc);
    self.msaa.deinit(gc);
    self.query_pool.deinit(gc);
}

pub fn beginFrame(self: Self, gc: GraphicsContext, framebuffer: vk.Framebuffer, cmdbuf: vk.CommandBuffer, index: u32, gpu_time: *f32) !void {
    try gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{},
        .p_inheritance_info = null,
    });

    const result = try self.query_pool.start(gc, cmdbuf, index);
    if (result > 0) {
        gpu_time.* = gpu_time.* * 0.8 + gc.props.limits.timestamp_period * @intToFloat(f32, result) * 0.000_000_1;
    }

    gc.vkd.cmdSetViewport(cmdbuf, 0, 1, &[_]vk.Viewport{.{
        .x = 0,
        .y = 0,
        .width = @intToFloat(f32, self.extent.width),
        .height = @intToFloat(f32, self.extent.height),
        .min_depth = 0,
        .max_depth = 1,
    }});
    const render_area = [_]vk.Rect2D{.{
        .offset = .{ .x = 0, .y = 0 },
        .extent = self.extent,
    }};
    gc.vkd.cmdSetScissor(cmdbuf, 0, 1, &render_area);

    const clear_value = [3]vk.ClearValue{
        .{
            .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
        },
        .{
            .depth_stencil = .{ .depth = 1, .stencil = 0 },
        },
        .{
            .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
        },
    };

    gc.vkd.cmdBeginRenderPass(cmdbuf, &.{
        .render_pass = self.render_pass,
        .framebuffer = framebuffer,
        .render_area = render_area[0],
        .clear_value_count = @truncate(u32, clear_value.len),
        .p_clear_values = @ptrCast([*]const vk.ClearValue, &clear_value),
    }, .@"inline");

    self.pipeline.bind(gc, cmdbuf);
}

pub fn endFrame(self: Self, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, index: u32) !void {
    gc.vkd.cmdEndRenderPass(cmdbuf);
    self.query_pool.end(gc, cmdbuf, index);
    return gc.vkd.endCommandBuffer(cmdbuf);
}

pub fn createFrameBuffer(self: *Self, gc: GraphicsContext, extent: vk.Extent2D, swap_image: vk.ImageView, label: ?[*:0]const u8) !vk.Framebuffer {
    try self.updateSize(gc, extent.width, extent.height);

    return gc.create(vk.FramebufferCreateInfo{
        .flags = .{},
        .render_pass = self.render_pass,
        .attachment_count = 3,
        .p_attachments = &[_]vk.ImageView{
            self.msaa.view,
            self.depth.view,
            swap_image,
        },
        .width = self.extent.width,
        .height = self.extent.height,
        .layers = 1,
    }, label);
}

fn updateSize(self: *Self, gc: GraphicsContext, width: u32, height: u32) !void {
    if (self.extent.width == width and self.extent.height == height) return;
    self.extent.width = width;
    self.extent.height = height;

    self.depth.deinit(gc);
    self.depth = try tex.Texture.createDepthStencilTexture(
        gc,
        self.extent.width,
        self.extent.height,
        self.label,
    );
    self.msaa.deinit(gc);
    self.msaa = try tex.Texture.createRenderTexture(
        gc,
        self.extent.width,
        self.extent.height,
        self.format,
        self.label,
    );
}

fn createRenderPass(gc: GraphicsContext, format: vk.Format, opts: Options, label: ?[*:0]const u8) !vk.RenderPass {
    _ = opts;
    const attachments = [_]vk.AttachmentDescription{
        // Msaa
        .{
            .flags = .{},
            .format = format,
            // Note: if sample count > 1, load_op should be clear or dont_care
            // store_op should be dont_care for better performace
            .samples = gc.getSampleCount(),
            .load_op = .dont_care,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .color_attachment_optimal,
        },
        // depth
        .{
            .flags = .{},
            .format = .d32_sfloat_s8_uint,
            // Note: if sample count > 1, load_op should be clear or dont_care
            // store_op should be dont_care for better performace
            .samples = gc.getSampleCount(),
            .load_op = .clear,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .depth_stencil_attachment_optimal,
        },
        // resolve output
        .{
            .flags = .{},
            .format = format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .present_src_khr,
        },
    };
    const subpass = vk.SubpassDescription{
        .flags = .{},
        .pipeline_bind_point = .graphics,
        .input_attachment_count = 0,
        .p_input_attachments = undefined,
        .color_attachment_count = 1,
        //Msaa
        .p_color_attachments = &[_]vk.AttachmentReference{.{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        }},
        //Depth
        .p_depth_stencil_attachment = &.{
            .attachment = 1,
            .layout = .attachment_optimal,
        },
        //Resovle output
        .p_resolve_attachments = &[_]vk.AttachmentReference{.{
            .attachment = 2,
            .layout = .color_attachment_optimal,
        }},
        .preserve_attachment_count = 0,
        .p_preserve_attachments = undefined,
    };

    return try gc.create(vk.RenderPassCreateInfo{
        .flags = .{},
        .attachment_count = @truncate(u32, attachments.len),
        .p_attachments = @ptrCast([*]const vk.AttachmentDescription, &attachments),
        .subpass_count = 1,
        .p_subpasses = @ptrCast([*]const vk.SubpassDescription, &subpass),
        .dependency_count = 0,
        .p_dependencies = undefined,
        // .dependency_count = 1,
        // .p_dependencies = @ptrCast([*]const vk.SubpassDependency, &sd),
    }, label);
}
