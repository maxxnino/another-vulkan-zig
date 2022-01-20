const std = @import("std");
const vk = @import("vulkan");
const tex = @import("texture.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const ShaderBinding = @import("ShaderBinding.zig");
const z = @import("zalgebra");
const Mat4 = z.Mat4;
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Vec4 = z.Vec4;

pipeline: vk.Pipeline,
render_pass: vk.RenderPass,
descriptor_set_layout: vk.DescriptorSetLayout,
pipeline_layout: vk.PipelineLayout,
depth: ?tex.DepthStencilTexture,
msaa: ?tex.RenderTarget,
extent: vk.Extent2D,
format: vk.Format,
label: ?[*:0]const u8,

const Self = @This();

pub const Vertex = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "normal"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "tex_coord"),
        },
    };

    pos: Vec3,
    normal: Vec3,
    tex_coord: Vec2,
};
pub const Options = struct {
    depth: bool = true,
    msaa: bool = true,
    ssaa: bool = true,
};

pub fn init(gc: GraphicsContext, extent: vk.Extent2D, shader_binding: ShaderBinding, format: vk.Format, opts: Options, label: ?[*:0]const u8) !Self {
    var renderer: Self = undefined;
    renderer.label = label;
    renderer.extent = extent;
    renderer.format = format;

    renderer.render_pass = try createRenderPass(gc, format, opts, label);
    renderer.descriptor_set_layout = try shader_binding.createDescriptorSetLayout(gc, label);
    renderer.pipeline_layout = try gc.create(vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &renderer.descriptor_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, label);

    renderer.pipeline = try createPipeline(
        gc,
        renderer.render_pass,
        renderer.pipeline_layout,
        shader_binding.getPipelineShaderStageCreateInfo(),
        label,
    );
    renderer.depth = try tex.DepthStencilTexture.init(
        gc,
        extent.width,
        extent.height,
        label,
    );

    renderer.msaa = try tex.RenderTarget.init(
        gc,
        extent.width,
        extent.height,
        format,
        label,
    );
    return renderer;
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.pipeline);
    gc.destroy(self.render_pass);
    gc.destroy(self.descriptor_set_layout);
    gc.destroy(self.pipeline_layout);
    if (self.depth) |d| d.deinit(gc);
    if (self.msaa) |m| m.deinit(gc);
}

pub fn beginFrame(self: Self, gc: GraphicsContext, framebuffer: vk.Framebuffer, cmdbuf: vk.CommandBuffer) !void {
    try gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{},
        .p_inheritance_info = null,
    });

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

    gc.vkd.cmdBindPipeline(cmdbuf, .graphics, self.pipeline);
}

pub fn endFrame(self: Self, gc: GraphicsContext, cmdbuf: vk.CommandBuffer) !void {
    _ = self;
    gc.vkd.cmdEndRenderPass(cmdbuf);
    return gc.vkd.endCommandBuffer(cmdbuf);
}

pub fn createFrameBuffer(self: *Self, gc: GraphicsContext, extent: vk.Extent2D, swap_image: vk.ImageView, label: ?[*:0]const u8) !vk.Framebuffer {
    try self.updateSize(gc, extent.width, extent.height);

    return gc.create(vk.FramebufferCreateInfo{
        .flags = .{},
        .render_pass = self.render_pass,
        .attachment_count = 3,
        .p_attachments = &[_]vk.ImageView{
            self.msaa.?.view,
            self.depth.?.view,
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

    if (self.depth) |d| {
        d.deinit(gc);
        self.depth = try tex.DepthStencilTexture.init(
            gc,
            self.extent.width,
            self.extent.height,
            self.label,
        );
    }
    if (self.msaa) |m| {
        m.deinit(gc);
        self.msaa = try tex.RenderTarget.init(
            gc,
            self.extent.width,
            self.extent.height,
            self.format,
            self.label,
        );
    }
}

fn createPipeline(
    gc: GraphicsContext,
    render_pass: vk.RenderPass,
    pipeline_layout: vk.PipelineLayout,
    shader_stages: []vk.PipelineShaderStageCreateInfo,
    label: ?[*:0]const u8,
) !vk.Pipeline {
    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .flags = .{},
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast([*]const vk.VertexInputBindingDescription, &Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_description.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .flags = .{},
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const pvsci = vk.PipelineViewportStateCreateInfo{
        .flags = .{},
        .viewport_count = 1,
        .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .flags = .{},
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .counter_clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .flags = .{},
        // enable msaa
        .rasterization_samples = gc.getSampleCount(),
        // enable sample shading
        .sample_shading_enable = vk.TRUE,
        .min_sample_shading = 0.2,
        .p_sample_mask = null,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .flags = .{},
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast([*]const vk.PipelineColorBlendAttachmentState, &pcbas),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };
    const pdssci = vk.PipelineDepthStencilStateCreateInfo{
        .flags = .{},
        .depth_test_enable = vk.TRUE,
        .depth_write_enable = vk.TRUE,
        .depth_compare_op = .less,
        .depth_bounds_test_enable = vk.FALSE,
        .stencil_test_enable = vk.FALSE,
        .front = undefined,
        .back = undefined,
        .min_depth_bounds = undefined,
        .max_depth_bounds = undefined,
    };
    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = @truncate(u32, shader_stages.len),
        .p_stages = shader_stages.ptr,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = &pdssci,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    return try gc.create(gpci, label);
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
        //Resovle output
        .p_resolve_attachments = &[_]vk.AttachmentReference{.{
            .attachment = 2,
            .layout = .color_attachment_optimal,
        }},
        //Depth
        .p_depth_stencil_attachment = &.{
            .attachment = 1,
            .layout = .attachment_optimal_khr,
        },
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
