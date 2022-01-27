const std = @import("std");
const vk = @import("vulkan");
const tex = @import("texture.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const ShaderBinding = @import("ShaderBinding.zig");
const Self = @This();

pipeline: vk.Pipeline,
descriptor_set_layout: vk.DescriptorSetLayout,
pipeline_layout: vk.PipelineLayout,

pub const Options = struct {
    msaa: bool = true,
    ssaa: bool = true,
};

pub fn createSkyboxPipeline(
    gc: GraphicsContext,
    render_pass: vk.RenderPass,
    shader_binding: ShaderBinding,
    opts: Options,
    label: ?[*:0]const u8,
) !Self {
    var self: Self = undefined;

    self.descriptor_set_layout = try shader_binding.createDescriptorSetLayout(gc, label);
    self.pipeline_layout = try gc.create(vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &self.descriptor_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, label);

    // create Pipeline
    const shader_stages = shader_binding.getPipelineShaderStageCreateInfo();

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
        .cull_mode = .{ .front_bit = true },
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
        .rasterization_samples = if (opts.msaa) gc.getSampleCount() else .{ .@"1_bit" = true },
        // enable sample shading
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
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
        .depth_write_enable = vk.FALSE,
        .depth_compare_op = .less_or_equal,
        .depth_bounds_test_enable = vk.FALSE,
        .stencil_test_enable = vk.FALSE,
        .front = std.mem.zeroes(vk.StencilOpState),
        .back = std.mem.zeroes(vk.StencilOpState),
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };
    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = @truncate(u32, shader_stages.len),
        .p_stages = shader_stages.ptr,
        .p_vertex_input_state = if (shader_binding.vertex_stage_info) |vi| &vi else null,
        .p_input_assembly_state = &piasci, //a
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci, //a
        .p_rasterization_state = &prsci, // a
        .p_multisample_state = &pmsci, //a
        .p_depth_stencil_state = &pdssci, //a
        .p_color_blend_state = &pcbsci, //
        .p_dynamic_state = &pdsci,
        .layout = self.pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    self.pipeline = try gc.create(gpci, label);

    return self;
}
pub fn createBasicPipeline(
    gc: GraphicsContext,
    render_pass: vk.RenderPass,
    shader_binding: ShaderBinding,
    opts: Options,
    label: ?[*:0]const u8,
) !Self {
    var self: Self = undefined;

    self.descriptor_set_layout = try shader_binding.createDescriptorSetLayout(gc, label);
    self.pipeline_layout = try gc.create(vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &self.descriptor_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, label);

    // create Pipeline
    const shader_stages = shader_binding.getPipelineShaderStageCreateInfo();

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
        .rasterization_samples = if (opts.msaa) gc.getSampleCount() else .{ .@"1_bit" = true },
        // enable sample shading
        .sample_shading_enable = if (opts.ssaa) vk.TRUE else vk.FALSE,
        .min_sample_shading = if (opts.ssaa) 0.2 else 1,
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
        .front = std.mem.zeroes(vk.StencilOpState),
        .back = std.mem.zeroes(vk.StencilOpState),
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };
    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = @truncate(u32, shader_stages.len),
        .p_stages = shader_stages.ptr,
        .p_vertex_input_state = if (shader_binding.vertex_stage_info) |vi| &vi else null,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = &pdssci,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = self.pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    self.pipeline = try gc.create(gpci, label);

    return self;
}

pub fn bind(self: Self, gc: GraphicsContext, command_buffer: vk.CommandBuffer) void {
    gc.vkd.cmdBindPipeline(command_buffer, .graphics, self.pipeline);
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.descriptor_set_layout);
    gc.destroy(self.pipeline_layout);
    gc.destroy(self.pipeline);
}
