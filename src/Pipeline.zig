const std = @import("std");
const vk = @import("vulkan");
const tex = @import("texture.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const DescriptorLayout = @import("DescriptorLayout.zig");
const Shader = @import("Shader.zig");
const Self = @This();

pipeline: vk.Pipeline,
descriptor_set_layout: vk.DescriptorSetLayout,
bindless_set_layout: vk.DescriptorSetLayout,
immutable_sampler_set_layout: vk.DescriptorSetLayout,
pipeline_layout: vk.PipelineLayout,
descriptor_template: vk.DescriptorUpdateTemplate,

pub const Options = struct {
    msaa: bool = true,
    ssaa: bool = true,
};

fn createLayout(
    self: *Self,
    allocator: std.mem.Allocator,
    gc: GraphicsContext,
    shaders: []Shader,
    push_constants: ?[]const vk.PushConstantRange,
    label: ?[*:0]const u8,
) !void {
    self.descriptor_set_layout = try Shader.createDescriptorSetLayout(allocator, gc, shaders, label);
    self.pipeline_layout = try gc.create(vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = 3,
        .p_set_layouts = &[_]vk.DescriptorSetLayout{
            self.bindless_set_layout,
            self.immutable_sampler_set_layout,
            self.descriptor_set_layout,
        },
        .push_constant_range_count = if (push_constants) |p| @truncate(u32, p.len) else 0,
        .p_push_constant_ranges = if (push_constants) |p| p.ptr else undefined,
    }, label);
}

pub fn createSkyboxPipeline(
    allocator: std.mem.Allocator,
    gc: GraphicsContext,
    render_pass: vk.RenderPass,
    shaders: []Shader,
    push_constants: ?[]const vk.PushConstantRange,
    opts: Options,
    bindless: DescriptorLayout,
    immutable_sampler: DescriptorLayout,
    label: ?[*:0]const u8,
) !Self {
    var self: Self = undefined;
    self.immutable_sampler_set_layout = immutable_sampler.layout;
    self.bindless_set_layout = bindless.layout;

    try self.createLayout(allocator, gc, shaders, push_constants, label);

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
    var shader_stages = std.BoundedArray(vk.PipelineShaderStageCreateInfo, 8){};
    for (shaders) |shader| {
        try shader_stages.append(shader.getPipelineShaderStageCreateInfo());
    }
    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = @truncate(u32, shader_stages.len),
        .p_stages = shader_stages.slice().ptr,
        .p_vertex_input_state = blk: {
            for (shaders) |shader| {
                if (shader.vertexInputState()) |*value| break :blk value;
            }
            break :blk null;
        },
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
pub fn createBasicPipeline(
    allocator: std.mem.Allocator,
    gc: GraphicsContext,
    render_pass: vk.RenderPass,
    shaders: []Shader,
    push_constants: ?[]const vk.PushConstantRange,
    opts: Options,
    bindless: DescriptorLayout,
    immutable_sampler: DescriptorLayout,
    label: ?[*:0]const u8,
) !Self {
    var self: Self = undefined;
    self.immutable_sampler_set_layout = immutable_sampler.layout;
    self.bindless_set_layout = bindless.layout;

    try self.createLayout(allocator, gc, shaders, push_constants, label);

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

    var shader_stages = std.BoundedArray(vk.PipelineShaderStageCreateInfo, 8){};
    for (shaders) |shader| {
        try shader_stages.append(shader.getPipelineShaderStageCreateInfo());
    }
    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = @truncate(u32, shader_stages.len),
        .p_stages = shader_stages.slice().ptr,
        .p_vertex_input_state = blk: {
            for (shaders) |shader| {
                if (shader.vertexInputState()) |*value| break :blk value;
            }
            break :blk null;
        },
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

pub fn pushConstant(self: Self, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, stage: vk.ShaderStageFlags, data: anytype) void {
    const size = @sizeOf(@TypeOf(data));
    std.debug.assert(size <= gc.props.limits.max_push_constants_size);
    gc.vkd.cmdPushConstants(
        cmdbuf,
        self.pipeline_layout,
        stage,
        0,
        size,
        @ptrCast(*const anyopaque, &data),
    );
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.descriptor_set_layout);
    gc.destroy(self.pipeline_layout);
    gc.destroy(self.pipeline);
}
