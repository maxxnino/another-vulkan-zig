const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const cgltf = @import("cgltf.zig");
const resources = @import("resources");
const vma = @import("vma.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Buffer = @import("Buffer.zig");
const Allocator = std.mem.Allocator;
const RenderCommand = @import("RenderCommand.zig");
const Camera = @import("Camera.zig");
const tex = @import("texture.zig");
const Texture2D = tex.Texture2D;
const srcToString = @import("util.zig").srcToString;

const z = @import("zalgebra");
const assert = std.debug.assert;
const Mat4 = z.Mat4;
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Vec4 = z.Vec4;

const app_name = "vulkan-zig triangle example";

const UniformBufferObject = struct {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
};

const Vertex = struct {
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

const Mesh = struct {
    index_offset: u32,
    vertex_offset: u32,
    num_indices: u32,
    num_vertices: u32,
};
// const vertices = [_]Vertex{
//     .{ .pos = .{ -0.5, -0.5, 0.0 }, .color = .{ 1.0, 0.0, 0.0 }, .uv = .{ 0.0, 0.0 } },
//     .{ .pos = .{ 0.5, -0.5, 0.0 }, .color = .{ 0.0, 1.0, 0.0 }, .uv = .{ 1.0, 0.0 } },
//     .{ .pos = .{ 0.5, 0.5, 0.0 }, .color = .{ 0.0, 0.0, 1.0 }, .uv = .{ 1.0, 1.0 } },
//     .{ .pos = .{ -0.5, 0.5, 0.0 }, .color = .{ 1.0, 1.0, 1.0 }, .uv = .{ 0.0, 1.0 } },

//     .{ .pos = .{ -0.5, -0.5, -0.5 }, .color = .{ 1.0, 0.0, 0.0 }, .uv = .{ 0.0, 0.0 } },
//     .{ .pos = .{ 0.5, -0.5, -0.5 }, .color = .{ 0.0, 1.0, 0.0 }, .uv = .{ 1.0, 0.0 } },
//     .{ .pos = .{ 0.5, 0.5, -0.5 }, .color = .{ 0.0, 0.0, 1.0 }, .uv = .{ 1.0, 1.0 } },
//     .{ .pos = .{ -0.5, 0.5, -0.5 }, .color = .{ 1.0, 1.0, 1.0 }, .uv = .{ 0.0, 1.0 } },
// };

pub fn main() !void {
    try glfw.init(.{});
    defer glfw.terminate();

    var extent = vk.Extent2D{ .width = 800, .height = 600 };

    const window = try glfw.Window.create(extent.width, extent.height, app_name, null, null, .{
        .client_api = .no_api,
    });
    defer window.destroy();

    const allocator = std.heap.page_allocator;

    const gc = try GraphicsContext.init(allocator, app_name, window);
    defer gc.deinit();

    std.debug.print("Using device: {s}\n", .{gc.deviceName()});

    var swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    var depth = try tex.DepthStencilTexture.init(
        gc,
        swapchain.extent.width,
        swapchain.extent.height,
        srcToString(@src()),
    );
    defer depth.deinit(gc);

    var msaa = try tex.RenderTarget.init(
        gc,
        swapchain.extent.width,
        swapchain.extent.height,
        swapchain.surface_format.format,
        srcToString(@src()),
    );
    defer msaa.deinit(gc);

    // ********** load Model **********
    var indices = std.ArrayList(u32).init(allocator);
    defer indices.deinit();
    var vertices = std.ArrayList(Vertex).init(allocator);
    defer vertices.deinit();
    var meshs = std.ArrayList(Mesh).init(allocator);
    defer meshs.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    appendGltfModel(arena.allocator(), &meshs, &vertices, &indices, "assets/untitled.gltf");
    //*********************************

    const bindings = [_]vk.DescriptorSetLayoutBinding{ .{
        .binding = 0,
        .descriptor_type = .uniform_buffer,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex_bit = true },
        .p_immutable_samplers = null,
    }, .{
        .binding = 1,
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .fragment_bit = true },
        .p_immutable_samplers = null,
    } };

    const descriptor_set_layout = try gc.create(vk.DescriptorSetLayoutCreateInfo{
        .flags = .{},
        .binding_count = @truncate(u32, bindings.len),
        .p_bindings = @ptrCast([*]const vk.DescriptorSetLayoutBinding, &bindings),
    }, srcToString(@src()));
    defer gc.destroy(descriptor_set_layout);

    const pipeline_layout = try gc.create(vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &descriptor_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, srcToString(@src()));
    defer gc.destroy(pipeline_layout);

    const render_pass = try createRenderPass(&gc, swapchain, depth);
    defer gc.destroy(render_pass);

    var pipeline = try createPipeline(&gc, pipeline_layout, render_pass);
    defer gc.destroy(pipeline);

    var framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain, depth, msaa);
    defer destroyFramebuffers(&gc, allocator, framebuffers);
    const frame_size = @truncate(u32, framebuffers.len);

    const vertex_buffer = try Buffer.init(gc, Buffer.CreateInfo{
        .size = @sizeOf(Vertex) * vertices.items.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));
    defer vertex_buffer.deinit(gc);

    const texture = try Texture2D.loadFromFile(gc, "assets/viking_room.png", .{ .anisotropy = true, .mip_map = true });
    defer texture.deinit(gc);

    var camera = Camera{
        .pitch = 30,
        .yaw = 220,
        .pos = Vec3.new(0, 2, 4),
    };

    var ubo_buffers = try allocator.alloc(Buffer, framebuffers.len);

    for (ubo_buffers) |*ubo| {
        ubo.* = try Buffer.init(gc, Buffer.CreateInfo{
            .size = @sizeOf(UniformBufferObject),
            .buffer_usage = .{ .uniform_buffer_bit = true },
            .memory_usage = .cpu_to_gpu,
            .memory_flags = .{},
        }, srcToString(@src()));
        try ubo.update(UniformBufferObject, gc, &[_]UniformBufferObject{.{
            .model = Mat4.identity(),
            .view = camera.getViewMatrix(),
            .proj = camera.getProjMatrix(swapchain.extent.width, swapchain.extent.height),
        }});
    }
    defer for (ubo_buffers) |ubo| {
        ubo.deinit(gc);
    };

    const index_buffer = try Buffer.init(gc, Buffer.CreateInfo{
        .size = @sizeOf(u32) * indices.items.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .index_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));
    defer index_buffer.deinit(gc);

    // uploadVertices
    try uploadData(Vertex, gc, vertex_buffer, vertices.items);
    //Upload indices
    try uploadData(u32, gc, index_buffer, indices.items);

    // Desciptor Set
    const pool_sizes = [_]vk.DescriptorPoolSize{ .{
        .@"type" = .uniform_buffer,
        .descriptor_count = frame_size,
    }, .{
        .@"type" = .combined_image_sampler,
        .descriptor_count = frame_size,
    } };
    const descriptor_pool = try gc.create(vk.DescriptorPoolCreateInfo{
        .flags = .{},
        .max_sets = frame_size,
        .pool_size_count = @truncate(u32, pool_sizes.len),
        .p_pool_sizes = @ptrCast([*]const vk.DescriptorPoolSize, &pool_sizes),
    }, srcToString(@src()));
    defer gc.destroy(descriptor_pool);
    var des_layouts = try allocator.alloc(vk.DescriptorSetLayout, frame_size);
    for (des_layouts) |*l| {
        l.* = descriptor_set_layout;
    }
    defer allocator.free(des_layouts);

    const dsai = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .descriptor_set_count = frame_size,
        .p_set_layouts = des_layouts.ptr,
    };
    var des_sets = try allocator.alloc(vk.DescriptorSet, frame_size);
    defer allocator.free(des_sets);
    try gc.vkd.allocateDescriptorSets(gc.dev, &dsai, des_sets.ptr);
    for (des_sets) |ds, i| {
        updateDescriptorSet(gc, ds, ubo_buffers[i], texture);
    }
    //End descriptor set

    var render_command = try RenderCommand.init(
        allocator,
        gc,
        framebuffers,
        render_pass,
        pipeline,
        pipeline_layout,
        swapchain.extent,
    );
    defer render_command.deinit(gc, allocator);
    try buildCommandBuffers(&render_command, gc, vertex_buffer.buffer, index_buffer.buffer, des_sets, meshs.items);
    //Timer
    var update_timer = try std.time.Timer.start();

    while (!window.shouldClose()) {
        const dt = @intToFloat(f32, update_timer.lap()) / @intToFloat(f32, std.time.ns_per_s);
        camera.moveCamera(window, dt);
        try ubo_buffers[swapchain.image_index].update(UniformBufferObject, gc, &[_]UniformBufferObject{.{
            .model = Mat4.identity(),
            .view = camera.getViewMatrix(),
            .proj = camera.getProjMatrix(swapchain.extent.width, swapchain.extent.height),
        }});

        const cmdbuf = render_command.command_buffers[swapchain.image_index];

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            const size = try window.getSize();
            extent.width = @intCast(u32, size.width);
            extent.height = @intCast(u32, size.height);
            try swapchain.recreate(extent);

            // recreate depth resource
            depth.deinit(gc);
            depth = try tex.DepthStencilTexture.init(
                gc,
                swapchain.extent.width,
                swapchain.extent.height,
                srcToString(@src()),
            );

            //

            destroyFramebuffers(&gc, allocator, framebuffers);
            framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain, depth, msaa);

            render_command.deinit(gc, allocator);
            render_command = try RenderCommand.init(
                allocator,
                gc,
                framebuffers,
                render_pass,
                pipeline,
                pipeline_layout,
                swapchain.extent,
            );
            try buildCommandBuffers(&render_command, gc, vertex_buffer.buffer, index_buffer.buffer, des_sets, meshs.items);
        }

        try glfw.pollEvents();
        if (window.getKey(.q) == .press) break;
    }

    try swapchain.waitForAllFences();
}

pub fn uploadData(comptime T: type, gc: GraphicsContext, buffer: Buffer, data: []const T) !void {
    const size = @sizeOf(T) * data.len;
    const stage_buffer = try Buffer.init(gc, .{
        .size = size,
        .buffer_usage = .{ .transfer_src_bit = true },
        .memory_usage = .cpu_to_gpu,
        .memory_flags = .{},
    }, srcToString(@src()));
    defer stage_buffer.deinit(gc);
    try stage_buffer.update(T, gc, data);
    try stage_buffer.copyToBuffer(buffer, gc);
}

fn updateDescriptorSet(gc: GraphicsContext, descriptor_set: vk.DescriptorSet, buffer: Buffer, texture: Texture2D) void {
    const dbi = vk.DescriptorBufferInfo{
        .buffer = buffer.buffer,
        .offset = 0,
        .range = buffer.size,
    };
    const dii = vk.DescriptorImageInfo{
        .sampler = texture.smapler,
        .image_view = texture.view,
        .image_layout = texture.image.layout,
    };
    const wds = [_]vk.WriteDescriptorSet{
        .{
            .dst_set = descriptor_set,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .uniform_buffer,
            .p_image_info = undefined,
            .p_buffer_info = @ptrCast([*]const vk.DescriptorBufferInfo, &dbi),
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = descriptor_set,
            .dst_binding = 1,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .combined_image_sampler,
            .p_image_info = @ptrCast([*]const vk.DescriptorImageInfo, &dii),
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    };
    gc.vkd.updateDescriptorSets(gc.dev, @truncate(u32, wds.len), @ptrCast([*]const vk.WriteDescriptorSet, &wds), 0, undefined);
}

fn buildCommandBuffers(
    render_command: *RenderCommand,
    gc: GraphicsContext,
    vertex_buffer: vk.Buffer,
    index_buffer: vk.Buffer,
    sets: []const vk.DescriptorSet,
    meshs: []const Mesh,
) !void {
    while (try render_command.begin(gc)) |cmdbuf| {
        const offset = [_]vk.DeviceSize{0};
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast([*]const vk.Buffer, &vertex_buffer), &offset);
        gc.vkd.cmdBindIndexBuffer(cmdbuf, index_buffer, 0, .uint32);
        gc.vkd.cmdBindDescriptorSets(
            cmdbuf,
            .graphics,
            render_command.pipeline_layout,
            0,
            1,
            @ptrCast([*]const vk.DescriptorSet, &sets[render_command.index - 1]),
            0,
            undefined,
        );
        for (meshs) |m| {
            gc.vkd.cmdDrawIndexed(cmdbuf, m.num_indices, 1, m.index_offset, @intCast(i32, m.vertex_offset), 0);
        }
        try render_command.end(gc);
    }
}

fn createFramebuffers(
    gc: *const GraphicsContext,
    allocator: Allocator,
    render_pass: vk.RenderPass,
    swapchain: Swapchain,
    depth: tex.DepthStencilTexture,
    msaa: tex.RenderTarget,
) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (framebuffers) |*fb| {
        fb.* = try gc.create(vk.FramebufferCreateInfo{
            .flags = .{},
            .render_pass = render_pass,
            .attachment_count = 3,
            .p_attachments = @ptrCast([*]const vk.ImageView, &[_]vk.ImageView{
                msaa.view,
                depth.view,
                swapchain.swap_images[i].view,
            }),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, srcToString(@src()));
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const GraphicsContext, allocator: Allocator, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);
    allocator.free(framebuffers);
}

fn createRenderPass(gc: *const GraphicsContext, swapchain: Swapchain, depth: tex.DepthStencilTexture) !vk.RenderPass {
    const attachments = [_]vk.AttachmentDescription{
        // Msaa
        .{
            .flags = .{},
            .format = swapchain.surface_format.format,
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
            .format = depth.image.format,
            // Note: if sample count > 1, load_op should be clear or dont_care
            // store_op should be dont_care for better performace
            .samples = gc.getSampleCount(),
            .load_op = .clear,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = depth.image.layout,
        },
        // resolve output
        .{
            .flags = .{},
            .format = swapchain.surface_format.format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .present_src_khr,
        },
    };
    //Msaa
    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };
    //Depth
    const depth_attachment_ref = vk.AttachmentReference{
        .attachment = 1,
        .layout = depth.image.layout,
    };
    //Resovle output
    const resolve_attachment_ref = vk.AttachmentReference{
        .attachment = 2,
        .layout = .color_attachment_optimal,
    };
    const subpass = vk.SubpassDescription{
        .flags = .{},
        .pipeline_bind_point = .graphics,
        .input_attachment_count = 0,
        .p_input_attachments = undefined,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast([*]const vk.AttachmentReference, &color_attachment_ref),
        .p_resolve_attachments = @ptrCast([*]const vk.AttachmentReference, &resolve_attachment_ref),
        .p_depth_stencil_attachment = &depth_attachment_ref,
        .preserve_attachment_count = 0,
        .p_preserve_attachments = undefined,
    };
    // const sd = vk.SubpassDependency{
    //     .src_subpass = vk.SUBPASS_EXTERNAL,
    //     .dst_subpass = 0,
    //     .src_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
    //     .dst_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
    //     .src_access_mask = .{},
    //     .dst_access_mask = .{ .color_attachment_write_bit = true, .depth_stencil_attachment_write_bit = true },
    //     .dependency_flags = .{},
    // };
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
    }, srcToString(@src()));
}

fn createPipeline(
    gc: *const GraphicsContext,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try gc.create(vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = resources.triangle_vert.len,
        .p_code = @ptrCast([*]const u32, resources.triangle_vert),
    }, srcToString(@src()));
    defer gc.destroy(vert);

    const frag = try gc.create(vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = resources.triangle_frag.len,
        .p_code = @ptrCast([*]const u32, resources.triangle_frag),
    }, srcToString(@src()));
    defer gc.destroy(frag);

    const pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .flags = .{},
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
            .p_specialization_info = null,
        },
        .{
            .flags = .{},
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
            .p_specialization_info = null,
        },
    };

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
        .stage_count = 2,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = &pdssci,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    return try gc.create(gpci, srcToString(@src()));
}
pub fn appendGltfModel(
    arena: Allocator,
    all_meshes: *std.ArrayList(Mesh),
    all_vertices: *std.ArrayList(Vertex),
    all_indices: *std.ArrayList(u32),
    path: [:0]const u8,
) void {
    var indices = std.ArrayList(u32).init(arena);
    var positions = std.ArrayList(Vec3).init(arena);
    var normals = std.ArrayList(Vec3).init(arena);
    var texcoords0 = std.ArrayList(Vec2).init(arena);
    // var tangents = std.ArrayList(Vec4).init(arena);

    const data = parseAndLoadGltfFile(path);
    defer cgltf.cgltf_free(data);

    const num_meshes = @truncate(u32, data.meshes_count);
    var mesh_index: u32 = 0;
    // const base_indices = @truncate(u32, all_indices.items.len);
    // const base_vertices = @truncate(u32, all_vertices.items.len);

    while (mesh_index < num_meshes) : (mesh_index += 1) {
        const num_prims = @intCast(u32, data.meshes[mesh_index].primitives_count);
        var prim_index: u32 = 0;

        while (prim_index < num_prims) : (prim_index += 1) {
            const pre_indices_len = indices.items.len;
            const pre_positions_len = positions.items.len;

            appendMeshPrimitive(data, mesh_index, prim_index, &indices, &positions, &normals, &texcoords0, null);

            all_meshes.append(.{
                .index_offset = @intCast(u32, pre_indices_len),
                .vertex_offset = @intCast(u32, pre_positions_len),
                .num_indices = @intCast(u32, indices.items.len - pre_indices_len),
                .num_vertices = @intCast(u32, positions.items.len - pre_positions_len),
            }) catch unreachable;
        }
    }

    all_indices.ensureTotalCapacity(indices.items.len) catch unreachable;
    for (indices.items) |index| {
        all_indices.appendAssumeCapacity(index);
    }

    all_vertices.ensureTotalCapacity(positions.items.len) catch unreachable;
    for (positions.items) |_, index| {
        all_vertices.appendAssumeCapacity(.{
            .pos = positions.items[index].scale(0.2), // NOTE(mziulek): Sponza requires scaling.
            // .pos = positions.items[index],
            .normal = normals.items[index],
            .tex_coord = texcoords0.items[index],
            // .tangent = tangents.items[index],
        });
    }
}

fn parseAndLoadGltfFile(gltf_path: [:0]const u8) *cgltf.Data {
    var data: *cgltf.Data = undefined;
    // Parse.
    {
        const result = cgltf.cgltf_parse_file(&.{}, gltf_path.ptr, &data);
        assert(result == .success);
    }
    // Load.
    {
        const result = cgltf.cgltf_load_buffers(&.{}, data, gltf_path.ptr);
        assert(result == .success);
    }
    return data;
}
fn appendMeshPrimitive(
    data: *cgltf.Data,
    mesh_index: u32,
    prim_index: u32,
    indices: *std.ArrayList(u32),
    positions: *std.ArrayList(Vec3),
    normals: ?*std.ArrayList(Vec3),
    texcoords0: ?*std.ArrayList(Vec2),
    tangents: ?*std.ArrayList(Vec4),
) void {
    assert(mesh_index < data.meshes_count);
    assert(prim_index < data.meshes[mesh_index].primitives_count);
    const num_vertices: u32 = @intCast(u32, data.meshes[mesh_index].primitives[prim_index].attributes[0].data.count);
    const num_indices: u32 = @intCast(u32, data.meshes[mesh_index].primitives[prim_index].indices.?.count);

    // Indices.
    {
        indices.ensureTotalCapacity(indices.items.len + num_indices) catch unreachable;

        const accessor = data.meshes[mesh_index].primitives[prim_index].indices.?;

        const buffer_view = accessor.buffer_view.?;
        assert(accessor.stride == buffer_view.stride or buffer_view.stride == 0);
        assert((accessor.stride * accessor.count) == buffer_view.size);

        const data_addr = @alignCast(4, @ptrCast([*]const u8, buffer_view.buffer.data.?) +
            accessor.offset + buffer_view.offset);

        if (accessor.stride == 1) {
            assert(accessor.component_type == .r_8u);
            const src = @ptrCast([*]const u8, data_addr);
            var i: u32 = 0;
            while (i < num_indices) : (i += 1) {
                indices.appendAssumeCapacity(src[i]);
            }
        } else if (accessor.stride == 2) {
            assert(accessor.component_type == .r_16u);
            const src = @ptrCast([*]const u16, data_addr);
            var i: u32 = 0;
            while (i < num_indices) : (i += 1) {
                indices.appendAssumeCapacity(src[i]);
            }
        } else if (accessor.stride == 4) {
            assert(accessor.component_type == .r_32u);
            const src = @ptrCast([*]const u32, data_addr);
            var i: u32 = 0;
            while (i < num_indices) : (i += 1) {
                indices.appendAssumeCapacity(src[i]);
            }
        } else {
            unreachable;
        }
    }

    // Attributes.
    {
        positions.resize(positions.items.len + num_vertices) catch unreachable;
        if (normals != null) normals.?.resize(normals.?.items.len + num_vertices) catch unreachable;
        if (texcoords0 != null) texcoords0.?.resize(texcoords0.?.items.len + num_vertices) catch unreachable;
        if (tangents != null) tangents.?.resize(tangents.?.items.len + num_vertices) catch unreachable;

        const num_attribs: u32 = @truncate(u32, data.meshes[mesh_index].primitives[prim_index].attributes_count);

        var attrib_index: u32 = 0;
        while (attrib_index < num_attribs) : (attrib_index += 1) {
            const attrib = &data.meshes[mesh_index].primitives[prim_index].attributes[attrib_index];
            const accessor = attrib.data;

            const buffer_view = accessor.buffer_view.?;
            assert(accessor.stride == buffer_view.stride or buffer_view.stride == 0);
            assert((accessor.stride * accessor.count) == buffer_view.size);

            const data_addr = @ptrCast([*]const u8, buffer_view.buffer.data.?) +
                accessor.offset + buffer_view.offset;

            if (attrib.type == .position) {
                assert(accessor.type == .vec3);
                assert(accessor.component_type == .r_32f);
                @memcpy(
                    @ptrCast([*]u8, &positions.items[positions.items.len - num_vertices]),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            } else if (attrib.type == .normal and normals != null) {
                assert(accessor.type == .vec3);
                assert(accessor.component_type == .r_32f);
                @memcpy(
                    @ptrCast([*]u8, &normals.?.items[normals.?.items.len - num_vertices]),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            } else if (attrib.type == .texcoord and texcoords0 != null) {
                assert(accessor.type == .vec2);
                assert(accessor.component_type == .r_32f);
                @memcpy(
                    @ptrCast([*]u8, &texcoords0.?.items[texcoords0.?.items.len - num_vertices]),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            }
            // else if (attrib.*.type == c.cgltf_attribute_type_tangent and tangents != null) {
            //     assert(accessor.*.type == c.cgltf_type_vec4);
            //     assert(accessor.*.component_type == c.cgltf_component_type_r_32f);
            //     @memcpy(
            //         @ptrCast([*]u8, &tangents.?.items[tangents.?.items.len - num_vertices]),
            //         data_addr,
            //         accessor.*.count * accessor.*.stride,
            //     );
            // }
        }
    }
}
