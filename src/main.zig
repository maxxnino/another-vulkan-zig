const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const cgltf = @import("binding/cgltf.zig");
const resources = @import("resources");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Buffer = @import("Buffer.zig");
const Allocator = std.mem.Allocator;
const Camera = @import("Camera.zig");
const tex = @import("texture.zig");
const Texture = tex.Texture;
const Shader = @import("Shader.zig");
const ShaderBinding = @import("ShaderBinding.zig");
const BasicRenderer = @import("BasicRenderer.zig");
const Pipeline = @import("Pipeline.zig");
const Vertex = @import("Vertex.zig");
const Model = @import("Model.zig");
const Mesh = Model.Mesh;
const srcToString = @import("util.zig").srcToString;

const command_pool = @import("command_pool.zig");
const DrawPool = command_pool.DrawPool;

const assert = std.debug.assert;
const z = @import("zalgebra");
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

const PushConstant = struct {
    texture_id: u32,
};

const Gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = false });

pub fn main() !void {
    try glfw.init(.{});
    defer glfw.terminate();

    const monitor = glfw.Monitor.getPrimary().?;
    const mode = try monitor.getVideoMode();
    const is_fullscreen = false;
    var extent: vk.Extent2D =
        if (is_fullscreen)
    .{ .width = mode.getWidth(), .height = mode.getHeight() } else .{ .width = 800, .height = 600 };

    const window = try glfw.Window.create(extent.width, extent.height, app_name, if (is_fullscreen) monitor else null, null, .{
        .client_api = .no_api,
    });
    defer window.destroy();
    var gpa = Gpa{};
    const allocator = gpa.allocator();

    // ********** load Model **********
    var indices = std.ArrayList(u32).init(allocator);
    defer indices.deinit();
    var vertices = std.ArrayList(Vertex).init(allocator);
    defer vertices.deinit();
    var meshs = std.ArrayList(Mesh).init(allocator);
    defer meshs.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    appendGltfModel(arena.allocator(), &meshs, &vertices, &indices, "assets/untitled.gltf");
    const viking_room = Model{
        .mesh_begin = 0,
        .mesh_end = @truncate(u32, meshs.items.len),
    };
    appendGltfModel(arena.allocator(), &meshs, &vertices, &indices, "assets/cube.gltf");
    const cube = Model{
        .mesh_begin = viking_room.mesh_end,
        .mesh_end = @truncate(u32, meshs.items.len),
    };
    arena.deinit();
    //*********************************

    const gc = try GraphicsContext.init(allocator, app_name, window);
    defer gc.deinit();

    std.debug.print("Using device: {s}\n", .{gc.deviceName()});

    var swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    // ********** BasicRenderer **********

    // Define the push constant range used by the pipeline layout
    // Note that the spec only requires a minimum of 128 bytes, so for passing larger blocks of data you'd use UBOs or SSBOs
    const push_constants = [_]vk.PushConstantRange{.{
        .stage_flags = .{ .fragment_bit = true },
        .offset = 0,
        .size = @sizeOf(PushConstant),
    }};
    const vert_shader = try Shader.createFromMemory(
        gc,
        allocator,
        resources.triangle_vert,
        "main",
        .{ .vertex_bit = true },
        &[_]Shader.BindingInfo{.{
            .binding = 0,
            .descriptor_type = .uniform_buffer,
        }},
        &.{
            .binding = &Vertex.binding_description,
            .attribute = &Vertex.attribute_description,
        },
        srcToString(@src()),
    );
    defer vert_shader.deinit(gc);

    const frag_shader = try Shader.createFromMemory(
        gc,
        allocator,
        resources.triangle_frag,
        "main",
        .{ .fragment_bit = true },
        &[_]Shader.BindingInfo{.{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
        }},
        null,
        srcToString(@src()),
    );
    defer frag_shader.deinit(gc);

    var shader_binding = ShaderBinding.init(allocator);
    defer shader_binding.deinit();
    try shader_binding.addShader(vert_shader);
    try shader_binding.addShader(frag_shader);

    var renderer = try BasicRenderer.init(
        gc,
        swapchain.extent,
        shader_binding,
        swapchain.surface_format.format,
        &push_constants,
        .{},
        srcToString(@src()),
    );
    defer renderer.deinit(gc);

    var framebuffers = try createFramebuffers(gc, allocator, swapchain, &renderer, srcToString(@src()));
    defer destroyFramebuffers(&gc, allocator, framebuffers);
    //*********************************

    const frame_size = @truncate(u32, framebuffers.len);

    const vertex_buffer = try Buffer.init(gc, Buffer.CreateInfo{
        .size = @sizeOf(Vertex) * vertices.items.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));
    defer vertex_buffer.deinit(gc);
    const textures = [_]Texture{
        try Texture.loadFromMemory(gc, .texture, &[_]u8{ 125, 125, 125, 255 }, 1, 1, 4, .{}, "default texture"),
        try Texture.loadFromFile(
            gc,
            .texture,
            "assets/viking_room.png",
            .{ .anisotropy = true, .mip_map = true },
        ),
    };
    defer for (textures) |texture| {
        texture.deinit(gc);
    };

    // ================= Skybox ===================
    const skybox_textures = [_]Texture{
        try Texture.loadFromMemory(gc, .cube_map, &[_]u8{
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
        }, 6, 1, 4, .{}, "default skybox"),
        try Texture.loadFromFile(
            gc,
            .cube_map,
            "assets/cube_map.png",
            .{ .anisotropy = true, .mip_map = true },
        ),
    };
    defer for (skybox_textures) |texture| {
        texture.deinit(gc);
    };
    const skybox_vert = try Shader.createFromMemory(
        gc,
        allocator,
        resources.skybox_vert,
        "main",
        .{ .vertex_bit = true },
        &[_]Shader.BindingInfo{.{
            .binding = 0,
            .descriptor_type = .uniform_buffer,
        }},
        &.{
            .binding = &[_]vk.VertexInputBindingDescription{.{
                .binding = 0,
                .stride = @sizeOf(Vertex),
                .input_rate = .vertex,
            }},
            .attribute = &[_]vk.VertexInputAttributeDescription{
                .{
                    .binding = 0,
                    .location = 0,
                    .format = .r32g32b32_sfloat,
                    .offset = 0,
                },
            },
        },
        srcToString(@src()),
    );
    defer skybox_vert.deinit(gc);

    const skybox_frag = try Shader.createFromMemory(
        gc,
        allocator,
        resources.skybox_frag,
        "main",
        .{ .fragment_bit = true },
        &[_]Shader.BindingInfo{.{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
        }},
        null,
        srcToString(@src()),
    );
    defer skybox_frag.deinit(gc);

    var skybox_binding = ShaderBinding.init(allocator);
    defer skybox_binding.deinit();
    try skybox_binding.addShader(skybox_vert);
    try skybox_binding.addShader(skybox_frag);
    const skybox_pipeline = try Pipeline.createSkyboxPipeline(
        gc,
        renderer.render_pass,
        skybox_binding,
        &push_constants,

        .{},
        "skybox" ++ srcToString(@src()),
    );
    defer skybox_pipeline.deinit(gc);
    // ============================================
    var camera = Camera{
        .pitch = 270,
        .yaw = 30,
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
    // const pool_size = frame_size;
    const pool_sizes = [_]vk.DescriptorPoolSize{ .{
        .@"type" = .uniform_buffer,
        .descriptor_count = frame_size,
    }, .{
        .@"type" = .combined_image_sampler,
        .descriptor_count = @truncate(u32, textures.len),
    } };
    const descriptor_pool = try gc.create(vk.DescriptorPoolCreateInfo{
        .flags = .{},
        .max_sets = frame_size + 1,
        .pool_size_count = @truncate(u32, pool_sizes.len),
        .p_pool_sizes = @ptrCast([*]const vk.DescriptorPoolSize, &pool_sizes),
    }, srcToString(@src()));

    defer gc.destroy(descriptor_pool);
    var des_layouts = try allocator.alloc(vk.DescriptorSetLayout, frame_size);
    for (des_layouts) |*l| {
        l.* = renderer.pipeline.descriptor_set_layout;
    }
    defer allocator.free(des_layouts);
    var dsai = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .descriptor_set_count = frame_size,
        .p_set_layouts = des_layouts.ptr,
    };
    var des_sets = try allocator.alloc(vk.DescriptorSet, frame_size);
    defer allocator.free(des_sets);
    try gc.vkd.allocateDescriptorSets(gc.dev, &dsai, des_sets.ptr);
    for (des_sets) |ds, i| {
        updateDescriptorSet(gc, ds, ubo_buffers[i]);
    }

    const des_variable_sets = vk.DescriptorSetVariableDescriptorCountAllocateInfo{
        .descriptor_set_count = 1,
        .p_descriptor_counts = &[_]u32{2},
    };
    for (des_layouts) |*l| {
        l.* = renderer.pipeline.bindless_set_layout;
    }
    dsai.descriptor_set_count = 1;
    dsai.p_next = @ptrCast(*const anyopaque, &des_variable_sets);
    var bindless_sets: vk.DescriptorSet = undefined;
    try gc.vkd.allocateDescriptorSets(gc.dev, &dsai, @ptrCast([*]vk.DescriptorSet, &bindless_sets));
    {
        const wds = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = bindless_sets,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 2,
                .descriptor_type = .combined_image_sampler,
                .p_image_info = &[_]vk.DescriptorImageInfo{ .{
                    .sampler = textures[0].smapler,
                    .image_view = textures[0].view,
                    .image_layout = textures[0].image.layout,
                }, .{
                    .sampler = textures[1].smapler,
                    .image_view = textures[1].view,
                    .image_layout = textures[1].image.layout,
                } },
                .p_buffer_info = undefined,
                .p_texel_buffer_view = undefined,
            },
        };
        gc.vkd.updateDescriptorSets(gc.dev, @truncate(u32, wds.len), @ptrCast([*]const vk.WriteDescriptorSet, &wds), 0, undefined);
    }

    // =====skybox descriptor ====
    // var skybox_des = try allocator.alloc(vk.DescriptorSet, frame_size);
    // defer allocator.free(skybox_des);
    // try gc.vkd.allocateDescriptorSets(gc.dev, &dsai, skybox_des.ptr);
    // for (skybox_des) |ds, i| {
    //     updateDescriptorSet(gc, ds, ubo_buffers[i], skybox_textures);
    // }
    //End descriptor set

    var draw_pool = try DrawPool.init(gc, srcToString(@src()));
    defer draw_pool.deinit(gc);

    //Timer
    var update_timer = try std.time.Timer.start();
    var press = false;

    var push_constant = PushConstant{ .texture_id = 0 };
    while (!window.shouldClose()) {
        const dt = @intToFloat(f32, update_timer.lap()) / @intToFloat(f32, std.time.ns_per_s);
        camera.moveCamera(window, dt);
        try ubo_buffers[swapchain.image_index].update(UniformBufferObject, gc, &[_]UniformBufferObject{.{
            .model = Mat4.identity(),
            .view = camera.getViewMatrix(),
            .proj = camera.getProjMatrix(swapchain.extent.width, swapchain.extent.height),
        }});
        if (window.getKey(.f) == .press) {
            press = true;
        }

        if (window.getKey(.f) == .release) {
            if (press) {
                push_constant.texture_id = (push_constant.texture_id + 1) % @truncate(u32, textures.len);
                press = false;
            }
        }
        const cmdbuf = draw_pool.createCommandBuffer();

        try buildCommandBuffers(
            gc,
            renderer,
            swapchain.image_index,
            framebuffers[swapchain.image_index],
            cmdbuf.cmd,
            vertex_buffer.buffer,
            index_buffer.buffer,
            des_sets,
            bindless_sets,
            meshs.items,
            viking_room,
            cube,
            // skybox_pipeline,
            // skybox_des,
            push_constant,
        );

        const state = swapchain.present(cmdbuf, &draw_pool) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            const size = try window.getSize();
            extent.width = @intCast(u32, size.width);
            extent.height = @intCast(u32, size.height);
            try swapchain.recreate(extent);

            destroyFramebuffers(&gc, allocator, framebuffers);
            framebuffers = try createFramebuffers(gc, allocator, swapchain, &renderer, srcToString(@src()));
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

fn updateDescriptorSet(gc: GraphicsContext, descriptor_set: vk.DescriptorSet, buffer: Buffer) void {
    const wds = [_]vk.WriteDescriptorSet{
        .{
            .dst_set = descriptor_set,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .uniform_buffer,
            .p_image_info = undefined,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = buffer.buffer,
                .offset = 0,
                .range = buffer.size,
            }},
            .p_texel_buffer_view = undefined,
        },
        // .{
        //     .dst_set = descriptor_set,
        //     .dst_binding = 1,
        //     .dst_array_element = 0,
        //     .descriptor_count = 2,
        //     .descriptor_type = .combined_image_sampler,
        //     .p_image_info = &[_]vk.DescriptorImageInfo{ .{
        //         .sampler = textures[0].smapler,
        //         .image_view = textures[0].view,
        //         .image_layout = textures[0].image.layout,
        //     }, .{
        //         .sampler = textures[1].smapler,
        //         .image_view = textures[1].view,
        //         .image_layout = textures[1].image.layout,
        //     } },
        //     .p_buffer_info = undefined,
        //     .p_texel_buffer_view = undefined,
        // },
    };
    gc.vkd.updateDescriptorSets(gc.dev, @truncate(u32, wds.len), @ptrCast([*]const vk.WriteDescriptorSet, &wds), 0, undefined);
}

fn buildCommandBuffers(
    gc: GraphicsContext,
    renderer: BasicRenderer,
    i: u32,
    framebuffer: vk.Framebuffer,
    cmdbuf: vk.CommandBuffer,
    vertex_buffer: vk.Buffer,
    index_buffer: vk.Buffer,
    sets: []const vk.DescriptorSet,
    bindless: vk.DescriptorSet,
    meshs: []const Mesh,
    viking_room: Model,
    cube: Model,
    // skybox: Pipeline,
    // skybox_des: []vk.DescriptorSet,
    push_constant: PushConstant,
) !void {
    try renderer.beginFrame(gc, framebuffer, cmdbuf);

    const offset = [_]vk.DeviceSize{0};
    gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast([*]const vk.Buffer, &vertex_buffer), &offset);
    gc.vkd.cmdBindIndexBuffer(cmdbuf, index_buffer, 0, .uint32);
    gc.vkd.cmdBindDescriptorSets(
        cmdbuf,
        .graphics,
        renderer.pipeline.pipeline_layout,
        0,
        2,
        &[_]vk.DescriptorSet{
            sets[i],
            bindless,
        },
        0,
        undefined,
    );
    gc.vkd.cmdPushConstants(
        cmdbuf,
        renderer.pipeline.pipeline_layout,
        .{ .fragment_bit = true },
        0,
        @sizeOf(PushConstant),
        @ptrCast(*const anyopaque, &push_constant),
    );
    viking_room.draw(gc, cmdbuf, meshs);

    // skybox.bind(gc, cmdbuf);
    // gc.vkd.cmdBindDescriptorSets(
    //     cmdbuf,
    //     .graphics,
    //     renderer.pipeline.pipeline_layout,
    //     0,
    //     1,
    //     @ptrCast([*]const vk.DescriptorSet, &skybox_des[i]),
    //     0,
    //     undefined,
    // );
    cube.draw(gc, cmdbuf, meshs);
    try renderer.endFrame(gc, cmdbuf);
}

fn createFramebuffers(
    gc: GraphicsContext,
    allocator: Allocator,
    swapchain: Swapchain,
    renderer: *BasicRenderer,
    label: ?[*:0]const u8,
) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (framebuffers) |*fb, j| {
        fb.* = try renderer.createFrameBuffer(gc, swapchain.extent, swapchain.swap_images[j].view, label);
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const GraphicsContext, allocator: Allocator, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);
    allocator.free(framebuffers);
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
    const base_indices = @truncate(u32, all_indices.items.len);
    const base_vertices = @truncate(u32, all_vertices.items.len);

    while (mesh_index < num_meshes) : (mesh_index += 1) {
        const num_prims = @intCast(u32, data.meshes[mesh_index].primitives_count);
        var prim_index: u32 = 0;

        while (prim_index < num_prims) : (prim_index += 1) {
            const pre_indices_len = indices.items.len;
            const pre_positions_len = positions.items.len;

            appendMeshPrimitive(data, mesh_index, prim_index, &indices, &positions, &normals, &texcoords0, null);

            all_meshes.append(.{
                .index_offset = @intCast(u32, base_indices + pre_indices_len),
                .vertex_offset = @intCast(i32, base_vertices + pre_positions_len),
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
