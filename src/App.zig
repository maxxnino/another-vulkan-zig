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
const Model = @import("Model.zig");
const Mesh = Model.Mesh;
const Window = @import("Window.zig");
const Self = @This();
const srcToString = @import("util.zig").srcToString;
const command_pool = @import("command_pool.zig");
const DrawPool = command_pool.DrawPool;
const assert = std.debug.assert;
const z = @import("zalgebra");
const Mat4 = z.Mat4;
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Vec4 = z.Vec4;

const app_name = "vulkan + glfw";
const UniformBufferObject = struct {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
};

const PushConstant = struct {
    texture_id: u32,
};
const Vertex = struct {
    position: Vec3,
    tex_coord: Vec2,
    normal: Vec3,
};

const VertexArray = std.MultiArrayList(Vertex);

allocator: std.mem.Allocator,
window: Window,
timer: std.time.Timer,
indices: std.ArrayList(u32),
vertices: VertexArray,
meshs: std.ArrayList(Mesh),
gc: GraphicsContext,
swapchain: Swapchain,
renderer: BasicRenderer,
framebuffers: []vk.Framebuffer,
vertex_buffer: [3]Buffer,
textures: [2]Texture,
skybox_textures: [3]Texture,
skybox_pipeline: Pipeline,
push_constant: [2]PushConstant,
camera: Camera,
ubo_buffers: []Buffer,
index_buffer: Buffer,
descriptor_pool: vk.DescriptorPool,
des_sets: []vk.DescriptorSet,
bindless_sets: vk.DescriptorSet,
draw_pool: DrawPool,
cube: Model,
viking_room: Model,
pub fn init(allocator: std.mem.Allocator) !Self {
    var self: Self = undefined;
    self.allocator = allocator;
    self.window = try Window.init(app_name, false, 800, 600);

    // ********** load Model **********
    self.indices = std.ArrayList(u32).init(allocator);
    self.vertices = VertexArray{};
    self.meshs = std.ArrayList(Mesh).init(allocator);
    var arena = std.heap.ArenaAllocator.init(allocator);
    appendGltfModel(allocator, arena.allocator(), &self.meshs, &self.vertices, &self.indices, "assets/untitled.gltf");
    self.viking_room = Model{
        .mesh_begin = 0,
        .mesh_end = @truncate(u32, self.meshs.items.len),
    };
    appendGltfModel(allocator, arena.allocator(), &self.meshs, &self.vertices, &self.indices, "assets/cube.gltf");
    self.cube = Model{
        .mesh_begin = self.viking_room.mesh_end,
        .mesh_end = @truncate(u32, self.meshs.items.len),
    };
    arena.deinit();
    //*********************************

    self.gc = try GraphicsContext.init(allocator, app_name, self.window.window);

    std.debug.print("Using device: {s}\n", .{self.gc.deviceName()});

    self.swapchain = try Swapchain.init(self.gc, allocator, .{ .width = self.window.width, .height = self.window.height });

    // ********** BasicRenderer **********

    // Define the push constant range used by the pipeline layout
    // Note that the spec only requires a minimum of 128 bytes, so for passing larger blocks of data you'd use UBOs or SSBOs
    const push_constants = [_]vk.PushConstantRange{.{
        .stage_flags = .{ .fragment_bit = true },
        .offset = 0,
        .size = @sizeOf(PushConstant),
    }};
    const vert_shader = try Shader.createFromMemory(
        self.gc,
        resources.triangle_vert,
        "main",
        .{ .vertex = Shader.getVertexInput(&.{ .position, .tex_coord, .normal }) },
        &[_]Shader.DescriptorBindingLayout{.{
            .binding = 0,
            .descriptor_type = .uniform_buffer,
        }},

        srcToString(@src()),
    );
    defer vert_shader.deinit(self.gc);

    const frag_shader = try Shader.createFromMemory(
        self.gc,
        resources.triangle_frag,
        "main",
        .{ .fragment = {} },
        &[_]Shader.DescriptorBindingLayout{.{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
        }},
        srcToString(@src()),
    );
    defer frag_shader.deinit(self.gc);

    var shader_binding = ShaderBinding.init(allocator);
    defer shader_binding.deinit();
    try shader_binding.addShader(vert_shader);
    try shader_binding.addShader(frag_shader);

    self.renderer = try BasicRenderer.init(
        self.gc,
        self.swapchain.extent,
        shader_binding,
        self.swapchain.surface_format.format,
        &push_constants,
        .{},
        srcToString(@src()),
    );

    self.framebuffers = try createFramebuffers(self.gc, allocator, self.swapchain, &self.renderer, srcToString(@src()));
    //*********************************

    const frame_size = @truncate(u32, self.framebuffers.len);

    self.vertex_buffer[0] = try Buffer.init(self.gc, Buffer.CreateInfo{
        .size = @sizeOf(Vec3) * self.vertices.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));
    self.vertex_buffer[1] = try Buffer.init(self.gc, Buffer.CreateInfo{
        .size = @sizeOf(Vec2) * self.vertices.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));
    self.vertex_buffer[2] = try Buffer.init(self.gc, Buffer.CreateInfo{
        .size = @sizeOf(Vec3) * self.vertices.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));
    self.textures = [_]Texture{
        try Texture.loadFromMemory(self.gc, .texture, &[_]u8{ 125, 125, 125, 255 }, 1, 1, 4, .{}, "default texture"),
        try Texture.loadFromFile(
            self.gc,
            .texture,
            "assets/viking_room.png",
            .{ .anisotropy = true, .mip_map = true },
        ),
    };

    // ================= Skybox ===================
    self.skybox_textures = [_]Texture{
        try Texture.loadFromMemory(self.gc, .cube_map, &[_]u8{
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
        }, 6, 1, 4, .{}, "default skybox"),
        try Texture.loadFromFile(
            self.gc,
            .cube_map,
            "assets/cube_map.png",
            .{ .anisotropy = true, .mip_map = true },
        ),
        try Texture.loadFromFile(
            self.gc,
            .cube_map,
            "assets/cube_map_2.png",
            .{ .anisotropy = true, .mip_map = true },
        ),
    };
    const skybox_vert = try Shader.createFromMemory(
        self.gc,
        resources.skybox_vert,
        "main",
        .{ .vertex = Shader.getVertexInput(&.{.position}) },
        &[_]Shader.DescriptorBindingLayout{.{
            .binding = 0,
            .descriptor_type = .uniform_buffer,
        }},
        srcToString(@src()),
    );
    defer skybox_vert.deinit(self.gc);

    const skybox_frag = try Shader.createFromMemory(
        self.gc,
        resources.skybox_frag,
        "main",
        .{ .fragment = {} },
        &[_]Shader.DescriptorBindingLayout{.{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
        }},
        srcToString(@src()),
    );
    defer skybox_frag.deinit(self.gc);

    var skybox_binding = ShaderBinding.init(allocator);
    defer skybox_binding.deinit();
    try skybox_binding.addShader(skybox_vert);
    try skybox_binding.addShader(skybox_frag);
    self.skybox_pipeline = try Pipeline.createSkyboxPipeline(
        self.gc,
        self.renderer.render_pass,
        skybox_binding,
        &push_constants,

        .{},
        "skybox" ++ srcToString(@src()),
    );
    // ============================================
    self.camera = Camera{
        .pitch = 270,
        .yaw = 30,
        .pos = Vec3.new(0, 2, 4),
    };

    self.ubo_buffers = try allocator.alloc(Buffer, self.framebuffers.len);

    for (self.ubo_buffers) |*ubo| {
        ubo.* = try Buffer.init(self.gc, Buffer.CreateInfo{
            .size = @sizeOf(UniformBufferObject),
            .buffer_usage = .{ .uniform_buffer_bit = true },
            .memory_usage = .cpu_to_gpu,
            .memory_flags = .{},
        }, srcToString(@src()));
        try ubo.upload(UniformBufferObject, self.gc, &[_]UniformBufferObject{.{
            .model = Mat4.identity(),
            .view = self.camera.getViewMatrix(),
            .proj = self.camera.getProjMatrix(self.swapchain.extent.width, self.swapchain.extent.height),
        }});
    }

    self.index_buffer = try Buffer.init(self.gc, Buffer.CreateInfo{
        .size = @sizeOf(u32) * self.indices.items.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .index_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));

    // uploadVertices
    const slice = self.vertices.slice();
    try self.vertex_buffer[0].upload(Vec3, self.gc, slice.items(.position));
    try self.vertex_buffer[1].upload(Vec2, self.gc, slice.items(.tex_coord));
    try self.vertex_buffer[2].upload(Vec3, self.gc, slice.items(.normal));
    //Upload indices
    try self.index_buffer.upload(u32, self.gc, self.indices.items);

    // Desciptor Set
    // const pool_size = frame_size;
    const pool_sizes = [_]vk.DescriptorPoolSize{ .{
        .@"type" = .uniform_buffer,
        .descriptor_count = frame_size,
    }, .{
        .@"type" = .combined_image_sampler,
        .descriptor_count = 64,
    } };
    self.descriptor_pool = try self.gc.create(vk.DescriptorPoolCreateInfo{
        .flags = .{},
        .max_sets = frame_size + 1,
        .pool_size_count = @truncate(u32, pool_sizes.len),
        .p_pool_sizes = @ptrCast([*]const vk.DescriptorPoolSize, &pool_sizes),
    }, srcToString(@src()));

    var des_layouts = try allocator.alloc(vk.DescriptorSetLayout, frame_size);
    defer allocator.free(des_layouts);
    for (des_layouts) |*l| {
        l.* = self.renderer.pipeline.descriptor_set_layout;
    }
    var dsai = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = frame_size,
        .p_set_layouts = des_layouts.ptr,
    };
    self.des_sets = try allocator.alloc(vk.DescriptorSet, frame_size);
    try self.gc.vkd.allocateDescriptorSets(self.gc.dev, &dsai, self.des_sets.ptr);
    for (self.des_sets) |ds, i| {
        updateDescriptorSet(self.gc, ds, self.ubo_buffers[i]);
    }

    const des_variable_sets = vk.DescriptorSetVariableDescriptorCountAllocateInfo{
        .descriptor_set_count = 1,
        .p_descriptor_counts = &[_]u32{64},
    };
    for (des_layouts) |*l| {
        l.* = self.renderer.pipeline.bindless_set_layout;
    }
    dsai.descriptor_set_count = 1;
    dsai.p_next = @ptrCast(*const anyopaque, &des_variable_sets);
    try self.gc.vkd.allocateDescriptorSets(self.gc.dev, &dsai, @ptrCast([*]vk.DescriptorSet, &self.bindless_sets));
    {
        const dii = [_]vk.DescriptorImageInfo{ .{
            .sampler = self.textures[0].smapler,
            .image_view = self.textures[0].view,
            .image_layout = self.textures[0].image.layout,
        }, .{
            .sampler = self.textures[1].smapler,
            .image_view = self.textures[1].view,
            .image_layout = self.textures[1].image.layout,
        }, .{
            .sampler = self.skybox_textures[0].smapler,
            .image_view = self.skybox_textures[0].view,
            .image_layout = self.skybox_textures[0].image.layout,
        }, .{
            .sampler = self.skybox_textures[1].smapler,
            .image_view = self.skybox_textures[1].view,
            .image_layout = self.skybox_textures[1].image.layout,
        }, .{
            .sampler = self.skybox_textures[2].smapler,
            .image_view = self.skybox_textures[2].view,
            .image_layout = self.skybox_textures[2].image.layout,
        } };
        const wds = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = self.bindless_sets,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = @truncate(u32, dii.len),
                .descriptor_type = .combined_image_sampler,
                .p_image_info = &dii,
                .p_buffer_info = undefined,
                .p_texel_buffer_view = undefined,
            },
        };
        self.gc.vkd.updateDescriptorSets(
            self.gc.dev,
            @truncate(u32, wds.len),
            @ptrCast([*]const vk.WriteDescriptorSet, &wds),
            0,
            undefined,
        );
    }
    //End descriptor set

    self.draw_pool = try DrawPool.init(self.gc, srcToString(@src()));

    //Timer
    self.timer = try std.time.Timer.start();
    self.push_constant = [_]PushConstant{
        .{ .texture_id = 0 },
        .{ .texture_id = 0 },
    };
    return self;
}

pub fn deinit(self: *Self) void {
    self.draw_pool.deinit(self.gc);
    self.gc.destroy(self.descriptor_pool);
    self.allocator.free(self.des_sets);
    for (self.ubo_buffers) |ubo| {
        ubo.deinit(self.gc);
    }
    self.allocator.free(self.ubo_buffers);
    self.skybox_pipeline.deinit(self.gc);
    for (self.skybox_textures) |texture| {
        texture.deinit(self.gc);
    }
    for (self.textures) |texture| {
        texture.deinit(self.gc);
    }
    self.index_buffer.deinit(self.gc);
    for (self.vertex_buffer) |vb| {
        vb.deinit(self.gc);
    }
    destroyFramebuffers(&self.gc, self.allocator, self.framebuffers);
    self.renderer.deinit(self.gc);
    self.swapchain.deinit(self.gc);
    self.gc.deinit();
    self.indices.deinit();
    self.vertices.deinit(self.allocator);
    self.meshs.deinit();
    self.window.deinit();
}

pub fn run(self: *Self) !void {
    while (self.window.pollEvents()) {
        if (self.window.isKey(.q, .just_press)) break;
        const dt = @intToFloat(f32, self.timer.lap()) / @intToFloat(f32, std.time.ns_per_s);

        self.camera.moveCamera(self.window, dt);
        try self.ubo_buffers[self.swapchain.image_index].upload(UniformBufferObject, self.gc, &[_]UniformBufferObject{.{
            .model = Mat4.identity(),
            .view = self.camera.getViewMatrix(),
            .proj = self.camera.getProjMatrix(self.swapchain.extent.width, self.swapchain.extent.height),
        }});
        if (self.window.isKey(.f, .just_press)) {
            self.push_constant[0].texture_id = (self.push_constant[0].texture_id + 1) % @truncate(u32, self.textures.len);
        }

        if (self.window.isKey(.g, .just_press)) {
            self.push_constant[1].texture_id = (self.push_constant[1].texture_id + 1) % @truncate(u32, self.skybox_textures.len);
        }
        const cmdbuf = self.draw_pool.createCommandBuffer();

        try buildCommandBuffers(
            self.gc,
            self.renderer,
            self.swapchain.image_index,
            self.framebuffers[self.swapchain.image_index],
            cmdbuf.cmd,
            self.vertex_buffer,
            self.index_buffer.buffer,
            self.des_sets,
            self.bindless_sets,
            self.meshs.items,
            self.viking_room,
            self.skybox_pipeline,
            self.cube,
            &self.push_constant,
        );

        const state = self.swapchain.present(self.gc, cmdbuf, &self.draw_pool) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            const size = try self.window.window.getSize();
            self.window.width = @intCast(u32, size.width);
            self.window.height = @intCast(u32, size.height);
            try self.swapchain.recreate(self.gc, .{ .width = self.window.width, .height = self.window.height });

            destroyFramebuffers(&self.gc, self.allocator, self.framebuffers);
            self.framebuffers = try createFramebuffers(self.gc, self.allocator, self.swapchain, &self.renderer, srcToString(@src()));
            try self.draw_pool.resetAll(self.gc);
        }
    }
    try self.swapchain.waitForAllFences(self.gc);
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
                .range = buffer.create_info.size,
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
    vertex_buffer: [3]Buffer,
    index_buffer: vk.Buffer,
    sets: []const vk.DescriptorSet,
    bindless: vk.DescriptorSet,
    meshs: []const Mesh,
    viking_room: Model,
    skybox: Pipeline,
    cube: Model,
    push_constant: []PushConstant,
) !void {
    try renderer.beginFrame(gc, framebuffer, cmdbuf);

    const offset = [_]vk.DeviceSize{ 0, 0, 0 };
    gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 3, &[_]vk.Buffer{
        vertex_buffer[0].buffer,
        vertex_buffer[1].buffer,
        vertex_buffer[2].buffer,
    }, &offset);
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
        @ptrCast(*const anyopaque, &push_constant[0]),
    );
    viking_room.draw(gc, cmdbuf, meshs);

    //draw skybox
    skybox.bind(gc, cmdbuf);
    gc.vkd.cmdPushConstants(
        cmdbuf,
        skybox.pipeline_layout,
        .{ .fragment_bit = true },
        0,
        @sizeOf(PushConstant),
        @ptrCast(*const anyopaque, &PushConstant{ .texture_id = push_constant[1].texture_id + 2 }),
    );
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
    base_allocator: Allocator,
    arena: Allocator,
    all_meshes: *std.ArrayList(Mesh),
    all_vertices: *VertexArray,
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
    const base_vertices = @truncate(u32, all_vertices.len);

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

    all_vertices.ensureTotalCapacity(base_allocator, positions.items.len) catch unreachable;
    for (positions.items) |_, index| {
        all_vertices.appendAssumeCapacity(.{
            .position = positions.items[index].scale(0.2), // NOTE(mziulek): Sponza requires scaling.
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
