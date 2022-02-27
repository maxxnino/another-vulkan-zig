const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const cgltf = @import("binding/cgltf.zig");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Buffer = @import("Buffer.zig");
const Allocator = std.mem.Allocator;
const Camera = @import("Camera.zig");
const tex = @import("texture.zig");
const Texture = tex.Texture;
const Shader = @import("Shader.zig");
const BasicRenderer = @import("BasicRenderer.zig");
const DescriptorLayout = @import("DescriptorLayout.zig");
const PipelineLayout = @import("PipelineLayout.zig");
const Pipeline = @import("Pipeline.zig");
const Model = @import("Model.zig");
const Mesh = Model.Mesh;
const VertexGen = @import("vertex.zig").VertexGen;
const Window = @import("Window.zig");
const Self = @This();
const srcToString = @import("util.zig").srcToString;
const command_pool = @import("command_pool.zig");
const DrawPool = command_pool.DrawPool;
const assert = std.debug.assert;
const z = @import("zalgebra");
const basisu = @import("binding/basisu.zig");
const Mat4 = z.Mat4;
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Vec4 = z.Vec4;

const Vertex = VertexGen(struct {
    position: z.Vector4(u16),
    texcoord: z.Vector2(u16),
    normal: z.Vector4(i8),
    tangent: z.Vector4(i8),

    pub fn format(component: anytype) vk.Format {
        return switch (component) {
            .position => .r16g16b16a16_unorm,
            .texcoord => .r16g16_unorm,
            .normal => .r8g8b8a8_snorm,
            .tangent => .r8g8b8a8_snorm,
        };
    }

    pub fn componentType(comptime at: cgltf.AttributeType) cgltf.ComponentType {
        return switch (at) {
            .position => .r_16u,
            .texcoord => .r_16u,
            .normal => .r_8,
            .tangent => .r_8,
            else => unreachable,
        };
    }
    pub fn accessorType(comptime at: cgltf.AttributeType) cgltf.Type {
        return switch (at) {
            .position => .vec3,
            .texcoord => .vec2,
            .normal => .vec3,
            .tangent => .vec4,
            else => unreachable,
        };
    }

    pub fn hasComponent(comptime at: cgltf.AttributeType) bool {
        return switch (at) {
            .position => true,
            .texcoord => true,
            .normal => true,
            .tangent => true,
            else => false,
        };
    }
});

const app_name = "vulkan + glfw";
const UniformBufferObject = struct {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
    camera_pos: Vec3,
};

const Light = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    const speed: f32 = 10;

    pub fn update(self: *Light, window: Window, dt: f32) void {
        var h: f32 = 0;
        var v: f32 = 0;
        if (window.isKey(.right, .press)) h += 1;
        if (window.isKey(.left, .press)) h -= 1;
        if (window.isKey(.up, .press)) v += 1;
        if (window.isKey(.down, .press)) v -= 1;
        self.x += h * dt * speed;
        self.z += v * dt * speed;
    }

    fn getPos(self: Light) Vec3 {
        return Vec3.new(self.x, self.y, self.z);
    }
};

const PushConstant = struct {
    base_color_id: u32,
    normal_id: u32 = 0,
    metalic_roughtness_id: u32 = 0,
    ao_id: u32 = 0,
    light: Vec3,
};

allocator: std.mem.Allocator,
window: Window,
timer: std.time.Timer,
indices: std.ArrayList(u32),
vertices: Vertex.MultiArrayList,
meshs: std.ArrayList(Mesh),
gc: GraphicsContext,
swapchain: Swapchain,
renderer: BasicRenderer,
framebuffers: []vk.Framebuffer,
vertex_buffer: Vertex.Buffer,
textures: [5]Texture,
skybox_textures: [1]Texture,
skybox_pipeline: Pipeline,
skybox_push_constant: PushConstant,
object_push_constant: PushConstant,
current_mat: u32 = 0,
camera: Camera,
ubo_buffers: []Buffer,
uniform_des: DescriptorLayout,
template: vk.DescriptorUpdateTemplate,
pipeline_layout: PipelineLayout,
index_buffer: Buffer,
descriptor_pool: vk.DescriptorPool,
bindless_sets: vk.DescriptorSet,
immutable_sampler_set: vk.DescriptorSet,
draw_pool: DrawPool,
cube: Model,
viking_room: Model,
light: Light,
gpu_time: f32,
pub fn init(allocator: std.mem.Allocator) !Self {
    var self: Self = undefined;
    self.allocator = allocator;
    self.gpu_time = 0;
    basisu.init();

    self.window = try Window.init(app_name, false, 800, 600);

    // ********** load Model **********
    self.indices = std.ArrayList(u32).init(allocator);
    self.vertices = Vertex.MultiArrayList{};
    self.meshs = std.ArrayList(Mesh).init(allocator);
    appendGltfModel(allocator, &self.meshs, &self.vertices, &self.indices, "assets/SciFiHelmet/SciFiHelmet_fast.gltf");
    // appendGltfModel(allocator, &self.meshs, &self.vertices, &self.indices, "assets/cube_fast.gltf");
    self.viking_room = Model{
        .mesh_begin = 0,
        .mesh_end = @truncate(u32, self.meshs.items.len),
    };
    appendGltfModel(allocator, &self.meshs, &self.vertices, &self.indices, "assets/cube_fast.gltf");
    self.cube = Model{
        .mesh_begin = self.viking_room.mesh_end,
        .mesh_end = @truncate(u32, self.meshs.items.len),
    };
    //*********************************

    self.gc = try GraphicsContext.init(allocator, app_name, self.window.window);

    std.log.info("Using device: {s}\n", .{self.gc.deviceName()});

    self.swapchain = try Swapchain.init(self.gc, allocator, .{ .width = self.window.width, .height = self.window.height });

    // ********** BasicRenderer **********

    const vert_shader = try Shader.createFromFile(
        self.gc,
        allocator,
        "zig-cache/shaders/shaders/triangle.vert.spv",
        "main",
        .{ .vertex = Vertex.inputDescription(&.{ .position, .texcoord, .normal, .tangent }) },
    );
    defer vert_shader.deinit(self.gc);

    const frag_shader = try Shader.createFromFile(
        self.gc,
        allocator,
        "zig-cache/shaders/shaders/triangle.frag.spv",
        "main",
        .{ .fragment = {} },
    );
    defer frag_shader.deinit(self.gc);

    const bindless = try DescriptorLayout.init(self.gc, .{
        .bindless = &[_]DescriptorLayout.BindingInfo.Bindless{.{
            .binding = 0,
            .count = 64,
            .stage = .{ .fragment_bit = true },
            .des_type = .sampled_image,
        }},
    }, "bindless " ++ srcToString(@src()));
    defer bindless.deinit(self.gc);

    const immutable = try DescriptorLayout.init(self.gc, .{
        .immutable_sampler = DescriptorLayout.BindingInfo.ImmutableSampler{
            .binding = 0,
            .stage = .{ .fragment_bit = true },
            .samplers = &.{self.gc.immutable_samplers},
        },
    }, "immutable " ++ srcToString(@src()));
    defer immutable.deinit(self.gc);

    self.uniform_des = try DescriptorLayout.init(self.gc, .{
        .template = &[_]DescriptorLayout.BindingInfo.Template{.{
            .binding = 0,
            .stage = .{ .fragment_bit = true, .vertex_bit = true },
            .des_type = .uniform_buffer,
        }},
    }, "uniform " ++ srcToString(@src()));
    // defer uniform_des.deinit(self.gc);

    // Define the push constant range used by the pipeline layout
    // Note that the spec only requires a minimum of 128 bytes, so for passing larger blocks of data you'd use UBOs or SSBOs
    const push_constant = vk.PushConstantRange{
        .stage_flags = .{ .fragment_bit = true },
        .offset = 0,
        .size = @sizeOf(PushConstant),
    };

    self.pipeline_layout = try PipelineLayout.init(
        self.gc,
        &.{ bindless, immutable, self.uniform_des },
        push_constant,
        "pipeline layout" ++ srcToString(@src()),
    );

    self.renderer = try BasicRenderer.init(
        self.gc,
        self.swapchain.extent,
        &.{ vert_shader, frag_shader },
        self.swapchain.surface_format.format,
        .{},
        self.pipeline_layout,
        "basic pipeline " ++ srcToString(@src()),
    );

    self.template = try self.uniform_des.createDescriptorTemplate(self.gc, self.pipeline_layout, 2);
    self.framebuffers = try createFramebuffers(self.gc, allocator, self.swapchain, &self.renderer, srcToString(@src()));
    //*********************************

    self.textures = [_]Texture{
        try Texture.loadFromMemory(self.gc, .srgb, &[_]u8{ 240, 161, 80, 255 }, 1, 1, 4, .{}, "default texture"),
        try Texture.loadCompressFromFile(
            allocator,
            self.gc,
            .srgb,
            "assets/SciFiHelmet/SciFiHelmet_BaseColor.basis",
        ),
        try Texture.loadCompressFromFile(
            allocator,
            self.gc,
            .unorm,
            "assets/SciFiHelmet/SciFiHelmet_Normal.basis",
        ),
        try Texture.loadCompressFromFile(
            allocator,
            self.gc,
            .unorm,
            "assets/SciFiHelmet/SciFiHelmet_MetallicRoughness.basis",
        ),
        try Texture.loadCompressFromFile(
            allocator,
            self.gc,
            .unorm,
            "assets/SciFiHelmet/SciFiHelmet_AmbientOcclusion.basis",
        ),
    };

    // ================= Skybox ===================
    self.skybox_textures = [_]Texture{
        // try Texture.loadFromMemory(self.gc, .cube_map, &[_]u8{
        //     50, 50, 50, 255,
        //     50, 50, 50, 255,
        //     50, 50, 50, 255,
        //     50, 50, 50, 255,
        //     50, 50, 50, 255,
        //     50, 50, 50, 255,
        // }, 6, 1, 4, .{}, "default skybox"),
        try Texture.loadCompressFromFile(
            allocator,
            self.gc,
            .cube_map,
            "assets/cube_map.basis",
        ),
        // try Texture.loadFromFile(
        //     self.gc,
        //     .cube_map,
        //     "assets/cube_map.png",
        //     .{},
        // ),
        // try Texture.loadCompressFromFile(
        //     allocator,
        //     self.gc,
        //     .cube_map,
        //     "assets/cube_map_2.basis",
        // ),
        // try Texture.loadFromFile(
        //     self.gc,
        //     .cube_map,
        //     "assets/cube_map_2.png",
        //     .{},
        // ),
    };
    const skybox_vert = try Shader.createFromFile(
        self.gc,
        allocator,
        "zig-cache/shaders/shaders/texturecubemap/skybox.vert.spv",
        "main",
        .{ .vertex = Vertex.inputDescription(&.{.position}) },
    );
    defer skybox_vert.deinit(self.gc);

    const skybox_frag = try Shader.createFromFile(
        self.gc,
        allocator,
        "zig-cache/shaders/shaders/texturecubemap/skybox.frag.spv",
        "main",
        .{ .fragment = {} },
    );
    defer skybox_frag.deinit(self.gc);

    self.skybox_pipeline = try Pipeline.createSkyboxPipeline(
        self.gc,
        self.renderer.render_pass,
        &.{ skybox_vert, skybox_frag },
        .{},
        self.pipeline_layout,
        "skybox " ++ srcToString(@src()),
    );
    // ============================================
    self.camera = Camera{
        .pitch = 270,
        .yaw = 30,
        .pos = Vec3.new(0, 2, 4),
    };
    self.light = .{};

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
            .camera_pos = self.camera.pos,
        }});
    }

    self.index_buffer = try Buffer.init(self.gc, Buffer.CreateInfo{
        .size = @sizeOf(u32) * self.indices.items.len,
        .buffer_usage = .{ .transfer_dst_bit = true, .index_buffer_bit = true },
        .memory_usage = .gpu_only,
        .memory_flags = .{},
    }, srcToString(@src()));

    // uploadVertices
    self.vertex_buffer = try Vertex.Buffer.init(self.gc, self.vertices.slice(), srcToString(@src()));
    //Upload indices
    try self.index_buffer.upload(u32, self.gc, self.indices.items);

    // Desciptor Set
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{
            .@"type" = .sampled_image,
            .descriptor_count = bindless.total,
        },
        .{
            .@"type" = .sampler,
            .descriptor_count = 1,
        },
    };
    self.descriptor_pool = try self.gc.create(vk.DescriptorPoolCreateInfo{
        .flags = .{},
        .max_sets = 2,
        .pool_size_count = @truncate(u32, pool_sizes.len),
        .p_pool_sizes = @ptrCast([*]const vk.DescriptorPoolSize, &pool_sizes),
    }, srcToString(@src()));

    // immutable_sampler_set
    self.immutable_sampler_set = try immutable.createDescriptorSet(self.gc, self.descriptor_pool, "immutable_sampler_set");

    // bindless_sets
    self.bindless_sets = try bindless.createDescriptorSet(self.gc, self.descriptor_pool, "bindless set");

    {
        var dii: [self.textures.len + self.skybox_textures.len]vk.DescriptorImageInfo = undefined;
        for (self.textures) |t, i| {
            dii[i] = .{
                .sampler = .null_handle,
                .image_view = t.view,
                .image_layout = t.image.layout,
            };
        }
        for (self.skybox_textures) |t, index| {
            const i = index + self.textures.len;

            dii[i] = .{
                .sampler = .null_handle,
                .image_view = t.view,
                .image_layout = t.image.layout,
            };
        }

        const wds = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = self.bindless_sets,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = @truncate(u32, dii.len),
                .descriptor_type = .sampled_image,
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

    self.object_push_constant = .{
        .base_color_id = 1,
        .normal_id = 2,
        .metalic_roughtness_id = 3,
        .ao_id = 4,
        .light = self.light.getPos(),
    };
    self.skybox_push_constant = .{
        .base_color_id = self.textures.len,
        .light = self.light.getPos(),
    };
    return self;
}

pub fn deinit(self: *Self) void {
    self.draw_pool.deinit(self.gc);
    self.gc.destroy(self.descriptor_pool);
    for (self.ubo_buffers) |ubo| {
        ubo.deinit(self.gc);
    }
    self.allocator.free(self.ubo_buffers);
    self.skybox_pipeline.deinit(self.gc);
    self.uniform_des.deinit(self.gc);
    self.pipeline_layout.deinit(self.gc);
    self.gc.destroy(self.template);
    for (self.skybox_textures) |texture| {
        texture.deinit(self.gc);
    }
    for (self.textures) |texture| {
        texture.deinit(self.gc);
    }
    self.index_buffer.deinit(self.gc);
    self.vertex_buffer.deinit(self.gc);
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
    var timer = try std.time.Timer.start();
    var cpu_time: f64 = 0;
    while (self.window.pollEvents()) {
        if (self.window.isKey(.q, .just_press)) break;
        const dt = @intToFloat(f32, self.timer.lap()) / @intToFloat(f32, std.time.ns_per_s);

        self.camera.moveCamera(self.window, dt);
        try self.ubo_buffers[self.swapchain.image_index].upload(UniformBufferObject, self.gc, &[_]UniformBufferObject{.{
            .model = Mat4.fromTranslate(Vec3.new(-0.5, -0.5, -0.5)),
            .view = self.camera.getViewMatrix(),
            .proj = self.camera.getProjMatrix(self.swapchain.extent.width, self.swapchain.extent.height),
            .camera_pos = self.camera.pos,
        }});

        if (self.window.isKey(.g, .just_press)) {
            const texture_len = @truncate(u32, self.textures.len);
            const sky_len = @truncate(u32, self.skybox_textures.len);
            const current = self.skybox_push_constant.base_color_id - texture_len;
            self.skybox_push_constant.base_color_id = ((current + 1) % sky_len) + texture_len;
        }
        if (self.window.isKey(.f, .just_press)) {
            const texture_len = @truncate(u32, self.textures.len);
            const sky_len = @truncate(u32, self.skybox_textures.len);
            var current = self.skybox_push_constant.base_color_id - texture_len;
            if (current > 0) current -= 1;
            self.skybox_push_constant.base_color_id = ((current) % sky_len) + texture_len;
        }

        self.light.update(self.window, dt);
        self.object_push_constant.light = self.light.getPos();

        // ================= Begin Draw ==============
        const command_buffer = self.draw_pool.createCommandBuffer();
        {
            const cmdbuf = command_buffer.cmd;
            const i = self.swapchain.image_index;
            try self.renderer.beginFrame(self.gc, self.framebuffers[i], cmdbuf, i, &self.gpu_time);

            self.vertex_buffer.bind(self.gc, cmdbuf, Vertex.Buffer.zero_offsets);
            self.gc.vkd.cmdBindIndexBuffer(cmdbuf, self.index_buffer.buffer, 0, .uint32);
            self.pipeline_layout.bindDescriptorSet(
                self.gc,
                cmdbuf,
                .graphics,
                0,
                &[_]vk.DescriptorSet{ self.bindless_sets, self.immutable_sampler_set },
            );
            self.pipeline_layout.pushConstant(self.gc, cmdbuf, self.object_push_constant);
            self.pipeline_layout.pushDescriptorSet(self.gc, cmdbuf, self.template, 2, &.{
                DescriptorLayout.DescriptorInfo.create(self.ubo_buffers[i]),
            });

            self.viking_room.draw(self.gc, cmdbuf, self.meshs.items);

            //draw skybox
            self.skybox_pipeline.bind(self.gc, cmdbuf);
            self.pipeline_layout.pushConstant(self.gc, cmdbuf, self.skybox_push_constant);
            self.cube.draw(self.gc, cmdbuf, self.meshs.items);
            try self.renderer.endFrame(self.gc, cmdbuf, i);
        }

        // ============= End Draw ===================

        const state = self.swapchain.present(self.gc, command_buffer, &self.draw_pool) catch |err| switch (err) {
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
        cpu_time = cpu_time * 0.8 + @intToFloat(f64, timer.lap()) * 0.000_000_1;
        var buffer: [64]u8 = undefined;
        const str = try std.fmt.bufPrintZ(&buffer, "Vulkan - Cpu: {d:.2}ms, Gpu: {d:.2}ms", .{ cpu_time, self.gpu_time });
        try self.window.window.setTitle(str.ptr);
    }
    try self.swapchain.waitForAllFences(self.gc);
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
    allocator: Allocator,
    all_meshes: *std.ArrayList(Mesh),
    all_vertices: *Vertex.MultiArrayList,
    all_indices: *std.ArrayList(u32),
    path: [:0]const u8,
) void {
    const data = parseAndLoadGltfFile(path);
    defer cgltf.cgltf_free(data);

    var current_indices_index = @truncate(u32, all_indices.items.len);
    var current_vertex_index = @truncate(u32, all_vertices.len);

    all_vertices.resize(allocator, all_vertices.len + data.accessors[0].count) catch unreachable;
    var slice = all_vertices.slice();

    const num_meshes = @truncate(u32, data.meshes_count);
    var mesh_index: u32 = 0;
    while (mesh_index < num_meshes) : (mesh_index += 1) {
        const num_prims = @intCast(u32, data.meshes[mesh_index].primitives_count);
        var prim_index: u32 = 0;

        while (prim_index < num_prims) : (prim_index += 1) {
            const mesh = appendMeshPrimitive(
                data,
                mesh_index,
                prim_index,
                current_indices_index,
                current_vertex_index,
                all_indices,
                slice,
            );

            all_meshes.append(mesh) catch unreachable;
            current_vertex_index += mesh.num_vertices;
            current_indices_index += mesh.num_indices;
        }
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
    current_indices_index: u32,
    current_vertex_index: u32,
    indices: *std.ArrayList(u32),
    slice: Vertex.MultiArrayList.Slice,
) Mesh {
    assert(mesh_index < data.meshes_count);
    const meshes = data.getMeshes();

    assert(prim_index < data.meshes[mesh_index].primitives_count);
    const primitives = meshes[mesh_index].getPrimitives();

    const attributes = primitives[prim_index].getAttribute();

    const num_vertices = @truncate(u32, attributes[0].data.count);
    const num_indices = @truncate(u32, primitives[prim_index].indices.?.count);

    // Indices.
    {
        indices.ensureTotalCapacity(current_indices_index + num_indices) catch unreachable;
        const accessor = primitives[prim_index].indices.?;

        const buffer_view = accessor.buffer_view.?;
        assert(accessor.stride == buffer_view.stride or buffer_view.stride == 0);
        assert((accessor.stride * accessor.count) == buffer_view.size);

        const data_addr = @alignCast(4, @ptrCast([*]const u8, buffer_view.buffer.data.?) +
            accessor.offset + buffer_view.offset);

        switch (accessor.component_type) {
            .r_8u => {
                assert(accessor.stride == 1);
                const src = @ptrCast([*]const u8, data_addr);
                var i: u32 = 0;
                while (i < num_indices) : (i += 1) {
                    indices.appendAssumeCapacity(src[i]);
                }
            },
            .r_16u => {
                assert(accessor.stride == 2);
                const src = @ptrCast([*]const u16, data_addr);
                var i: u32 = 0;
                while (i < num_indices) : (i += 1) {
                    indices.appendAssumeCapacity(src[i]);
                }
            },
            .r_32u => {
                assert(accessor.stride == 4);
                const src = @ptrCast([*]const u32, data_addr);
                var i: u32 = 0;
                while (i < num_indices) : (i += 1) {
                    indices.appendAssumeCapacity(src[i]);
                }
            },
            else => unreachable,
        }
    }

    // Attributes.
    const end_vertex_index = current_vertex_index + num_vertices;

    for (attributes) |attrib| {
        const accessor = attrib.data;

        const buffer_view = accessor.buffer_view.?;
        assert(accessor.stride == buffer_view.stride or buffer_view.stride == 0);
        assert((accessor.stride * accessor.count) == buffer_view.size);

        const data_addr = @ptrCast([*]const u8, buffer_view.buffer.data.?) +
            accessor.offset + buffer_view.offset;

        switch (attrib.type) {
            .position => if (Vertex.hasComponent(.position)) {
                assert(accessor.type == Vertex.accessorType(.position));
                assert(accessor.component_type == Vertex.componentType(.position));

                var buffer = slice.items(.position)[current_vertex_index..end_vertex_index];
                @memcpy(
                    @ptrCast([*]u8, buffer.ptr),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            },
            .normal => if (Vertex.hasComponent(.normal)) {
                assert(accessor.type == Vertex.accessorType(.normal));
                assert(accessor.component_type == Vertex.componentType(.normal));

                var buffer = slice.items(.normal)[current_vertex_index..end_vertex_index];
                @memcpy(
                    @ptrCast([*]u8, buffer.ptr),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            },
            .tangent => if (Vertex.hasComponent(.tangent)) {
                assert(accessor.type == Vertex.accessorType(.tangent));
                assert(accessor.component_type == Vertex.componentType(.tangent));

                var buffer = slice.items(.tangent)[current_vertex_index..end_vertex_index];
                @memcpy(
                    @ptrCast([*]u8, buffer.ptr),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            },
            .texcoord => if (Vertex.hasComponent(.texcoord)) {
                assert(accessor.type == Vertex.accessorType(.texcoord));
                assert(accessor.component_type == Vertex.componentType(.texcoord));

                var buffer = slice.items(.texcoord)[current_vertex_index..end_vertex_index];
                @memcpy(
                    @ptrCast([*]u8, buffer.ptr),
                    data_addr,
                    accessor.count * accessor.stride,
                );
            },
            else => std.log.info("Parse gltf: not using {s}", .{@tagName(attrib.type)}),
        }
    }
    return .{
        .index_offset = current_indices_index,
        .vertex_offset = @intCast(i32, current_vertex_index),
        .num_indices = num_indices,
        .num_vertices = num_vertices,
    };
}
