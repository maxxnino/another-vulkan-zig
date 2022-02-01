const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Self = @This();

pub const Component = enum {
    position,
    tex_coord,
    normal,
    tangent,
    color,

    pub fn format(component: Component) vk.Format  {
        return switch (component) {
            .position => .r32g32b32_sfloat,
            .tex_coord => .r32g32_sfloat,
            .normal => .r32g32b32_sfloat,
            .tangent => .r32g32b32a32_sfloat,
            .color => .r32g32b32_sfloat,
        };
    }

    pub fn stride(component: Component) u32 {
        return switch (component) {
            .position => @sizeOf([3]f32),
            .tex_coord => @sizeOf([2]f32),
            .normal => @sizeOf([3]f32),
            .tangent => @sizeOf([4]f32),
            .color => @sizeOf([3]f32),
        };
    }
};

pub const VertexInput = struct {
    input_binding: []const vk.VertexInputBindingDescription,
    input_attribute: []const vk.VertexInputAttributeDescription,
};

pub fn getVertexInput(comptime components: []const Component) VertexInput {
    comptime var bd: [components.len]vk.VertexInputBindingDescription = undefined;
    comptime var ad: [components.len]vk.VertexInputAttributeDescription = undefined;
    inline for (components) |component, i| {
        const location = @enumToInt(component);
        bd[i].binding = location;
        bd[i].stride = comptime Component.stride(component);
        bd[i].input_rate = .vertex;

        ad[i].binding = location;
        ad[i].location = location;
        ad[i].format = comptime Component.format(component);
        ad[i].offset = 0;
    }
    return .{
        .input_binding = &bd,
        .input_attribute = &ad,
    };
}
pub const DescriptorBindingLayout = struct {
    binding: u32,
    descriptor_type: vk.DescriptorType,
};
pub const Stage = union(enum) {
    vertex: VertexInput,
    fragment,
};

module: vk.ShaderModule,
stage: Stage,
entry: [*:0]const u8,
layouts: []const DescriptorBindingLayout,

pub fn createFromMemory(
    gc: GraphicsContext,
    buffer: []const u8,
    entry: [*:0]const u8,
    stage: Stage,
    layouts: []const DescriptorBindingLayout,
    label: ?[*:0]const u8,
) !Self {
    // verifyBinding(allocator, bindings_info);
    var shader: Self = undefined;

    shader.module = try gc.create(vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = buffer.len,
        .p_code = @ptrCast([*]const u32, @alignCast(@alignOf(u32), buffer.ptr)),
    }, label);
    shader.stage = stage;
    shader.entry = entry;
    shader.layouts = layouts;
    return shader;
}

pub fn createFromFile(
    gc: GraphicsContext,
    allocator: std.mem.Allocator,
    file_path: [*:0]const u8,
    stage: Stage,
    entry: [*:0]const u8,
    layouts: []const DescriptorBindingLayout,
) !Self {
    const file = try std.fs.cwd().openFileZ(file_path, .{});
    defer file.close();

    const buffer = try file.readToEndAlloc(allocator, std.math.maxInt(u32));
    defer allocator.free(buffer);
    return try createFromMemory(gc, buffer, stage, entry, layouts, file_path);
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.module);
}

pub fn verifyBinding(map: std.AutoHashMap(u32, void), layouts: []const DescriptorBindingLayout) void {
    defer map.clearRetainingCapacity();
    if (builtin.mode == .Debug) {
        for (layouts) |i| {
            map.putNoClobber(i.binding, {}) catch unreachable;
        }
    }
}
