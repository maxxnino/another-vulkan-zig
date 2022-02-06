module: vk.ShaderModule,
stage: Stage,
entry: [*:0]const u8,
layouts: []const DescriptorBindingLayout,

const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const VertexInputDescription = @import("vertex.zig").VertexInputDescription;
const Self = @This();

pub const DescriptorBindingLayout = struct {
    binding: u32,
    descriptor_type: vk.DescriptorType,
};
pub const Stage = union(enum) {
    vertex: VertexInputDescription,
    fragment,
};

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
