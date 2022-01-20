const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Self = @This();

pub const BindingInfo = struct {
    binding: u32,
    descriptor_type: vk.DescriptorType,
};

module: vk.ShaderModule,
entry: [*:0]const u8,
stage: vk.ShaderStageFlags,
bindings_info: []const BindingInfo,

pub fn createFromMemory(
    gc: GraphicsContext,
    allocator: std.mem.Allocator,
    buffer: []const u8,
    entry: [*:0]const u8,
    stage: vk.ShaderStageFlags,
    bindings_info: []const BindingInfo,
    label: ?[*:0]const u8,
) !Self {
    verifyBinding(allocator, bindings_info);
    var shader: Self = undefined;

    shader.module = try gc.create(vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = buffer.len,
        .p_code = @ptrCast([*]const u32, @alignCast(@alignOf(u32), buffer.ptr)),
    }, label);
    shader.entry = entry;
    shader.stage = stage;
    shader.bindings_info = bindings_info;
    return shader;
}

pub fn createFromFile(
    gc: GraphicsContext,
    allocator: std.mem.Allocator,
    file_path: [*:0]const u8,
    entry: [*:0]const u8,
    stage: vk.ShaderStageFlags,
) !Self {
    const file = try std.fs.cwd().openFileZ(file_path, .{});
    defer file.close();

    const buffer = try file.readToEndAlloc(allocator, std.math.maxInt(u32));
    defer allocator.free(buffer);
    return try createFromMemory(gc, allocator, buffer, entry, stage, file_path);
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.module);
}

fn verifyBinding(allocator: std.mem.Allocator, bindings_info: []const BindingInfo) void {
    if (builtin.mode == .Debug) {
        var map = std.AutoHashMap(u32, void).init(allocator);
        defer map.deinit();
        for (bindings_info) |i| {
            map.putNoClobber(i.binding, {}) catch unreachable;
        }
    }
}
