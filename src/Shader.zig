module: vk.ShaderModule,
stage: Stage,
entry: [*:0]const u8,

const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const VertexInputDescription = @import("vertex.zig").VertexInputDescription;
const Self = @This();

pub const Stage = union(enum) {
    vertex: VertexInputDescription,
    fragment,
};

pub fn createFromMemory(
    gc: GraphicsContext,
    buffer: []const u8,
    entry: [*:0]const u8,
    stage: Stage,
    label: ?[*:0]const u8,
) !Self {
    // verifyBinding(allocator, bindings_info);
    var shader: Self = undefined;
    _ = std.math.divExact(usize, buffer.len, 4) catch unreachable;
    shader.module = try gc.create(vk.ShaderModuleCreateInfo{
        .flags = .{},
        .code_size = buffer.len,
        .p_code = @ptrCast([*]const u32, @alignCast(@alignOf(u32), buffer)),
    }, label);
    shader.stage = stage;
    shader.entry = entry;

    return shader;
}
// pub fn createDescriptorSetLayout(allocator: std.mem.Allocator, gc: GraphicsContext, shaders: []Self, label: ?[*:0]const u8) !vk.DescriptorSetLayout {
//     var layout = std.AutoArrayHashMap(u32, vk.DescriptorSetLayoutBinding).init(allocator);
//     defer layout.deinit();

//     for (shaders) |shader| {
//         const stage_flags = shader.getStageFlags();
//         for (shader.layouts) |info| {
//             if (layout.getEntry(info.binding)) |dslb| {
//                 // Same binding but different type,
//                 std.debug.assert(dslb.value_ptr.descriptor_type == info.descriptor_type);
//                 dslb.value_ptr.stage_flags = dslb.value_ptr.stage_flags.merge(stage_flags);
//             } else {
//                 try layout.put(info.binding, .{
//                     .binding = info.binding,
//                     .descriptor_type = info.descriptor_type,
//                     .descriptor_count = 1,
//                     .stage_flags = stage_flags,
//                     .p_immutable_samplers = null,
//                 });
//             }
//         }
//     }
// //     gc.vkd.createDescriptorUpdateTemplate();
// //     []vk.DescriptorUpdateTemplateEntry{
// //         .dst_binding = i,
// //         .dst_array_element = 0,
// //         .descriptor_count = 1,
// //         .descriptor_type = descriptor_type,
// //         .offset = @sizeOf(DescriptorInfo) * i,
// //         .stride = @sizeOf(DescriptorInfo),
// //     };

// //     vk.DescriptorUpdateTemplateCreateInfo{
// //         .flags = .{},
// //         .descriptor_update_entry_count = u32,
// //         .p_descriptor_update_entries = [*]const DescriptorUpdateTemplateEntry,
// //         .template_type = .push_descriptors_khr,
// //         .descriptor_set_layout = descriptor_set_layout,
// //         .pipeline_bind_point = .graphics,
// //         .pipeline_layout = pipeline_layout,
// //         .set = 2,
// //     };
//     const dslb = layout.values();
//     return gc.create(vk.DescriptorSetLayoutCreateInfo{
//         .flags = .{ .push_descriptor_bit_khr = true },
//         .binding_count = @truncate(u32, dslb.len),
//         .p_bindings = @ptrCast([*]const vk.DescriptorSetLayoutBinding, dslb.ptr),
//     }, label);
// }

pub fn createFromFile(
    gc: GraphicsContext,
    allocator: std.mem.Allocator,
    file_path: [*:0]const u8,
    entry: [*:0]const u8,
    stage: Stage,
) !Self {
    const file = try std.fs.cwd().openFileZ(file_path, .{});
    defer file.close();

    const buffer = try file.readToEndAlloc(allocator, std.math.maxInt(u32));
    defer allocator.free(buffer);
    return try createFromMemory(gc, buffer, entry, stage, file_path);
}

pub fn vertexInputState(self: Self) ?vk.PipelineVertexInputStateCreateInfo {
    return switch (self.stage) {
        .vertex => |v| .{
            .flags = .{},
            .vertex_binding_description_count = @truncate(u32, v.binding.len),
            .p_vertex_binding_descriptions = v.binding.ptr,
            .vertex_attribute_description_count = @truncate(u32, v.attribute.len),
            .p_vertex_attribute_descriptions = v.attribute.ptr,
        },
        else => null,
    };
}

pub fn getPipelineShaderStageCreateInfo(self: Self) vk.PipelineShaderStageCreateInfo {
    return vk.PipelineShaderStageCreateInfo{
        .flags = .{},
        .stage = switch (self.stage) {
            .vertex => .{ .vertex_bit = true },
            .fragment => .{ .fragment_bit = true },
        },
        .module = self.module,
        .p_name = self.entry,
        .p_specialization_info = null,
    };
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.module);
}
