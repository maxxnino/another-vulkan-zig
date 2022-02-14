shaders: []Shader,
layout: LayoutMap,
shader_stage: ShaderStage,
vertex_stage_info: ?vk.PipelineVertexInputStateCreateInfo = null,

const std = @import("std");
const vk = @import("vulkan");
const assert = std.debug.assert;

const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Shader = @import("Shader.zig");
const Self = @This();
const LayoutMap = std.AutoArrayHashMap(u32, vk.DescriptorSetLayoutBinding);
const ShaderStage = std.ArrayList(vk.PipelineShaderStageCreateInfo);

pub fn init(allocator: std.mem.Allocator) Self {
    return .{
        .shaders = &.{},
        .layout = LayoutMap.init(allocator),
        .shader_stage = ShaderStage.init(allocator),
    };
}

pub fn addShader(self: *Self, shader: Shader) !void {
    const stage_flags: vk.ShaderStageFlags = switch (shader.stage) {
        .vertex => |v| blk: {
            //Already create vertex_stage_info
            assert(self.vertex_stage_info == null);
            self.vertex_stage_info = .{
                .flags = .{},
                .vertex_binding_description_count = @truncate(u32, v.binding.len),
                .p_vertex_binding_descriptions = v.binding.ptr,
                .vertex_attribute_description_count = @truncate(u32, v.attribute.len),
                .p_vertex_attribute_descriptions = v.attribute.ptr,
            };

            break :blk .{ .vertex_bit = true };
        },
        .fragment => .{ .fragment_bit = true },
    };

    // DescriptorSetLayoutBinding
    for (shader.layouts) |info| {
        if (self.layout.getEntry(info.binding)) |dslb| {
            // Same binding but different type,
            assert(dslb.value_ptr.descriptor_type == info.descriptor_type);
            dslb.value_ptr.stage_flags = dslb.value_ptr.stage_flags.merge(stage_flags);
        } else {
            try self.layout.put(info.binding, .{
                .binding = info.binding,
                .descriptor_type = info.descriptor_type,
                .descriptor_count = 1,
                .stage_flags = stage_flags,
                .p_immutable_samplers = null,
            });
        }
    }
    // PipelineShaderStageCreateInfo
    try self.shader_stage.append(.{
        .flags = .{},
        .stage = stage_flags,
        .module = shader.module,
        .p_name = shader.entry,
        .p_specialization_info = null,
    });
}

pub fn createDescriptorSetLayout(self: Self, gc: GraphicsContext, label: ?[*:0]const u8) !vk.DescriptorSetLayout {
    const dslb = self.layout.values();
    return gc.create(vk.DescriptorSetLayoutCreateInfo{
        .flags = .{ .push_descriptor_bit_khr = true },
        .binding_count = @truncate(u32, dslb.len),
        .p_bindings = @ptrCast([*]const vk.DescriptorSetLayoutBinding, dslb.ptr),
    }, label);
}

pub fn getPipelineShaderStageCreateInfo(self: Self) []vk.PipelineShaderStageCreateInfo {
    return self.shader_stage.items;
}

pub fn deinit(self: *Self) void {
    self.layout.deinit();
    self.shader_stage.deinit();
}
