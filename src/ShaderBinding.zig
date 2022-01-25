const std = @import("std");
const vk = @import("vulkan");
const assert = std.debug.assert;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Shader = @import("Shader.zig");
const Self = @This();

const LayoutMap = std.AutoArrayHashMap(u32, vk.DescriptorSetLayoutBinding);
const ShaderStage = std.ArrayList(vk.PipelineShaderStageCreateInfo);

shaders: []Shader,
layout: LayoutMap,
shader_stage: ShaderStage,
vertex_stage_info: ?vk.PipelineVertexInputStateCreateInfo = null,

pub fn init(allocator: std.mem.Allocator) Self {
    return .{
        .shaders = &.{},
        .layout = LayoutMap.init(allocator),
        .shader_stage = ShaderStage.init(allocator),
    };
}

pub fn addShader(self: *Self, shader: Shader) !void {
    //create PipelineVertexInputStateCreateInfo if shader_stage is vertex_bit
    if (shader.stage.contains(.{ .vertex_bit = true })) {
        if (shader.vertex_info) |vi| {
            //Already create vertex_stage_info
            assert(self.vertex_stage_info == null);
            self.vertex_stage_info = .{
                .flags = .{},
                .vertex_binding_description_count = @truncate(u32, vi.binding.len),
                .p_vertex_binding_descriptions = vi.binding.ptr,
                .vertex_attribute_description_count = @truncate(u32, vi.attribute.len),
                .p_vertex_attribute_descriptions = vi.attribute.ptr,
            };
        }
    }

    // build DescriptorSetLayoutBinding
    for (shader.bindings_info) |info| {
        if (self.layout.getEntry(info.binding)) |dslb| {
            // Same binding but different type,
            assert(dslb.value_ptr.descriptor_type == info.descriptor_type);

            dslb.value_ptr.stage_flags = dslb.value_ptr.stage_flags.merge(shader.stage);
        } else {
            try self.layout.put(info.binding, .{
                .binding = info.binding,
                .descriptor_type = info.descriptor_type,
                .descriptor_count = 1,
                .stage_flags = shader.stage,
                .p_immutable_samplers = null,
            });
        }
    }

    // build PipelineShaderStageCreateInfo array
    try self.shader_stage.append(.{
        .flags = .{},
        .stage = shader.stage,
        .module = shader.module,
        .p_name = shader.entry,
        .p_specialization_info = null,
    });
}

pub fn createDescriptorSetLayout(self: Self, gc: GraphicsContext, label: ?[*:0]const u8) !vk.DescriptorSetLayout {
    const dslb = self.layout.values();
    return gc.create(vk.DescriptorSetLayoutCreateInfo{
        .flags = .{},
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
