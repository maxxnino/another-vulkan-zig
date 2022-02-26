layout: vk.DescriptorSetLayout,
info: BindingInfo,
template: vk.DescriptorUpdateTemplate,
total: u32,

const std = @import("std");
const vk = @import("vulkan");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Self = @This();

const capacity = 8;
const bindless_flags = [_]vk.DescriptorBindingFlags{.{
    // .update_after_bind_bit = true,
    // .update_unused_while_pending_bit = true,
    .partially_bound_bit = true,
    // .variable_descriptor_count_bit = true,
}} ** capacity;

pub const DescriptorType = enum {
    bindless,
    template,
    immutable_sampler,
};

pub const BindingInfo = union(DescriptorType) {
    bindless: []const Bindless,
    template: []const Template,
    immutable_sampler: ImmutableSampler,

    pub const Bindless = struct {
        binding: u32,
        count: u32,
        stage: vk.ShaderStageFlags,
        des_type: vk.DescriptorType,
    };

    pub const Template = struct {
        binding: u32,
        stage: vk.ShaderStageFlags,
        des_type: vk.DescriptorType,
    };

    pub const ImmutableSampler = struct {
        binding: u32,
        samplers: []vk.Sampler,
        stage: vk.ShaderStageFlags,
    };
};

pub fn init(gc: GraphicsContext, info: BindingInfo, label: ?[*:0]const u8) !Self {
    var self: Self = undefined;
    self.total = 0;
    self.info = info;
    self.template = .null_handle;

    var combine_binding: std.BoundedArray(vk.DescriptorSetLayoutBinding, capacity) = .{};
    switch (info) {
        .bindless => |bindings| {
            for (bindings) |binding| {
                combine_binding.append(.{
                    .binding = binding.binding,
                    .descriptor_type = binding.des_type,
                    .descriptor_count = binding.count,
                    .stage_flags = binding.stage,
                    .p_immutable_samplers = null,
                }) catch unreachable;
                self.total += binding.count;
            }
        },
        .template => |bindings| {
            for (bindings) |binding| {
                combine_binding.append(.{
                    .binding = binding.binding,
                    .descriptor_type = binding.des_type,
                    .descriptor_count = 1,
                    .stage_flags = binding.stage,
                    .p_immutable_samplers = null,
                }) catch unreachable;
                self.total += 1;
            }
        },
        .immutable_sampler => |is| {
            self.total += @truncate(u32, is.samplers.len);
            combine_binding.append(.{
                .binding = is.binding,
                .descriptor_type = .sampler,
                .descriptor_count = self.total,
                .stage_flags = is.stage,
                .p_immutable_samplers = is.samplers.ptr,
            }) catch unreachable;
        },
    }

    const binding_slice = combine_binding.slice();

    self.layout = try gc.create(vk.DescriptorSetLayoutCreateInfo{
        .flags = if (info == .template) .{ .push_descriptor_bit_khr = true } else .{},
        .binding_count = @truncate(u32, binding_slice.len),
        .p_bindings = binding_slice.ptr,
        .p_next = if (info == .bindless) @ptrCast(*const anyopaque, &vk.DescriptorSetLayoutBindingFlagsCreateInfo{
            .binding_count = @truncate(u32, binding_slice.len),
            .p_binding_flags = &bindless_flags,
        }) else null,
    }, label);

    return self;
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.layout);
}

pub fn createDescriptorSet(self: Self, gc: GraphicsContext, pool: vk.DescriptorPool, label: ?[*:0]const u8) !vk.DescriptorSet {
    var dsai = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = pool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &self.layout),
        .p_next = if (self.info == .bindless)
            @ptrCast(*const anyopaque, &vk.DescriptorSetVariableDescriptorCountAllocateInfo{
                .descriptor_set_count = 1,
                .p_descriptor_counts = &[_]u32{self.total},
            })
        else
            null,
    };

    var descriptor_set: vk.DescriptorSet = .null_handle;
    try gc.vkd.allocateDescriptorSets(gc.dev, &dsai, @ptrCast([*]vk.DescriptorSet, &descriptor_set));
    try gc.markHandle(descriptor_set, .descriptor_set, label);
    return descriptor_set;
}

pub const DescriptorInfo = union(enum) {
    image: vk.DescriptorImageInfo,
    buffer: vk.DescriptorBufferInfo,
};

pub fn createDescriptorTemplate(self: Self, gc: GraphicsContext, pipeline_layout: vk.PipelineLayout, set: u32) !vk.DescriptorUpdateTemplate {
    var combine_entry: std.BoundedArray(vk.DescriptorUpdateTemplateEntry, capacity) = .{};
    switch (self.info) {
        .template => |bindings| {
            for (bindings) |binding, i| {
                combine_entry.append(.{
                    .dst_binding = binding.binding,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = binding.des_type,
                    .offset = @sizeOf(DescriptorInfo) * i,
                    .stride = @sizeOf(DescriptorInfo),
                }) catch unreachable;
            }
        },
        else => unreachable,
    }
    const slice = combine_entry.slice();
    return gc.create(vk.DescriptorUpdateTemplateCreateInfo{
        .flags = .{},
        .descriptor_update_entry_count = @truncate(u32, slice.len),
        .p_descriptor_update_entries = slice.ptr,
        .template_type = .push_descriptors_khr,
        .descriptor_set_layout = self.layout,
        .pipeline_bind_point = .graphics,
        .pipeline_layout = pipeline_layout,
        .set = set,
    }, "update template");
}
