layout: vk.DescriptorSetLayout,
descriptor_type: DescriptorType,
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
    normal,
    immutable_sampler,
};

pub const BindingInfo = union(DescriptorType) {
    bindless: []const BindlessAndNormal,
    normal: []const BindlessAndNormal,
    immutable_sampler: ImmutableSampler,

    pub const BindlessAndNormal = struct {
        binding: u32,
        count: u32,
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
            self.descriptor_type = .bindless;
        },
        .normal => |bindings| {
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
            self.descriptor_type = .normal;
        },
        .immutable_sampler => |is| {
            self.total += @truncate(u32, is.samplers.len);
            self.descriptor_type = .immutable_sampler;
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
        .flags = .{},
        .binding_count = @truncate(u32, binding_slice.len),
        .p_bindings = binding_slice.ptr,
        .p_next = if (self.descriptor_type == .bindless) @ptrCast(*const anyopaque, &vk.DescriptorSetLayoutBindingFlagsCreateInfo{
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
        .p_next = if (self.descriptor_type == .bindless)
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
