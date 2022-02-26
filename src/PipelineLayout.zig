layout: vk.PipelineLayout,
// template: vk.DescriptorUpdateTemplate,
push_constant_stage: vk.ShaderStageFlags,

const std = @import("std");
const vk = @import("vulkan");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const DescriptorLayout = @import("DescriptorLayout.zig");
const Self = @This();

const capacity = 8;

pub fn init(
    gc: GraphicsContext,
    des_layouts: []const DescriptorLayout,
    push_constant: ?vk.PushConstantRange,
    label: ?[*:0]const u8,
) !Self {
    std.debug.assert(des_layouts.len > 0);

    var self: Self = undefined;
    self.push_constant_stage = .{};
    const len = @truncate(u32, des_layouts.len);
    var combine_des_layout = std.BoundedArray(vk.DescriptorSetLayout, capacity){};

    for (des_layouts) |l| {
        std.debug.assert(l.layout != .null_handle);
        combine_des_layout.append(l.layout) catch unreachable;
    }
    if (push_constant) |p| {
        self.push_constant_stage = p.stage_flags;
    }

    self.layout = try gc.create(vk.PipelineLayoutCreateInfo{
        .flags = .{},
        .set_layout_count = len,
        .p_set_layouts = combine_des_layout.slice().ptr,
        .push_constant_range_count = if (push_constant != null) 1 else 0,
        .p_push_constant_ranges = if (push_constant) |p| &[_]vk.PushConstantRange{p} else undefined,
    }, label);
    return self;
}

pub fn pushConstant(self: Self, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, data: anytype) void {
    const size = @sizeOf(@TypeOf(data));
    std.debug.assert(size <= gc.props.limits.max_push_constants_size);
    std.debug.assert(self.push_constant_stage.toInt() > 0);
    gc.vkd.cmdPushConstants(
        cmdbuf,
        self.layout,
        self.push_constant_stage,
        0,
        size,
        @ptrCast(*const anyopaque, &data),
    );
}

pub fn deinit(self: Self, gc: GraphicsContext) void {
    gc.destroy(self.layout);
}

pub fn bindDescriptorSet(
    self: Self,
    gc: GraphicsContext,
    command_buffer: vk.CommandBuffer,
    pipeline_bind_point: vk.PipelineBindPoint,
    first_set: u32,
    descriptor_sets: []const vk.DescriptorSet,
) void {
    gc.vkd.cmdBindDescriptorSets(
        command_buffer,
        pipeline_bind_point,
        self.layout,
        first_set,
        @truncate(u32, descriptor_sets.len),
        descriptor_sets.ptr,
        0,
        undefined,
    );
}

pub fn pushDescriptorSet(
    self: Self,
    gc: GraphicsContext,
    command_buffer: vk.CommandBuffer,
    descriptor_update_template: vk.DescriptorUpdateTemplate,
    set: u32,
    data: []const DescriptorLayout.DescriptorInfo,
) void {
    gc.vkd.cmdPushDescriptorSetWithTemplateKHR(
        command_buffer,
        descriptor_update_template,
        self.layout,
        set,
        @ptrCast(*const anyopaque, data.ptr),
    );
}

// pub const DescriptorInfo = union(enum) {
//     image: vk.DescriptorImageInfo,
//     buffer: vk.DescriptorBufferInfo,

//     pub fn create(resource: anytype) DescriptorInfo {
//         const T = @TypeOf(resource);
//         if (T == Buffer) {
//             return .{ .buffer = .{
//                 .buffer = resource.buffer,
//                 .offset = 0,
//                 .range = resource.create_info.size,
//             } };
//         }
//         if (T == Image) {
//             return .{ .image = .{
//                 .sampler = .null_handle,
//                 .image_view = resource.view,
//                 .image_layout = resource.image.layout,
//             } };
//         }
//         @compileError("expect Buffer or Image but get " ++ @typeName(T));
//     }
// };

// pub fn createDescriptorTemplate(self: Self, gc: GraphicsContext, pipeline_layout: vk.PipelineLayout, set: u32) !vk.DescriptorUpdateTemplate {
//     var combine_entry: std.BoundedArray(vk.DescriptorUpdateTemplateEntry, capacity) = .{};
//     switch (self.info) {
//         .template => |bindings| {
//             for (bindings) |binding, i| {
//                 combine_entry.append(.{
//                     .dst_binding = binding.binding,
//                     .dst_array_element = 0,
//                     .descriptor_count = 1,
//                     .descriptor_type = binding.des_type,
//                     .offset = @sizeOf(DescriptorInfo) * i,
//                     .stride = @sizeOf(DescriptorInfo),
//                 }) catch unreachable;
//             }
//         },
//         else => unreachable,
//     }
//     const slice = combine_entry.slice();
//     return gc.create(vk.DescriptorUpdateTemplateCreateInfo{
//         .flags = .{},
//         .descriptor_update_entry_count = @truncate(u32, slice.len),
//         .p_descriptor_update_entries = slice.ptr,
//         .template_type = .push_descriptors_khr,
//         .descriptor_set_layout = self.layout,
//         .pipeline_bind_point = .graphics,
//         .pipeline_layout = pipeline_layout,
//         .set = set,
//     }, "update template");
// }
