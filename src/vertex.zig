const std = @import("std");
const vk = @import("vulkan");
const vma = @import("binding/vma.zig");
const builtin = @import("builtin");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const VulkanBuffer = @import("Buffer.zig");

pub const VertexInputDescription = struct {
    binding: []const vk.VertexInputBindingDescription,
    attribute: []const vk.VertexInputAttributeDescription,
};

pub fn VertexGen(comptime Vertex: type) type {
    return struct {
        const Self = @This();
        pub const MultiArrayList = std.MultiArrayList(Vertex);
        pub usingnamespace Vertex;

        const Component = MultiArrayList.Field;
        pub fn inputDescription(comptime components: []const Component) VertexInputDescription {
            comptime var bd: [components.len]vk.VertexInputBindingDescription = undefined;
            comptime var ad: [components.len]vk.VertexInputAttributeDescription = undefined;
            inline for (components) |component, i| {
                const ComponentType = FieldType(component);
                const location = @enumToInt(component);
                bd[i].binding = location;
                bd[i].stride = @sizeOf(ComponentType);
                bd[i].input_rate = .vertex;

                ad[i].binding = location;
                ad[i].location = location;
                ad[i].format = comptime Vertex.format(component);
                ad[i].offset = 0;
            }
            return .{
                .binding = &bd,
                .attribute = &ad,
            };
        }
        const vertex_fields = std.meta.fields(Vertex);
        const component_fields = std.meta.fields(Component);

        pub const Buffer = struct {
            pub const zero_offsets = [_]vk.DeviceSize{0} ** component_fields.len;

            buffers: [component_fields.len]VulkanBuffer,
            bind_buffer: [component_fields.len]vk.Buffer,

            pub fn init(gc: GraphicsContext, slice: MultiArrayList.Slice, label: ?[*:0]const u8) !Buffer {
                var self: Buffer = undefined;

                inline for (component_fields) |field, i| {
                    const component = @intToEnum(Component, field.value);
                    const ComponentType = FieldType(component);

                    self.buffers[i] = try VulkanBuffer.init(gc, VulkanBuffer.CreateInfo{
                        .size = @sizeOf(ComponentType) * slice.len,
                        .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
                        .memory_usage = .gpu_only,
                        .memory_flags = .{},
                    }, label);

                    try self.buffers[field.value].upload(FieldType(component), gc, slice.items(component));

                    self.bind_buffer[i] = self.buffers[i].buffer;
                }

                return self;
            }

            pub fn bind(self: Buffer, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, offsets: [component_fields.len]vk.DeviceSize) void {
                gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, component_fields.len, &self.bind_buffer, &offsets);
            }

            pub fn deinit(self: Buffer, gc: GraphicsContext) void {
                for (self.buffers) |buffer| {
                    buffer.deinit(gc);
                }
            }
        };

        fn FieldType(comptime component: Component) type {
            return std.meta.fieldInfo(Vertex, component).field_type;
        }
    };
}
