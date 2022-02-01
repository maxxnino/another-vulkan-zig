const std = @import("std");
const vk = @import("vulkan");
const vma = @import("binding/vma.zig");
const builtin = @import("builtin");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Buffer = @import("Buffer.zig");
const z = @import("zalgebra");
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Vec4 = z.Vec4;
const Self = @This();

pub const Vertex = struct {
    position: Vec3,
    tex_coord: Vec2,
    normal: Vec3,
};

pub const Component = VertexArray.Field;
pub const VertexArray = std.MultiArrayList(Vertex);

pub const InputDescription = struct {
    binding: []const vk.VertexInputBindingDescription,
    attribute: []const vk.VertexInputAttributeDescription,

    pub fn get(comptime components: []const Component) InputDescription {
        comptime var bd: [components.len]vk.VertexInputBindingDescription = undefined;
        comptime var ad: [components.len]vk.VertexInputAttributeDescription = undefined;
        inline for (components) |component, i| {
            const location = @enumToInt(component);
            bd[i].binding = location;
            bd[i].stride = comptime toStride(component);
            bd[i].input_rate = .vertex;

            ad[i].binding = location;
            ad[i].location = location;
            ad[i].format = comptime toFormat(component);
            ad[i].offset = 0;
        }
        return .{
            .binding = &bd,
            .attribute = &ad,
        };
    }
};

const vertex_fields = std.meta.fields(Vertex);
const component_fields = std.meta.fields(Component);

pub const VertexBuffer = struct {
    pub const zero_offsets = [_]vk.DeviceSize{0} ** component_fields.len;

    buffers: [component_fields.len]Buffer,
    bind_buffer: [component_fields.len]vk.Buffer,

    pub fn init(gc: GraphicsContext, slice: VertexArray.Slice, label: ?[*:0]const u8) !VertexBuffer {
        var self: VertexBuffer = undefined;

        inline for (component_fields) |field, i| {
            const component = @intToEnum(Component, field.value);
            self.buffers[i] = try Buffer.init(gc, Buffer.CreateInfo{
                .size = toStride(component) * slice.len,
                .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
                .memory_usage = .gpu_only,
                .memory_flags = .{},
            }, label);

            try self.buffers[field.value].upload(FieldType(component), gc, slice.items(component));

            self.bind_buffer[i] = self.buffers[i].buffer;
        }

        return self;
    }

    pub fn bind(self: VertexBuffer, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, offsets: [component_fields.len]vk.DeviceSize) void {
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, component_fields.len, &self.bind_buffer, &offsets);
    }

    pub fn deinit(self: VertexBuffer, gc: GraphicsContext) void {
        for (self.buffers) |buffer| {
            buffer.deinit(gc);
        }
    }
};

fn toFormat(comptime component: Component) vk.Format {
    const T = FieldType(component);
    if (T == Vec3) return .r32g32b32_sfloat;
    if (T == Vec2) return .r32g32_sfloat;
    if (T == Vec4) return .r32g32b32a32_sfloat;

    @compileError("Not support field: " ++ @tagName(component) ++ " with type: " ++ @typeName(T));
}

fn toStride(comptime component: Component) u32 {
    return @sizeOf(FieldType(component));
}

fn FieldType(comptime component: Component) type {
    return std.meta.fieldInfo(Vertex, component).field_type;
}
