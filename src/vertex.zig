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

const vertex_fields = std.meta.fields(Vertex);
const component_fields = std.meta.fields(Component);

pub const Component = enum {
    position,
    tex_coord,
    normal,
    // tangent,
    // color,

    pub fn format(component: Component) vk.Format {
        return switch (component) {
            .position => .r32g32b32_sfloat,
            .tex_coord => .r32g32_sfloat,
            .normal => .r32g32b32_sfloat,
            // .tangent => .r32g32b32a32_sfloat,
            // .color => .r32g32b32_sfloat,
        };
    }

    pub fn stride(component: Component) u32 {
        return switch (component) {
            .position => @sizeOf(Vec3),
            .tex_coord => @sizeOf(Vec2),
            .normal => @sizeOf(Vec3),
            // .tangent => @sizeOf(Vec4),
            // .color => @sizeOf(Vec3),
        };
    }
};

pub const InputDescription = struct {
    binding: []const vk.VertexInputBindingDescription,
    attribute: []const vk.VertexInputAttributeDescription,

    pub fn get(comptime components: []const Component) InputDescription {
        comptime var bd: [components.len]vk.VertexInputBindingDescription = undefined;
        comptime var ad: [components.len]vk.VertexInputAttributeDescription = undefined;
        inline for (components) |component, i| {
            const location = @enumToInt(component);
            bd[i].binding = location;
            bd[i].stride = comptime Component.stride(component);
            bd[i].input_rate = .vertex;

            ad[i].binding = location;
            ad[i].location = location;
            ad[i].format = comptime Component.format(component);
            ad[i].offset = 0;
        }
        return .{
            .binding = &bd,
            .attribute = &ad,
        };
    }
};

pub const Vertex = struct {
    position: Vec3,
    tex_coord: Vec2,
    normal: Vec3,

    comptime {
        if (vertex_fields.len != component_fields.len) @compileError("Vertex and Component lenght mismatch");
        for (component_fields) |c, i| {
            if (!std.mem.eql(u8, c.name, vertex_fields[i].name))
                @compileError("Vertex and Component name mismatch " ++ vertex_fields[i].name ++ " - " ++ c.name);
        }
    }
};

pub const VertexBuffer = struct {
    pub const zero_offsets = [_]vk.DeviceSize{0} ** component_fields.len;

    buffers: [component_fields.len]Buffer,
    bind_buffer: [component_fields.len]vk.Buffer,

    pub fn get(self: VertexBuffer, component: Component) Buffer {
        return self.buffers[@enumToInt(component)];
    }

    pub fn init(gc: GraphicsContext, num_vertex: vk.DeviceSize, label: ?[*:0]const u8) !VertexBuffer {
        var self: VertexBuffer = undefined;

        inline for (component_fields) |_, i| {
            self.buffers[i] = try Buffer.init(gc, Buffer.CreateInfo{
                .size = @intToEnum(Component, i).stride() * num_vertex,
                .buffer_usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
                .memory_usage = .gpu_only,
                .memory_flags = .{},
            }, label);
            self.bind_buffer[i] = self.buffers[i].buffer;
        }

        return self;
    }

    pub fn bind(self: VertexBuffer, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, offsets: [component_fields.len]vk.DeviceSize) void {
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, component_fields.len, &self.bind_buffer, &offset);
    }

    pub fn deinit(self: VertexBuffer, gc: GraphicsContext) void {
        for (self.buffers) |buffer| {
            buffer.deinit(gc);
        }
    }
};
