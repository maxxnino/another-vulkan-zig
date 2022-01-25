const vk = @import("vulkan");
const z = @import("zalgebra");
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Self = @This();

pub const binding_description = [_]vk.VertexInputBindingDescription{.{
    .binding = 0,
    .stride = @sizeOf(Self),
    .input_rate = .vertex,
}};

pub const attribute_description = [_]vk.VertexInputAttributeDescription{
    .{
        .binding = 0,
        .location = 0,
        .format = .r32g32b32_sfloat,
        .offset = @offsetOf(Self, "pos"),
    },
    .{
        .binding = 0,
        .location = 1,
        .format = .r32g32b32_sfloat,
        .offset = @offsetOf(Self, "normal"),
    },
    .{
        .binding = 0,
        .location = 2,
        .format = .r32g32_sfloat,
        .offset = @offsetOf(Self, "tex_coord"),
    },
};

pos: Vec3,
normal: Vec3,
tex_coord: Vec2,
