const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const vma = @import("binding/vma.zig");
const z = @import("zalgebra");
const assert = std.debug.assert;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Buffer = @import("Buffer.zig");
const Allocator = std.mem.Allocator;
const Mat4 = z.Mat4;
const Vec3 = z.Vec3;
const Vec2 = z.Vec2;
const Camera = @This();

pitch: f32,
yaw: f32,
pos: Vec3,
quat: z.Quat = z.Quat.zero(),
z_near: f32 = 0.1,
z_far: f32 = 100,
fov_in_degrees: f32 = 60,
const rotate_speed: f32 = 85;
const move_speed: f32 = 2;

pub fn getViewMatrix(self: Camera) Mat4 {
    const target = self.quat.rotateVec(Vec3.forward());
    return z.lookAt(self.pos, self.pos.add(target), Vec3.up());
}

pub fn getProjMatrix(self: Camera, width: u32, height: u32) Mat4 {
    var proj = z.perspective(
        self.fov_in_degrees,
        @intToFloat(f32, width) / @intToFloat(f32, height),
        self.z_near,
        self.z_far,
    );
    proj.data[1][1] *= -1;
    return proj;
}

pub fn moveCamera(self: *Camera, window: glfw.Window, dt: f32) void {
    var x_dir: f32 = 0;
    var y_dir: f32 = 0;

    if (window.getKey(.j) == .press) y_dir += dt;
    if (window.getKey(.k) == .press) y_dir -= dt;
    if (window.getKey(.h) == .press) x_dir += dt;
    if (window.getKey(.l) == .press) x_dir -= dt;

    // limit pitch values between about +/- 85ish degrees
    self.yaw += x_dir * rotate_speed;
    self.pitch += y_dir * rotate_speed;
    self.pitch = std.math.clamp(self.pitch, -85, 85);
    self.yaw = std.math.mod(f32, self.yaw, 360) catch unreachable;

    var move_dir = Vec3.zero();
    if (window.getKey(.w) == .press) move_dir.z += dt;
    if (window.getKey(.s) == .press) move_dir.z -= dt;
    if (window.getKey(.a) == .press) move_dir.x += dt;
    if (window.getKey(.d) == .press) move_dir.x -= dt;
    if (window.getKey(.space) == .press) move_dir.y += dt;
    if (window.getKey(.left_control) == .press) move_dir.y -= dt;

    self.quat = z.Quat.fromEulerAngle(Vec3.new(self.pitch, self.yaw, 0));
    const translation = self.quat.rotateVec(move_dir.scale(move_speed));
    self.pos = self.pos.add(translation);
}
