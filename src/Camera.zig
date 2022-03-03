const std = @import("std");
const z = @import("zalgebra");
const Window = @import("Window.zig");
const Mat4 = z.Mat4;
const Vec3 = z.Vec3;
const Camera = @This();

pitch: f32,
yaw: f32,
pos: Vec3,
quat: z.Quat = z.Quat.zero(),
z_near: f32 = 0.01,
fov_in_degrees: f32 = 60,
const rotate_speed: f32 = 85;
const move_speed: f32 = 2;

pub fn getViewMatrix(self: Camera) Mat4 {
    const target = self.quat.rotateVec(Vec3.forward());
    return z.lookAt(self.pos, self.pos.add(target), Vec3.up());
}

pub fn getProjMatrix(self: Camera, width: u32, height: u32) Mat4 {
    const f = 1.0 / std.math.tan(z.toRadians(self.fov_in_degrees * 0.5));
    const aspect_ratio = @intToFloat(f32, width) / @intToFloat(f32, height);
    return  Mat4.fromSlice(&.{
        f / aspect_ratio, 0,  0,           0,
        0,                -f, 0,           0,
        0,                0,  0,           1,
        0,                0,  self.z_near, 0,
    });
}

pub fn moveCamera(self: *Camera, window: Window, dt: f32) void {
    var x_dir: f32 = 0;
    var y_dir: f32 = 0;

    if (window.isKey(.j, .press)) y_dir -= dt;
    if (window.isKey(.k, .press)) y_dir += dt;
    if (window.isKey(.h, .press)) x_dir -= dt;
    if (window.isKey(.l, .press)) x_dir += dt;

    // limit pitch values between about +/- 85ish degrees
    self.yaw += x_dir * rotate_speed;
    self.pitch += y_dir * rotate_speed;
    self.pitch = std.math.clamp(self.pitch, -85, 85);
    self.yaw = std.math.mod(f32, self.yaw, 360) catch unreachable;

    var move_dir = Vec3.zero();
    if (window.isKey(.w, .press)) move_dir.z -= dt;
    if (window.isKey(.s, .press)) move_dir.z += dt;
    if (window.isKey(.a, .press)) move_dir.x += dt;
    if (window.isKey(.d, .press)) move_dir.x -= dt;
    if (window.isKey(.space, .press)) move_dir.y += dt;
    if (window.isKey(.left_control, .press)) move_dir.y -= dt;

    self.quat = z.Quat.fromEulerAngle(Vec3.new(self.pitch, self.yaw, 0));
    const translation = self.quat.rotateVec(move_dir.scale(move_speed));
    self.pos = self.pos.add(translation);
}
