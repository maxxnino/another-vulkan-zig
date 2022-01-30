const std = @import("std");
const glfw = @import("glfw");
const Self = @This();
const KeyboardSet = blk: {
    @setEvalBranchQuota(1500);
    break :blk std.EnumSet(glfw.Key);
};
const MouseSet = std.EnumSet(glfw.mouse_button.MouseButton);

window: glfw.Window,
width: u32,
height: u32,
fullscreen: bool,
keyboard: KeyboardSet,
mouse: MouseSet,
cursor_pos: glfw.Window.CursorPos,
scroll_delta: glfw.Window.CursorPos,
support_raw_mouse_motion: bool,
enable_raw_mouse_motion: bool,

pub const Action = enum {
    press,
    release,
    just_press,
};

pub fn init(app_name: [*:0]const u8, is_fullscreen: bool, width: u32, height: u32) !Self {
    try glfw.init(.{});

    var self: Self = undefined;
    const monitor = glfw.Monitor.getPrimary().?;
    const mode = try monitor.getVideoMode();
    self.fullscreen = is_fullscreen;
    self.width = if (self.fullscreen) mode.getWidth() else width;
    self.height = if (self.fullscreen) mode.getHeight() else height;
    self.window = try glfw.Window.create(self.width, self.height, app_name, if (self.fullscreen) monitor else null, null, .{
        .client_api = .no_api,
    });
    self.cursor_pos = try self.window.getCursorPos();
    self.support_raw_mouse_motion = glfw.rawMouseMotionSupported();
    self.enable_raw_mouse_motion = false;
    self.keyboard = KeyboardSet{};
    self.mouse = MouseSet{};
    return self;
}

pub fn setDataAndCallBack(self: *Self) void {
    self.window.setUserPointer(Self, self);
    self.window.setKeyCallback(Self.keyboardCallBack);
    self.window.setCursorPosCallback(Self.cursorPosCallback);
    self.window.setMouseButtonCallback(Self.mouseButtonCallback);
    self.window.setScrollCallback(Self.scrollCallback);
}

pub fn deinit(self: Self) void {
    self.window.destroy();
    glfw.terminate();
}

pub fn pollEvents(self: *Self) bool {
    self.resetKeyAndMouseState();
    glfw.pollEvents() catch unreachable;
    return !self.window.shouldClose();
}

fn resetKeyAndMouseState(self: *Self) void {
    self.keyboard.bits = comptime @TypeOf(self.keyboard.bits).initEmpty();
    self.mouse.bits = comptime @TypeOf(self.mouse.bits).initEmpty();
}

pub fn isKey(self: Self, key: glfw.Key, action: Action) bool {
    return switch (action) {
        .just_press => self.keyboard.contains(key),
        .press => self.window.getKey(key) == .press,
        .release => self.window.getKey(key) == .release,
    };
}

pub fn isMouse(self: Self, button: glfw.mouse_button.MouseButton, action: Action) bool {
    return switch (action) {
        .just_press => self.mouse.contains(button),
        .press => self.window.getMouseButton(button) == .press,
        .release => self.window.getMouseButton(button) == .release,
    };
}

pub fn togglRawMouseMotion(self: *Self) void {
    if (self.support_raw_mouse_motion) {
        self.enable_raw_mouse_motion = !self.enable_raw_mouse_motion;
        self.window.setInputModeRawMouseMotion(self.enable_raw_mouse_motion) catch unreachable;
        if (self.enable_raw_mouse_motion)
            self.window.setInputModeCursor(.disabled) catch unreachable
        else
            self.window.setInputModeCursor(.normal) catch unreachable;
    }
}

fn keyboardCallBack(window: glfw.Window, key: glfw.Key, scancode: i32, action: glfw.Action, mods: glfw.Mods) void {
    _ = mods;
    _ = scancode;
    if (action == .press) {
        var self = window.getUserPointer(*Self).?;
        self.keyboard.insert(key);
    }
}

fn cursorPosCallback(window: glfw.Window, xpos: f64, ypos: f64) void {
    var self = window.getUserPointer(*Self).?;
    self.cursor_pos.xpos = xpos;
    self.cursor_pos.ypos = ypos;
}

fn mouseButtonCallback(window: glfw.Window, button: glfw.mouse_button.MouseButton, action: glfw.Action, mods: glfw.Mods) void {
    _ = mods;
    if (action == .press) {
        return window.getUserPointer(*Self).?.mouse.insert(button);
    }
}

fn scrollCallback(window: glfw.Window, xoffset: f64, yoffset: f64) void {
    var self = window.getUserPointer(*Self).?;
    self.scroll_delta.xpos += xoffset;
    self.scroll_delta.ypos += yoffset;
}
