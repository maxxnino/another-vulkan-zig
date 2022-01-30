const std = @import("std");
const App = @import("App.zig");

const Gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = false });

pub fn main() !void {
    var gpa = Gpa{};
    defer _ = gpa.deinit();
    defer _ = gpa.detectLeaks();

    var app = try App.init(gpa.allocator());
    defer app.deinit();
    app.window.setDataAndCallBack();
    try app.run();
}
