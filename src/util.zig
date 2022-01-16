const std = @import("std");

pub fn srcToString(comptime src: std.builtin.SourceLocation) [*:0]const u8{
    // comptime var buffer: [10]u8 = undefined;
    // const pos = comptime std.fmt.formatIntBuf(&buffer, src.line, 10, .lower, .{});
    const line_info = comptime std.fmt.comptimePrint("{}:{}", .{src.line, src.column});
    return src.file ++ " -> " ++ src.fn_name ++ ":" ++ line_info;
}
