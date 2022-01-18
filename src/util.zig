const std = @import("std");

pub fn srcToString(comptime src: std.builtin.SourceLocation) [*:0]const u8{
    const line_info = comptime std.fmt.comptimePrint("{}:{}", .{src.line, src.column});
    return src.file ++ " -> " ++ src.fn_name ++ ":" ++ line_info;
}
