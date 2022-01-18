const std = @import("std");
pub const StbImage = StbImageType(u8);
pub const StbImageU16 = StbImageType(u16);
pub const StbImageF32 = StbImageType(f32);

pub fn StbImageType(comptime T: type) type {
    const load_tuple = blk: {
        if (T == u8) break :blk .{ stbi_load, stbi_load_from_memory };
        if (T == u16) break :blk .{ stbi_load_16, stbi_load_16_from_memory };
        if (T == f32) break :blk .{ stbi_loadf, stbi_load_from_memory };
        @compileError("Only support u8/u16/f32");
    };

    return struct {
        pixels: []T,
        width: u32,
        height: u32,
        channels: u32,
        const Image = @This();

        pub fn loadFromFile(filename: [*:0]const u8, desired_channels: Channel) !Image {
            var x: i32 = 0;
            var y: i32 = 0;
            var c: i32 = 0;
            if (load_tuple[0](filename, &x, &y, &c, desired_channels)) |pixels| {
                std.debug.assert(@enumToInt(desired_channels) == c);
                const width = @intCast(u32, x);
                const height = @intCast(u32, y);
                const channels = @intCast(u32, c);
                return Image{
                    .pixels = pixels[0 .. width * height * channels],
                    .width = width,
                    .height = height,
                    .channels = channels,
                };
            }

            return error.CantLoadImage;
        }

        pub fn loadFromMemory(buffer: []const u8, desired_channels: Channel) !Image {
            var x: i32 = 0;
            var y: i32 = 0;
            var c: i32 = 0;
            if (load_tuple[1](buffer, @intCast(i32, buffer.len), &x, &y, &c, desired_channels)) |pixels| {
                const width = @intCast(u32, x);
                const height = @intCast(u32, y);
                const channels = @intCast(u32, c);
                return Image{
                    .pixels = pixels[0 .. width * height * channels],
                    .width = width,
                    .height = height,
                    .channels = channels,
                };
            }

            return error.CantLoadImage;
        }

        pub fn free(self: Image) void {
            stbi_image_free(@ptrCast(*anyopaque, self.pixels.ptr));
        }

        pub fn bytePerRow(self: Image) u32 {
            return self.width * self.channels;
        }

        pub fn totalByte(self: Image) u32 {
            return self.width * self.height * self.channels;
        }
    };
}
pub const Channel = enum(i32) {
    default = 0,
    grey = 1,
    grey_alpha = 2,
    rgb = 3,
    rgb_alpha = 4,
};

pub extern fn stbi_load(filename: [*]const u8, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: Channel) ?[*]u8;
pub extern fn stbi_load_from_memory(
    buffer: [*]const u8,
    len: i32,
    x: *i32,
    y: *i32,
    channels_in_file: *i32,
    desired_channels: Channel,
) ?[*]u8;
pub extern fn stbi_load_gif_from_memory(
    buffer: [*]const u8,
    len: i32,
    delays: [*][*]i32,
    x: *i32,
    y: *i32,
    z: *i32,
    comp: *i32,
    req_comp: i32,
) ?[*]u8;
pub extern fn stbi_load_16_from_memory(
    buffer: [*]const u8,
    len: i32,
    x: *i32,
    y: *i32,
    channels_in_file: *i32,
    desired_channels: Channel,
) ?[*]u16;
pub extern fn stbi_load_16(filename: [*]const u8, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: Channel) ?[*]u16;
pub extern fn stbi_loadf_from_memory(
    buffer: [*]const u8,
    len: i32,
    x: *i32,
    y: *i32,
    channels_in_file: *i32,
    desired_channels: Channel,
) ?[*]f32;
pub extern fn stbi_loadf(filename: [*]const u8, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: Channel) ?[*]f32;
pub extern fn stbi_hdr_to_ldr_gamma(gamma: f32) void;
pub extern fn stbi_hdr_to_ldr_scale(scale: f32) void;
pub extern fn stbi_ldr_to_hdr_gamma(gamma: f32) void;
pub extern fn stbi_ldr_to_hdr_scale(scale: f32) void;
pub extern fn stbi_is_hdr_from_memory(buffer: [*]const u8, len: i32) i32;
pub extern fn stbi_is_hdr(filename: [*]const u8) i32;
pub extern fn stbi_failure_reason() [*]const u8;
pub extern fn stbi_image_free(retval_from_stbi_load: *anyopaque) void;
pub extern fn stbi_info_from_memory(buffer: [*]const u8, len: i32, x: *i32, y: *i32, comp: *i32) i32;
pub extern fn stbi_is_16_bit_from_memory(buffer: [*]const u8, len: i32) i32;
pub extern fn stbi_info(filename: [*]const u8, x: *i32, y: *i32, comp: *i32) i32;
pub extern fn stbi_is_16_bit(filename: [*]const u8) i32;
pub extern fn stbi_set_unpremultiply_on_load(flag_true_if_should_unpremultiply: i32) void;
pub extern fn stbi_convert_iphone_png_to_rgb(flag_true_if_should_convert: i32) void;
pub extern fn stbi_set_flip_vertically_on_load(flag_true_if_should_flip: i32) void;
pub extern fn stbi_set_unpremultiply_on_load_thread(flag_true_if_should_unpremultiply: i32) void;
pub extern fn stbi_convert_iphone_png_to_rgb_thread(flag_true_if_should_convert: i32) void;
pub extern fn stbi_set_flip_vertically_on_load_thread(flag_true_if_should_flip: i32) void;

// pub extern fn stbi_zlib_decode_malloc_guesssize(buffer: [*]const u8, len: i32, initial_size: i32, outlen: [*c]i32) [*c]u8;
// pub extern fn stbi_zlib_decode_malloc_guesssize_headerflag(buffer: [*]const u8, len: i32, initial_size: i32, outlen: [*c]i32, parse_header: i32) [*c]u8;
// pub extern fn stbi_zlib_decode_malloc(buffer: [*]const u8, len: i32, outlen: [*c]i32) [*c]u8;
// pub extern fn stbi_zlib_decode_buffer(obuffer: [*c]u8, olen: i32, ibuffer: [*c]const u8, ilen: i32) i32;
// pub extern fn stbi_zlib_decode_noheader_malloc(buffer: [*]const u8, len: i32, outlen: [*c]i32) [*c]u8;
// pub extern fn stbi_zlib_decode_noheader_buffer(obuffer: [*c]u8, olen: i32, ibuffer: [*c]const u8, ilen: i32) i32;
// pub extern fn stbi_load_from_file_16(f: [*c]File, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: Channel) [*]c_ushort;
// pub extern fn stbi_load_from_file(f: [*c]File, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: Channel) [*]u8;
// pub extern fn stbi_loadf_from_file(f: [*c]File, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: Channel) [*]f32;
// pub extern fn stbi_is_hdr_from_file(f: [*c]File) i32;
// pub extern fn stbi_info_from_file(f: [*c]File, x: *i32, y: *i32, comp: [*c]i32) i32;
// pub extern fn stbi_is_16_bit_from_file(f: [*c]File) i32;
// pub const File = extern struct {
//     _ptr: [*c]u8,
//     _cnt: c_int,
//     _base: [*c]u8,
//     _flag: c_int,
//     _file: c_int,
//     _charbuf: c_int,
//     _bufsiz: c_int,
//     _tmpfname: [*c]u8,
// };
