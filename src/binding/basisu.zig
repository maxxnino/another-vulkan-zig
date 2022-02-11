/// the total_blocks in ImageInfo only for the first mip level
/// this function return total block include all mip levels
pub fn getTotalBlock(image_info: ImageInfo) u32 {
    var current_block = image_info.total_blocks;
    var current_value = current_block;
    var i: u32 = 1;
    while (i < image_info.total_levels) : (i += 1) {
        current_value = current_value >> 2;
        current_block += current_value;
    }
    return current_block;
}

const ImageInfo = extern struct {
    image_index: u32,
    total_levels: u32,

    orig_width: u32,
    orig_height: u32,

    width: u32,
    height: u32,

    num_blocks_x: u32,
    num_blocks_y: u32,
    total_blocks: u32,

    first_slice_index: u32,

    // true if the image has alpha data
    is_alpha: bool,
    // true if the image is an I-Frame
    is_iframe: bool,
};

const ImageLevelInfo = extern struct {
    image_index: u32,
    level_index: u32,

    orig_width: u32,
    orig_height: u32,

    width: u32,
    height: u32,

    num_blocks_x: u32,
    num_blocks_y: u32,
    total_blocks: u32,

    first_slice_index: u32,

    rgb_file_ofs: u32,
    rgb_file_len: u32,
    alpha_file_ofs: u32,
    alpha_file_len: u32,

    // true if the image has alpha data
    alpha_flag: bool,
    // true if the image is an I-Frame
    iframe_flag: bool,
};
pub const CompressedFormat = enum(i32) {

    /// ETC1-2
    /// Opaque only, returns RGB or alpha data if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified
    etc1_rgb = 0,
    /// Opaque+alpha, ETC2_EAC_A8 block followed by a ETC1 block, alpha channel will be opaque for opaque .basis files
    etc2_rgba = 1,

    /// BC1-5, BC7 (desktop, some mobile devices)
    /// Opaque only, no punchthrough alpha support yet, transcodes alpha slice if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified
    bc1_rgb = 2,
    /// Opaque+alpha, BC4 followed by a BC1 block, alpha channel will be opaque for opaque .basis files
    bc3_rgba = 3,
    /// Red only, alpha slice is transcoded to output if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified
    bc4_r = 4,
    /// XY: Two BC4 blocks, X=R and Y=Alpha, .basis file should have alpha data (if not Y will be all 255's)
    bc5_rg = 5,
    /// RGB or RGBA, mode 5 for ETC1S, modes (1,2,3,5,6,7) for UASTC
    bc7_rgba = 6,

    /// PVRTC1 4bpp (mobile, PowerVR devices)
    /// Opaque only, RGB or alpha if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified, nearly lowest quality of any texture format.
    pvrtc1_4_rgb = 8,
    /// Opaque+alpha, most useful for simple opacity maps. If .basis file doesn't have alpha cTFPVRTC1_4_RGB will be used instead. Lowest quality of any supported texture format.
    pvrtc1_4_rgba = 9,

    /// ASTC (mobile, Intel devices, hopefully all desktop GPU's one day)
    /// Opaque+alpha, ASTC 4x4, alpha channel will be opaque for opaque .basis files. Transcoder uses RGB/RGBA/L/LA modes, void extent, and up to two ([0,47] and [0,255]) endpoint precisions.
    astc_4x4_rgba = 10,

    /// ATC (mobile, Adreno devices, this is a niche format)
    /// Opaque, RGB or alpha if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified. ATI ATC (GL_ATC_RGB_AMD)
    atc_rgb = 11,
    /// Opaque+alpha, alpha channel will be opaque for opaque .basis files. ATI ATC (GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD)
    atc_rgba = 12,

    /// FXT1 (desktop, Intel devices, this is a super obscure format)
    /// Opaque only, uses exclusively CC_MIXED blocks. Notable for having a 8x4 block size. GL_3DFX_texture_compression_FXT1 is supported on Intel integrated GPU's (such as HD 630).
    fxt1_rgb = 17,
    /// Punch-through alpha is relatively easy to support, but full alpha is harder. This format is only here for completeness so opaque-only is fine for now.
    /// See the BASISU_USE_ORIGINAL_3DFX_FXT1_ENCODING macro in basisu_transcoder_internal.h.
    /// Opaque-only, almost BC1 quality, much faster to transcode and supports arbitrary texture dimensions (unlike PVRTC1 RGB).
    pvrtc2_4_rgb = 18,
    /// Opaque+alpha, slower to encode than cTFPVRTC2_4_RGB. Premultiplied alpha is highly recommended, otherwise the color channel can leak into the alpha channel on transparent blocks.
    pvrtc2_4_rgba = 19,

    /// R only (ETC2 EAC R11 unsigned)
    etc2_eac_r11 = 20,
    /// RG only (ETC2 EAC RG11 unsigned), R=opaque.r, G=alpha - for tangent space normal maps
    etc2_eac_rg11 = 21,

    /// Uncompressed (raw pixel) formats
    /// 32bpp RGBA image stored in raster (not block) order in memory, R is first byte, A is last byte.
    rgba32 = 13,
    /// 16bpp RGB image stored in raster (not block) order in memory, R at bit position 11
    rgb565 = 14,
    /// 16bpp RGB image stored in raster (not block) order in memory, R at bit position 0
    bgr565 = 15,
    /// 16bpp RGBA image stored in raster (not block) order in memory, R at bit position 12, A at bit position 0
    rgba4444 = 16,

    /// Previously, the caller had some control over which BC7 mode the transcoder output. We've simplified this due to UASTC, which supports numerous modes.
    /// Opaque only, RGB or alpha if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified. Highest quality of all the non-ETC1 formats.
    pub const bc7_m6_rgb = CompressedFormat.bc7_rgba;
    /// Opaque+alpha, alpha channel will be opaque for opaque .basis files
    pub const bc7_m5_rgbA = CompressedFormat.bc7_rgba;
    pub const bc7_m6_opaque_only = CompressedFormat.bc7_rgba;
    pub const bc7_m5 = CompressedFormat.bc7_rgba;
    pub const bc7_alT = 7;

    pub const astc_4x4 = CompressedFormat.astc_4x4_RGBA;

    pub const atc_rgba_interpolated_alpha = CompressedFormat.atc_rgba;
};

/// basisu_transcoder_init() MUST be called before a .basis file can be transcoded.
pub extern fn init() void;

// start_transcoding() must be called before calling transcode_slice() or transcode_image_level().
// For ETC1S files, this call decompresses the selector/endpoint codebooks,
// so ideally you would only call this once per .basis file (not each image/mipmap level).
pub extern fn start(data: [*]const u8, size: u32) void;

/// Returns the total number of images in the basis file (always 1 or more).
/// Note that the number of mipmap levels for each image may differ, and that images may have different resolutions.
pub extern fn totalImages(data: [*]const u8, size: u32) u32;

/// Returns information about the specified image.
pub extern fn imageInfo(data: [*]const u8, size: u32, image_index: u32) ImageInfo;

/// Returns information about the specified image's mipmap level.
pub extern fn imageLevelInfo(data: [*]const u8, size: u32, image_index: u32, level_index: u32) ImageLevelInfo;

/// get bytes per blocks base on texture format
pub extern fn bytesPerBlock(fmt: CompressedFormat) u32;

/// transcode_image_level() decodes a single mipmap level from the .basis file 
/// to any of the supported output texture formats.
/// It'll first find the slice(s) to transcode, then call transcode_slice() one or two times 
/// to decode both the color and alpha texture data (or RG texture data from two slices for BC5).
/// If the .basis file doesn't have alpha slices, the output alpha blocks will be set to fully opaque (all 255's).
/// Currently, to decode to PVRTC1 the basis texture's dimensions in pixels must be a power of 2,
/// due to PVRTC1 format requirements.
/// output_blocks_buf_size_in_blocks_or_pixels should be at least the image level's total_blocks
/// (num_blocks_x * num_blocks_y), or the total number of output pixels if fmt==cTFRGBA32.
/// output_row_pitch_in_blocks_or_pixels: Number of blocks or pixels per row. 
/// If 0, the transcoder uses the slice's num_blocks_x or orig_width (NOT num_blocks_x * 4). Ignored for PVRTC1 (due to texture swizzling).
/// output_rows_in_pixels: Ignored unless fmt is uncompressed (cRGBA32, etc.).
/// The total number of output rows in the output buffer. If 0, the transcoder assumes the slice's orig_height (NOT num_blocks_y * 4).
/// Notes:
/// - basisu_transcoder_init() must have been called first to initialize 
/// the transcoder lookup tables before calling this function.
/// - This method assumes the output texture buffer is readable. In some cases to handle alpha, 
/// the transcoder will write temporary data to the output texture in a first pass, which will be read in a second pass.
pub extern fn transcodeImageLevel(
    data: [*]const u8,
    size: u32,
    image_level_info: *const ImageLevelInfo,
    output_blocks: *anyopaque,
    fmt: CompressedFormat,
) bool;
