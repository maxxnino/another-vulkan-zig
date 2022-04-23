pub const FileType = enum(i32) {
    invalid = 0,
    gltf = 1,
    glb = 2,
    _,
};
pub const Result = enum(i32) {
    success = 0,
    data_too_short = 1,
    unknown_format = 2,
    invalid_json = 3,
    invalid_gltf = 4,
    invalid_options = 5,
    file_not_found = 6,
    io_error = 7,
    out_of_memory = 8,
    legacy_gltf = 9,
    _,
};
pub const MemoryOption = extern struct {
    alloc: ?fn (*anyopaque, usize) callconv(.C) ?*anyopaque = null,
    free: ?fn (*anyopaque, *anyopaque) callconv(.C) void = null,
    user_data: ?*anyopaque = null,
};
pub const FileOption = extern struct {
    read: ?fn (*const MemoryOption, *const FileOption, [*:0]const u8, [*c]usize, [*c]?*anyopaque) callconv(.C) Result = null,
    release: ?fn (*const MemoryOption, *const FileOption, ?*anyopaque) callconv(.C) void = null,
    user_data: ?*anyopaque = null,
};
pub const Option = extern struct {
    type: FileType = .invalid,
    json_token_count: usize = 0,
    memory: MemoryOption = .{},
    file: FileOption = .{},
};

pub const BufferViewType = enum(i32) {
    invalid = 0,
    indices = 1,
    vertices = 2,
    _,
};
pub const AttributeType = enum(i32) {
    invalid = 0,
    position = 1,
    normal = 2,
    tangent = 3,
    texcoord = 4,
    color = 5,
    joints = 6,
    weights = 7,
    _,
};
pub const ComponentType = enum(i32) {
    invalid = 0,
    r_8 = 1,
    r_8u = 2,
    r_16 = 3,
    r_16u = 4,
    r_32u = 5,
    r_32f = 6,
    _,
};
pub const Type = enum(i32) {
    invalid = 0,
    scalar = 1,
    vec2 = 2,
    vec3 = 3,
    vec4 = 4,
    mat2 = 5,
    mat3 = 6,
    mat4 = 7,
    _,
};
pub const PrimitiveType = enum(i32) {
    points = 0,
    lines = 1,
    line_loop = 2,
    line_strip = 3,
    triangles = 4,
    triangle_strip = 5,
    triangle_fan = 6,
    _,
};
pub const AlphaMode = enum(i32) {
    @"opaque" = 0,
    mask = 1,
    blend = 2,
    _,
};
pub const AnimationPathType = enum(i32) {
    invalid = 0,
    translation = 1,
    rotation = 2,
    scale = 3,
    weights = 4,
    _,
};
pub const InterpolationType = enum(i32) {
    linear = 0,
    step = 1,
    cubic_spline = 2,
    _,
};
pub const CameraType = enum(i32) {
    invalid = 0,
    perspective = 1,
    orthographic = 2,
    _,
};
pub const LightType = enum(i32) {
    invalid = 0,
    directional = 1,
    point = 2,
    spot = 3,
    _,
};
pub const DataFreeMethod = enum(i32) {
    none = 0,
    file_release = 1,
    memory_free = 2,
    _,
};
pub const Extra = extern struct {
    start_offset: usize,
    end_offset: usize,
};
pub const Extensions = extern struct {
    name: [*:0]u8,
    data: [*c]u8,
};
pub const Buffer = extern struct {
    name: ?[*:0]u8,
    size: usize,
    uri: ?[*:0]u8,
    data: ?*anyopaque,
    data_free_method: DataFreeMethod,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const CompressionMode = enum(i32) {
    invalid = 0,
    attributes = 1,
    triangles = 2,
    indices = 3,
};
pub const CompressionFilter = enum(i32) {
    none = 0,
    octahedral = 1,
    quaternion = 2,
    exponential = 3,
};
pub const MeshoptCompression = extern struct {
    buffer: *Buffer,
    offset: usize,
    size: usize,
    stride: usize,
    count: usize,
    mode: CompressionMode,
    filter: CompressionFilter,
};
pub const BufferView = extern struct {
    name: ?[*:0]u8,
    buffer: *Buffer,
    offset: usize,
    size: usize,
    stride: usize,
    type: BufferViewType,
    data: ?*anyopaque,
    has_meshopt_compression: bool,
    meshopt_compression: MeshoptCompression,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const AccessorSparse = extern struct {
    count: usize,
    indices_buffer_view: *BufferView,
    indices_byte_offset: usize,
    indices_component_type: ComponentType,
    values_buffer_view: *BufferView,
    values_byte_offset: usize,
    extras: Extra,
    indices_extras: Extra,
    values_extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
    indices_extensions_count: usize,
    indices_extensions: ?[*]Extensions,
    values_extensions_count: usize,
    values_extensions: ?[*]Extensions,
};
pub const Accessor = extern struct {
    name: ?[*:0]u8,
    component_type: ComponentType,
    normalized: bool,
    type: Type,
    offset: usize,
    count: usize,
    stride: usize,
    buffer_view: ?*BufferView,
    has_min: bool,
    min: [16]f32,
    has_max: bool,
    max: [16]f32,
    is_sparse: bool,
    sparse: AccessorSparse,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Attribute = extern struct {
    name: ?[*:0]u8,
    type: AttributeType,
    index: i32,
    data: *Accessor,
};
pub const Image = extern struct {
    name: ?[*:0]u8,
    uri: ?[*:0]u8,
    buffer_view: ?*BufferView,
    mime_type: ?[*:0]u8,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Sampler = extern struct {
    name: ?[*:0]u8,
    mag_filter: i32,
    min_filter: i32,
    wrap_s: i32,
    wrap_t: i32,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Texture = extern struct {
    name: ?[*:0]u8,
    image: ?*Image,
    sampler: ?*Sampler,
    has_basisu: bool,
    basisu_image: ?*Image,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const TextureTransform = extern struct {
    offset: [2]f32,
    rotation: f32,
    scale: [2]f32,
    has_texcoord: bool,
    texcoord: i32,
};
pub const TextureView = extern struct {
    texture: *Texture,
    texcoord: i32,
    scale: f32,
    has_transform: bool,
    transform: TextureTransform,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const PbrMetallicRoughness = extern struct {
    base_color_texture: TextureView,
    metallic_roughness_texture: TextureView,
    base_color_factor: [4]f32,
    metallic_factor: f32,
    roughness_factor: f32,
    extras: Extra,
};
pub const PbrSpecularGlossiness = extern struct {
    diffuse_texture: TextureView,
    specular_glossiness_texture: TextureView,
    diffuse_factor: [4]f32,
    specular_factor: [3]f32,
    glossiness_factor: f32,
};
pub const ClearCoat = extern struct {
    clearcoat_texture: TextureView,
    clearcoat_roughness_texture: TextureView,
    clearcoat_normal_texture: TextureView,
    clearcoat_factor: f32,
    clearcoat_roughness_factor: f32,
};
pub const Transmission = extern struct {
    transmission_texture: TextureView,
    transmission_factor: f32,
};
pub const Ior = extern struct {
    ior: f32,
};
pub const Specular = extern struct {
    specular_texture: TextureView,
    specular_color_texture: TextureView,
    specular_color_factor: [3]f32,
    specular_factor: f32,
};
pub const Volume = extern struct {
    thickness_texture: TextureView,
    thickness_factor: f32,
    attenuation_color: [3]f32,
    attenuation_distance: f32,
};
pub const Sheen = extern struct {
    sheen_color_texture: TextureView,
    sheen_color_factor: [3]f32,
    sheen_roughness_texture: TextureView,
    sheen_roughness_factor: f32,
};
pub const EmissiveStrength = extern struct {
    emissive_strength: f32,
};
pub const Material = extern struct {
    name: ?[*:0]u8,
    has_pbr_metallic_roughness: bool,
    has_pbr_specular_glossiness: bool,
    has_clearcoat: bool,
    has_transmission: bool,
    has_volume: bool,
    has_ior: bool,
    has_specular: bool,
    has_sheen: bool,
    has_emissive_strength: bool,
    pbr_metallic_roughness: PbrMetallicRoughness,
    pbr_specular_glossiness: PbrSpecularGlossiness,
    clearcoat: ClearCoat,
    ior: Ior,
    specular: Specular,
    sheen: Sheen,
    transmission: Transmission,
    volume: Volume,
    emissive_strength: EmissiveStrength,
    normal_texture: TextureView,
    occlusion_texture: TextureView,
    emissive_texture: TextureView,
    emissive_factor: [3]f32,
    alpha_mode: AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    unlit: bool,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const MaterialMapping = extern struct {
    variant: usize,
    material: *Material,
    extras: Extra,
};
pub const MorphTarget = extern struct {
    attributes: [*]Attribute,
    attributes_count: usize,
};
pub const DracoMeshCompression = extern struct {
    buffer_view: *BufferView,
    attributes: [*]Attribute,
    attributes_count: usize,
};
pub const Primitive = extern struct {
    type: PrimitiveType,
    indices: ?*Accessor,
    material: ?*Material,
    attributes: [*]Attribute,
    attributes_count: usize,
    targets: ?[*]MorphTarget,
    targets_count: usize,
    extras: Extra,
    has_draco_mesh_compression: bool,
    draco_mesh_compression: DracoMeshCompression,
    mappings: [*]MaterialMapping,
    mappings_count: usize,
    extensions_count: usize,
    extensions: ?[*]Extensions,

    pub fn getAttribute(self: Primitive) []const Attribute {
        return self.attributes[0..self.attributes_count];
    }
};
pub const Mesh = extern struct {
    name: ?[*:0]u8,
    primitives: [*]Primitive,
    primitives_count: usize,
    weights: [*]f32,
    weights_count: usize,
    target_names: ?[*][*:0]u8,
    target_names_count: usize,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,

    pub fn getPrimitives(self: Mesh) []const Primitive {
        return self.primitives[0..self.primitives_count];
    }
};
pub const Skin = extern struct {
    name: ?[*:0]u8,
    joints: [*]*Node,
    joints_count: usize,
    skeleton: ?*Node,
    inverse_bind_matrices: ?*Accessor,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const CameraPerspective = extern struct {
    has_aspect_ratio: bool,
    aspect_ratio: f32,
    yfov: f32,
    has_zfar: bool,
    zfar: f32,
    znear: f32,
    extras: Extra,
};
pub const CameraOrthographic = extern struct {
    xmag: f32,
    ymag: f32,
    zfar: f32,
    znear: f32,
    extras: Extra,
};
const CameraMode = extern union {
    perspective: CameraPerspective,
    orthographic: CameraOrthographic,
};
pub const Camera = extern struct {
    name: ?[*:0]u8,
    type: CameraType,
    data: CameraMode,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Light = extern struct {
    name: ?[*:0]u8,
    color: [3]f32,
    intensity: f32,
    type: LightType,
    range: f32,
    spot_inner_cone_angle: f32,
    spot_outer_cone_angle: f32,
    extras: Extra,
};
pub const Node = extern struct {
    name: ?[*:0]u8,
    parent: *Node,
    children: ?[*]*Node,
    children_count: usize,
    skin: ?*Skin,
    mesh: ?*Mesh,
    camera: ?*Camera,
    light: ?*Light,
    weights: [*]f32,
    weights_count: usize,
    has_translation: bool,
    has_rotation: bool,
    has_scale: bool,
    has_matrix: bool,
    translation: [3]f32,
    rotation: [4]f32,
    scale: [3]f32,
    matrix: [16]f32,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Scene = extern struct {
    name: ?[*:0]u8,
    nodes: ?[*]*Node,
    nodes_count: usize,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const AnimationSampler = extern struct {
    input: *Accessor,
    output: *Accessor,
    interpolation: InterpolationType,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const AnimationChannel = extern struct {
    sampler: *AnimationSampler,
    target_node: *Node,
    target_path: AnimationPathType,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Animation = extern struct {
    name: ?[*:0]u8,
    samplers: [*]AnimationSampler,
    samplers_count: usize,
    channels: [*]AnimationChannel,
    channels_count: usize,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const MaterialVariant = extern struct {
    name: ?[*:0]u8,
    extras: Extra,
};
pub const Asset = extern struct {
    copyright: ?[*:0]u8,
    generator: ?[*:0]u8,
    version: [*:0]u8,
    min_version: ?[*:0]u8,
    extras: Extra,
    extensions_count: usize,
    extensions: ?[*]Extensions,
};
pub const Data = extern struct {
    file_type: FileType,
    file_data: *anyopaque,
    asset: Asset,
    meshes: [*]Mesh,
    meshes_count: usize,
    materials: [*]Material,
    materials_count: usize,
    accessors: [*]Accessor,
    accessors_count: usize,
    buffer_views: [*]BufferView,
    buffer_views_count: usize,
    buffers: [*]Buffer,
    buffers_count: usize,
    images: [*]Image,
    images_count: usize,
    textures: [*]Texture,
    textures_count: usize,
    samplers: [*]Sampler,
    samplers_count: usize,
    skins: [*]Skin,
    skins_count: usize,
    cameras: [*]Camera,
    cameras_count: usize,
    lights: [*]Light,
    lights_count: usize,
    nodes: [*]Node,
    nodes_count: usize,
    scenes: [*]Scene,
    scenes_count: usize,
    scene: *Scene,
    animations: [*]Animation,
    animations_count: usize,
    variants: [*]MaterialVariant,
    variants_count: usize,
    extras: Extra,
    data_extensions_count: usize,
    data_extensions: ?[*]Extensions,
    extensions_used: [*][*:0]u8,
    extensions_used_count: usize,
    extensions_required: [*][*:0]u8,
    extensions_required_count: usize,
    json: [*]const u8,
    json_size: usize,
    bin: ?*const anyopaque,
    bin_size: usize,
    memory: MemoryOption,
    file: FileOption,

    pub fn getMeshes(self: Data) []const Mesh {
        return self.meshes[0..self.meshes_count];
    }

};
pub extern fn cgltf_parse(options: *const Option, data: *const anyopaque, size: usize, out_data: **Data) Result;
pub extern fn cgltf_parse_file(options: *const Option, path: [*:0]const u8, out_data: **Data) Result;
pub extern fn cgltf_load_buffers(options: *const Option, data: *Data, gltf_path: [*:0]const u8) Result;
pub extern fn cgltf_load_buffer_base64(options: *const Option, size: usize, base64: [*c]const u8, out_data: [*c]?*anyopaque) Result;
pub extern fn cgltf_decode_string(string: [*:0]u8) usize;
pub extern fn cgltf_decode_uri(uri: [*:0]u8) usize;
pub extern fn cgltf_validate(data: *Data) Result;
pub extern fn cgltf_free(data: *Data) void;
pub extern fn cgltf_node_transform_local(node: *const Node, out_matrix: [*]f32) void;
pub extern fn cgltf_node_transform_world(node: *const Node, out_matrix: [*]f32) void;
pub extern fn cgltf_accessor_read_float(accessor: *const Accessor, index: usize, out: *f32, element_size: usize) bool;
pub extern fn cgltf_accessor_read_uint(accessor: *const Accessor, index: usize, out: *u32, element_size: usize) bool;
pub extern fn cgltf_accessor_read_index(accessor: *const Accessor, index: usize) usize;
pub extern fn cgltf_num_components(@"type": Type) usize;
pub extern fn cgltf_accessor_unpack_floats(accessor: *const Accessor, out: [*]f32, float_count: usize) usize;
pub extern fn cgltf_copy_extras_json(data: [*c]const Data, extras: [*c]const Extra, dest: [*c]u8, dest_size: [*c]usize) Result;
