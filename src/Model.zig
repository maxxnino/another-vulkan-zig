const vk = @import("vulkan");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Self = @This();

mesh_begin: u32,
mesh_end: u32,

pub const Mesh = struct {
    index_offset: u32,
    vertex_offset: i32,
    num_indices: u32,
    num_vertices: u32,
};

pub fn draw(self: Self, gc: GraphicsContext, cmdbuf: vk.CommandBuffer, meshs: []const Mesh) void {
    var i: u32 = self.mesh_begin;
    while (i < self.mesh_end) : (i += 1) {
        gc.vkd.cmdDrawIndexed(cmdbuf, meshs[i].num_indices, 1, meshs[i].index_offset, meshs[i].vertex_offset, 0);
    }
}
