const std = @import("std");
const glfw = @import("libs/mach-glfw/build.zig");
const vkgen = @import("libs/vulkan-zig/generator/index.zig");
const zigvulkan = @import("libs/vulkan-zig/build.zig");

const glfw_pkg = std.build.Pkg{
    .name = "glfw",
    .path = .{ .path = "libs/mach-glfw/src/main.zig" },
};
const vulkan_pkg = std.build.Pkg{
    .name = "vulkan",
    .path = .{ .path = "zig-cache/vk.zig" },
};
const resources_pkg = std.build.Pkg{
    .name = "resources",
    .path = .{ .path = "zig-cache/resources.zig" },
};

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("mach-glfw-vulkan-example", "src/main.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const compile_shader = b.option(bool, "shader", "Compile shader when build") orelse false;
    const generate_vulkan = b.option(bool, "gen", "Generate vk.zig") orelse false;

    // vulkan-zig: Create a step that generates vk.zig (stored in zig-cache) from the provided vulkan registry.
    exe.addPackage(blk: {
        if (generate_vulkan) {
            const gen = vkgen.VkGenerateStep.init(b, "libs/vulkan-zig/examples/vk.xml", "vk.zig");
            break :blk gen.package;
        } else {
            break :blk vulkan_pkg;
        }
    });

    // shader resources, to be compiled using glslc
    exe.addPackage(blk: {
        if (compile_shader) {
            const res = zigvulkan.ResourceGenStep.init(b, "resources.zig");
            res.addShader("triangle_vert", "shaders/triangle.vert");
            res.addShader("triangle_frag", "shaders/triangle.frag");
            break :blk res.package;
        } else {
            break :blk resources_pkg;
        }
    });

    // mach-glfw
    glfw.link(b, exe, .{});
    exe.addPackage(glfw_pkg);

    // single header file
    exe.linkLibCpp();
    exe.addIncludeDir("libs");
    exe.addCSourceFile("libs/vk_mem_alloc.cpp", &.{"-std=c++14"});

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
