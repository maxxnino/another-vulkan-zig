const std = @import("std");
const glfw = @import("libs/mach-glfw/build.zig");
const vkgen = @import("libs/vulkan-zig/generator/index.zig");
const Step = std.build.Step;
const Builder = std.build.Builder;

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
const zalgebra_pkg = std.build.Pkg{
    .name = "zalgebra",
    .path = .{ .path = "libs/zalgebra/src/main.zig" },
};

pub const ResourceGenStep = struct {
    step: Step,
    shader_step: *vkgen.ShaderCompileStep,
    builder: *Builder,
    package: std.build.Pkg,
    output_file: std.build.GeneratedFile,
    resources: std.ArrayList(u8),

    pub fn init(builder: *Builder, out: []const u8) *ResourceGenStep {
        const self = builder.allocator.create(ResourceGenStep) catch unreachable;
        const full_out_path = std.fs.path.join(builder.allocator, &[_][]const u8{
            builder.build_root,
            builder.cache_root,
            out,
        }) catch unreachable;

        self.* = .{
            .step = Step.init(.custom, "resources", builder.allocator, make),
            .shader_step = vkgen.ShaderCompileStep.init(builder, &[_][]const u8{ "glslc", "--target-env=vulkan1.1" }, "shaders"),
            .builder = builder,
            .package = .{
                .name = "resources",
                .path = .{ .generated = &self.output_file },
                .dependencies = null,
            },
            .output_file = .{
                .step = &self.step,
                .path = full_out_path,
            },
            .resources = std.ArrayList(u8).init(builder.allocator),
        };

        self.step.dependOn(&self.shader_step.step);
        return self;
    }

    fn renderPath(path: []const u8, writer: anytype) void {
        const separators = &[_]u8{ std.fs.path.sep_windows, std.fs.path.sep_posix };
        var i: usize = 0;
        while (std.mem.indexOfAnyPos(u8, path, i, separators)) |j| {
            writer.writeAll(path[i..j]) catch unreachable;
            switch (std.fs.path.sep) {
                std.fs.path.sep_windows => writer.writeAll("\\\\") catch unreachable,
                std.fs.path.sep_posix => writer.writeByte(std.fs.path.sep_posix) catch unreachable,
                else => unreachable,
            }

            i = j + 1;
        }
        writer.writeAll(path[i..]) catch unreachable;
    }

    pub fn addShader(self: *ResourceGenStep, name: []const u8, source: []const u8) void {
        const shader_out_path = self.shader_step.add(source);
        var writer = self.resources.writer();

        writer.print("pub const {s} = @embedFile(\"", .{name}) catch unreachable;
        renderPath(shader_out_path, writer);
        writer.writeAll("\");\n") catch unreachable;
    }

    fn make(step: *Step) !void {
        const self = @fieldParentPtr(ResourceGenStep, "step", step);
        const cwd = std.fs.cwd();

        const dir = std.fs.path.dirname(self.output_file.path.?).?;
        try cwd.makePath(dir);
        try cwd.writeFile(self.output_file.path.?, self.resources.items);
    }
};

pub fn build(b: *Builder) void {
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
            const res = ResourceGenStep.init(b, "resources.zig");
            res.addShader("triangle_vert", "shaders/triangle.vert");
            res.addShader("triangle_frag", "shaders/triangle.frag");
            res.addShader("skybox_vert", "shaders/texturecubemap/skybox.vert");
            res.addShader("skybox_frag", "shaders/texturecubemap/skybox.frag");
            break :blk res.package;
        } else {
            break :blk resources_pkg;
        }
    });

    // mach-glfw
    glfw.link(b, exe, .{});
    exe.addPackage(glfw_pkg);
    exe.addPackage(zalgebra_pkg);

    // single header file
    exe.linkLibCpp();
    exe.addIncludeDir("libs");
    exe.addCSourceFile("libs/vk_mem_alloc.cpp", &.{"-std=c++14"});
    exe.addCSourceFile("libs/cgltf.c", &.{"-std=c99"});
    exe.addCSourceFile("libs/stb_image.c", &.{"-std=c99"});

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
