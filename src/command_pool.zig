const std = @import("std");
const vk = @import("vulkan");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const DrawPool = CommandPool(2, 6);

const State = enum {
    free,
    using,
    done,
};

const CommandResult = struct {
    cmd: vk.CommandBuffer,
    index: u4,
};

fn SmallPool(comptime size: u4) type {
    return struct {
        const Self = @This();
        pool: vk.CommandPool,
        command_buffers: [size]vk.CommandBuffer,
        state: [size]State,
        total: u4,
        current: u4,

        pub fn init(gc: GraphicsContext, label: ?[*:0]const u8) !Self {
            var self: Self = undefined;
            self.state = .{.free} ** size;
            self.total = 0;
            self.current = 0;

            self.pool = try gc.create(vk.CommandPoolCreateInfo{
                .flags = .{},
                .queue_family_index = gc.graphics_queue.family,
            }, label);

            try gc.vkd.allocateCommandBuffers(gc.dev, &.{
                .command_pool = self.pool,
                .level = .primary,
                .command_buffer_count = size,
            }, &self.command_buffers);
            return self;
        }

        pub fn deinit(self: Self, gc: GraphicsContext) void {
            gc.destroy(self.pool);
        }

        pub fn reset(self: *Self, gc: GraphicsContext) !void {
            for (self.state) |*s| {
                assert(s.* == .done);
                s.* = .free;
            }
            self.total = 0;
            self.current = 0;
            try gc.vkd.resetCommandPool(gc.dev, self.pool, .{});
        }

        pub fn done(self: *Self, gc: GraphicsContext, command_index: u4) !bool {
            self.state[command_index] = .done;
            self.total += 1;
            if (self.total == size) {
                try self.reset(gc);
                return true;
            }
            return false;
        }

        pub fn get(self: *Self) CommandResult {
            defer self.current += 1;
            assert(self.state[self.current] == .free);
            self.state[self.current] = .using;
            return .{
                .cmd = self.command_buffers[self.current],
                .index = self.current,
            };
        }

        pub fn isFull(self: Self) bool {
            return self.current >= size;
        }
    };
}

pub const CommandBuffer = struct {
    cmd: vk.CommandBuffer,
    index: u4,
    pool_index: u4,
};

pub fn CommandPool(comptime total_pool: u4, comptime pool_size: u4) type {
    return struct {
        const Self = @This();
        const Pool = SmallPool(pool_size);
        pools: [total_pool]Pool,
        current: u4,

        pub fn init(gc: GraphicsContext, label: ?[*:0]const u8) !Self {
            var self: Self = undefined;
            self.current = 0;

            for (self.pools) |*p| {
                p.* = try Pool.init(gc, label);
            }

            return self;
        }

        pub fn deinit(self: Self, gc: GraphicsContext) void {
            for (self.pools) |p| {
                p.deinit(gc);
            }
        }

        pub fn done(self: *Self, gc: GraphicsContext, cmd: CommandBuffer) !void {
            // TODO:
            _ = try self.pools[cmd.pool_index].done(gc, cmd.index);
        }

        pub fn createCommandBuffer(self: *Self) CommandBuffer {
            if (self.pools[self.current].isFull()) {
                self.current += 1;
                if (self.current == total_pool) {
                    self.current = 0;
                }
            }

            assert(!self.pools[self.current].isFull());
            const r = self.pools[self.current].get();
            return .{
                .cmd = r.cmd,
                .index = r.index,
                .pool_index = self.current,
            };
        }
    };
}
