const std = @import("std");
const c = @cImport({
    @cInclude("ggml/ggml.h");
});

pub fn main() !void {
    const params = .{
        .mem_size   = 128*1024*1024,
        .mem_buffer = null,
        .no_alloc   = false,
    };

    const ctx0 = c.ggml_init(params);
    defer c.ggml_free(ctx0);

    const t1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 10);
    const t2 = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I16, 10, 20);
    const t3 = c.ggml_new_tensor_3d(ctx0, c.GGML_TYPE_I32, 10, 20, 30);

    try std.testing.expect(t1.*.n_dims == 1);
    try std.testing.expect(t1.*.ne[0]  == 10);
    try std.testing.expect(t1.*.nb[1]  == 10*@sizeOf(f32));

    try std.testing.expect(t2.*.n_dims == 2);
    try std.testing.expect(t2.*.ne[0]  == 10);
    try std.testing.expect(t2.*.ne[1]  == 20);
    try std.testing.expect(t2.*.nb[1]  == 10*@sizeOf(i16));
    try std.testing.expect(t2.*.nb[2]  == 10*20*@sizeOf(i16));

    try std.testing.expect(t3.*.n_dims == 3);
    try std.testing.expect(t3.*.ne[0]  == 10);
    try std.testing.expect(t3.*.ne[1]  == 20);
    try std.testing.expect(t3.*.ne[2]  == 30);
    try std.testing.expect(t3.*.nb[1]  == 10*@sizeOf(i32));
    try std.testing.expect(t3.*.nb[2]  == 10*20*@sizeOf(i32));
    try std.testing.expect(t3.*.nb[3]  == 10*20*30*@sizeOf(i32));

    c.ggml_print_objects(ctx0);

    _ = try std.io.getStdIn().reader().readByte();
}
