const std = @import("std");
const c = @cImport({
    @cInclude("ggml/ggml.h");
});

pub fn main() !void {
    const n_threads = 2;

    const params = .{
        .mem_size   = 128*1024*1024,
        .mem_buffer = null,
        .no_alloc   = false,
    };

    const ctx0 = c.ggml_init(params);
    defer c.ggml_free(ctx0);

    {
        const x = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);

        c.ggml_set_param(ctx0, x);

        const a = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        const b = c.ggml_mul(ctx0, x, x);
        const f = c.ggml_mul(ctx0, b, a);

        // a*x^2
        // 2*a*x

        c.ggml_print_objects(ctx0);

        const gf = c.ggml_build_forward(f);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x, 2.0);
        _ = c.ggml_set_f32(a, 3.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(f.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("f     = {d:.6}\n", .{c.ggml_get_f32_1d(f, 0)});
        std.debug.print("df/dx = {d:.6}\n", .{c.ggml_get_f32_1d(x.*.grad, 0)});

        try std.testing.expect(c.ggml_get_f32_1d(f, 0)          ==  12.0);
        try std.testing.expect(c.ggml_get_f32_1d(x.*.grad, 0)   ==  12.0);

        _ = c.ggml_set_f32(x, 3.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(f.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("f     = {d:.6}\n", .{c.ggml_get_f32_1d(f, 0)});
        std.debug.print("df/dx = {d:.6}\n", .{c.ggml_get_f32_1d(x.*.grad, 0)});

        try std.testing.expect(c.ggml_get_f32_1d(f, 0)          ==  27.0);
        try std.testing.expect(c.ggml_get_f32_1d(x.*.grad, 0)   ==  18.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-1-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-1-backward.dot");
    }

    /////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        const x3 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);

        _ = c.ggml_set_f32(x1, 3.0);
        _ = c.ggml_set_f32(x2, 1.0);
        _ = c.ggml_set_f32(x3, 0.0);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);

        const y = c.ggml_add(ctx0, c.ggml_mul(ctx0, x1, x1), c.ggml_mul(ctx0, x1, x2));

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6}\n", .{c.ggml_get_f32_1d(x1.*.grad, 0)});
        std.debug.print("df/dx2 = {d:.6}\n", .{c.ggml_get_f32_1d(x2.*.grad, 0)});

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  12.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  7.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  3.0);

        const g1 = x1.*.grad;
        const g2 = x2.*.grad;

        const gbb = c.ggml_build_backward(ctx0, @constCast(&gb), true);

        c.ggml_graph_reset(@constCast(&gb));
        _ = c.ggml_set_f32(g1.*.grad, 1.0);
        _ = c.ggml_set_f32(g2.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gbb), n_threads);

        std.debug.print("H * [1, 1] = [ {d:.6} {d:.6} ]\n", .{c.ggml_get_f32_1d(x1.*.grad, 0), c.ggml_get_f32_1d(x2.*.grad, 0)});

        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  1.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-2-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-2-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);

        const y = c.ggml_mul(ctx0, c.ggml_add(ctx0, c.ggml_mul(ctx0, x1, x1), c.ggml_mul(ctx0, x1, x2)), x1);

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x1, 3.0);
        _ = c.ggml_set_f32(x2, 4.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6}\n", .{c.ggml_get_f32_1d(x1.*.grad, 0)});
        std.debug.print("df/dx2 = {d:.6}\n", .{c.ggml_get_f32_1d(x2.*.grad, 0)});

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  63.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  51.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  9.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-3-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-3-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        const x3 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);
        c.ggml_set_param(ctx0, x3);

        const y = c.ggml_mul(ctx0, c.ggml_mul(ctx0, c.ggml_mul(ctx0, x1, x1), c.ggml_mul(ctx0, x2, x2)), x3);

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x1, 1.0);
        _ = c.ggml_set_f32(x2, 2.0);
        _ = c.ggml_set_f32(x3, 3.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6}\n", .{c.ggml_get_f32_1d(x1.*.grad, 0)});
        std.debug.print("df/dx2 = {d:.6}\n", .{c.ggml_get_f32_1d(x2.*.grad, 0)});
        std.debug.print("df/dx3 = {d:.6}\n", .{c.ggml_get_f32_1d(x3.*.grad, 0)});

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  12.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  24.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  12.0);
        try std.testing.expect(c.ggml_get_f32_1d(x3.*.grad, 0)  ==  4.0);

        const g1 = x1.*.grad;
        const g2 = x2.*.grad;
        const g3 = x3.*.grad;

        const gbb = c.ggml_build_backward(ctx0, @constCast(&gb), true);

        c.ggml_graph_reset(@constCast(&gb));
        _ = c.ggml_set_f32(g1.*.grad, 1.0);
        _ = c.ggml_set_f32(g2.*.grad, 1.0);
        _ = c.ggml_set_f32(g3.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gbb), n_threads);

        std.debug.print("H * [1, 1, 1] = [ {d:.6} {d:.6} {d:.6}]\n",
            .{
                c.ggml_get_f32_1d(x1.*.grad, 0),
                c.ggml_get_f32_1d(x2.*.grad, 0),
                c.ggml_get_f32_1d(x3.*.grad, 0),
            });

        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  56.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  34.0);
        try std.testing.expect(c.ggml_get_f32_1d(x3.*.grad, 0)  ==  12.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-4-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-4-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);

        const y = c.ggml_sum(ctx0, c.ggml_mul(ctx0, x1, x2));

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x1, 3.0);
        _ = c.ggml_set_f32(x2, 5.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x1.*.grad, 0),
                c.ggml_get_f32_1d(x1.*.grad, 1),
                c.ggml_get_f32_1d(x1.*.grad, 2),
            });
        std.debug.print("df/dx2 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x2.*.grad, 0),
                c.ggml_get_f32_1d(x2.*.grad, 1),
                c.ggml_get_f32_1d(x2.*.grad, 2),
            });

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  45.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  5.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 1)  ==  5.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 1)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 2)  ==  5.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 2)  ==  3.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-5-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-5-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);

        const y =
            c.ggml_sum(ctx0,
                    c.ggml_add(ctx0,
                        c.ggml_mul(ctx0, x1, x2),
                        c.ggml_mul(ctx0,
                            c.ggml_repeat(ctx0, c.ggml_new_f32(ctx0, -2.0), x1),
                            c.ggml_mul(ctx0, x1, x1)
                            )
                        )
                    );

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x1, 3.0);
        _ = c.ggml_set_f32(x2, 5.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x1.*.grad, 0),
                c.ggml_get_f32_1d(x1.*.grad, 1),
                c.ggml_get_f32_1d(x1.*.grad, 2),
            });
        std.debug.print("df/dx2 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x2.*.grad, 0),
                c.ggml_get_f32_1d(x2.*.grad, 1),
                c.ggml_get_f32_1d(x2.*.grad, 2),
            });

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  -9.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  -7.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 1)  ==  -7.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 2)  ==  -7.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 1)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 2)  ==  3.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-6-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-6-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);

        const y =
            c.ggml_sum(ctx0,
                    c.ggml_sub(ctx0,
                        c.ggml_mul(ctx0, x1, x2),
                        c.ggml_mul(ctx0,
                            c.ggml_mul(ctx0, x1, x1),
                            c.ggml_repeat(ctx0, c.ggml_new_f32(ctx0, -2.0), x1)
                            )
                        )
                    );

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x1, 3.0);
        _ = c.ggml_set_f32(x2, 5.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x1.*.grad, 0),
                c.ggml_get_f32_1d(x1.*.grad, 1),
                c.ggml_get_f32_1d(x1.*.grad, 2),
            });
        std.debug.print("df/dx2 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x2.*.grad, 0),
                c.ggml_get_f32_1d(x2.*.grad, 1),
                c.ggml_get_f32_1d(x2.*.grad, 2),
            });

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  99.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  17.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 1)  ==  17.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 2)  ==  17.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 1)  ==  3.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 2)  ==  3.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-7-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-7-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        const x1 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);
        const x2 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 3);

        c.ggml_set_param(ctx0, x1);
        c.ggml_set_param(ctx0, x2);

        const y =
            c.ggml_abs(ctx0,
                    c.ggml_sub(ctx0, x1, x2)
                    );

        const gf = c.ggml_build_forward(y);
        const gb = c.ggml_build_backward(ctx0, @constCast(&gf), false);

        _ = c.ggml_set_f32(x1, 3.0);
        _ = c.ggml_set_f32(x2, 5.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x1.*.grad, 0),
                c.ggml_get_f32_1d(x1.*.grad, 1),
                c.ggml_get_f32_1d(x1.*.grad, 2),
            });
        std.debug.print("df/dx2 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x2.*.grad, 0),
                c.ggml_get_f32_1d(x2.*.grad, 1),
                c.ggml_get_f32_1d(x2.*.grad, 2),
            });

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  2.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  -1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 1)  ==  -1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 2)  ==  -1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 1)  ==  1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 2)  ==  1.0);

        _ = c.ggml_set_f32(x1, 7.0);
        _ = c.ggml_set_f32(x2, 5.0);

        c.ggml_graph_reset(@constCast(&gf));
        _ = c.ggml_set_f32(y.*.grad, 1.0);

        c.ggml_graph_compute_with_ctx(ctx0, @constCast(&gb), n_threads);

        std.debug.print("y      = {d:.6}\n", .{c.ggml_get_f32_1d(y, 0)});
        std.debug.print("df/dx1 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x1.*.grad, 0),
                c.ggml_get_f32_1d(x1.*.grad, 1),
                c.ggml_get_f32_1d(x1.*.grad, 2),
            });
        std.debug.print("df/dx2 = {d:.6} {d:.6} {d:.6}\n",
            .{
                c.ggml_get_f32_1d(x2.*.grad, 0),
                c.ggml_get_f32_1d(x2.*.grad, 1),
                c.ggml_get_f32_1d(x2.*.grad, 2),
            });

        try std.testing.expect(c.ggml_get_f32_1d(y, 0)          ==  2.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 0)  ==  1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 1)  ==  1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x1.*.grad, 2)  ==  1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 0)  ==  -1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 1)  ==  -1.0);
        try std.testing.expect(c.ggml_get_f32_1d(x2.*.grad, 2)  ==  -1.0);

        c.ggml_graph_dump_dot(&gf, null, "test1-8-forward.dot");
        c.ggml_graph_dump_dot(&gb, &gf,  "test1-8-backward.dot");
    }

    _ = try std.io.getStdIn().reader().readByte();
}
