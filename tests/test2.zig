const std = @import("std");
const Thread = std.Thread;
const c = @cImport({
    @cInclude("ggml/ggml.h");
});

fn is_close(a: f32, b: f32, epsilon: f32) bool {
    return std.math.fabs(a - b) < epsilon;
}

pub fn main() !void {
    const params = .{
        .mem_size   = 128*1024*1024,
        .mem_buffer = null,
        .no_alloc   = false,
    };

    var opt_params = c.ggml_opt_default_params(c.GGML_OPT_LBFGS);
    
    const nthreads = try Thread.getCpuCount();
    opt_params.n_threads = @intCast(nthreads);
    std.debug.print("test2: n_threads:{}\n", .{opt_params.n_threads});

    const xi = [_]f32{  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0 };
    const yi = [_]f32{ 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0 };

    const n = xi.len;

    const ctx0 = c.ggml_init(params);
    defer c.ggml_free(ctx0);

    const x = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, n);
    const y = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, n);

    for (0..n) |i| {
        const x_data_pointer: [*]f32 = @ptrCast(@alignCast(x.*.data));
        x_data_pointer[i] = xi[i];
        const y_data_pointer: [*]f32 = @ptrCast(@alignCast(y.*.data));
        y_data_pointer[i] = yi[i];
    }

    {
        const t0 = c.ggml_new_f32(ctx0, 0.0);
        const t1 = c.ggml_new_f32(ctx0, 0.0);

        // initialize auto-diff parameters:
        _ = c.ggml_set_param(ctx0, t0);
        _ = c.ggml_set_param(ctx0, t1);

        // f = sum_i[(t0 + t1*x_i - y_i)^2]/(2n)
        const f =
            c.ggml_div(ctx0,
                    c.ggml_sum(ctx0,
                        c.ggml_sqr(ctx0,
                            c.ggml_sub(ctx0,
                                c.ggml_add(ctx0,
                                    c.ggml_mul(ctx0, x, c.ggml_repeat(ctx0, t1, x)),
                                    c.ggml_repeat(ctx0, t0, x)),
                                y)
                            )
                        ),
                    c.ggml_new_f32(ctx0, @as(f32, 2.0)*n));

        const res = c.ggml_opt(null, opt_params, f);

        std.debug.print("t0 = {d:.6}\n", .{c.ggml_get_f32_1d(t0, 0)});
        std.debug.print("t1 = {d:.6}\n", .{c.ggml_get_f32_1d(t1, 0)});

        try std.testing.expect(res == c.GGML_OPT_OK);
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t0, 0),  5.0, 1e-3));
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t1, 0), 10.0, 1e-3));
    }

    {
        const t0 = c.ggml_new_f32(ctx0, -1.0);
        const t1 = c.ggml_new_f32(ctx0,  9.0);

        _ = c.ggml_set_param(ctx0, t0);
        _ = c.ggml_set_param(ctx0, t1);

        // f = 0.5*sum_i[abs(t0 + t1*x_i - y_i)]/n
        const f =
            c.ggml_mul(ctx0,
                    c.ggml_new_f32(ctx0, @as(f32, 1.0)/(2*n)),
                    c.ggml_sum(ctx0,
                        c.ggml_abs(ctx0,
                            c.ggml_sub(ctx0,
                                c.ggml_add(ctx0,
                                    c.ggml_mul(ctx0, x, c.ggml_repeat(ctx0, t1, x)),
                                    c.ggml_repeat(ctx0, t0, x)),
                                y)
                            )
                        )
                    );


        const res = c.ggml_opt(null, opt_params, f);

        try std.testing.expect(res == c.GGML_OPT_OK);
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t0, 0),  5.0, 1e-2));
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t1, 0), 10.0, 1e-2));
    }

    {
        const t0 = c.ggml_new_f32(ctx0,  5.0);
        const t1 = c.ggml_new_f32(ctx0, -4.0);

        _ = c.ggml_set_param(ctx0, t0);
        _ = c.ggml_set_param(ctx0, t1);

        // f = t0^2 + t1^2
        const f =
            c.ggml_add(ctx0,
                    c.ggml_sqr(ctx0, t0),
                    c.ggml_sqr(ctx0, t1)
                    );

        const res = c.ggml_opt(null, opt_params, f);

        try std.testing.expect(res == c.GGML_OPT_OK);
        try std.testing.expect(is_close(c.ggml_get_f32_1d(f,  0), 0.0, 1e-3));
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t0, 0), 0.0, 1e-3));
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t1, 0), 0.0, 1e-3));
    }

    /////////////////////////////////////////

    {
        const t0 = c.ggml_new_f32(ctx0, -7.0);
        const t1 = c.ggml_new_f32(ctx0,  8.0);

        _ = c.ggml_set_param(ctx0, t0);
        _ = c.ggml_set_param(ctx0, t1);

        // f = (t0 + 2*t1 - 7)^2 + (2*t0 + t1 - 5)^2
        const f =
            c.ggml_add(ctx0,
                    c.ggml_sqr(ctx0,
                        c.ggml_sub(ctx0,
                            c.ggml_add(ctx0,
                                t0,
                                c.ggml_mul(ctx0, t1, c.ggml_new_f32(ctx0, 2.0))),
                            c.ggml_new_f32(ctx0, 7.0)
                            )
                        ),
                    c.ggml_sqr(ctx0,
                        c.ggml_sub(ctx0,
                            c.ggml_add(ctx0,
                                c.ggml_mul(ctx0, t0, c.ggml_new_f32(ctx0, 2.0)),
                                t1),
                            c.ggml_new_f32(ctx0, 5.0)
                            )
                        )
                    );

        const res = c.ggml_opt(null, opt_params, f);

        try std.testing.expect(res == c.GGML_OPT_OK);
        try std.testing.expect(is_close(c.ggml_get_f32_1d(f,  0), 0.0, 1e-3));
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t0, 0), 1.0, 1e-3));
        try std.testing.expect(is_close(c.ggml_get_f32_1d(t1, 0), 3.0, 1e-3));
    }

    _ = try std.io.getStdIn().reader().readByte();
}
