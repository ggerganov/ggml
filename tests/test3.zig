const std = @import("std");
const Thread = std.Thread;
const c = @cImport({
    @cInclude("stdlib.h");
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

    const NP = 1 << 12;
    const NF = 1 << 8;

    const ctx0 = c.ggml_init(params);
    defer c.ggml_free(ctx0);

    const F = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, NF, NP);
    const l = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, NP);

    // regularization weight
    const lambda = c.ggml_new_f32(ctx0, 1e-5);

    c.srand(0);

    const l_data_pointer: [*]f32 = @ptrCast(@alignCast(l.*.data));
    const f_data_pointer: [*]f32 = @ptrCast(@alignCast(F.*.data));
    for (0..NP) |j| {
        const ll = if (j < NP/2) @as(f32, 1.0) else @as(f32, -1.0);
        l_data_pointer[j] = ll;
        
        for (0..NF) |i| {
            const c_rand: f32 = @floatFromInt(c.rand());
            f_data_pointer[j*NF + i] = 
                ((if (ll > 0 and i < NF/2) @as(f32, 1.0) else 
                    if (ll < 0 and i >= NF/2) @as(f32, 1.0) else @as(f32, 0.0)) + 
                        (c_rand/c.RAND_MAX - 0.5) * 0.1) / (0.5 * NF);
        }
    }

    {
        // initial guess
        const x = c.ggml_set_f32(c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, NF), 0.0);

        c.ggml_set_param(ctx0, x);

        // f = sum[(fj*x - l)^2]/n + lambda*|x^2|
        const f =
            c.ggml_add(ctx0,
                    c.ggml_div(ctx0,
                        c.ggml_sum(ctx0,
                            c.ggml_sqr(ctx0,
                                c.ggml_sub(ctx0,
                                    c.ggml_mul_mat(ctx0, F, x),
                                    l)
                                )
                            ),
                        c.ggml_new_f32(ctx0, @as(f32, NP))
                        ),
                    c.ggml_mul(ctx0,
                        c.ggml_sum(ctx0, c.ggml_sqr(ctx0, x)),
                        lambda)
                    );

        const res = c.ggml_opt(null, opt_params, f);

        try std.testing.expect(res == c.GGML_OPT_OK);

        const x_data_pointer: [*]f32 = @ptrCast(@alignCast(x.*.data));
        // print results
        for (0..16) |i| {
            std.debug.print("x[{d:3}] = {d:.6}\n", .{i, x_data_pointer[i]});
        }
        std.debug.print("...\n", .{});
        for (NF - 16..NF) |i| {
            std.debug.print("x[{d:3}] = {d:.6}\n", .{i, x_data_pointer[i]});
        }
        std.debug.print("\n", .{});

        for (0..NF) |i| {
            if (i < NF/2) {
                try std.testing.expect(is_close(x_data_pointer[i], 1.0, 1e-2));
            } else {
                try std.testing.expect(is_close(x_data_pointer[i], -1.0, 1e-2));
            }
        }
    }

    _ = try std.io.getStdIn().reader().readByte();
}
