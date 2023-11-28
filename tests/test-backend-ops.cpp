#include <cstring>
#include <functional>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-backend-impl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>


static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    for (size_t i = 0; i < size; i++) {
        data[i] = distribution(generator);
    }

    ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
}

static std::vector<float> tensor_to_float(const struct ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
                    float v;
                    if (t->type == GGML_TYPE_F16) {
                        v = (float) ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]);
                    } else if (t->type == GGML_TYPE_F32) {
                        v = *(float *) &buf[i];
                    } else {
                        GGML_ASSERT(false);
                    }
                    tv.push_back(v);
                }
            }
        }
    }

    return tv;
}

static double cosine_similarity(const float * v1, const float * v2, size_t n) {
    double dot = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return -1.0f;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        dot  += v1[i]*v2[i];
        mag1 += v1[i]*v1[i];
        mag2 += v2[i]*v2[i];
    }

    return dot/sqrt(mag1*mag2);
}

static float distance(const float * v1, const float * v2, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return INFINITY;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        d += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }

    return sqrt(d);
}

static float vec_len(const float * v, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v[i])) {
            return INFINITY;
        }
        if (std::isinf(v[i])) {
            continue;
        }
        d += v[i]*v[i];
    }

    return sqrt(d);
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct test_case {
    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    bool eval(ggml_backend_t backend1, ggml_backend_t backend2) {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * out = build_graph(ctx);

        // check if backends support op
        for (ggml_backend_t backend : {backend1, backend2}) {
            if (!ggml_backend_supports_op(backend, out)) {
                //printf("Backend [%s] does not support op %s, skipping\n", ggml_backend_name(backend), ggml_op_desc(out));
                ggml_free(ctx);
                return true;
            }
        }

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);

        // build graph
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);
            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > 1e-6) {
                printf("Error: %s: %f\n", ggml_op_desc(t1), err);
                return false;
            }
            return true;
        };

        ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, nullptr);

        printf("  %s: OK\n", ggml_op_desc(out));

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return true; // or false if failed
    }
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;

    test_unary(ggml_unary_op op, ggml_type type = GGML_TYPE_F32) : op(op), type(type) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        static const int n = 10;

        ggml_tensor * in = ggml_new_tensor_1d(ctx, type, n);

        ggml_tensor * out = ggml_unary(ctx, in, op);

        return out;
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n = 10; // cols
    const int m = 5;  // rows
    const int r = 3;  // rows to get

    test_get_rows(ggml_type type = GGML_TYPE_F32) : type(type) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {

        ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, r);

        ggml_tensor * in = ggml_new_tensor_2d(ctx, type, n, m);

        ggml_tensor * out = ggml_get_rows(ctx, in, rows);

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // rows
                std::vector<int> data(r);
                for (int i = 0; i < r; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

void test_backend(ggml_backend_t backend) {
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();

    // test unary ops
    for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
        test_unary test((ggml_unary_op) op);
        test.eval(backend_cpu, backend);
    }

    // test get_rows
    {
        test_get_rows test;
        test.eval(backend_cpu, backend);
    }

    // TODO:
    // ggml_repeat, ggml_dup, ggml_add, ggml_mul,
    // ggml_norm, ggml_rms_norm, ggml_mul_mat, ggml_scale,
    // ggml_sqr, ggml_clamp, ggml_cpy, ggml_cont,
    // ggml_diag_mask_inf, ggml_soft_max, ggml_rope, ggml_alibi, ggml_im2col
}

int main() {
    // enumerate backends
    printf("Testing %zu backends\n\n", ggml_backend_reg_get_count());

    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        printf("Backend %zu/%zu (%s)\n", i + 1, ggml_backend_reg_get_count(), ggml_backend_reg_get_name(i));

        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);
        GGML_ASSERT(backend != NULL);
        printf("  Backend name: %s\n", ggml_backend_name(backend));

        test_backend(backend);

        ggml_backend_free(backend);

        printf("  OK\n\n");
    }
}
