#include <cstring>
#include <functional>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-backend-impl.h>
#include <array>
#include <memory>
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

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<uint16_t> data16(size);
        for (size_t i = 0; i < size; i++) {
            data16[i] = ggml_fp32_to_fp16(data[i]);
        }
        ggml_backend_tensor_set(tensor, data16.data(), 0, size * sizeof(uint16_t));
    } else {
        GGML_ASSERT(false);
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    // access elements by index to avoid gaps in views
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

/*
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
*/

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

//static std::string var_to_str(ggml_unary_op unary_op) {
//    return ggml_unary_op_name(unary_op);
//}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, ...) VAR_TO_STR(a) + "," + VAR_TO_STR(__VA_ARGS__)
#define VARS_TO_STR3(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR2(__VA_ARGS__)
#define VARS_TO_STR4(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR3(__VA_ARGS__)
#define VARS_TO_STR5(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR4(__VA_ARGS__)
#define VARS_TO_STR6(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR5(__VA_ARGS__)
#define VARS_TO_STR7(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR6(__VA_ARGS__)
#define VARS_TO_STR8(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR7(__VA_ARGS__)
#define VARS_TO_STR9(a, ...) VAR_TO_STR(a) + "," + VARS_TO_STR8(__VA_ARGS__)
#define VARS_TO_STR10(a,...) VAR_TO_STR(a) + "," + VARS_TO_STR9(__VA_ARGS__)
#define VARS_TO_STR11(a,...) VAR_TO_STR(a) + "," + VARS_TO_STR10(__VA_ARGS__)


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
    virtual std::string vars() {
        return "";
    }

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
                printf("  %s: not supported\n", ggml_op_desc(out));
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
        bool ok = true;

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);
            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > 1e-6) {
                printf("Error: %s: NMSE = %f\n", ggml_op_desc(t1), err);
                *(bool *) user_data = false;
            }
            return true;
       };

        ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ok);

        printf("  %s(%s): ", ggml_op_desc(out), vars().c_str());
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return ok;
    }
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_unary(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 10, 10, 10})
        : op(op), type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_unary(ctx, in, op);
        return out;
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get

    std::string vars() override {
        return VARS_TO_STR4(type, n, m, r);
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3)
        : type(type), n(n), m(m), r(r) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_2d(ctx, type, n, m);
        ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, r);
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

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            std::array<int, 4> nr = {2, 2, 2, 2})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_repeat(ctx, src, target);
        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_dup(ctx, src);
        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR3(type_src, type_dst, ne);
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type_src(type_src), type_dst(type_dst), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, ne.data());
        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        return out;
    }
};

// GGML_OP_CONT
struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        src = ggml_transpose(ctx, src);
        ggml_tensor * out = ggml_cont(ctx, src);

        return out;
    }
};

// GGML_OP_ADD
struct test_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int,4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    test_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_add(ctx, a, b);
        return out;
    }
};

// GGML_OP_MUL
struct test_mul : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int,4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    test_mul(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_mul(ctx, a, b);
        return out;
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, type, 1);
        ggml_tensor * out = ggml_scale(ctx, a, scale);
        return out;
    }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_rms_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4

    std::string vars() override {
        return VARS_TO_STR7(type_a, type_b, m, n, k, bs, nr);
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2})
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0]*nr[0], bs[1]*nr[1]);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        return out;
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sqr(ctx, a);
        return out;
    }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        return out;
    }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_soft_max(ctx, a);
        return out;
    }
};

// GGML_OP_ROPE
struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    int n_dims;
    int mode;
    int n_ctx;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, n_dims, mode, n_ctx);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1},
            int n_dims = 10, int mode = 0, int n_ctx = 512)
        : type(type), ne(ne), n_dims(n_dims), mode(mode), n_ctx(n_ctx) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne[2]);
        ggml_tensor * out = ggml_rope(ctx, a, pos, n_dims, mode, n_ctx);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(ne[2]);
                for (int i = 0; i < ne[2]; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, ne[2] * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_ALIBI
struct test_alibi : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    int n_past;
    int n_head;
    float bias_max;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, n_past, n_head, bias_max);
    }

    test_alibi(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int n_past = 512, int n_head = 10, float bias_max = 0.5f)
        : type(type), ne(ne), n_past(n_past), n_head(n_head), bias_max(bias_max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_alibi(ctx, a, n_past, n_head, bias_max);
        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;
    const int s0;
    const int s1;
    const int p0;
    const int p1;
    const int d0;
    const int d1;
    const bool is_2D;

    std::string vars() override {
        return VARS_TO_STR11(type_a, type_b, ne_a, ne_b, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_a = GGML_TYPE_F16, ggml_type type_b = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 10},
            std::array<int64_t, 4> ne_b = {10, 10, 10, 10},
            int s0 = 1, int s1 = 1,
            int p0 = 0, int p1 = 0,
            int d0 = 1, int d1 = 1,
            bool is_2D = false)
        : type_a(type_a), type_b(type_b), ne_a(ne_a), ne_b(ne_b), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type_a, 4, ne_a.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type_b, 4, ne_b.data());
        ggml_tensor * out = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D);
        return out;
    }
};

static bool test_backend(ggml_backend_t backend) {
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();

    std::vector<std::unique_ptr<test_case>> test_cases;

    // unary ops
    for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
        test_cases.emplace_back(new test_unary((ggml_unary_op) op));
    }

    test_cases.emplace_back(new test_get_rows());
    test_cases.emplace_back(new test_repeat());
    test_cases.emplace_back(new test_dup());
    test_cases.emplace_back(new test_cpy());
    test_cases.emplace_back(new test_cont());

    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 1, 1}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 1}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 1}));
    //test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 1, 1, 1})); // broadcasting dim 0 is not supported
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 1, 1}));
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 1}));
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 2}));
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 2}));
    test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 2, 2}));
    //test_cases.emplace_back(new test_add(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 2, 2, 2}));

    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 1, 1}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 1}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 1}));
    //test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 1, 1, 1})); // broadcasting dim 0 is not supported
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 1, 1}));
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 1}));
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 2}));
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 2}));
    test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 2, 2}));
    //test_cases.emplace_back(new test_mul(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 2, 2, 2}));

    test_cases.emplace_back(new test_scale());
    test_cases.emplace_back(new test_norm());
    test_cases.emplace_back(new test_rms_norm());
    test_cases.emplace_back(new test_mul_mat());
    test_cases.emplace_back(new test_sqr());
    test_cases.emplace_back(new test_clamp());
    test_cases.emplace_back(new test_diag_mask_inf());
    test_cases.emplace_back(new test_soft_max());
    test_cases.emplace_back(new test_rope());
    test_cases.emplace_back(new test_alibi());
    test_cases.emplace_back(new test_im2col());

    size_t n_ok = 0;
    for (auto & test : test_cases) {
        if (test->eval(backend, backend_cpu)) {
            n_ok++;
        }
    }

    printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

    ggml_backend_free(backend_cpu);

    return n_ok == test_cases.size();
}

int main() {
    // enumerate backends
    printf("Testing %zu backends\n\n", ggml_backend_reg_get_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        printf("Backend %zu/%zu (%s)\n", i + 1, ggml_backend_reg_get_count(), ggml_backend_reg_get_name(i));

        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);
        GGML_ASSERT(backend != NULL);
        printf("  Backend name: %s\n", ggml_backend_name(backend));

        bool ok = test_backend(backend);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_reg_get_count());
    if (n_ok != ggml_backend_reg_get_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    } else {
        printf("\033[1;32mOK\033[0m\n");
        return 0;
    }
}
