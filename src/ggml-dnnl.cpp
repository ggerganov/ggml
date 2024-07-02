#include "ggml-dnnl.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <tuple>
#include <utility>
#include <unordered_map>

#include <dnnl.hpp>

static const char* ggml_backend_dnnl_name_str = "DNNL";

struct ggml_backend_dnnl_context {
    dnnl::engine engine;
    dnnl::stream stream;
    std::unordered_map<const ggml_tensor*, dnnl::memory> tensor_to_mem;
    std::unordered_map<const ggml_tensor*, dnnl::primitive_desc> tensor_to_prim;

    ggml_backend_dnnl_context(size_t dev_idx = 0)
        : engine{dnnl::engine::kind::cpu, dev_idx}
        , stream{engine}
    {}
};

static dnnl::memory::data_type ggml_type_to_dnnl_dtype(enum ggml_type type) {
    using dt = dnnl::memory::data_type;
    switch (type) {
        case GGML_TYPE_F32:
            return dt::f32;
        case GGML_TYPE_F16:
            return dt::f16;
        case GGML_TYPE_I8:
            return dt::s8;
        case GGML_TYPE_I32:
            return dt::s32;
        case GGML_TYPE_F64:
            return dt::f64;
        case GGML_TYPE_BF16:
            return dt::bf16;
        default:
            return dt::undef;
    }
}

static bool ggml_dnnl_type_supported(enum ggml_type type) {
    return ggml_type_to_dnnl_dtype(type) != dnnl::memory::data_type::undef;
}

static bool ggml_dnnl_tensor_supported(const struct ggml_tensor * t) {
    auto type = t->type;

    if (!ggml_dnnl_type_supported(type)) {
        return false;
    }
    return true;
}

static void* get_dnnl_memory_handle(const struct ggml_tensor * t) {
    return t->data;
}

static dnnl::memory::desc ggml_tensor_to_dnnl_md(const struct ggml_tensor * t, bool transpose = false,
                                          size_t ndims = GGML_MAX_DIMS) {
    GGML_ASSERT(ggml_dnnl_tensor_supported(t));
    using dims_t = dnnl::memory::dims;

    const auto tensor_type = t->type;
    auto dt = ggml_type_to_dnnl_dtype(tensor_type);
    auto type_size = ggml_type_size(tensor_type);

    dims_t adims(ndims);
    dims_t strides(ndims);

    for (size_t i = 0; i < ndims; i++ ) {
        adims[ndims - 1 - i] = t->ne[i];
        strides[ndims - 1 - i] = t->nb[i] / type_size;
    }

    if (transpose) {
        std::swap(adims[ndims-1], adims[ndims-2]);
        std::swap(strides[ndims-1], strides[ndims-2]);
    }

    for (size_t i = ndims; i < GGML_MAX_DIMS; i++) {
        GGML_ASSERT(t->nb[i] == t->nb[i-1] * t->ne[i-1]);
        adims[0] *= t->ne[i];
    }

    return dnnl::memory::desc{adims, dt, strides};
}

static dnnl::memory ggml_tensor_to_dnnl_mem(ggml_backend_dnnl_context * ctx, const struct ggml_tensor * t,
                                     bool transpose = false,
                                     size_t ndims = GGML_MAX_DIMS) {
    auto t_md = ggml_tensor_to_dnnl_md(t, transpose, ndims);
    auto t_mem = dnnl::memory{t_md, ctx->engine, get_dnnl_memory_handle(t)};

    return t_mem;
}

static dnnl::memory ggml_backend_dnnl_reorder_to(ggml_backend_dnnl_context * ctx,
                                                 dnnl::memory& src_mem,
                                                 const dnnl::memory::desc& dst_md) {
    auto dst_mem = dnnl::memory{dst_md, ctx->engine};
    auto rdr = dnnl::reorder{src_mem, dst_mem};
    rdr.execute(ctx->stream, src_mem, dst_mem);
    ctx->stream.wait();
    return dst_mem;
}

static dnnl::primitive_desc ggml_backend_dnnl_create_matmul_pd(ggml_backend_dnnl_context * ctx,
                                                  const struct ggml_tensor * dst,
                                                  const struct ggml_tensor * src,
                                                  const struct ggml_tensor * weights,
                                                  const struct ggml_tensor * bias_add = nullptr) {
    bool is_ip = weights->ne[2] == 1 && weights->ne[3] == 1;
    size_t ndims = is_ip ? 2 : GGML_MAX_DIMS;

    auto dst_md = ggml_tensor_to_dnnl_md(dst, false, ndims);
    auto src_md = ggml_tensor_to_dnnl_md(src, false, ndims);
    auto weights_md = ggml_tensor_to_dnnl_md(weights, !is_ip, ndims);
    auto weights_md_any = weights_md;

    if (weights->buffer && weights->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        weights_md_any = dnnl::memory::desc{weights_md.get_dims(), weights_md.get_data_type(), dnnl::memory::format_tag::any};
    }

    dnnl::memory::desc bias_md;
    if (bias_add != nullptr) {
        bias_md = ggml_tensor_to_dnnl_md(bias_add, false, is_ip ? 1 : ndims);
    }

    static const dnnl::primitive_attr default_attr;
    if (is_ip) {
        return dnnl::inner_product_forward::primitive_desc{ctx->engine, dnnl::prop_kind::forward_inference, src_md, weights_md_any, bias_md, dst_md, default_attr, true};
    } else {
        return dnnl::matmul::primitive_desc{ctx->engine, src_md, weights_md_any, bias_md, dst_md, default_attr, true};
    }
}

static dnnl::memory ggml_backend_dnnl_get_weights_mem(ggml_backend_dnnl_context * ctx, const struct ggml_tensor * weights, const dnnl::primitive_desc& pd) {
    auto is_ip = pd.get_kind() == dnnl::primitive::kind::inner_product;
    size_t ndims = is_ip ? 2 : GGML_MAX_DIMS;

    auto weights_target_md = pd.weights_desc();

    // lookup in cache
    if (weights->buffer && weights->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        auto it = ctx->tensor_to_mem.find(weights);
        if (it == ctx->tensor_to_mem.end() || weights_target_md != it->second.get_desc()) {
            // reorder and store to cache
            auto weights_mem = ggml_tensor_to_dnnl_mem(ctx, weights, !is_ip, ndims);
            if (weights_mem.get_desc() != weights_target_md) {
                weights_mem = ggml_backend_dnnl_reorder_to(ctx, weights_mem, weights_target_md);
            }
            bool emplaced;
            std::tie(it, emplaced) = ctx->tensor_to_mem.emplace(weights, weights_mem);
            if (!emplaced) {
                it->second = weights_mem;
            }
            //GGML_ASSERT(emplaced);
        }

        GGML_ASSERT(weights_target_md == it->second.get_desc());
        return it->second;
    } else {
        // reorder not store to cache
        auto weights_mem = ggml_tensor_to_dnnl_mem(ctx, weights, !is_ip, ndims);
        if (weights_mem.get_desc() != weights_target_md) {
            return ggml_backend_dnnl_reorder_to(ctx, weights_mem, weights_target_md);
        } else {
            return weights_mem;
        }
    }
}

// helper function to determine if it is better to use DNNL or not
// for large matrices, DNNL is faster
static bool ggml_backend_dnnl_use_dnnl_impl(ggml_backend_dnnl_context * ctx,
                                            const struct ggml_tensor * dst,
                                            const struct ggml_tensor * src,
                                            const struct ggml_tensor * weights,
                                            const struct ggml_tensor * bias_add = nullptr) {
#if 0
    return true;
    GGML_UNUSED(dst);
#else
    GGML_UNUSED(ctx);

    const int64_t ne10 = src->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t mm_threshold = 32;

    bool ok = (true
        && dst->op != GGML_OP_MUL_MAT_ID
        && ggml_dnnl_tensor_supported(src)
        && ggml_dnnl_tensor_supported(weights)
        && ggml_dnnl_tensor_supported(dst)
        && ggml_is_contiguous(src)
        && ggml_is_contiguous(weights)
        && (ne0 >= mm_threshold && ne1 >= mm_threshold && ne10 >= mm_threshold)
        );
    if (!ok) {
        return false;
    }

    auto pd = ggml_backend_dnnl_create_matmul_pd(ctx, dst, src, weights, bias_add);
    if (!pd) {
        return false;
    }
    ctx->tensor_to_prim[dst] = pd;
    ggml_backend_dnnl_get_weights_mem(ctx, weights, pd);

    return true;
#endif
}

static bool ggml_backend_dnnl_use_dnnl(ggml_backend_dnnl_context * ctx, const struct ggml_tensor * dst) {
    const struct ggml_tensor * src = dst->src[1];
    const struct ggml_tensor * weights = dst->src[0];
    return ggml_backend_dnnl_use_dnnl_impl(ctx, dst, src, weights, nullptr);
}

static bool ggml_backend_dnnl_use_dnnl_bias_add(ggml_backend_dnnl_context * ctx, const struct ggml_tensor * dst) {
#if 0
    return false;
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
#else
    const struct ggml_tensor * mul_mat = dst->src[0];
    const struct ggml_tensor * bias = dst->src[1];
    

    // NOTE: src0 - weights, src1 - input
    const struct ggml_tensor * src = mul_mat->src[1];
    const struct ggml_tensor * weights = mul_mat->src[0];

    return (mul_mat->op == GGML_OP_MUL_MAT 
           && bias->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS
           && ggml_backend_dnnl_use_dnnl_impl(ctx, dst, src, weights, bias)
    );
#endif
}

static ggml_status ggml_backend_dnnl_mul_mat_impl(ggml_backend_dnnl_context * ctx,
                                                        struct ggml_tensor * dst,
                                                  const struct ggml_tensor * src,
                                                  const struct ggml_tensor * weights,
                                                  const struct ggml_tensor * bias_add = nullptr) {
    GGML_TENSOR_LOCALS(int64_t, ne_s, src, ne)
    GGML_TENSOR_LOCALS(int64_t, ne_w, weights, ne)
    GGML_TENSOR_LOCALS(int64_t, ne_d, dst, ne)

    GGML_ASSERT(ne_d0 == ne_w1);
    GGML_ASSERT(ne_d1 == ne_s1);
    GGML_ASSERT(ne_d2 == ne_s2);
    GGML_ASSERT(ne_d3 == ne_s3);

    auto it = ctx->tensor_to_prim.find(dst);

    dnnl::primitive_desc pd;
    if (it != ctx->tensor_to_prim.end()) {
        pd = it->second;
    } else {
        pd = ggml_backend_dnnl_create_matmul_pd(ctx, dst, src, weights, bias_add);
    }
    GGML_ASSERT(pd);
    auto dst_mem = dnnl::memory{pd.dst_desc(), pd.get_engine(), get_dnnl_memory_handle(dst)};
    auto src_mem = dnnl::memory{pd.src_desc(), pd.get_engine(), get_dnnl_memory_handle(src)};
    auto weights_mem = ggml_backend_dnnl_get_weights_mem(ctx, weights, pd);

    dnnl::memory bias_mem{nullptr};
    if (bias_add) {
        bias_mem = dnnl::memory{pd.weights_desc(1), pd.get_engine(), get_dnnl_memory_handle(bias_add)};
    }

    auto prim = dnnl::primitive{pd};
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,     src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS,    bias_mem},
        {DNNL_ARG_DST,     dst_mem},
    };
    prim.execute(ctx->stream, args);
    ctx->stream.wait();

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_dnnl_mul_mat(ggml_backend_dnnl_context * ctx, struct ggml_tensor * dst) {
    // NOTE: src0 - weights, src1 - input
    const struct ggml_tensor * src = dst->src[1];
    const struct ggml_tensor * weights = dst->src[0];
    return ggml_backend_dnnl_mul_mat_impl(ctx, dst, src, weights, nullptr);
}

static ggml_status ggml_backend_dnnl_mul_mat_bias(ggml_backend_dnnl_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * mul_mat = dst->src[0];
    const struct ggml_tensor * bias = dst->src[1];

    // NOTE: src0 - weights, src1 - input
    const struct ggml_tensor * src = mul_mat->src[1];
    const struct ggml_tensor * weights = mul_mat->src[0];
    
    return ggml_backend_dnnl_mul_mat_impl(ctx, dst, src, weights, bias);
}

// backend interface

GGML_CALL static const char * ggml_backend_dnnl_name(ggml_backend_t backend) {
    return ggml_backend_dnnl_name_str;

    GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_dnnl_free(ggml_backend_t backend) {
    ggml_backend_dnnl_context * ctx = (ggml_backend_dnnl_context *)backend->context;
    delete ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_dnnl_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_dnnl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_dnnl_context * ctx = (ggml_backend_dnnl_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
            {
                ggml_tensor * node2 = i+1 < cgraph->n_nodes ? cgraph->nodes[i+1]: nullptr;
                if (node2 && node2->op == GGML_OP_ADD && ggml_backend_dnnl_use_dnnl_bias_add(ctx, node2)) {
                    ggml_backend_dnnl_mul_mat_bias(ctx, node2);
                    ++i;
                } else {
                    ggml_backend_dnnl_mul_mat(ctx, node);
                }
                break;
            }

            // case GGML_OP_OUT_PROD:
            //     ggml_backend_dnnl_out_prod(ctx, node);
            //     break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_dnnl_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    // const struct ggml_tensor * src0 = op->src[0];
    // const struct ggml_tensor * src1 = op->src[1];

    ggml_backend_dnnl_context * ctx = (ggml_backend_dnnl_context *)backend->context;

    return ((op->op == GGML_OP_MUL_MAT  && ggml_backend_dnnl_use_dnnl(ctx, op))
            || (op->op == GGML_OP_ADD && ggml_backend_dnnl_use_dnnl_bias_add(ctx, op)));
    //  ||
    //        (op->op == GGML_OP_OUT_PROD && op->src[0]->type == GGML_TYPE_F32 &&
    //                                       op->src[1]->type == GGML_TYPE_F32 &&
    //                                       ggml_is_matrix(src0) &&
    //                                       ggml_is_matrix(src1) &&
    //                                       ggml_is_contiguous(src0) &&
    //                                       (ggml_is_contiguous(src1) || ggml_is_transposed(src1)));

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_dnnl_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
}

static struct ggml_backend_i dnnl_backend_i = {
    /* .get_name                = */ ggml_backend_dnnl_name,
    /* .free                    = */ ggml_backend_dnnl_free,
    /* .get_default_buffer_type = */ ggml_backend_dnnl_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_dnnl_graph_compute,
    /* .supports_op             = */ ggml_backend_dnnl_supports_op,
    /* .supports_buft           = */ ggml_backend_dnnl_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_dnnl_guid(void) {
    static ggml_guid guid = {0xce, 0x8e, 0x0c, 0x82, 0xfe, 0x2f, 0x11, 0xee, 0x9f, 0x1a, 0xef, 0xb0, 0x5d, 0xa0, 0xc4, 0x4a};
    return &guid;
}

ggml_backend_t ggml_backend_dnnl_init(void) {
    ggml_backend_dnnl_context * ctx = new ggml_backend_dnnl_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_dnnl_guid(),
        /* .interface = */ dnnl_backend_i,
        /* .context   = */ ctx,
    };
    return backend;
}

GGML_CALL bool ggml_backend_is_dnnl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_dnnl_guid());
}

