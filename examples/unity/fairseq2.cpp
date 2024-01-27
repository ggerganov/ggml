#include <algorithm>
#include <fnmatch.h>
#include <iostream>
#include <math.h>
#include <queue>
#include <unordered_map>

#include "feature-fbank.h"
#include "feature-window.h"
#include "fairseq2.h"
#include "ggml.h"
#include "ggml-alloc.h"

#include <numeric>

ggml_tensor* ggml_detach(ggml_tensor* a) {
    a->op = GGML_OP_NONE;
    std::fill(a->src, a->src + GGML_MAX_SRC, nullptr);
    return a;
}

// generate_sequence uses ggml_context and ggml_allocr to reuse memory buffers across steps.
// This can lead to dangling pointers, which don't segfault, but instead read garbage data.
// Enabling this flag allows to explictly reset memory buffers, making it more explicit
// when we read garbage data.
// It also prints memory usage information, which is useful to
#define DEBUG_MEM_USAGE DEBUG
std::size_t MB = 1024 * 1024;

void printf_mem_usage(ggml_context* ctx, std::string name) {
#if DEBUG_MEM_USAGE
    double mb = 1024.0 * 1024.0;
    printf(
        "%s: memory used = %8.2f MB, memory reserved = %8.2f Mb\n",
        name.c_str(),
        ggml_used_mem(ctx) / mb,
        ggml_get_mem_size(ctx) / mb
    );
#endif
}

#define SWAP(x, y) \
    auto tmp_ ## x = x; x = y; y = tmp_ ## x;


#define GGML_ASSERT_SHAPE(x, ne0, ne1, ne2, ne3) \
    GGML_ASSERT((ne0 == -1 || x->ne[0] == ne0) && (ne1 == -1 || x->ne[1] == ne1) && (ne2 == -1 || x->ne[2] == ne2) && (ne3 == -1 || x->ne[3] == ne3));

/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc() {
    // pre-allocate some memory to write hyperparameters and tensors pointers
    auto* model = new fairseq2_model;
    model->tensors_ctx = nullptr;
    return model;
}

extern "C" void fairseq2_kv_cache_alloc(fairseq2_model& model, ggml_context* kv_cache_ctx, int beam_size, int max_seq_len) {
    // Note: we only allocate the masks, proper kv cache allocation is delayed.
    GGML_ASSERT(kv_cache_ctx);
    GGML_ASSERT(!ggml_get_no_alloc(kv_cache_ctx));  // We need to be able to alloc the kv_cache buffers
    auto attn_glob = "text_decoder.*_attn.k_proj.weight";
    FORCE_ALLOC(self_attn_mask, kv_cache_ctx, ggml_new_tensor_2d(kv_cache_ctx, GGML_TYPE_F32, max_seq_len, max_seq_len));
    self_attn_mask = ggml_diag_mask_inf_inplace(kv_cache_ctx, self_attn_mask, 0);
    ggml_format_name(self_attn_mask, "self_attn_mask[%d]", max_seq_len);

    for (auto named_tensor : model.tensors) {
        const std::string& name = named_tensor.first;
        if (::fnmatch(attn_glob, name.c_str(), 0) == FNM_NOMATCH)
            continue;
        // create a cache entry without the ".k_proj.weight" suffix
        const std::string& shortname = name.substr(0, name.size() - 14);
        KeyValueTensor& kv = model.kv_cache[shortname];
        kv.step_nr = 0;

        kv.full_k = nullptr;
        kv.full_v = nullptr;
        kv.self_attn_mask = self_attn_mask;
    }
}

extern "C" void fairseq2_kv_cache_reset(const fairseq2_model& model) {
    // TODO: use a dedicated allocator, so that kv_cache.clear actually frees the memory
    model.kv_cache.clear();
}


bool has_kv_cache(const fairseq2_model& model) {
    return model.kv_cache.size() > 0;
}


inline ggml_tensor* ggml_squeeze(ggml_context* ctx, ggml_tensor* x, int dim) {
    int n_dims = ggml_n_dims(x);
    GGML_ASSERT(dim >= 0);
    GGML_ASSERT(dim < n_dims);
    GGML_ASSERT(x->ne[dim] == 1);
    return ggml_flatten_1d(ctx, x, dim);
}

inline ggml_tensor* ggml_unsqueeze(ggml_context* ctx, ggml_tensor* x, int dim) {
    return ggml_unflatten_1d(ctx, x, dim, 1);
}


// copy k and v to kv cache
// kv.full_k[step_nr] = k;
// kv.full_v[step_nr] = v;
void append_to_prev_kv(const fairseq2_model& model, const std::string& prefix, ggml_tensor** k, ggml_tensor** v, ggml_tensor** self_attn_mask) {
    KeyValueTensor& kv = model.kv_cache[prefix];
    int step_nr = kv.step_nr;
    ggml_context* ctx = model.ctx;
    // We need to force allocation here, otherwise the kv_cache buffers can be reused
    bool no_alloc_save = ggml_get_no_alloc(ctx);
    ggml_set_no_alloc(ctx, false);
    int n_steps = (*k)->ne[1];
    int k_proj, batch_size;

    if (kv.full_k != nullptr) {
        // (N, S_kv, K_proj)
        k_proj = kv.full_k->ne[0];
        batch_size = kv.full_k->ne[2];
        ggml_detach(kv.full_k);
        ggml_detach(kv.full_v);
        kv.full_k = ggml_squeeze(ctx, ggml_concat(ctx, ggml_unsqueeze(ctx, kv.full_k, 1), ggml_unsqueeze(ctx, *k, 1)), 1);
        kv.full_v = ggml_squeeze(ctx, ggml_concat(ctx, ggml_unsqueeze(ctx, kv.full_v, 1), ggml_unsqueeze(ctx, *v, 1)), 1);
    } else {
        GGML_ASSERT(step_nr == 0);
        k_proj = (*k)->ne[0];
        batch_size = (*v)->ne[2];
        kv.full_k = ggml_dup(ctx, *k);
        kv.full_v = ggml_dup(ctx, *v);
    }
    *k = kv.full_k;
    *v = kv.full_v;
    ggml_format_name(kv.full_k, "%s.k (step=%d)", prefix.c_str(), step_nr);
    ggml_format_name(kv.full_v, "%s.v (step=%d)", prefix.c_str(), step_nr);
    step_nr += n_steps;

    GGML_ASSERT_SHAPE(kv.full_k, k_proj, step_nr, batch_size, 1);

    // qk is (B * H, Sq, Sk) == (B*H, 1, Sk) in incremental mode
    // we return the Sq slice of the (Sq, Sk) attention mask
    if (self_attn_mask != nullptr) {
        *self_attn_mask = ggml_slice(
            ctx, ggml_slice(ctx, kv.self_attn_mask, 0, 0, step_nr),
            1, step_nr - 1, step_nr
        );
    }

    kv.step_nr = step_nr;
    ggml_set_no_alloc(ctx, no_alloc_save);
}

// variant of ggml_get_rows that allows for a with more than 2 dims.
ggml_tensor* ggml_get_rows2(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    int flattened = 0;
    GGML_ASSERT(ggml_n_dims(a) <= 3);
    if (ggml_n_dims(a) == 3) {
        flattened = a->ne[0];
        a = ggml_flatten_1d(ctx, a, 0);
    }
    a = ggml_get_rows(ctx, a, b);
    if (flattened) {
        a = ggml_unflatten_1d(ctx, a, 0, flattened);
    }
    return a;
}


void _reorder_kv_cache(ggml_context* ctx, ggml_cgraph* gf, KeyValueTensor& kv, ggml_tensor* new_order) {
    // GGML_ASSERT(ctx == kv.full_k->con);
    if (kv.full_k != nullptr) {
        ggml_detach(kv.full_k);
        const char* name = kv.full_k->name;
        kv.full_k = ggml_get_rows2(ctx, kv.full_k, new_order);
        ggml_build_forward_expand(gf, kv.full_k);
        ggml_format_name(kv.full_k, "%s (sorted)", name);
    }

    if (kv.full_v != nullptr) {
        ggml_detach(kv.full_v);
        const char* name = kv.full_v->name;
        kv.full_v = ggml_get_rows2(ctx, kv.full_v, new_order);
        ggml_build_forward_expand(gf, kv.full_v);
        ggml_format_name(kv.full_v, "%s (sorted)", name);
    }
}


void reorder_kv_cache(const fairseq2_model& model, ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* new_order) {
    auto self_attn_glob = "*.self_attn";
    for (auto& named_kv : model.kv_cache) {
        if (::fnmatch(self_attn_glob, named_kv.first.c_str(), 0) == FNM_NOMATCH)
            continue;

        _reorder_kv_cache(ctx, gf, named_kv.second, new_order);
    }
}


inline double model_layer_config_d(const fairseq2_model& model, std::string name) {
    const std::int64_t* data = &model.layer_config.at(name);
    double val = *(const double*)data;
    return val;
}

extern "C" double fairseq2_model_layer_config_double(const fairseq2_model& model, const char* name) {
    return model_layer_config_d(model, std::string(name));
}

extern "C" std::int64_t fairseq2_model_layer_config_int(const fairseq2_model& model, const char* name) {
    return model.layer_config.at(std::string(name));
}


extern "C" void fairseq2_model_free(fairseq2_model* model) {
    if (model->tensors_ctx) ggml_free(model->tensors_ctx);
    // delete model;
}

extern "C" void fairseq2_model_set_inference_ctx(fairseq2_model* model, ggml_context* ctx) {
    model->ctx = ctx;
}

extern "C" std::string* std_string_alloc(char* c_str) {
    return new std::string(c_str);
}

extern "C" void std_string_free(std::string* str) {
    delete str;
}

bool has_layer(fairseq2_model& model, const std::string& name) {
    return model.tensors.find(name) != model.tensors.end();
}

ggml_tensor* mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    if (b->ne[1] == 1 && b->ne[2] > 1 && ggml_n_dims(a) == 2) {
        // `b` has shape (B, 1, D).
        // if `a` is (D_out, D), then we do one matmul for the full batch.
        b = ggml_flatten_1d(ctx, b, 1);
        return ggml_unflatten_1d(ctx, ggml_mul_mat(ctx, a, b), 1, 1);
    }
    // there is also the k * q matmul -> (D, 1, B) * (D, 1, B) -> (1, 1, B)
    // not sure what's the best way to compute this with BLAS

    return ggml_mul_mat(ctx, a, b);  // (d_out)
}


extern "C" ggml_tensor* Linear_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input  // (d_in)
) {
    // Note: for now we assumed un-batched input
    ggml_tensor* weight = model.tensors[prefix + ".weight"];  // (d_in, d_out)
    GGML_ASSERT(weight != nullptr);
    ggml_tensor* out = mul_mat(model.ctx, weight, input);  // (d_out)
    ggml_tensor* bias = model.tensors[prefix + ".bias"];  // (d_out)
    if (bias == nullptr) return out;

    return ggml_add(model.ctx, out, bias);
}

extern "C" ggml_tensor* LayerNorm_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input
) {
    ggml_tensor* weight = model.tensors[prefix + ".weight"];
    GGML_ASSERT(weight != nullptr);
    ggml_tensor* bias = model.tensors[prefix + ".bias"];
    GGML_ASSERT(bias != nullptr);

    auto ctx = model.ctx;
    double eps = model_layer_config_d(model, prefix + ".eps");

    input = ggml_norm(ctx, input, /*eps*/eps);
    return ggml_add_inplace(
        ctx,
        ggml_mul_inplace(ctx, ggml_repeat(ctx, weight, input), input),
        ggml_repeat(ctx, bias, input)
    );
}


extern "C" ggml_tensor* StandardFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    seqs = Linear_forward(model, prefix + ".inner_proj", seqs);
    // inner_activation = ReLu // TODO: allow other activation
    seqs = ggml_relu_inplace(model.ctx, seqs);

    if (has_layer(model, prefix + ".inner_layer_norm")) {
        seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    }

    seqs = Linear_forward(model, prefix + ".output_proj", seqs);
    return seqs;
}

extern "C" ggml_tensor* SiluFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    seqs = Linear_forward(model, prefix + ".inner_proj", seqs);
    seqs = ggml_silu(model.ctx, seqs);

    if (has_layer(model, prefix + ".inner_layer_norm")) {
        seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    }

    seqs = Linear_forward(model, prefix + ".output_proj", seqs);
    return seqs;
}

ggml_tensor* ggml_flatten_1d(ggml_context* ctx, ggml_tensor* x, int dim) {
    int n_dims = ggml_n_dims(x);
    GGML_ASSERT(dim >= 0);
    GGML_ASSERT(dim < n_dims);
    GGML_ASSERT(ggml_is_contiguous(x));
    // Nothing to do
    if (dim == n_dims - 1) return x;

    if (n_dims == 2) {
        return ggml_reshape_1d(ctx, x, x->ne[0] * x->ne[1]);
    } else if (n_dims == 3) {
        if (dim == 0) {
            return ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]);
        } else { // dim == 1
            return ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
        }
    } else { // n_dims == 4
        if (dim == 0) {
            return ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
        } else if (dim == 1) {
            return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);
        } else { // dim == 2
            return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
        }
    }
}

ggml_tensor* ggml_unflatten_1d(ggml_context* ctx, ggml_tensor* x, int dim, int num_el) {
    int n_dims = ggml_n_dims(x);
    GGML_ASSERT(dim >= 0);
    GGML_ASSERT(dim < n_dims);
    GGML_ASSERT(n_dims < 4);
    GGML_ASSERT(x->ne[dim] % num_el == 0);
    GGML_ASSERT(x->nb[dim + 1] == x->nb[dim] * x->ne[dim]);  // `x` isn't contiguous along `dim`
    if (n_dims == 1) {
        return ggml_view_2d(ctx, x, num_el, x->ne[0] / num_el, x->nb[0] * num_el, 0);
    } else if (n_dims == 2) {
        if (dim == 0) {
            return ggml_view_3d(ctx, x, num_el, x->ne[0] / num_el, x->ne[1], x->nb[0] * num_el, x->nb[1], 0);
        } else { // dim == 1
            return ggml_view_3d(ctx, x, x->ne[0], num_el, x->ne[1] / num_el, x->nb[1], num_el * x->nb[1], 0);
        }
    } else { // (n_dims == 3)
        if (dim == 0) {
            return ggml_view_4d(ctx, x, num_el, x->ne[0] / num_el, x->ne[1], x->ne[2], x->nb[0] * num_el, x->nb[1], x->nb[2], 0);
        } else if (dim == 1) {
            return ggml_view_4d(ctx, x, x->ne[0], num_el, x->ne[1] / num_el, x->ne[2], x->nb[1], num_el * x->nb[1], x->nb[2], 0);
        } else { // dim == 2
            return ggml_view_4d(ctx, x, x->ne[0], x->ne[1], num_el, x->ne[2] / num_el, x->nb[1], x->nb[2], num_el * x->nb[2], 0);
        }
    }
}


ggml_tensor* _reshape_num_head(ggml_context* ctx, ggml_tensor* x, int head_dim) {
    // (B, S, dim) -> (B, S, H, H_dim)
    x = ggml_unflatten_1d(ctx, x, 0, head_dim);
    x = ggml_permute(ctx, x, 0, 2, 1, 3); // (B, H, S, H_dim)
    x = ggml_cont(ctx, x);
    x = ggml_flatten_1d(ctx, x, 2);  // (B * H, S, H_dim)
    return x;
}

/// (B, Sk, dim) -> // (B?, H, H_dim, Sk)
ggml_tensor* _reshape_num_head_values(ggml_context* ctx, ggml_tensor* v, int head_dim ) {
    // (B, Sk, dim) -> (B, Sk, H, H_dim)
    v = ggml_unflatten_1d(ctx, v, 0, head_dim);
    v = ggml_permute(ctx, v, 1, 2, 0, 3);  // (B?, H, H_dim, Sk)
    v = ggml_cont(ctx, v);
    v = ggml_flatten_1d(ctx, v, 2);  // (B * H, S, H_dim)
    return v;
}


// flash_attn doesn't work for cross attention because it assumes Q <= K
// and it seems to yield slightly different scores than expected, and thus a different beam search
# define UNITY_FLASH_ATTN 0

extern "C" ggml_tensor* MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (slen, d_in)
    ggml_tensor* keys,  // (klen, d_in)
    ggml_tensor* values,  // (klen, d_out)
    ggml_tensor* attn_mask // (klen, slen)
) {
    int model_dim = queries->ne[0];
    int num_heads = model.layer_config.at(prefix + ".num_heads");
    int head_dim = model_dim / num_heads;
    GGML_ASSERT(model_dim % num_heads == 0);

    ggml_context* ctx = model.ctx;
    ggml_tensor* q = Linear_forward(model, prefix + ".q_proj", queries); // (B, S, H * H_dim)
    q = _reshape_num_head(ctx, q, head_dim);  // (B * H, S, H_dim)
    ggml_set_name(q, "q");

    ggml_tensor *k, *v;
    if (!has_kv_cache(model)) {
        k = Linear_forward(model, prefix + ".k_proj", keys);
        ggml_set_name(k, "k");
        v = Linear_forward(model, prefix + ".v_proj", values);
        ggml_set_name(v, "v");
    } else {
        bool encoder_decoder_attn = keys == values && keys != queries;
        if (encoder_decoder_attn) {
            // The K and V tensors of an encoder-decoder attention (i.e. the
            // projected encoder outputs) remain static during evaluation.

            KeyValueTensor& kv_cache = model.kv_cache[prefix];
            if (kv_cache.step_nr == 0) {
                // If possible we use the ctx dedicated to kv_cache here,
                // because the enc dec attention is typically long lived.
                if (model.enc_kv_cache_ctx) model.ctx = model.enc_kv_cache_ctx;
                k = Linear_forward(model, prefix + ".k_proj", keys);
                ggml_set_name(k, "k");
                v = Linear_forward(model, prefix + ".v_proj", values);
                ggml_set_name(v, "v");
                // Note we are only storing a pointer to the buffer, not the full graph
                kv_cache.full_k = ggml_detach(ggml_dup_inplace(model.ctx, k));
                ggml_format_name(kv_cache.full_k, "%s.k_cache", prefix.c_str());
                kv_cache.full_v = ggml_detach(ggml_dup_inplace(model.ctx, v));
                ggml_format_name(kv_cache.full_v, "%s.v_cache", prefix.c_str());
                kv_cache.step_nr = keys->ne[1];
                model.ctx = ctx;
            } else {
                k = kv_cache.full_k;
                v = kv_cache.full_v;
                GGML_ASSERT(keys->ne[1] == k->ne[1]);  // cache content doesn't match the input sequence
                GGML_ASSERT(values->ne[1] == v->ne[1]); // cache content doesn't match the input sequence
            }
        } else { // self attention
            // (1, K) -> (N, 1, K_proj)
            k = Linear_forward(model, prefix + ".k_proj", keys);
            ggml_set_name(k, "k");
            // (1, V) -> (N, 1, V_proj)
            v = Linear_forward(model, prefix + ".v_proj", values);
            ggml_set_name(v, "v");

            append_to_prev_kv(model, prefix, &k, &v, &attn_mask);
        }
    }
    k = _reshape_num_head(ctx, k, head_dim);  // (B * H, Sk, H_dim)
    v = _reshape_num_head_values(ctx, v, head_dim); // (B * H, H_dim, Sk)
    v = ggml_cont(ctx, v);

#if UNITY_FLASH_ATTN
    // For flash_attn, we assume either no masks, or triangular masks.
    ggml_tensor* attn = ggml_flash_attn(ctx, q, k, v, /*masked*/attn_mask != nullptr);  // (B * H, S, H_dim)
    ggml_set_name(attn, "attn");
    attn = ggml_unflatten_1d(ctx, attn, 2, num_heads);  // (B, H, H_dim, S)
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3); // (B, S, H, H_dim)
#else
    // (B * H, Sk, H_dim) x (B * H, S, H_dim) -> (B * H, S, Sk)
    ggml_tensor* qk = mul_mat(ctx, k, q);
    ggml_set_name(qk, "qk");
    qk = ggml_scale(ctx, qk, 1.0f/sqrtf(float(head_dim)));
    ggml_set_name(qk, "qk_scaled");

    if (attn_mask) qk = ggml_add_inplace(ctx, qk, attn_mask);
    // TODO: upgrade qk to float32 if needed
    ggml_tensor* attn_weights = ggml_soft_max(ctx, qk);  // (B * H, S, Sk)
    ggml_set_name(attn_weights, "attn_weights");

    // (B * H, S, Sk) x (B * H, H_dim, Sk) -> (B * H, H_dim, S)
    ggml_tensor* attn = mul_mat(ctx, attn_weights, v);
    ggml_set_name(attn, "attn");
    attn = ggml_unflatten_1d(ctx, attn, 2, num_heads);  // (B, H, H_dim, S)
    attn = ggml_permute(ctx, attn, 2, 0, 1, 3); // (B, S, H, H_dim)
#endif  // UNITY_FLASH_ATTN
    attn = ggml_cont(ctx, attn);
    attn = ggml_flatten_1d(ctx, attn, 0); // (B, S, H * H_dim)
    // out -> (B, S, d_out)
    ggml_tensor* out = Linear_forward(model, prefix + ".output_proj", attn);
    ggml_set_name(out, "out");

    return out;
}


extern "C" ggml_tensor* StandardTransformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    auto norm_order = model.layer_config.at(prefix + ".norm_order");

    // _forward_self_attn(seqs, padding_mask)
    auto residual = seqs;
    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // TODO: add padding_mask to MultiheadAttention_forward
    GGML_ASSERT(padding_mask == nullptr);
    seqs = MultiheadAttention_forward(
        model,
        prefix + ".self_attn",
        seqs,
        seqs,
        seqs,
        /*attn_mask=*/nullptr
    );

    if (has_layer(model, prefix + ".self_attn_norm"))
        seqs = LayerNorm_forward(model, prefix + ".self_attn_norm", seqs);

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // _forward_ffn(seqs)
    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);

    // TODO: if self.residual_scale is not None:
    // residual = self.residual_scale * residual

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* WaveformToFbank_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* waveform
) {
    // Hardcoding: num_bins 80, sample rate 16k, always standardize
    ggml_context* ctx = model.ctx;
    knf::MelBanksOptions mel_opts{};
    mel_opts.num_bins = 80;

    knf::FrameExtractionOptions frame_opts{};
    frame_opts.samp_freq = 16000;

    knf::FbankOptions opts{};
    opts.frame_opts = frame_opts;
    opts.mel_opts = mel_opts;


    std::vector<float_t> signal_frame{};
    std::int32_t num_frames = knf::NumFrames(/*num_samples=*/waveform->ne[0], frame_opts);
    FORCE_ALLOC(output, ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 80, num_frames));
    knf::FbankComputer native_(opts);
    knf::FeatureWindowFunction window_fn_(native_.GetFrameOptions());

    for (std::int32_t frame_nr = 0; frame_nr < num_frames; ++frame_nr) {
        signal_frame.resize(0);

        // Extract the frame from the waveform tensor.
        knf::ExtractWindow(
            /*sample_offset=*/0,
            (float *)(waveform->data),
            waveform->ne[0],
            frame_nr,
            frame_opts,
            window_fn_,
            &signal_frame);

        native_.Compute(
            /*signal_raw_log_energy=*/0, /*vtln_warp=*/1.0, &signal_frame, ((float *)(output->data) + frame_nr * 80));
    }
    output = ggml_dup(ctx, ggml_transpose(ctx, output));
    output = ggml_norm(ctx, output, 1e-5);
    output = ggml_dup(ctx, ggml_transpose(ctx, output));
    if (output->ne[1] % 2 == 1) {
        output = ggml_dup(ctx, ggml_slice(ctx, output, 1, 0, output->ne[1]-1));
    }
    output = ggml_reshape_2d(ctx, output, output->ne[0] * 2, output->ne[1] / 2);
    return output;
}

// TODO: Check if it's possible to merge with standard MHA
extern "C" ggml_tensor* RelativePositionMHA_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    ggml_context* ctx = model.ctx;

    ggml_tensor* residual = seqs;
    seqs = LayerNorm_forward(model, prefix + "_layer_norm", seqs);
    // self_attn: qkv
    ggml_tensor* Qcur = Linear_forward(model, prefix + ".q_proj", seqs);
    ggml_tensor* Kcur = Linear_forward(model, prefix + ".k_proj", seqs);
    ggml_tensor* Vcur = Linear_forward(model, prefix + ".v_proj", seqs);

    // self_attn: rel_pos SDPA
    int32_t S = seqs->ne[1];
    int32_t H = 16; // TODO: Make this configurable
    int32_t n_ctx = 4096;
    int32_t K_h = seqs->ne[0] / H;

    int32_t start_index = n_ctx - S;
    int32_t end_index = n_ctx + S - 1;

    int num_indices = end_index - start_index;

    FORCE_ALLOC(rows, ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_indices));
    for (int i = 0; i < num_indices; i++) {
        ((int32_t *)rows->data)[i] = start_index + i;
    }

    // self_attn: load pos_enc weights & compute_r
    // In fairseq2 pos_enc weights are calculated on the fly, since some more custom operators might be needed to enable this,
    // we store the results (fixed) in checkpoint as model.audio_enc_pos_enc_w and load directly.
    ggml_tensor* r = ggml_get_rows(ctx, model.tensors["speech_encoder.pos_enc"], rows);
    r = mul_mat(ctx, model.tensors[prefix + ".sdpa.r_proj.weight"], r);
    r = ggml_dup(ctx, ggml_permute(ctx, ggml_unflatten_1d(ctx, r, 0, K_h), 0, 2, 1, 3));

    ggml_tensor* u_bias = ggml_reshape_3d(ctx, model.tensors[prefix + ".sdpa.u_bias"], K_h, 1, H);
    ggml_tensor* v_bias = ggml_reshape_3d(ctx, model.tensors[prefix + ".sdpa.v_bias"], K_h, 1, H);

    // self_attn: Permute QKV

    // (H * K_h, S) -> (K_h, H, S) -> (K_h, S, H)
    ggml_tensor* Q = ggml_cont(ctx, ggml_permute(ctx, ggml_unflatten_1d(ctx, Qcur, 0, K_h), 0, 2, 1, 3));
    // (H * K_h, S) -> (K_h, H, S) -> (K_h, S, H)
    ggml_tensor* K = ggml_cont(ctx, ggml_permute(ctx, ggml_unflatten_1d(ctx, Kcur, 0, K_h), 0, 2, 1, 3));
    // (H * K_h, S) -> (K_h, H, S) -> (H, S, K_h)
    ggml_tensor* V = ggml_cont(ctx, ggml_permute(ctx, ggml_unflatten_1d(ctx, Vcur, 0, K_h), 1, 2, 0, 3));


    ggml_tensor* q_with_u_bias = ggml_add_inplace(ctx, ggml_dup(ctx, Q), u_bias); // (K_h, S, H)
    ggml_tensor* q_with_v_bias = ggml_add_inplace(ctx, Q, v_bias); // (K_h, S, H)

    ggml_tensor* ac = mul_mat(ctx, K, q_with_u_bias);
    ggml_tensor* bd = mul_mat(ctx, r, q_with_v_bias);

    // self_attn: shift_bd. Logic follows https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/nn/transformer/relative_attention.py#L161
    bd = ggml_dup(ctx, ggml_permute(ctx, bd, 2, 1, 0, 3)); // H, S, 2S-1

    FORCE_ALLOC(pad, ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, S, 1));
    pad = ggml_set_f32(pad, 0.0);

    bd = ggml_concat(ctx, pad, bd); // bd[i][j][0] == 0, (H, S, 2S)
    bd = ggml_dup(ctx, ggml_permute(ctx, bd, 2, 1, 0, 3)); // (2S, S, H)
    bd = ggml_reshape_3d(ctx, bd, S, 2 * S, H);  // (S, 2S, H)
    // discard the first set of positive positions
    bd = ggml_dup(ctx, ggml_slice(ctx, bd, 1, 1, 2 * S));
    // shifts each row by an extra step
    bd = ggml_reshape_3d(ctx, bd, 2 * S - 1, S, H);
    // Discard positions used for shift.
    bd = ggml_slice(ctx, bd, 0, 0, S);

    // self_attn: compute attn / weights
    ggml_tensor* attn_weights = ggml_add_inplace(ctx, ac, bd);
    FORCE_ALLOC(attn_scale, ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1));
    ggml_set_f32(attn_scale, 1.0 / pow(K_h, 0.5));
    attn_weights = ggml_mul_inplace(ctx, attn_weights, ggml_repeat(ctx, attn_scale, attn_weights));
    attn_weights = ggml_soft_max(ctx, attn_weights);

    ggml_tensor* attn = mul_mat(ctx, V, attn_weights); // K_h, S, H
    attn = ggml_dup(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    ggml_tensor* attn_2d = ggml_reshape_2d(ctx, attn, K_h * H, S);

    ggml_tensor* attn_out = mul_mat(ctx, model.tensors[prefix + ".output_proj.weight"], attn_2d);
    attn_out = ggml_add_inplace(
        ctx,
        attn_out,
        ggml_repeat(ctx, model.tensors[prefix + ".output_proj.bias"], attn_out)
    );
    attn_out = ggml_add_inplace(ctx, attn_out, residual);
    return attn_out;
}

extern "C" ggml_tensor* ConvModule_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
        ggml_context* ctx = model.ctx;
        ggml_tensor* residual = seqs;
        seqs = LayerNorm_forward(model, prefix + "_layer_norm", seqs);
        // conv: Use matmul for pointwise conv 1 - kernel_size=1, no padding case
        seqs = mul_mat(ctx, model.tensors[prefix + ".pointwise_conv1.weight"], seqs);

        // conv: GLU
        seqs = ggml_glu(ctx, seqs);
        seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));

        // S x C -> (S+K-1) x C -> K x S x C -> S x C
        int K = model.tensors[prefix + ".depthwise_conv.weight"]->ne[0];

        seqs = ggml_depthwise_conv(ctx, model.tensors[prefix + ".depthwise_conv.weight"], seqs, K / 2);

        // conv: Custom implementation of batch norm
        seqs = ggml_batch_norm(ctx, seqs, model.tensors[prefix + ".batch_norm.weight"], model.tensors[prefix + ".batch_norm.bias"], model.tensors[prefix + ".batch_norm.running_mean"], model.tensors[prefix + ".batch_norm.running_var"], 1e-5);

        // conv: SiLU actvation
        seqs = ggml_silu_inplace(ctx, seqs);
        seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));

        // conv: Use matmul for pointwise conv 2 - kernel_size=1, no padding case
        seqs = mul_mat(ctx, model.tensors[prefix + ".pointwise_conv2.weight"], seqs);

        // conv: + residual
        seqs = ggml_add_inplace(ctx, seqs, residual);
        return seqs;
}

extern "C" ggml_tensor* StandardConformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    FORCE_ALLOC(ffn_scale, ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1));
    ggml_set_f32(ffn_scale, 0.5f);
    ggml_tensor* residual = seqs;
    seqs = LayerNorm_forward(model, prefix + ".ffn1_layer_norm", seqs);
    seqs = SiluFeedForwardNetwork_forward(model, prefix + ".ffn1", seqs);
    seqs = ggml_mul_inplace(ctx, seqs, ggml_repeat(ctx, ffn_scale, seqs));
    seqs = ggml_add_inplace(ctx, seqs, residual);
    seqs = RelativePositionMHA_forward(model, prefix + ".self_attn", seqs);
    seqs = ConvModule_forward(model, prefix + ".conv", seqs);
    residual = seqs;
    seqs = LayerNorm_forward(model, prefix + ".ffn2_layer_norm", seqs);
    seqs = SiluFeedForwardNetwork_forward(model, prefix + ".ffn2", seqs);
    seqs = ggml_mul_inplace(ctx, seqs, ggml_repeat(ctx, ffn_scale, seqs));
    seqs = ggml_add_inplace(ctx, seqs, residual);
    seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);
    return seqs;
}

extern "C" ggml_tensor* StandardConformerEncoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    seqs = WaveformToFbank_forward(model, prefix, seqs);
    seqs = LayerNorm_forward(model, prefix + "_frontend.post_extract_layer_norm", seqs);
    seqs = Linear_forward(model, prefix + "_frontend.model_dim_proj", seqs);
    int layer_idx = 0;

    std::string layer_name = prefix + ".inner.layers." + std::to_string(layer_idx);

    while (has_layer(model, layer_name)) {
        seqs = StandardConformerEncoderLayer_forward(
            model, layer_name, seqs, padding_mask
        );
        ggml_set_name(seqs, ("x_enc_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".inner.layers." + std::to_string(layer_idx);
    }

    seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    ggml_tensor* residual = seqs;
    seqs = Linear_forward(model, prefix + ".proj1", seqs);
    seqs = ggml_relu_inplace(ctx, seqs);
    seqs = Linear_forward(model, prefix + ".proj2", seqs);
    FORCE_ALLOC(ffn_scale, ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1));
    ggml_set_f32(ffn_scale, 0.5f);
    seqs = ggml_mul(ctx, ggml_repeat(ctx, ffn_scale, seqs), seqs);
    seqs = ggml_add_inplace(ctx, seqs, residual);
    layer_idx = 0;
    layer_name = prefix + ".adaptor_layers." + std::to_string(layer_idx);
    while (has_layer(model, layer_name)) {
        seqs = StandardConformerEncoderAdaptorLayer_forward(
            model, layer_name, seqs, padding_mask
        );
        ggml_set_name(seqs, ("x_ada_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".adaptor_layers." + std::to_string(layer_idx);
    }
    seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* StandardConformerEncoderAdaptorLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    ggml_tensor* residual = seqs;
    residual = LayerNorm_forward(model, prefix + ".residual_layer_norm", residual);
    residual = ggml_dup(ctx, ggml_permute(ctx, residual, 1, 0, 2, 3));
    ggml_tensor* residual_conv_weight = model.tensors[prefix + ".residual_conv.weight"];
    // ggml_tensor* from = model.tensors[prefix + ".residual_conv.weight"];
    // FORCE_ALLOC(residual_conv_weight, ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, from->ne[0], from->ne[1], from->ne[2]));
    // ggml_fp32_to_fp16_row((float*)model.tensors[prefix + ".residual_conv.weight"]->data, (ggml_fp16_t*)residual_conv_weight->data, from->ne[0] * from->ne[1] * from->ne[2]);
    residual = ggml_conv_1d(ctx, residual_conv_weight, residual, 8, 4, 1);
    residual = ggml_dup(ctx, ggml_permute(ctx, residual, 1, 0, 2, 3));
    residual = ggml_add_inplace(ctx, ggml_repeat(ctx, model.tensors[prefix + ".residual_conv.bias"], residual), residual);
    residual = ggml_glu(ctx, residual);

    seqs = LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);
    seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));
    ggml_tensor* self_attn_conv_weight = model.tensors[prefix + ".self_attn_conv.weight"];
    // from = model.tensors[prefix + ".self_attn_conv.weight"];
    // FORCE_ALLOC(self_attn_conv_weight, ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, from->ne[0], from->ne[1], from->ne[2]));
    // ggml_fp32_to_fp16_row((float*)model.tensors[prefix + ".self_attn_conv.weight"]->data, (ggml_fp16_t*)residual_conv_weight->data, from->ne[0] * from->ne[1] * from->ne[2]);
    seqs = ggml_conv_1d(ctx, self_attn_conv_weight, seqs, 8, 4, 1);
    seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));
    seqs = ggml_add_inplace(ctx, seqs, ggml_repeat(ctx, model.tensors[prefix + ".self_attn_conv.bias"], seqs));
    seqs = ggml_glu(ctx, seqs);

    seqs = MultiheadAttention_forward(
        model,
        prefix + ".self_attn",
        seqs,
        seqs,
        seqs,
        /*attention masks=*/nullptr
    );
    seqs = ggml_add_inplace(ctx, seqs, residual);
    residual = seqs;
    seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);
    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);
    seqs = ggml_add_inplace(ctx, seqs, residual);
    return seqs;
}


/// ggml_slice(X, -1, start, end) is equivalent to X[start:end]
/// ggml_slice(X, 0, start, end) is equivalent to X[..., start:end]
ggml_tensor* ggml_slice(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    int axis,
    int64_t start,
    int64_t end
) {
    int64_t ne[4];
    std::copy(a->ne, a->ne + 4, ne);
    if (axis < 0) axis = ggml_n_dims(a) + axis;
    if (start < 0) start = ne[axis] + start;
    if (end <= 0) end = ne[axis] + end;
    GGML_ASSERT(0 <= start);
    GGML_ASSERT(start < end);
    GGML_ASSERT(end <= ne[axis]);


    ne[axis] = end - start;
    std::size_t offset = a->nb[axis] * start;

    std::size_t* nb = a->nb;
    ggml_tensor* result = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
    ggml_format_name(result, "%s [(%d)%ld:%ld]", a->name, axis, start, end);
    return result;
}

ggml_tensor* ggml_select(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    int axis,
    int64_t index
) {
    int64_t ne[GGML_MAX_DIMS];
    std::copy(a->ne, a->ne + GGML_MAX_DIMS, ne);

    if (axis < 0) axis = ggml_n_dims(a) + axis;
    if (index < 0) index = ne[axis] + index;
    GGML_ASSERT(0 <= index);
    GGML_ASSERT(index < ne[axis]);

    std::copy(a->ne + axis + 1, a->ne + GGML_MAX_DIMS, ne + axis);

    std::size_t offset = a->nb[axis] * index;
    std::size_t* nb = a->nb;
    GGML_ASSERT(GGML_MAX_DIMS == 4);
    ggml_tensor* result = ggml_view_3d(ctx, a, ne[0], ne[1], ne[2], nb[1], nb[2], offset);
    ggml_format_name(result, "%s [(%d)%ld]", a->name, axis, index);
    return result;
}


// Inplace computation of PositionalEmbedding
extern "C" ggml_tensor* PositionalEmbedding_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* embeds
) {
    // This only work with the simple pos encoders
    int seq_len = embeds->ne[1];
    ggml_tensor* full_pos_embeds = model.tensors[prefix];

    int start_step = 0;
    if (has_kv_cache(model)) {
        start_step = model.kv_cache[prefix].step_nr++;
    }
    ggml_tensor* pos_embeds = ggml_slice(model.ctx, full_pos_embeds, /*axis*/1, start_step, seq_len + start_step);
    return ggml_add(model.ctx, embeds, pos_embeds);
}

extern "C" ggml_tensor* TransformerEmbeddingFrontend_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    GGML_ASSERT(ggml_n_dims(seqs) < GGML_MAX_DIMS);
    ggml_context* ctx = model.ctx;
    ggml_tensor* embed_weights = model.tensors[prefix + ".embed.weight"];
    GGML_ASSERT(embed_weights != nullptr);
    ggml_tensor* embeds;
    if (ggml_n_dims(seqs) == 1) {
        embeds = ggml_get_rows(ctx, embed_weights, seqs);
    } else {
        // ggml_get_rows isn't very flexible, we have to handle the reshape ourselves.
        ggml_tensor* flat_seqs = seqs;
        if (!ggml_is_contiguous(seqs)) {
            flat_seqs = ggml_cont(ctx, flat_seqs);
        }
        flat_seqs = ggml_reshape_1d(ctx, flat_seqs, ggml_nelements(seqs));
        embeds = ggml_get_rows(ctx, embed_weights, flat_seqs);
        embeds = ggml_reshape_4d(ctx, embeds, embed_weights->ne[0], seqs->ne[0], seqs->ne[1], seqs->ne[2]);
    }

    // padding mask ?
    // padding_mask = to_padding_mask(embeds, seq_lens)

    if (has_layer(model, prefix + ".pos_encoder")) {
        embeds = PositionalEmbedding_forward(model, prefix + ".pos_encoder", embeds);
    }

    if (has_layer(model, prefix + ".layer_norm")) {
        embeds = LayerNorm_forward(model, prefix + ".layer_norm", embeds);
    }

    return embeds;
}

extern "C" ggml_tensor* StandardTransformerEncoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    int layer_idx = 0;
    std::string layer_name = prefix + ".layers." + std::to_string(layer_idx);
    while (has_layer(model, layer_name)) {
        seqs = StandardTransformerEncoderLayer_forward(
            model, layer_name, seqs, padding_mask
        );

        ggml_set_name(seqs, ("x_enc_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".layers." + std::to_string(layer_idx);
    }

    if (has_layer(model, prefix + ".layer_norm"))
        seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* StandardTransformerDecoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* self_attn_mask,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask
) {
    ggml_context* ctx = model.ctx;
    auto norm_order = model.layer_config.at(prefix + ".norm_order");

    // _forward_self_attn(seqs, padding_mask)
    auto residual = seqs;
    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    seqs = MultiheadAttention_forward(
        model,
        prefix + ".self_attn",
        seqs,
        seqs,
        seqs,
        /*attn_mask=*/self_attn_mask
    );

    if (has_layer(model, prefix + ".self_attn_norm"))
        seqs = LayerNorm_forward(model, prefix + ".self_attn_norm", seqs);

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // _forward_encoder_decoder_attn
    if (! has_layer(model, prefix + ".encoder_decoder_attn")) {
        // `encoder_output` must be `None` for decoder-only attention.
        GGML_ASSERT(encoder_output == nullptr);
        return seqs;
    }

    // `encoder_output` must not be `None` for encoder-decoder attention.
    GGML_ASSERT(encoder_output != nullptr);

    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".encoder_decoder_attn_layer_norm", seqs);


    seqs = MultiheadAttention_forward(
        model,
        prefix + ".encoder_decoder_attn",
        seqs,
        encoder_output,
        encoder_output,
        /*attention masks=*/encoder_padding_mask
    );

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".encoder_decoder_attn_layer_norm", seqs);

    // _forward_ffn(seqs)
    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);

    // TODO:
    // if self.residual_scale is not None:
    // residual = self.residual_scale * residual

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* causal_attention_mask(ggml_context* ctx, ggml_tensor* seqs) {
    auto seq_len = seqs->ne[1];
    // TODO: allow other ggml_type
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
    return ggml_diag_mask_inf(ctx, mask, 0);
}

extern "C" ggml_tensor* StandardTransformerDecoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask
) {
    int layer_idx = 0;
    std::string layer_name = prefix + ".layers." + std::to_string(layer_idx);
    ggml_tensor* self_attn_mask = causal_attention_mask(model.ctx, seqs);
    while (has_layer(model, layer_name)) {
        seqs = StandardTransformerDecoderLayer_forward(
            model, layer_name, seqs, self_attn_mask, encoder_output, encoder_padding_mask
        );

        ggml_set_name(seqs, ("x_dec_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".layers." + std::to_string(layer_idx);
    }

    if (has_layer(model, prefix + ".layer_norm"))
        seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);

    return seqs;
}


int _determine_max_seq_len(const SequenceGeneratorJob& job, int source_seq_len) {
    auto opts = job.opts;
    int max_seq_len = -1;
    if (source_seq_len <= 0 || opts.soft_max_seq_len_a <= 0) {
        max_seq_len = opts.hard_max_seq_len;
    } else {
        max_seq_len = std::min(opts.hard_max_seq_len, int(opts.soft_max_seq_len_a * source_seq_len) + opts.soft_max_seq_len_b);
    }

    if (opts.min_seq_len > max_seq_len) {
        printf(
            "The effective maximum sequence length must be greater than or equal to `min_seq_len` (%d), but is %d instead. Adjust your soft and hard maximum sequence length limits.\n",
            opts.min_seq_len,
            max_seq_len
        );
        GGML_ASSERT(opts.min_seq_len <= max_seq_len);
    }

    int prefix_seq_len = job.prefix_seq->ne[0];
    if (prefix_seq_len >= max_seq_len) {
        printf(
            "The effective maximum sequence length must be greater than `prefix_seq_len` (%d), but is %d instead.\n",
            prefix_seq_len,
            max_seq_len
        );
        GGML_ASSERT(prefix_seq_len < max_seq_len);
    }

    return max_seq_len;
}

void _fan_out_encoder_output(
    ggml_context* ctx,
    ggml_tensor** encoder_output_out,
    ggml_tensor** encoder_padding_mask_out,
    int beam_size
) {
    // (S_enc, M)
    ggml_tensor* encoder_output = *encoder_output_out;
    ggml_tensor* encoder_padding_mask = *encoder_padding_mask_out;

    // (B, S_enc, M)
    ggml_tensor* shape = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, encoder_output->ne[0], encoder_output->ne[1], beam_size);
    // (S_enc, M) -> (B, S_enc, M)
    *encoder_output_out = ggml_repeat(ctx, encoder_output, shape);
    // (S_enc) -> (B, S_enc)
    if (encoder_padding_mask != nullptr) {
        ggml_tensor* shape_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, encoder_padding_mask->ne[0], 1, beam_size);
        *encoder_padding_mask_out = ggml_repeat(ctx, encoder_padding_mask, shape_mask);
    }
}

ggml_tensor* ggml_log_softmax(ggml_context* ctx, ggml_tensor* logits) {
    // TODO: this isn't the most precise way of doing this
    return ggml_log_inplace(ctx, ggml_soft_max_inplace(ctx, logits));
}

ggml_tensor* ggml_expand_2d(ggml_context* ctx, ggml_tensor* x, int64_t ne0, int64_t ne1) {
    ggml_tensor* shape = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, ne0, ne1);
    ggml_type true_type = x->type;
    ggml_tensor* y = ggml_repeat(ctx, x, shape);
    y->type = true_type;
    return y;
}

void _bootstrap_seqs_and_scores(
    fairseq2_model& model,
    const SequenceGeneratorJob& job,
    ggml_tensor* full_seqs,
    ggml_tensor* scores,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask,
    ggml_tensor* lid_scores,
    int n_threads,
    const std::vector<int>& lang_ids
) {
    // Returns LID score map
    int prefix_seq_len = job.prefix_seq->ne[0];
    int max_seq_len = scores->ne[0];
    int beam_size = scores->ne[1];
    GGML_ASSERT(prefix_seq_len > 0);
    ggml_context* ctx = model.ctx;
    if (prefix_seq_len == 1) {
        // We only have one token in prefix, we won't compute decoding scores,
        // we just need to copy the token to seqs.
        // Note: it also means the enc_kv_cache will be populated later.
        ggml_tensor* seqs = ggml_slice(ctx, full_seqs, 0, 0, prefix_seq_len);
        ggml_set_i32(seqs, ggml_get_i32_1d(job.prefix_seq, 0));
        return;
    }

    // full_seqs[:, : prefix_seq_len] = job.prefix_seq;
    ggml_tensor* seqs = ggml_slice(ctx, full_seqs, 0, 0, prefix_seq_len);
    seqs = ggml_cpy(ctx, ggml_repeat(ctx, job.prefix_seq, seqs), seqs);

    // We have to bootstrap the model with the already fanned-out encoder
    // output to correctly initialize its incremental state.
    // Note: we don't start decoding the last prefix token just yet.
    seqs = ggml_slice(ctx, seqs, 0, 0, prefix_seq_len - 1);

    // Bootstrap the model state with prefix sequence.
    seqs = TransformerEmbeddingFrontend_forward(model, "text_decoder_frontend", seqs);
    ggml_tensor* decoder_output = StandardTransformerDecoder_forward(
        model,
        "text_decoder",
        seqs,
        /*padding_mask*/ nullptr,
        encoder_output,
        encoder_padding_mask
    );

    // logits, lprobs: (N, S_pfx - 1, V)
    ggml_tensor* logits = Linear_forward(model, "final_proj", decoder_output);
    int vocab_size = logits->ne[0];
    ggml_tensor* lprobs = ggml_log_softmax(ctx, ggml_slice(ctx, logits, 1, 0, 1));
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, lprobs);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    full_seqs->type = GGML_TYPE_I32;
    job.prefix_seq->type = GGML_TYPE_I32;
    // For LID
    for (std::size_t i = 0; i < lang_ids.size(); ++i) {
        ggml_set_f32_1d(lid_scores, i, std::exp(ggml_get_f32_1d(lprobs, lang_ids[i])));
    }

    // Fetch scores of next steps from "lprobs"
    float p_score = 0;
    for (int i = 1; i < prefix_seq_len; ++i) {
        int p = 0;
        if (ggml_get_i32_1d(job.prefix_seq, i) == model.vocab.token_to_id["<unk>"]) {
            // If tgt_lang is unk, use the most probable lang tag predicted by model
            int max_value = std::numeric_limits<float>::min();
            for (std::size_t j = 0; j < lang_ids.size(); j++) {
                if(ggml_get_f32_1d(lprobs, lang_ids[j]) > max_value) {
                    max_value = ggml_get_f32_1d(lprobs, lang_ids[j]);
                    p = lang_ids[j];
                }
            }
        } else {
            p = ggml_get_i32_1d(job.prefix_seq, i);
        }
        p_score += ggml_get_f32_1d(lprobs, i * vocab_size + p);
        for (int b = 0; b < beam_size; ++b) {
            // scores: (N, S)
            // Note: First step (e.g. BOS)'s score is always 0.
            ggml_set_f32_1d(scores, b * max_seq_len + i, p_score);
        }
    }
}

/// Finds the topk indices, and write the winning indices in "candidate_indices" array.
int topk(
    ggml_tensor* lprobs,  // (B, V)
    std::int64_t k,
    ggml_tensor* candidate_indices
) {
    // Take the best 2 x `beam_size` predictions. We'll choose the first
    // `beam_size` of these which don't predict EOS to continue with.
    // (N, 2 x B)
    // `vocab_size` - 1 to never select PAD.
    std::int64_t K = std::min(k, ggml_nelements(lprobs));
    auto comp = [lprobs](std::int32_t a, std::int32_t b) {
        return ggml_get_f32_1d(lprobs, a) > ggml_get_f32_1d(lprobs, b);
    };
    GGML_ASSERT(ggml_nelements(candidate_indices) >= k);
    auto cand = (std::int32_t*)candidate_indices->data;
    std::partial_sort(cand, cand + K, cand + ggml_nelements(lprobs), comp);

    return K;
}

void _tweak_lprobs(const SequenceGeneratorJob& job, ggml_tensor* lprobs, int step_nr, int max_seq_len, std::size_t vocab_size) {
    std::size_t beam_size = job.opts.beam_size;
    std::size_t eos_idx = job.eos_idx;

    // Do not allow EOS before reaching the minimum sequence length.
    if (step_nr < job.opts.min_seq_len) {
        // lprobs[:, :, self.eos_idx] = -INFINITY;
        for (std::size_t i = 0; i < beam_size; ++i)
            ggml_set_f32_1d(lprobs, vocab_size * i + eos_idx, -INFINITY);
    }

    // If we have reached the maximum length, force the last step to be EOS.
    if (step_nr == max_seq_len - 2) {
        // lprobs[:, :, : self.eos_idx]       = -torch.inf
        // lprobs[:, :,   self.eos_idx + 1 :] = -torch.inf
        for (std::size_t b = 0; b < beam_size; ++b) {
            std::size_t t = 0;
            for (t = 0; t < eos_idx; ++t)
                ggml_set_f32_1d(lprobs, vocab_size * b + t, -INFINITY);
            for (t = eos_idx + 1; t < vocab_size; ++t)
                ggml_set_f32_1d(lprobs, vocab_size * b + t, -INFINITY);
        }
    }

    // Never allow PAD.
    std::size_t pad_idx = job.pad_idx;
    for (std::size_t i = 0; i < beam_size; ++i)
        ggml_set_f32_1d(lprobs, vocab_size * i + pad_idx, -INFINITY);

    // Apply UNK penalty.
    if (job.unk_idx >= 0 && job.opts.unk_penalty != 0) {
        // lprobs[:, :, self.unk_idx] -= self.opts.unk_penalty
        auto lprobs_raw = ggml_get_data_f32(lprobs);
        for (std::size_t i = 0; i < beam_size; ++i)
            lprobs_raw[vocab_size * i + job.unk_idx] -= job.opts.unk_penalty;
    }
}



/// Copies the sequence and scores of a given candidate beam.
void _finalize_hypothesis(
    const SequenceGeneratorJob& job,
    ggml_context* ctx,
    int step_nr,
    std::int32_t beam,
    std::int32_t token,
    float eos_score,
    ggml_tensor* seqs, // (beam_size, seq_len)
    ggml_tensor* scores, // (beam_size, seq_len)
    ggml_tensor* lid_scores,
    Hypothesis* hypothesis
) {
    ggml_tensor* seq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, step_nr + 2);
    hypothesis->seq = seq;
    ggml_tensor* step_scores = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, step_nr + 2);
    hypothesis->step_scores = step_scores;

    auto tok = (std::int32_t*)seq->data;
    for (int i = 0; i < step_nr + 1; ++i) {
        tok[i] = ggml_get_i32_1d(seqs, seqs->ne[0] * beam + i);
    }
    tok[step_nr + 1] = token;

    // Convert from cumulative to per-step scores.
    auto sc = (float*)step_scores->data;
    float last_score = eos_score;
    for (int i = step_nr; i >= 0; --i) {
        float sc0 = ggml_get_f32_1d(scores, scores->ne[0] * beam + i);
        sc[i + 1] = last_score - sc0;
        last_score = sc0;
    }
    sc[0] = 0;

    if (job.opts.normalize_scores)
        // Skip first EOS since it is always 0 and skews normalization.
        eos_score /= (float)std::pow((step_nr + 1), job.opts.len_penalty);
    hypothesis->score = eos_score;
    hypothesis->lid_scores = lid_scores;
}

// Uses ggml_context to store any object.
#define GGML_CTX_ALLOC(ctx, Type, n) \
    (Type*)(ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Type) * n)->data);


ggml_context* ctx_from_buffer(std::vector<uint8_t>& buffer) {
    return ggml_init({
        /*.mem_size   =*/ static_cast<std::size_t>(buffer.capacity()),
        /*.mem_buffer =*/ buffer.data(),
        /*.no_alloc   =*/ false,
    });
}

ggml_allocr* new_arena_allocr(std::vector<uint8_t>& buffer) {
    return ggml_allocr_new(buffer.data(), buffer.capacity(), 8);
}



/// Generates a translation for a single sequence
/// The results Hypothesis are written inside `result_ctx`.
extern "C" Hypothesis* generate_sequence(
    fairseq2_model& model,
    const SequenceGeneratorJob& job,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask,
    ggml_context* result_ctx,
    int n_threads
) {
    // Pre allocate memory buffers.
    // * step_ctx: contains metadata for the model graph, as well as some explicit
    // buffers for the lprobs tweaking.
    // * prev_step_ctx: is an additional buffer because we need some results from previous steps,
    // to compute next step. Notably self attention kv cache.
    // * search_ctx contains tensors that should live for the full search,
    // like encoder kv cache.
    // * step_alloc contains buffer for the forward pass of the model.
    // Split mem_mb into the different context we need to use.
    int mem_mb = job.opts.mem_mb;
    std::vector<uint8_t> local_bufs[4] = {
        std::vector<uint8_t>(mem_mb * MB * 3 / 10),  // step_ctx
        std::vector<uint8_t>(mem_mb * MB * 3 / 10),  // prev_step_ctx
        std::vector<uint8_t>(mem_mb * MB * 3 / 10),  // search_ctx
        std::vector<uint8_t>(mem_mb * MB * 1 / 10),  // step_alloc
    };
    ggml_allocr* step_alloc = new_arena_allocr(local_bufs[3]);

    std::vector<int> lang_ids;
    if (job.prefix_seq->ne[0] > 1) {
        for (const auto& kv : model.vocab.token_to_id) {
            if (kv.first.substr(0, 2) == "__" && kv.first.substr(kv.first.size() - 2) == "__") {
                lang_ids.push_back(kv.second);
            }
        }
        std::sort(lang_ids.begin(), lang_ids.end());
    }
    ggml_tensor* embed = model.tensors["text_decoder_frontend.embed.weight"];
    std::size_t vocab_size = embed->ne[1];
    std::size_t beam_size = job.opts.beam_size;
    ggml_detach(encoder_output);
    int source_seq_len = encoder_output->ne[1];
    int max_seq_len = _determine_max_seq_len(job, source_seq_len);

    ggml_context* search_ctx = ctx_from_buffer(local_bufs[2]);
    ggml_context* original_ctx = model.ctx;
    fairseq2_kv_cache_alloc(model, search_ctx, beam_size, max_seq_len);

    // (S_enc, M) -> (B, S_enc, M)
    model.ctx = search_ctx;
    _fan_out_encoder_output(search_ctx, &encoder_output, &encoder_padding_mask, beam_size);

    // Allocate results in the context provided by the caller.
    ggml_set_no_alloc(result_ctx, false);
    Hypothesis* finished_searches_begin = GGML_CTX_ALLOC(result_ctx, Hypothesis, beam_size);
    Hypothesis* finished_searches = finished_searches_begin;
    for (std::size_t i = 0; i < beam_size; ++i) finished_searches[i] = {nullptr, -INFINITY, nullptr};
    Hypothesis* finished_searches_end = finished_searches + beam_size;

    // Initialize buffers. (B, S)
    ggml_tensor* seqs = ggml_new_tensor_2d(search_ctx, GGML_TYPE_I32, max_seq_len, beam_size);
    ggml_set_i32(seqs, 0);
    ggml_set_name(seqs, "seqs_0");
    ggml_tensor* scores = ggml_new_tensor_2d(search_ctx, GGML_TYPE_F32, max_seq_len, beam_size);
    ggml_set_name(scores, "scores_0");
    ggml_set_f32(scores, 0.0);
    int prefix_seq_len = job.prefix_seq->ne[0];
    int start_step = prefix_seq_len - 1;
    ggml_context* prev_step_ctx = ctx_from_buffer(local_bufs[(start_step + 1) % 2]);
    ggml_context* step_ctx = ctx_from_buffer(local_bufs[start_step % 2]);
    GGML_ASSERT(step_ctx != search_ctx);
    model.enc_kv_cache_ctx = search_ctx;
    ggml_tensor* lid_scores = ggml_new_tensor_1d(result_ctx, GGML_TYPE_F32, 1); // Dummy initialization to get rid of warnings
    if (lang_ids.size()) {
        lid_scores = ggml_new_tensor_1d(result_ctx, GGML_TYPE_F32, lang_ids.size());
    } 
    // Multilingual models: Bootstrap LID scores
    _bootstrap_seqs_and_scores(
        model, job, seqs, scores, encoder_output, encoder_padding_mask, lid_scores, n_threads, lang_ids
    );

    // Holds the indices of beams (a beam can occur more than once) that we
    // should continue with in the next step.
    ggml_tensor* beam_indices = ggml_new_tensor_1d(search_ctx, GGML_TYPE_I32, beam_size);
    ggml_tensor* next_tokens = ggml_new_tensor_1d(search_ctx, GGML_TYPE_I32, beam_size);
    ggml_tensor* next_scores = ggml_new_tensor_1d(search_ctx, GGML_TYPE_F32, beam_size);

    // Array with integers up to 'vocab_size * beam_size' to represent next beams to explore
    ggml_tensor* candidate_indices = ggml_new_tensor_1d(search_ctx, GGML_TYPE_I32, vocab_size * beam_size);
    for (std::size_t i = 0; i < vocab_size * beam_size; ++i)
        ((int32_t *)(candidate_indices->data))[i] = i;

    printf_mem_usage(search_ctx, "search_ctx");

    for (int step_nr = start_step; step_nr < max_seq_len - 1; ++step_nr) {
        model.ctx = step_ctx;
        ggml_set_no_alloc(step_ctx, true); // Use allocr for the model forward pass
        int p = 0;
        if (step_nr == start_step) {
            // Find the most probable lang_tok and assign it to all beams, when prefix_seq[1] is <unk>
            if (lang_ids.size() && ggml_get_i32_1d(job.prefix_seq, 1) == model.vocab.token_to_id["<unk>"]) {
                float max_lprob = std::numeric_limits<float>::min();
                for(std::size_t j = 0; j < lang_ids.size(); j++) {
                    auto val = ggml_get_f32_1d(lid_scores, j);
                    if (val > max_lprob) {
                        max_lprob = val;
                        p = lang_ids[j];
                    }
                }
                for (std::size_t k = 0; k < beam_size; k++) {
                    ggml_set_i32_1d(seqs, k * vocab_size + step_nr, p);
                }
            }
        }
        ggml_tensor* prev_token = ggml_slice(step_ctx, seqs, 0, step_nr, step_nr + 1);

        ggml_tensor* decoder_input = TransformerEmbeddingFrontend_forward(model, "text_decoder_frontend", prev_token);
        ggml_tensor* decoder_output = StandardTransformerDecoder_forward(
            model,
            "text_decoder",
            decoder_input,
            nullptr,  // We never generate PAD.
            encoder_output,
            encoder_padding_mask
        ); // (B, 1, D)

        decoder_output = ggml_flatten_1d(step_ctx, decoder_output, 0);  // (B, model_dim)
        // Force logits to be allocated in step_ctx, not in step_alloc.
        ggml_set_no_alloc(step_ctx, false);
        ggml_tensor* logits = Linear_forward(model, "final_proj", decoder_output);  // (B, vocab_size)
        ggml_tensor* lprobs = ggml_log_softmax(step_ctx, logits);

        // Compute lprobs here so we can modify it in place in the lprob tweaking phase
        // TODO: use ggml properly compute the tweaks
        struct ggml_cgraph * gf = ggml_new_graph(step_ctx);
        ggml_build_forward_expand(gf, lprobs);
        std::size_t fwd_mem = ggml_allocr_alloc_graph(step_alloc, gf);
        GGML_UNUSED(fwd_mem);
        ggml_graph_compute_with_ctx(step_ctx, gf, n_threads);
        ggml_detach(lprobs);
        ggml_allocr_reset(step_alloc);
#if DEBUG_MEM_USAGE
        printf("beam search step %d. Graph.n_nodes: %d.\n", step_nr, gf.n_nodes);
        printf("  Fwd mem: %.1fMB, reserved %.1fMb\n", fwd_mem/(double)MB, local_bufs[3].capacity()/(double)MB);
        std::fill(local_bufs[3].begin(), local_bufs[3].end(), 0xAA);
#endif
        _tweak_lprobs(job, lprobs, step_nr, max_seq_len, vocab_size);

        ggml_tensor* last_scores = ggml_slice(step_ctx, scores, 0, step_nr, step_nr+1);
        if (step_nr == start_step) {
            // At the initial step, all hypotheses are equally likely, so we use
            // only the first beam.
            lprobs = ggml_slice(step_ctx, lprobs, 1, 0, 1);
            lprobs = ggml_cont(step_ctx, lprobs);
            // The first step always indicates the beginning of the sequence and has no score.
            if (step_nr > 0) {
                last_scores = ggml_slice(step_ctx, last_scores, 1, 0, 1);
                lprobs = ggml_add_inplace(step_ctx, lprobs, ggml_repeat(step_ctx, last_scores, lprobs));
            }
        } else {
            // Make probabilities contain cumulative scores for each hypothesis.
            lprobs = ggml_add_inplace(step_ctx, lprobs, ggml_repeat(step_ctx, last_scores, lprobs));
        }
        ggml_build_forward_expand(gf, lprobs);
        ggml_graph_compute_with_ctx(step_ctx, gf, n_threads);

        // Determine (beam, token) candidates for the next step.
        // (N, 2 x B)
        std::int64_t K = topk(
            lprobs, std::min(2 * beam_size, vocab_size - 1), candidate_indices
        );

        std::size_t ongoing_beams = 0;
        for (std::int32_t i = 0; i < K; ++i) {
            int c = ggml_get_f32_1d(candidate_indices, i);
            std::int32_t beam = c / vocab_size;
            std::int32_t token = c % vocab_size;
            float tok_score = ggml_get_f32_1d(lprobs, c);

            // Detect beams that reached the minimum length and that end with an EOS.
            bool eos = token == job.eos_idx;
            eos &= tok_score != -INFINITY;
            if (eos) {
                _finalize_hypothesis(job, result_ctx, step_nr, beam, token, tok_score, seqs, scores, lid_scores, finished_searches++);
                if (finished_searches == finished_searches_end)
                    goto end_of_beam_search;
                continue;
            }

            ggml_set_f32_1d(beam_indices, ongoing_beams, beam);
            ggml_set_f32_1d(next_tokens, ongoing_beams, token);
            ggml_set_f32_1d(next_scores, ongoing_beams, tok_score);
            ongoing_beams += 1;
            if (ongoing_beams >= beam_size) break;
        }

        // Reorder beams in the `seq` and `score` buffers. The same beam can
        // be selected more than once.
        // (B, S), (B) -> (B, S)
        // don't use allocr API, cause it might reuse a kv cache buffer several time.
        ggml_set_no_alloc(step_ctx, false);
        ggml_tensor* new_seqs = ggml_get_rows(step_ctx, seqs, beam_indices);
        ggml_tensor* new_scores = ggml_get_rows(step_ctx, scores, beam_indices);
        struct ggml_cgraph * gf_reorder = ggml_new_graph(step_ctx);
        ggml_build_forward_expand(gf_reorder, new_seqs);
        ggml_build_forward_expand(gf_reorder, new_scores);
        reorder_kv_cache(model, step_ctx, gf_reorder, beam_indices);
        ggml_graph_compute_with_ctx(step_ctx, gf_reorder, n_threads);
        seqs = ggml_detach(new_seqs);
        scores = ggml_detach(new_scores);

        // seqs[:, step_nr + 1] = next_tokens
        // scores[:, step_nr + 1] = next_scores
        for (std::size_t i = 0; i < beam_size; ++i) {
            ((std::int32_t*)seqs->data)[step_nr + 1 + i * max_seq_len] = ggml_get_i32_1d(next_tokens, i);
            ((float*)scores->data)[step_nr + 1 + i * max_seq_len] = ggml_get_f32_1d(next_scores, i);
        }

        printf_mem_usage(step_ctx, "step_ctx");
        ggml_free(prev_step_ctx);
        prev_step_ctx = step_ctx;
#if DEBUG_MEM_USAGE
        std::fill(local_bufs[(step_nr + 1) % 2].begin(), local_bufs[(step_nr + 1) % 2].end(), 0xAA);
#endif
        step_ctx = ctx_from_buffer(local_bufs[(step_nr + 1) % 2]);
    }

end_of_beam_search:
    // Ensure that hypotheses are sorted by decreasing scores before returning.
    std::sort(
        finished_searches_begin,
        finished_searches_end,
        [](Hypothesis a, Hypothesis b) { return a.score > b.score; }
    );

    printf_mem_usage(search_ctx, "search_ctx");
    fairseq2_kv_cache_reset(model);
    model.ctx = original_ctx;
    return finished_searches_begin;
}

extern "C" Hypothesis* _testing_return_hypothesis_ptr(ggml_context* ctx) {
    Hypothesis* result = GGML_CTX_ALLOC(ctx, struct Hypothesis, 2);

    result[0] = {ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), 3.14f, (ggml_tensor*)result};
    ggml_set_i32_1d(result[0].seq, 0, 314);

    result[1] = {ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), 4.21f, nullptr};
    ggml_set_i32_1d(result[1].seq, 0, 421);

    return result;
}

// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4



struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    std::size_t n;
    llama_vocab::id id;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

static std::size_t utf8_len(char src) {
    const std::size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    std::size_t size;
    llama_vocab::id id;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string& input_text, ggml_tensor* output) {
        llama_vocab::id unk_idx = vocab.token_to_id.at("<unk>");

        // split string into utf8 chars
        int index = 0;
        std::size_t offs = 0;
        // This is kind of annoying, but needed because with SPM,
        // characters following a space have a special meaning.
        // And the algorithm rely on substrings to do the lookups.
        std::string text = input_text;
        bool need_extra_space = text.size() > 0 && text[0] != ' ';
        if (need_extra_space) text = " " + text;

        while (offs < text.size()) {
            std::size_t len = utf8_len(text[offs]);
            std::size_t n = std::min(len, text.size() - offs);

            auto token = vocab.token_to_id.find(std::string(text, offs, n));
            llama_vocab::id id = token == vocab.token_to_id.end() ? unk_idx : token->second;
            llm_symbol sym = {
                /*prev*/ index - 1,
                /*next*/ offs + n == text.size() ? -1 : index + 1,
                /*text*/ text.c_str() + offs,
                /*n*/ n,
                /*id*/ id
            };
            offs += n;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (std::size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];
            const std::string text = std::string(left_sym.text, left_sym.n + right_sym.n);

            // if one of the symbols already got merged, skip it.
            if (
                left_sym.n == 0
                || right_sym.n == 0
                || left_sym.n + right_sym.n != bigram.size
            ) continue;

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            left_sym.id = bigram.id;
            right_sym.n = 0;

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        llama_vocab::id* out = (llama_vocab::id*)output->data;
        int out_step = sizeof(llama_vocab::id) / output->nb[0];
        int num_tokens = 0;
        for (int i = 0; i > -1; i = symbols[i].next) {
            llm_symbol& symbol = symbols[i];
            *(out + num_tokens * out_step) = symbol.id;
            num_tokens += 1;
        }
        *(out + num_tokens * out_step) = vocab.token_to_id.at("</s>");
        num_tokens += 1;
        output->ne[0] = num_tokens;
    }

private:

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        llama_vocab::id id = token->second;
        if (static_cast<std::size_t>(id) >= vocab.id_to_token.size()) {
            return;
        }

        const auto& tok_data = vocab.id_to_token[id];
        llm_bigram_spm bigram = {
            /*left */ left,
            /*right*/ right,
            /*score*/ tok_data.score,
            /*size */ text.size(),
            /*id */ id
        };
        work_queue.push(bigram);
    }

    const llama_vocab& vocab;
    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
};


extern "C" void fairseq2_spm_tokenize(fairseq2_model* model, const char* text, ggml_tensor* out) {
    llm_tokenizer_spm spm = {model->vocab};
    spm.tokenize(std::string(text), out);
}


extern "C" std::size_t fairseq2_spm_detokenize(fairseq2_model* model, ggml_tensor* tokens, char* out) {
    bool no_tgt_vocab = model->tgt_vocab.id_to_token.empty();
    int eos_idx = no_tgt_vocab ? model->vocab.token_to_id["</s>"] : model->tgt_vocab.token_to_id["</s>"];
    int sent_len = tokens->ne[0];
    std::size_t written = 0;
    std::vector<llama_vocab::token_data> id_to_token = no_tgt_vocab ? model->vocab.id_to_token : model->tgt_vocab.id_to_token;
    for (int i = 0; i < sent_len; ++i) {
        int id = ggml_get_i32_1d(tokens, i);
        // Don't print the EOS token but only if it appear at the end.
        if (i == sent_len - 1 && eos_idx == id) break;
        std::string token = no_tgt_vocab ? model->vocab.id_to_token.at(id).text : model->tgt_vocab.id_to_token.at(id).text;
        // Skip the first space outputted.
        auto begin = token.begin();
        if (i == 0 && token.size() > 0 && token[0] == ' ') begin += 1;
        std::copy(begin, token.end(), out);
        std::size_t n = token.end() - begin;
        written += n;
        out += n;
    }
    *out = '0';
    return written;
}


// TODO: Unify with the above?
std::pair<std::vector<std::string>, std::vector<float>> fairseq2_spm_detokenize(
        fairseq2_model* model,
        ggml_tensor* tokens,
        ggml_tensor* scores,
        char* out) {
    bool no_tgt_vocab = model->tgt_vocab.id_to_token.empty();
    int eos_idx = no_tgt_vocab ? model->vocab.token_to_id["</s>"] : model->tgt_vocab.token_to_id["</s>"];
    int sent_len = tokens->ne[0];
    std::size_t written = 0;
    std::vector<float> word_scores;
    std::vector<float> subword_scores;
    std::vector<std::string> result_text;
    std::string curr_token = "";
    for (int i = 0; i < sent_len; ++i) {
        int id = ggml_get_i32_1d(tokens, i);
        // Don't print the EOS token but only if it appear at the end.
        if (i == sent_len - 1 && eos_idx == id) break;

        std::string token = no_tgt_vocab ? model->vocab.id_to_token.at(id).text : model->tgt_vocab.id_to_token.at(id).text;
        float score = ggml_get_f32_1d(scores, i+2); // 2 is prefix size
        if(token[0] == ' ') {
            // reset word score
            if(subword_scores.size() > 0) {
                float avg = std::accumulate(subword_scores.begin(), subword_scores.end(), 0.0f) / subword_scores.size();
                word_scores.push_back(avg);
                subword_scores.clear();
                result_text.push_back(curr_token);
            }
            curr_token = token.substr(1);
        } else {
            curr_token += token;
        }
        subword_scores.push_back(score);
        // Skip the first space outputted.
        auto begin = token.begin();
        if (i == 0 && token.size() > 0 && token[0] == ' ') begin += 1;
        std::copy(begin, token.end(), out);
        std::size_t n = token.end() - begin;
        written += n;
        out += n;

    }
    if(subword_scores.size() > 0) {
        word_scores.push_back(*std::min_element(subword_scores.begin(), subword_scores.end()));
        subword_scores.clear();
        result_text.push_back(curr_token);
    }
    *out = '0';
    return std::make_pair(result_text, word_scores);
}
