#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include "ggml.h"
#include "feature-fbank.h"

#include "ggml-alloc.h"

#define FORCE_ALLOC(name, ctx, ggml_new_tensor)\
    bool name ## _save_no_alloc_ = ggml_get_no_alloc(ctx); \
    ggml_set_no_alloc(ctx, false); \
    ggml_tensor* name = ggml_new_tensor; \
    ggml_set_no_alloc(ctx, name ## _save_no_alloc_);

typedef int32_t llama_token;

extern "C" enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};


struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::unordered_map<token, id> special_tokens_cache;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1;
    id special_pad_id = -1;

    int special_add_bos = -1; // -1 unknown, 1 add, 0 don't add.
    int special_add_eos = -1; // -1 unknown, 1 add, 0 don't add.

    id linefeed_id       = 13;
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;

    int find_bpe_rank(std::string token_left, std::string token_right) const {
        GGML_ASSERT(token_left.find(" ") == std::string::npos);
        GGML_ASSERT(token_left.find("\n") == std::string::npos);
        GGML_ASSERT(token_right.find(" ") == std::string::npos);
        GGML_ASSERT(token_right.find("\n") == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }
};


struct KeyValueTensor {
    ggml_tensor* full_k;
    ggml_tensor* full_v;
    ggml_tensor* self_attn_mask;
    int step_nr;
};

struct fairseq2_model {
    // Context containing all tensors memory
    ggml_context* tensors_ctx = nullptr;

    // Named tensors, all tensors should belong to tensors_ctx
    std::unordered_map<std::string, struct ggml_tensor *> tensors = {};

    // Hashmap containing model hyper-parameters.
    std::unordered_map<std::string, std::int64_t> hparams = {};

    // Hashmap containing layers hyper-parameters.
    // Normally those can be inferred from hparams, but it avoids doing this logic in GGML
    std::unordered_map<std::string, std::int64_t> layer_config = {};

    // Vocabulary for text transcription and translation APIs
    llama_vocab vocab;

    // Optional target vocabulary for bilingual models
    llama_vocab tgt_vocab;

    // KV cache for attention layers
    mutable std::unordered_map<std::string, KeyValueTensor> kv_cache = {};

    // an inference context, not managed by this object
    // TODO: is this the best place to store this or should we also pass this to all forward methods ?
    ggml_context* ctx = nullptr;

    ggml_context* enc_kv_cache_ctx = nullptr;
};

double fairseq2_model_layer_config_double(const fairseq2_model& model, std::string name);

/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc();
// free the models and all its owned tensors
extern "C" void fairseq2_model_free(fairseq2_model* model);
extern "C" void fairseq2_model_set_inference_ctx(fairseq2_model* model, ggml_context* ctx);
extern "C" void fairseq2_kv_cache_reset(const fairseq2_model& model);
ggml_context* ctx_from_buffer(std::vector<uint8_t>& buffer);

extern "C" std::string* std_string_alloc(char* c_str);
extern "C" void std_string_free(std::string* str);

extern "C" ggml_tensor* WaveformToFbank_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* waveform 
);
extern "C" ggml_tensor* ggml_slice(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    int axis,
    int64_t start,
    int64_t end
);

/// Merge the given dimension and the previous one in the tensor.
/// (..., num_heads, N, ...) -> (..., num_heads * N, ...)
/// dim is the position of the resulting merged dimension
/// ggml_flatten_1d(x, d) <==> torch.flatten(x, -1-d-1, -1-d0
extern "C" ggml_tensor* ggml_flatten_1d(ggml_context* ctx, ggml_tensor* x, int dim);

/// Split the given dimension.
/// (..., K * N, ...) -> (..., K, N, ...)
/// dim is the position of the output dimension with the given number of element (N).
extern "C" ggml_tensor* ggml_unflatten_1d(ggml_context* ctx, ggml_tensor* x, int dim, int num_el);

extern "C" ggml_tensor* Linear_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input
);

extern "C" ggml_tensor* LayerNorm_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input
);

extern "C" ggml_tensor* StandardFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
);

extern "C" ggml_tensor* SiluFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
);

extern "C" ggml_tensor* MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (slen, d_in)
    ggml_tensor* keys,  // (klen, d_in)
    ggml_tensor* values,  // (klen, d_out)
    ggml_tensor* attn_mask // (klen, slen)
);


extern "C" ggml_tensor* PositionalEmbedding_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* embeds
);

extern "C" ggml_tensor* TransformerEmbeddingFrontend_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
);

extern "C" ggml_tensor* StandardTransformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);

extern "C" ggml_tensor* StandardTransformerEncoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);

extern "C" ggml_tensor* RelativePositionMHA_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
);

extern "C" ggml_tensor* ConvModule_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
);

extern "C" ggml_tensor* StandardConformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);

extern "C" ggml_tensor* StandardConformerEncoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);

extern "C" ggml_tensor* StandardConformerEncoderAdaptorLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);

extern "C" ggml_tensor* StandardConformerEncoderAdaptor_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);
// Specifies the Layer Normalization order.
// see fairseq2/nn/transformer/norm_order.py
enum TransformerNormOrder {
    TRANSFORMER_NORM_ORDER_POST = 0,
    TRANSFORMER_NORM_ORDER_PRE = 1,
    TRANSFORMER_NORM_ORDER_PRE_WITH_NORMFORMER = 2
};



/// Holds the options to pass to a sequence generator.
struct SequenceGeneratorOptions {
    /// The beam size.
    int beam_size = 5;

    /// The minimum length of generated sequences (including prefix sequence).
    int min_seq_len = 1;

    /// The terms ``a`` and ``b`` of ``ax + b`` where ``x`` is the source
    /// sequence length. The generated sequences (including prefix sequence) will
    /// have the maximum length of ``min(hard_max_seq_len, ax + b)``. See also
    /// ``hard_max_seq_len``.
    float soft_max_seq_len_a = 1;
    int soft_max_seq_len_b = 200;

    /// The hard limit on maximum length of generated sequences.
    int hard_max_seq_len = 1024;

    /// The length penalty, where values less than 1.0 favor shorter, values
    /// greater than 1.0 favor longer sequences.
    float len_penalty = 1.0;

    /// The unknown symbol penalty, where values less than 0 produce more UNKs,
    /// values greater than 0 produce fewer UNKs.
    float unk_penalty = 0.0;

    /// If ``True``, normalizes scores by the length of generated sequences.
    bool normalize_scores = true;

    // memory needed is largely a fn of model size + sentence length and beam_size
    int mem_mb = 256;
};


struct SequenceGeneratorJob {
    SequenceGeneratorOptions opts;
    ggml_tensor* prefix_seq;
    std::int32_t pad_idx;
    std::int32_t unk_idx;
    std::int32_t bos_idx;
    std::int32_t eos_idx;
    std::int32_t num_threads;
};

/// Represents a hypothesis produced by a sequence generator.
struct Hypothesis {
    /// The generated sequence.
    ggml_tensor* seq;

    /// The score of the hypothesis.
    float score;

    /// The score of each individual sequence step.
    ggml_tensor* step_scores;

    /// The score of each lang tok at first decoding step, serving as LID 
    ggml_tensor* lid_scores;
};


extern "C" Hypothesis* generate_sequence(
    fairseq2_model& model,
    const SequenceGeneratorJob& opts,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask,
    ggml_context* result_ctx,
    int threads
);

extern "C" void fairseq2_spm_tokenize(fairseq2_model* model, const char* text, ggml_tensor* out);
extern "C" std::size_t fairseq2_spm_detokenize(fairseq2_model* model, ggml_tensor* tokens, char* out);

std::pair<std::vector<std::string>, std::vector<float>> fairseq2_spm_detokenize(fairseq2_model* model, ggml_tensor* tokens, ggml_tensor* scores, char* out);
