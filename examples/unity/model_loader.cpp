#include "model_loader.h"
#include <string>

#define DEBUG_MODEL_LOAD 0

std::ifstream open_ggml_file(const char* fname) {
    printf("%s: loading model from '%s'\n", __func__, fname);

    auto fin = std::ifstream(std::string(fname), std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }

    std::uint32_t magic;
    fin.read((char*)&magic, 4);
    if (magic != GGML_FILE_MAGIC) {
        fprintf(stderr, "%s: invalid model file '%s' (bad header %d)\n", __func__, fname, magic);
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }
    return fin;
}

void register_prefix(fairseq2_model &model, const std::string& name) {
    std::size_t i = name.find_last_of('.');
    while(i != std::string::npos && i > 0) {
        std::string prefix = name.substr(0, i);
        auto prev_tensor = model.tensors.find(prefix);
        if (prev_tensor != model.tensors.end()) {
            GGML_ASSERT(prev_tensor->second == nullptr);
        }
        model.tensors[prefix] = nullptr;
        i = name.find_last_of('.', i - 1);
    }
}


std::int64_t
model_loader::load_model_weights(fairseq2_model &model, std::ifstream &fin)
{
    std::int64_t num_tensor = 0;
    std::int64_t f32_tensor_size = 0;
    fin.read((char*) &num_tensor, sizeof(num_tensor));
    fin.read((char*) &f32_tensor_size, sizeof(f32_tensor_size));

    // TODO: it might be interesting to allow the caller to not upcast the weights to float32.
    // Note this require changing the on disk format
    bool as_float32 = true;
    struct ggml_init_params params = {
        /*.mem_size   =*/ static_cast<size_t>(f32_tensor_size + (num_tensor + 1) * (int64_t)ggml_tensor_overhead()),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    model.tensors_ctx = ggml_init(params);

    size_t model_size = 0;
    for (int i = 0; i < num_tensor; ++i) {
        std::string name = get_name(fin);
        if (name.length() == 0)
            break;
        auto tensor = load_tensor_value(fin, model.tensors_ctx, as_float32);
        if (tensor == nullptr) {
            // Abort in case of error, the input stream is corrupted at this point.
            printf("Error while reading tensor %s\n", name.c_str() );
            throw std::invalid_argument("Error while reading tensor from file.");
        }
        register_prefix(model, name);
        ggml_set_name(tensor, name.c_str());
        model.tensors[name] = tensor;
        if (DEBUG_MODEL_LOAD) {
            printf("%s [%5ld, %5ld], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), tensor->ne[0], tensor->ne[1], ggml_type_name(tensor->type), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
        }
        model_size += ggml_nbytes(tensor);
    }

    double mb = 1024.0 * 1024.0;
    printf("%s: model size: %8.2f MB, memory used: %8.2f MB, memory reserved: %8.2f MB\n",
        __func__,
        model_size / mb,
        ggml_used_mem(model.tensors_ctx) / mb,
        ggml_get_mem_size(model.tensors_ctx) / mb
    );

    return ggml_get_mem_size(model.tensors_ctx);
}

void assert_endianness() {
    union {
        unsigned int i;
        char c[4];
    } un;
    un.i = 0x12345678;

    if (un.c[0] == 0x78 && un.c[3] == 0x12) {
        printf("little-endian\n");
    }
    else if (un.c[0] == 0x12 && un.c[3] == 0x78) {
        printf("big-endian\n");
        GGML_ASSERT(false); // model_loader.cpp assumes the system is little-endian
    }
    else {
        printf("unknown-endian\n");
        GGML_ASSERT(false); // model_loader.cpp assumes the system is little-endian
    }
}


void model_loader::load_hparams(std::unordered_map<std::string, std::int64_t>& hparams, std::ifstream &fin)
{
    std::int64_t num_params = 0;
    fin.read(reinterpret_cast<char*>(&num_params), sizeof num_params);
    GGML_ASSERT(fin.gcount() == 8);

    hparams.reserve(num_params);

    std::int64_t value;
    for (int i = 0; i < num_params; ++i) {
        std::string name = get_name(fin);
        if (name.length() == 0)
            break;
        fin.read((char*) &value, sizeof(value));
        hparams[name] = value;
    }
}

void model_loader::load_vocab(llama_vocab& vocab, std::ifstream &fin)
{
    // vocab.special_bos_id = 1;
    // vocab.special_eos_id = 2;
    // vocab.special_unk_id = 0;
    // vocab.special_sep_id = -1;
    // vocab.special_pad_id = -1;

    std::int64_t vocab_size = 0;
    fin.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    // GGML_ASSERT(fin.gcount() == 8);
    if (vocab_size == 0) {
        return;
    }

    vocab.token_to_id.reserve(vocab_size);
    vocab.id_to_token.reserve(vocab_size);

    std::string packed_vocab = get_name(fin);
    std::int64_t ctx_size = vocab_size * sizeof(float) + vocab_size + 2 * ggml_tensor_overhead();
    ctx_size *= 2;
    ggml_context* ctx = ggml_init(ggml_init_params{static_cast<size_t>(ctx_size), nullptr, false});
    ggml_tensor* lengths_tensor = load_tensor_value(fin, ctx, true);
    std::int8_t* lengths = (std::int8_t*)lengths_tensor->data;
    ggml_tensor* scores_tensor = load_tensor_value(fin, ctx, true);
    float* scores = ggml_get_data_f32(scores_tensor);

    int64_t offset = 0;
    for (int i = 0; i < vocab_size; ++i) {
        // TODO: we should use string view instead of copying each word in a new string
        std::string word = packed_vocab.substr(offset, lengths[i]);
        vocab.token_to_id[word] = i;
        vocab.id_to_token.push_back({word, scores[i], LLAMA_TOKEN_TYPE_NORMAL});
        offset += lengths[i] + 1;
    }
    // Since we copied lengths and scores, we don't need the context anymore.
    ggml_free(ctx);

    // vocab.linefeed_id = llama_byte_to_token(vocab, '\n');
    // TODO: special tokens stuff ?
}

ggml_tensor* load_tensor_value(std::ifstream &fin, ggml_context* ctx, bool as_float32)
{
    int32_t n_dims = 0;
    int32_t raw_type = 0;

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char *>(&raw_type),  sizeof(raw_type));
    ggml_type type = ggml_type(raw_type);

    if (n_dims <= 0 || n_dims > GGML_MAX_DIMS || raw_type < 0 || raw_type > GGML_TYPE_COUNT) {
        return nullptr;
    }
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
    }

    ggml_tensor* tensor;
    if (as_float32 && type == GGML_TYPE_F16) {
        // read quantized weights from disk, and convert them to f32.
        tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
        ggml_fp16_t buf[128];
        int num_el = ggml_nelements(tensor);
        for (int i = 0; i < num_el; i += 128) {
            int block_size = std::min(128, num_el - i);
            fin.read(reinterpret_cast<char *>(&buf), ggml_type_size(type) * block_size);
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)&buf, (float*)tensor->data + i, block_size);
        }
    } else {
        tensor = ggml_new_tensor(ctx, type, n_dims, ne);
        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
    }
    return tensor;
}

std::string
model_loader::get_name(std::ifstream& fin)
{
    std::uint32_t length = 0;
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    if (length == 0)
        return "";

    std::string name(length, 0);
    fin.read(&name[0], length);

    return name;
}

extern "C" int load_fairseq2_ggml_file(fairseq2_model& model, const char* fname) {
    model_loader loader;
    assert_endianness();
    auto fin = open_ggml_file(fname);
    loader.load_hparams(model.hparams, fin);
    loader.load_hparams(model.layer_config, fin);
    loader.load_vocab(model.vocab, fin);
    loader.load_model_weights(model, fin);
    
    // load optional target vocabulary in cases of bilingual models
    loader.load_vocab(model.tgt_vocab, fin);
    return 0;
}
