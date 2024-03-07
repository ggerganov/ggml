#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static const char * magika_labels[] = {
    "ai",                 "apk",                "appleplist",         "asm",                "asp",
    "batch",              "bmp",                "bzip",               "c",                  "cab",
    "cat",                "chm",                "coff",               "crx",                "cs",
    "css",                "csv",                "deb",                "dex",                "dmg",
    "doc",                "docx",               "elf",                "emf",                "eml",
    "epub",               "flac",               "gif",                "go",                 "gzip",
    "hlp",                "html",               "ico",                "ini",                "internetshortcut",
    "iso",                "jar",                "java",               "javabytecode",       "javascript",
    "jpeg",               "json",               "latex",              "lisp",               "lnk",
    "m3u",                "macho",              "makefile",           "markdown",           "mht",
    "mp3",                "mp4",                "mscompress",         "msi",                "mum",
    "odex",               "odp",                "ods",                "odt",                "ogg",
    "outlook",            "pcap",               "pdf",                "pebin",              "pem",
    "perl",               "php",                "png",                "postscript",         "powershell",
    "ppt",                "pptx",               "python",             "pythonbytecode",     "rar",
    "rdf",                "rpm",                "rst",                "rtf",                "ruby",
    "rust",               "scala",              "sevenzip",           "shell",              "smali",
    "sql",                "squashfs",           "svg",                "swf",                "symlinktext",
    "tar",                "tga",                "tiff",               "torrent",            "ttf",
    "txt",                "unknown",            "vba",                "wav",                "webm",
    "webp",               "winregistry",        "wmf",                "xar",                "xls",
    "xlsb",               "xlsx",               "xml",                "xpi",                "xz",
    "yaml",               "zip",                "zlibstream"
};

struct magika_hparams {
    const int block_size = 4096;
    const int beg_size = 512;
    const int mid_size = 512;
    const int end_size = 512;
    const int min_file_size_for_dl = 16;
    const int n_label = 113;
    const float f_norm_eps = 0.001f;
    const int padding_token = 256;
};

struct magika_model {
    ~magika_model() {
        ggml_backend_buffer_free(buf_w);
        ggml_backend_free(backend);
        ggml_free(ctx_w);
    }

    magika_hparams hparams;

    struct ggml_tensor * dense_w;
    struct ggml_tensor * dense_b;

    struct ggml_tensor * layer_norm_gamma;
    struct ggml_tensor * layer_norm_beta;

    struct ggml_tensor * dense_1_w;
    struct ggml_tensor * dense_1_b;

    struct ggml_tensor * dense_2_w;
    struct ggml_tensor * dense_2_b;

    struct ggml_tensor * layer_norm_1_gamma;
    struct ggml_tensor * layer_norm_1_beta;

    struct ggml_tensor * target_label_w;
    struct ggml_tensor * target_label_b;

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context * ctx_w = nullptr;
};

struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
    if (!tensor) {
        fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name);
        throw std::runtime_error("ggml_get_tensor() failed");
    }
    return tensor;
}

bool magika_model_load(const std::string & fname, magika_model & model) {
    auto & ctx = model.ctx_w;

    struct gguf_init_params params = {
        /*.no_alloc   =*/ true,
        /*.ctx        =*/ &ctx,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    model.buf_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!model.buf_w) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors() failed\n", __func__);
        gguf_free(ctx_gguf);
        return false;
    }

    try {
        model.dense_w = checked_get_tensor(ctx, "dense/kernel:0");
        model.dense_b = checked_get_tensor(ctx, "dense/bias:0");

        model.layer_norm_gamma = checked_get_tensor(ctx, "layer_normalization/gamma:0");
        model.layer_norm_beta  = checked_get_tensor(ctx, "layer_normalization/beta:0");

        model.dense_1_w = checked_get_tensor(ctx, "dense_1/kernel:0");
        model.dense_1_b = checked_get_tensor(ctx, "dense_1/bias:0");

        model.dense_2_w = checked_get_tensor(ctx, "dense_2/kernel:0");
        model.dense_2_b = checked_get_tensor(ctx, "dense_2/bias:0");

        model.layer_norm_1_gamma = checked_get_tensor(ctx, "layer_normalization_1/gamma:0");
        model.layer_norm_1_beta  = checked_get_tensor(ctx, "layer_normalization_1/beta:0");

        model.target_label_w = checked_get_tensor(ctx, "target_label/kernel:0");
        model.target_label_b = checked_get_tensor(ctx, "target_label/bias:0");
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", __func__, e.what());
        gguf_free(ctx_gguf);
        return false;
    }

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen() failed\n", __func__);
        gguf_free(ctx_gguf);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(ctx_gguf);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

        //printf("%-30s: [%3ld, %3ld, %3ld, %3ld] %s\n",
        //    name,
        //    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        //    ggml_type_name(tensor->type));

        std::vector<uint8_t> buf(ggml_nbytes(tensor));
        if (fseek(f, offs, SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek() failed\n", __func__);
            gguf_free(ctx_gguf);
            fclose(f);
            return false;
        }

        if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
            fprintf(stderr, "%s: fread() failed\n", __func__);
            gguf_free(ctx_gguf);
            fclose(f);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());
    }

    fclose(f);

    gguf_free(ctx_gguf);

    return true;
}

struct ggml_cgraph * magika_graph(
    const magika_model & model,
    const int n_files) {

    const auto & hparams = model.hparams;

    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, 1536, n_files); // one-hot
    ggml_set_name(input, "input");
    ggml_set_input(input);

    struct ggml_tensor * cur;

    // dense
    cur = ggml_mul_mat(ctx, model.dense_w, input);
    cur = ggml_add(ctx, cur, model.dense_b); // [128, 1536, n_files]
    cur = ggml_gelu(ctx, cur);

    // reshape
    cur = ggml_reshape_3d(ctx, cur, 512, 384, n_files); // [384, 512, n_files]
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    // layer normalization
    cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
    cur = ggml_mul(ctx, cur, model.layer_norm_gamma); // [384, 512, n_files]
    cur = ggml_add(ctx, cur, model.layer_norm_beta);  // [384, 512, n_files]

    // dense_1
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    cur = ggml_mul_mat(ctx, model.dense_1_w, cur);
    cur = ggml_add(ctx, cur, model.dense_1_b); // [256, 384, n_files]
    cur = ggml_gelu(ctx, cur);

    // dense_2
    cur = ggml_mul_mat(ctx, model.dense_2_w, cur);
    cur = ggml_add(ctx, cur, model.dense_2_b); // [256, 384, n_files]
    cur = ggml_gelu(ctx, cur);

    // global_max_pooling1d
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur)); // [384, 256, n_files]
    cur = ggml_pool_1d(ctx, cur, GGML_OP_POOL_MAX, 384, 384, 0); // [1, 256, n_files]
    cur = ggml_reshape_2d(ctx, cur, 256, n_files); // [256, n_files]

    // layer normalization 1
    cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
    cur = ggml_mul(ctx, cur, model.layer_norm_1_gamma); // [256, n_files]
    cur = ggml_add(ctx, cur, model.layer_norm_1_beta);  // [256, n_files]

    // target_label
    cur = ggml_mul_mat(ctx, model.target_label_w, cur);
    cur = ggml_add(ctx, cur, model.target_label_b); // [n_label, n_files]
    cur = ggml_soft_max(ctx, cur); // [n_label, n_files]
    ggml_set_name(cur, "target_label_probs");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

bool magika_eval(
    struct magika_model & model,
    const std::vector<std::string> & fnames) {

    const auto & hparams = model.hparams;

    static ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    struct ggml_cgraph * gf = magika_graph(model, fnames.size());

    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        return false;
    }

    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input");

    for (size_t i = 0; i < fnames.size(); i++) {
        FILE * f = fopen(fnames[i].c_str(), "rb");
        if (!f) {
            fprintf(stderr, "%s: fopen() failed\n", __func__);
            return false;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);

        // the buffer is padded with the padding_token if the file is smaller than the block size
        std::vector<int> buf(1536, hparams.padding_token);
        std::vector<uint8_t> read_buf(std::max(hparams.beg_size, std::max(hparams.mid_size, hparams.end_size)));

        // read beg
        fseek(f, 0, SEEK_SET);
        int n_read = fread(read_buf.data(), 1, hparams.beg_size, f);
        for (int j = 0; j < n_read; j++) {
            // pad at the end
            buf[j] = read_buf[j];
        }

        // read mid
        long mid_offs = std::max(0L, (fsize - hparams.mid_size) / 2);
        fseek(f, mid_offs, SEEK_SET);
        n_read = fread(read_buf.data(), 1, hparams.mid_size, f);
        for (int j = 0; j < n_read; j++) {
            // pad at both ends
            long mid_idx = hparams.beg_size + (hparams.mid_size / 2) - n_read / 2 + j;
            buf[mid_idx] = read_buf[j];
        }

        // read end
        long end_offs = std::max(0L, fsize - hparams.end_size);
        fseek(f, end_offs, SEEK_SET);
        n_read = fread(read_buf.data(), 1, hparams.end_size, f);
        for (int j = 0; j < n_read; j++) {
            // pad at the beginning
            int end_idx = hparams.beg_size + hparams.mid_size + hparams.end_size - n_read + j;
            buf[end_idx] = read_buf[j];
        }

        fclose(f);

        const size_t inp_bytes = hparams.beg_size + hparams.mid_size + hparams.end_size;

        // convert to one-hot
        std::vector<float> one_hot(257*inp_bytes);
        for (size_t j = 0; j < inp_bytes; j++) {
            one_hot[257*j + buf[j]] = 1.0f;
        }

        ggml_backend_tensor_set(input, one_hot.data(), 257*inp_bytes*i*sizeof(float), 257*inp_bytes*sizeof(float));
    }

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return false;
    }

    struct ggml_tensor * target_label_probs = ggml_graph_get_tensor(gf, "target_label_probs");

    // print probabilities for the top labels of each file
    for (size_t i = 0; i < fnames.size(); i++) {
        std::vector<float> probs(hparams.n_label);
        ggml_backend_tensor_get(target_label_probs, probs.data(), hparams.n_label*i*sizeof(float), hparams.n_label*sizeof(float));

        // sort the probabilities
        std::vector<int> idx(hparams.n_label);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&probs](int i1, int i2) { return probs[i1] > probs[i2]; });

        // print the top labels
        const int top_n = 5;
        printf("%-30s: ", fnames[i].c_str());
        for (int j = 0; j < top_n; j++) {
            printf("%s (%.2f%%) ", magika_labels[idx[j]], probs[idx[j]]*100);
        }
        printf("\n");
    }

    return true;
}

int main(int argc, const char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model> <file1> [<file2> ...]\n", argv[0]);
        return 1;
    }

    const char * model_fname = argv[1];
    std::vector<std::string> fnames;
    for (int i = 2; i < argc; i++) {
        fnames.push_back(argv[i]);
    }

    magika_model model;
    if (!magika_model_load(model_fname, model)) {
        fprintf(stderr, "magika_model_load() failed\n");
        return 1;
    }

    magika_eval(model, fnames);

    return 0;
}
