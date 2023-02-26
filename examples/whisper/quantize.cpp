#include "ggml/ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

#define QK 32

size_t ggml_quantize_q4_0(float * src, void * dst, int n, int k) {
    const int nb = k / QK;
    const size_t row_size = nb*(sizeof(float) + sizeof(uint8_t)*QK/2);

    assert(k % QK == 0);

    uint8_t pp[QK/2];

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) {
        float   * pd = (float *)   (pdst + (j/k)*row_size);
        uint8_t * pb = (uint8_t *) (pd + nb);

        for (int i = 0; i < nb; i++) {
            float amax = 0.0f; // absolute max

            {
                for (int l = 0; l < QK; l++) {
                    const float v = src[j + i*QK + l];
                    amax = std::max(amax, fabsf(v));
                }

                const float d = amax / ((1 << 3) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                pd[i] = d;

                for (int l = 0; l < QK; l += 2) {
                    const float v0 = (src[j + i*QK + l + 0])*id;
                    const float v1 = (src[j + i*QK + l + 1])*id;

                    const uint8_t vi0 = ((int8_t) (round(v0))) + 8;
                    const uint8_t vi1 = ((int8_t) (round(v1))) + 8;

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb + i*QK/2, pp, sizeof(pp));
            }
        }
    }

    return (n/k)*row_size;
}

size_t ggml_quantize_q4_1(float * src, void * dst, int n, int k) {
    const int nb = k / QK;
    const size_t row_size = nb*(2*sizeof(float) + sizeof(uint8_t)*QK/2);

    assert(k % QK == 0);

    uint8_t pp[QK/2];

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) {
        float   * pm = (float *)   (pdst + (j/k)*row_size);
        float   * pd = (float *)   (pm + nb);
        uint8_t * pb = (uint8_t *) (pd + nb);

        //printf("n = %d, k = %d, nb = %d, row_size = %d, j = %d, pm = %p, pd = %p, pb = %p\n", n, k, nb, row_size, j, pm, pd, pb);

        for (int i = 0; i < nb; i++) {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();

            {
                for (int l = 0; l < QK; l++) {
                    const float v = src[j + i*QK + l];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                const float d = (max - min) / ((1 << 4) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                pm[i] = min;
                pd[i] = d;

                for (int l = 0; l < QK; l += 2) {
                    const float v0 = (src[j + i*QK + l + 0] - min)*id;
                    const float v1 = (src[j + i*QK + l + 1] - min)*id;

                    const uint8_t vi0 = round(v0);
                    const uint8_t vi1 = round(v1);

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb + i*QK/2, pp, sizeof(pp));
            }
        }
    }

    return (n/k)*row_size;
}

// default hparams (Whisper tiny)
struct whisper_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t f16           = 1;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// quantize a model
bool whisper_model_quantize(const std::string & fname_inp, const std::string & fname_out, int itype) {
    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
        case 2: type = GGML_TYPE_Q4_0; break;
        case 3: type = GGML_TYPE_Q4_1; break;
        default: fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype); return 1;
    };

    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1) {
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, type);
        return false;
    }

    gpt_vocab vocab;

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *) &magic, sizeof(magic));
    }

    whisper_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        finp.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        finp.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        finp.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        finp.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        finp.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        finp.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        finp.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        finp.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        finp.read((char *) &hparams.f16,           sizeof(hparams.f16));

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stderr, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stderr, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stderr, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stderr, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stderr, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stderr, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stderr, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stderr, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stderr, "%s: f16           = %d\n", __func__, hparams.f16);

        fout.write((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        fout.write((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        fout.write((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        fout.write((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        fout.write((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        fout.write((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        fout.write((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        fout.write((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        fout.write((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        fout.write((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        fout.write((char *) &itype,                 sizeof(hparams.f16));
    }

    // load mel filters
    {
        whisper_filters filters;

        finp.read ((char *) &filters.n_mel, sizeof(filters.n_mel));
        fout.write((char *) &filters.n_mel, sizeof(filters.n_mel));
        finp.read ((char *) &filters.n_fft, sizeof(filters.n_fft));
        fout.write((char *) &filters.n_fft, sizeof(filters.n_fft));

        filters.data.resize(filters.n_mel * filters.n_fft);
        finp.read ((char *) filters.data.data(), filters.data.size() * sizeof(float));
        fout.write((char *) filters.data.data(), filters.data.size() * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        finp.read ((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_vocab, sizeof(n_vocab));

        //if (n_vocab != hparams.n_vocab) {
        //    fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
        //    return false;
        //}

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word.resize(len);
            finp.read ((char *) word.data(), len);
            fout.write((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t>     data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float>       data_f32;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (finp.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = { 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read (&name[0], length);

            {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%48s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ftype_str[ftype]);
            }

            // regexes of tensor names to not be quantized
            const std::vector<std::string> k_names = {
                //"encoder.*",
                "encoder.conv1.bias",
                "encoder.conv2.bias",
                "encoder.positional_embedding",
                "decoder.positional_embedding",
            };

            bool quantize = true;
            for (const auto & s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = false;
                    break;
                }
            }

            // quantize only 2D and 3D tensors
            quantize &= (n_dims == 2);

            if (quantize) {
                if (ftype != 0 && ftype != 1) {
                    fprintf(stderr, "%s: unsupported ftype %d for integer quantization\n", __func__, ftype);
                    return false;
                }

                if (ftype == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = itype;
            } else {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements*bpe);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            if (quantize) {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;

                switch (type) {
                    case GGML_TYPE_Q4_0:
                        {
                            cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0]);
                        } break;
                    case GGML_TYPE_Q4_1:
                        {
                            cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0]);
                        } break;
                    default:
                        {
                            fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                            return false;
                        }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.3f MB -> %8.3f MB\n", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
            } else {
                printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);
    }

    finp.close();
    fout.close();

    return true;
}

// usage:
//  ./gpt-2-quantize models/gpt-2-117M/ggml-model.bin models/gpt-2-117M/ggml-model-quant.bin type
//
int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        fprintf(stderr, "  type = 2 - q4_0\n");
        fprintf(stderr, "  type = 3 - q4_1\n");
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!whisper_model_quantize(fname_inp, fname_out, itype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}
