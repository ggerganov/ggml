#include "common-ggml.h"

#include <regex>
#include <map>

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
};

void ggml_print_ftypes(FILE * fp) {
    for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
        fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
    }
}

enum ggml_ftype ggml_parse_ftype(const char * str) {
    enum ggml_ftype ftype;
    if (str[0] == 'q') {
        const auto it = GGML_FTYPE_MAP.find(str);
        if (it == GGML_FTYPE_MAP.end()) {
            fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
            return GGML_FTYPE_UNKNOWN;
        }
        ftype = it->second;
    } else {
        ftype = (enum ggml_ftype) atoi(str);
    }

    return ftype;
}

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip) {

    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };

    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ggml_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    std::vector<int64_t> hist_all(1 << 4, 0);

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read (&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
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

            ttype = qtype;
        } else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements*bpe);
            finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            switch ((ggml_type) ttype) {
                case GGML_TYPE_Q4_0:
                    {
                        cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q4_1:
                    {
                        cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q5_0:
                    {
                        cur_size = ggml_quantize_q5_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q5_1:
                    {
                        cur_size = ggml_quantize_q5_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q8_0:
                    {
                        cur_size = ggml_quantize_q8_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                hist_all[i] += hist_cur[i];
            }

            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                printf("%5.3f ", hist_cur[i] / (float)nelements);
            }
            printf("\n");
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ggml_type_name(qtype));

    {
        int64_t sum_all = 0;
        for (int i = 0; i < (int) hist_all.size(); ++i) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (int i = 0; i < (int) hist_all.size(); ++i) {
            printf("%5.3f ", hist_all[i] / (float)sum_all);
        }
        printf("\n");
    }

    return true;
}

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

void ggml_graph_export_leaf(const struct ggml_tensor * tensor, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;

    fprintf(fout, "%-6s %-12s %8d %8lld %8lld %8lld %8lld %16zu %16zu %16zu %16zu %16p %16s\n",
            ggml_type_name(tensor->type),
            ggml_op_name  (tensor->op),
            tensor->n_dims,
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);
}

void ggml_graph_export_node(const struct ggml_tensor * tensor, const char * arg, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;

    fprintf(fout, "%-6s %-6s %-12s %8d %8lld %8lld %8lld %8lld %16zu %16zu %16zu %16zu %8d %16p %16s\n",
            arg,
            ggml_type_name(tensor->type),
            ggml_op_name  (tensor->op),
            tensor->n_dims,
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->n_tasks,
            tensor->data,
            tensor->name);
}

void ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname) {
    // print
    {
        FILE * fout = stdout;

        fprintf(fout, "\n");
        fprintf(fout, "%-16s %8x\n", "magic",   GGML_FILE_MAGIC);
        fprintf(fout, "%-16s %8d\n", "version", GGML_FILE_VERSION);
        fprintf(fout, "%-16s %8d\n", "leafs",   cgraph->n_leafs);
        fprintf(fout, "%-16s %8d\n", "nodes",   cgraph->n_nodes);

        // header
        fprintf(fout, "\n");
        fprintf(fout, "%-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %16s %16s\n",
                "TYPE", "OP", "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "NAME");

        for (int i = 0; i < cgraph->n_leafs; ++i) {
            ggml_graph_export_leaf(cgraph->leafs[i], fout);

            GGML_ASSERT(cgraph->leafs[i]->op   == GGML_OP_NONE);
            GGML_ASSERT(cgraph->leafs[i]->src0 == NULL);
            GGML_ASSERT(cgraph->leafs[i]->src1 == NULL);
        }

        // header
        fprintf(fout, "\n");
        fprintf(fout, "%-6s %-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %8s %16s %16s\n",
                "ARG", "TYPE", "OP", "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "NTASKS", "DATA", "NAME");

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_graph_export_node(cgraph->nodes[i], "DST", fout);

            if (cgraph->nodes[i]->src0) {
                ggml_graph_export_node(cgraph->nodes[i]->src0, "SRC0", fout);
            }

            if (cgraph->nodes[i]->src1) {
                ggml_graph_export_node(cgraph->nodes[i]->src1, "SRC1", fout);
            }

            for (int j = 0; j < GGML_MAX_OPT; ++j) {
                if (cgraph->nodes[i]->opt[j]) {
                    ggml_graph_export_node(cgraph->nodes[i]->opt[j], "OPT", fout);
                }
            }

            fprintf(fout, "\n");
        }

        fprintf(fout, "\n");
    }

    // write binary data
    {
        FILE * fout = fopen(fname, "wb");

        if (!fout) {
            fprintf(stderr, "%s: failed to open %s\n", __func__, fname);
            return;
        }

        // header
        {
            uint32_t magic   = GGML_FILE_MAGIC;
            uint32_t version = GGML_FILE_VERSION;
            uint32_t leafs   = cgraph->n_leafs;
            uint32_t nodes   = cgraph->n_nodes;

            fwrite(&magic,   sizeof(uint32_t), 1, fout);
            fwrite(&version, sizeof(uint32_t), 1, fout);
            fwrite(&leafs,   sizeof(uint32_t), 1, fout);
            fwrite(&nodes,   sizeof(uint32_t), 1, fout);
        }

        fclose(fout);
    }
}
