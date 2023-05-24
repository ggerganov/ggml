#pragma once

#include "ggml.h"

#include <fstream>
#include <vector>
#include <string>

enum ggml_ftype ggml_parse_ftype(const char * str);

void ggml_print_ftypes(FILE * fp = stderr);

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);

void        ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
ggml_cgraph ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
