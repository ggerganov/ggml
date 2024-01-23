// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// MIT_LICENSE file in the root directory of this source tree.

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "fairseq2.h"


class model_loader {
public:
    std::int64_t load_model_weights(fairseq2_model &model, std::ifstream &fin);

    void load_hparams(std::unordered_map<std::string, std::int64_t>& hparams, std::ifstream &fin);

    void load_vocab(llama_vocab& vocab, std::ifstream &fin);

private:
    std::string get_name(std::ifstream &fin);
};

ggml_tensor* load_tensor_value(std::ifstream &fin, ggml_context* ctx, bool as_float32);

std::ifstream open_ggml_file(const char* fname);

extern "C" int load_fairseq2_ggml_file(fairseq2_model& model, const char* fname);
