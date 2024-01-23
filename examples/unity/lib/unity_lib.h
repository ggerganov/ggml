#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "math.h"
#include "model_loader.h"
#include "fairseq2.h"

#include <thread>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

struct Result {
    std::vector<std::string> transcription;
    std::vector<float> word_confidence_scores;
    std::unordered_map<std::string, float> lid_scores;
    int err;
};

struct ggml_cgraph * unity_speech_encoder(
    fairseq2_model& model,
    struct ggml_tensor * speech_input
);

struct ggml_cgraph * unity_text_encoder(
    fairseq2_model& model,
    struct ggml_tensor * text_input
);

Hypothesis* unity_decode(
    fairseq2_model& model,
    const SequenceGeneratorOptions& opts,
    int tgt_lang_idx,
    ggml_tensor* encoder_output,
    int n_threads
);

extern "C" fairseq2_model unity_init_model(const char* model_path);

extern "C" Result unity_eval_speech(
    fairseq2_model& model, 
    std::vector<float>& data, 
    SequenceGeneratorOptions opts, 
    std::string tgt_lang, 
    int n_threads
);

extern "C" Result unity_eval_text(
    fairseq2_model& model,  
    const std::string& text, 
    SequenceGeneratorOptions opts, 
    std::string tgt_lang, 
    int n_threads
);
