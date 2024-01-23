#include "unity_lib.h"
#include <algorithm>
#include <stdexcept>


struct ggml_cgraph * unity_text_encoder(
        fairseq2_model & model,
        struct ggml_tensor * text_input) {
    ggml_context* ctx0 = model.ctx;
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* seqs = TransformerEmbeddingFrontend_forward(model, "text_encoder_frontend", text_input);
    ggml_tensor* encoder_output = StandardTransformerEncoder_forward(
        model,
        "text_encoder",
        seqs,
        nullptr  // TODO: handle padding mask
    );
    encoder_output = ggml_dup(model.ctx, encoder_output);
    ggml_build_forward_expand(gf, encoder_output);
    return gf;
}

struct ggml_cgraph * unity_speech_encoder(
        fairseq2_model& model,
        struct ggml_tensor * speech_input) {
    ggml_context* ctx0 = model.ctx;
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* seqs = StandardConformerEncoder_forward(model, "speech_encoder", speech_input, nullptr);
    seqs = ggml_dup(model.ctx, seqs);
    ggml_build_forward_expand(gf, seqs);
    return gf;
}

Hypothesis* unity_decode(
        fairseq2_model& model,
        const SequenceGeneratorOptions& opts,
        int tgt_lang_idx,
        ggml_tensor* encoder_output,
        int n_threads
) {
    SequenceGeneratorJob job = {
        opts,
        /*prefix_seq*/ nullptr,
        /*pad_idx*/model.vocab.token_to_id["<pad>"],
        /*unk_idx*/model.vocab.token_to_id["<unk>"],
        /*bos_idx*/model.vocab.token_to_id["<s>"],
        /*eos_idx*/model.vocab.token_to_id["</s>"],
        /*num_threads*/n_threads,
    };
    int prefix_seq_len = tgt_lang_idx ? 2 : 1;
    FORCE_ALLOC(prefix_seq, model.ctx, ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, prefix_seq_len));
    ((int *)prefix_seq->data)[0]  = job.eos_idx;
    if (tgt_lang_idx != 0) { // multilingual case
        ((int *)prefix_seq->data)[1]  = tgt_lang_idx;
    }
    job.prefix_seq = prefix_seq;
    return generate_sequence(model, job, encoder_output, nullptr, model.ctx, n_threads);
}

extern "C" fairseq2_model unity_init_model(const char* model_path) {
    fairseq2_model model;
    load_fairseq2_ggml_file(model, model_path);
    return model;
}

//  struct as return - transcription, CE score, LID 
extern "C" Result unity_eval_speech(fairseq2_model& model, std::vector<float>& data, SequenceGeneratorOptions opts, std::string tgt_lang, int n_threads) {
    Result result;
    // The ctx_size_mb mostly depends of input length and model dim.
    int ctx_size_mb = opts.mem_mb;
    auto encoder_buf = std::vector<uint8_t>(8 * 1024 * 1024);  // this is only for tensor metadata, it can be small
    auto encoder_fwd_buf = std::vector<uint8_t>(ctx_size_mb * 1024 * 1024);
    ggml_allocr* fwd_alloc = ggml_allocr_new(encoder_fwd_buf.data(), encoder_fwd_buf.capacity(), 8);
    int tgt_lang_idx;
    if (tgt_lang == "unk") {
        tgt_lang_idx = model.vocab.token_to_id["<unk>"];
    } else {
        auto tgt_lang_ptr = model.vocab.token_to_id.find("__" + tgt_lang + "__"); 
        if (tgt_lang_ptr == model.vocab.token_to_id.end()) {
            std::cerr << "Unknown language " << tgt_lang << "\n";
            result.err = 1;
            return result;
        }
        tgt_lang_idx = tgt_lang_ptr->second;
    }


    // Reset the ggml_context
    model.ctx = ctx_from_buffer(encoder_buf);
    ggml_set_no_alloc(model.ctx, true);
    ggml_tensor* seqs = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, data.size(), 1);
    seqs->data = data.data();

    // Audio encoder
    ggml_cgraph* gf = unity_speech_encoder(model, seqs);
    ggml_allocr_alloc_graph(fwd_alloc, gf);
    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);
    // encoder_output is valid until we call `ggml_allocr_reset(fwd_alloc)`
    ggml_tensor* encoder_output = gf->nodes[gf->n_nodes - 1];

    // Beam search decoding
    const Hypothesis* hypo = unity_decode(model, opts, tgt_lang_idx, encoder_output, n_threads);

    // Drop language and bos token.
    ggml_tensor* tokens = ggml_slice(model.ctx, hypo[0].seq, 0, 2, 0);

    // Collect result string
    char result_str[4096];
    std::pair<std::vector<std::string>, std::vector<float>> p = fairseq2_spm_detokenize(&model, tokens, hypo[0].step_scores, (char*)&result_str);
    std::vector<std::string> result_tokens = p.first;
    std::vector<float> word_scores = p.second;

    std::unordered_map<std::string, float> lid_scores;
    std::vector<int> lang_ids;
    for (const auto& kv : model.vocab.token_to_id) {
        if (kv.first.substr(0, 2) == "__" && kv.first.substr(kv.first.size() - 2) == "__") {
            lang_ids.push_back(kv.second);
        }
    }
    std::sort(lang_ids.begin(), lang_ids.end());
    for (size_t i = 0; i < lang_ids.size(); ++i) {
        lid_scores[model.vocab.id_to_token[lang_ids[i]].text] = ggml_get_f32_1d(hypo[0].lid_scores, i); 
    }
    result.transcription = result_tokens;
    result.word_confidence_scores = word_scores;
    result.lid_scores = lid_scores;
    result.err = 0;
    ggml_free(model.ctx);
    ggml_allocr_reset(fwd_alloc);
    return result;
}


extern "C" Result unity_eval_text(fairseq2_model& model, const std::string& text, SequenceGeneratorOptions opts, std::string tgt_lang, int n_threads) {
    Result result;
    // The ctx_size_mb mostly depends of input length and model dim.
    int ctx_size_mb = opts.mem_mb;
    auto encoder_buf = std::vector<uint8_t>(ctx_size_mb * 1024 * 1024);
    auto encoder_fwd_buf = std::vector<uint8_t>(ctx_size_mb * 1024 * 1024);
    ggml_allocr* fwd_alloc = ggml_allocr_new(encoder_fwd_buf.data(), encoder_fwd_buf.capacity(), 8);
    int tgt_lang_idx = 0;
    if (model.hparams["multilingual"] != 0) {
        auto tgt_lang_ptr = model.vocab.token_to_id.find("__" + tgt_lang + "__"); 
        if (tgt_lang_ptr == model.vocab.token_to_id.end()) {
            std::cerr << "Unknown language " << tgt_lang << "\n";
            result.err = 1;
            return result;
        }
        tgt_lang_idx = tgt_lang_ptr->second;
    }

    // tokenize the input text
    model.ctx = ctx_from_buffer(encoder_buf);
    ggml_set_no_alloc(model.ctx, false);
    ggml_tensor* tokens_tensor = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, 64);
    ggml_set_no_alloc(model.ctx, true);
    fairseq2_spm_tokenize(&model, text.c_str(), tokens_tensor);
    
    // Text encoder
    ggml_cgraph* gf = unity_text_encoder(model, tokens_tensor);
    ggml_allocr_alloc_graph(fwd_alloc, gf);
    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);
    ggml_tensor* encoder_output = gf->nodes[gf->n_nodes - 1];
    
    // Beam search decoding
    const Hypothesis* hypo = unity_decode(model, opts, tgt_lang_idx, encoder_output, n_threads);
    
    // Drop language and bos token for multilingual, or only bos token for the bilingual model
    int token_offset = (model.hparams["multilingual"] != 0) ? 2 : 1;
    ggml_tensor* tgt_tokens = ggml_slice(model.ctx, hypo[0].seq, 0, token_offset, 0);

    // Collect result string
    char result_str[4096];

    std::pair<std::vector<std::string>, std::vector<float>> p = fairseq2_spm_detokenize(&model, tgt_tokens, hypo[0].step_scores, (char*)&result_str);
    std::vector<std::string> result_tokens = p.first;
    std::vector<float> word_scores = p.second;

    std::unordered_map<std::string, float> lid_scores;
    if (model.hparams["multilingual"] != 0) {
        std::vector<int> lang_ids;
        for (const auto& kv : model.vocab.token_to_id) {
            if (kv.first.substr(0, 2) == "__" && kv.first.substr(kv.first.size() - 2) == "__") {
                lang_ids.push_back(kv.second);
            }
        }
        std::sort(lang_ids.begin(), lang_ids.end());
        for (size_t i = 0; i < lang_ids.size(); ++i) {
            lid_scores[model.vocab.id_to_token[lang_ids[i]].text] = ggml_get_f32_1d(hypo[0].lid_scores, i); 
        }
        result.lid_scores = lid_scores;
    }
    result.transcription = result_tokens;
    result.word_confidence_scores = word_scores;
    result.err = 0;
    ggml_free(model.ctx);
    ggml_allocr_reset(fwd_alloc);
    return result;
}
