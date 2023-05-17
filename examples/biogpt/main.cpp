#include "biogpt.h"
#include "ggml.h"

#include <random>
#include <string> 
#include <vector>

int main(int argc, char **argv) {
    const int64_t t_main_start_us = ggml_time_us();

    biogpt_params params;

    if (biogpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if(params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    int64_t t_load_us = 0;

    biogpt_vocab vocab;
    biogpt_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if(!biogpt_model_load(params.model, model, vocab, params.verbosity)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<biogpt_vocab::id> embed_inp = gpt_tokenize(vocab, params.prompt, params.lang);

    params.n_predict = std::min(params.n_predict, model.hparams.n_positions - (int) embed_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embed_inp.size());
    for (int i = 0; i < std::min(8, (int) embed_inp.size()); i++) {
        printf("%d ", embed_inp[i]);
    }
    printf("\n\n");

    std::vector<biogpt_vocab::id> embed;

    // determine the required inference memory per token
    size_t mem_per_token = 0;
    biogpt_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = embed.size(); i < (int) embed_inp.size() + params.n_predict; i++) {
        // predict
        if (embed.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if(!biogpt_eval(model, params.n_threads, n_past, embed, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embed.size();
        embed.clear();

        if (i >= (int) embed_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            biogpt_vocab::id id = 0;

            // generation
            {
                const int64_t t_start_sample_us = ggml_time_us();
                id = biogpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            embed.push_back(id);
        } else {
            for (int k = i; k < (int) embed_inp.size(); k++) {
                embed.push_back(embed_inp[k]);
                if ((int) embed.size() > params.n_batch) {
                    break;
                }
            }
            i += embed.size() - 1;
        }

        std::vector<std::string> tokens;
        for (auto id : embed) {
            tokens.push_back(vocab.id_to_token[id]);
        }
        std::string decoded_word = gpt_decode(tokens, params.lang);
        printf("%s ", decoded_word.c_str());
        fflush(stdout);

        // end of text token
        if (embed.back() == model.hparams.n_vocab) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}