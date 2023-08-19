#include "ggml/ggml.h"

#include "common-ggml.h"
#include "common.h"

#include <string>
#include <map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


// no defaults for now
struct mpt_hparams {
    int d_model      = 0;
    int max_seq_len  = 0;
    int n_heads      = 0;
    int n_layers     = 0;
    int n_vocab      = 0;
    float alibi_bias_max = 0;
    float clip_qkv       = 0;
    int ftype        = 0;
    int n_ctx        = 0;
};

struct mpt_layer {
    // pre normalization
    struct ggml_tensor * norm_1_weight;

    // attention
    struct ggml_tensor * c_attn_wqkv_weight;
    struct ggml_tensor * c_attn_out_proj_weight;

    // post normalization
    struct ggml_tensor * norm_2_weight;

    // ff
    struct ggml_tensor * ffn_up_proj;
    struct ggml_tensor * ffn_down_proj;
};

struct mpt_model {
    mpt_hparams hparams;

    struct ggml_tensor * wte_weight;    // position embedding
    struct ggml_tensor * norm_f_weight; // language model head

    std::vector<mpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct mpt_params {
    int n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    int seed           = -1; // RNG seed
    int n_predict      = 200; // new tokens to predict
    int n_batch        = 8; // batch size for prompt processing
    int n_ctx          = 512;

    std::string model      = ""; // model path
	
    // sampling parameters
    int top_k          = 0;
    float   top_p          = 1.0f;
    float   temp           = 0.8f;
    int repeat_last_n  = 64;
    float   repeat_penalty = 1.02f;
};

struct Mpt {
    Mpt(mpt_params params);
	
	// overload to get tokens one-by-one while ggml runs
	virtual void OnNewTokenProcessed(const std::string& token);
	
	// overload to get logs
    virtual void OnLogMessage(const std::string& information);

	// Push in message - get message+network output
    std::string Process(const std::string& message);
	
	// Push in message - get tokens vector (it has tokens count in it!)
	std::vector<int> TokenizeMessage(const std::string& message);
	
	// Push in tokens - get string results
	std::string ProcessTokenizedMessage(const std::vector<int>& embd_inp);	
	
	// Only logs as output. Computes some magic from https://huggingface.co/docs/transformers/perplexity
    void LogPerplexity(const std::string& message);
	
	 // Get some random prompt - may not follow the format expected by your trained model
    std::string GetRandomMessage();
	
    virtual ~Mpt();    
private:
    gpt_vocab vocab;
    mpt_model model; 
    mpt_params params; 
    std::mt19937 rng;  
};