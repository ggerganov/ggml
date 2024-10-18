#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

struct ggml_opt_new_context {
    ggml_backend_t backend;
    ggml_backend_buffer_t buf;
    struct ggml_context * ctx;
    bool ctx_owned;
    std::mt19937 rng;

    struct ggml_tensor * inputs;
    struct ggml_tensor * outputs;
    struct ggml_tensor * labels;

    struct ggml_tensor * loss;
    struct ggml_tensor * pred;
    struct ggml_tensor * ncorrect;

    struct ggml_cgraph * gf;
    struct ggml_cgraph * gb_grad;
    struct ggml_cgraph * gb_opt;

    int32_t opt_period;
    int32_t opt_i;
};

struct ggml_opt_new_dataset {
    struct ggml_context * ctx;
    struct ggml_tensor  * data;
    struct ggml_tensor  * labels;

    int64_t ndata;
    int64_t ndata_shard;
    size_t  nbs_data;
    size_t  nbs_labels;

    std::vector<int64_t> permutation;
};

struct ggml_opt_new_result {
    int64_t              ndata      = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;
};

struct ggml_opt_new_params ggml_opt_new_default_params(
        ggml_backend_t       backend,
        struct ggml_tensor * inputs,
        struct ggml_tensor * logits) {
    return {
        /*backend    =*/ backend,
        /*ctx        =*/ nullptr,
        /*inputs     =*/ inputs,
        /*logits     =*/ logits,
        /*forward_only =*/ false,
        /*opt_period =*/ 1,
        /*adamw      =*/ {
            /*alpha      =*/ 0.001f,
            /*beta1      =*/ 0.9f,
            /*beta2      =*/ 0.999f,
            /*eps        =*/ 1e-8f,
            /*wd         =*/ 0.0f,
        },
    };
}

struct ggml_opt_new_context * ggml_opt_new_init(struct ggml_opt_new_params params) {
    struct ggml_opt_new_context * result = new struct ggml_opt_new_context;
    result->backend    = params.backend;
    result->inputs     = params.inputs;
    result->outputs     = params.logits;
    result->opt_period = params.opt_period;
    result->opt_i      = 0;

    if (params.ctx) {
        result->ctx = params.ctx;
        result->ctx_owned = false;
    } else {
        // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
        const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
        struct ggml_init_params ctx_params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(ctx_params);
        result->ctx_owned = true;
    }

    result->gf = ggml_new_graph_custom(result->ctx, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.

    ggml_set_input(result->inputs);

    ggml_set_output(result->outputs);
    ggml_build_forward_expand(result->gf, result->outputs);

    result->labels = ggml_dup_tensor(result->ctx, result->outputs);
    ggml_set_input(result->labels);

    result->loss = ggml_cross_entropy_loss(result->ctx, result->outputs, result->labels);
    ggml_set_output(result->loss);
    ggml_set_loss(result->loss);
    ggml_build_forward_expand(result->gf, result->loss);

    result->pred = ggml_argmax(result->ctx, result->outputs);
    ggml_set_output(result->pred);
    ggml_build_forward_expand(result->gf, result->pred);

    result->ncorrect = ggml_count_equal(result->ctx, result->pred, ggml_argmax(result->ctx, result->labels));
    ggml_set_output(result->ncorrect);
    ggml_build_forward_expand(result->gf, result->ncorrect);

    if (params.forward_only) {
        result->gb_grad = nullptr;
        result->gb_opt  = nullptr;

        result->buf = ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);

        return result;
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    result->gb_grad = ggml_graph_dup(result->ctx, result->gf);
    ggml_build_backward_expand(result->ctx, result->gf, result->gb_grad, params.opt_period > 1);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    result->gb_opt = ggml_graph_dup(result->ctx, result->gb_grad);

    for (int i = result->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = result->gf->nodes[i];

        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            struct ggml_tensor * opt_step = ggml_opt_step_adamw(
                result->ctx, node, node->grad, params.adamw.alpha, params.adamw.beta1, params.adamw.beta2, params.adamw.eps, params.adamw.wd);
            ggml_build_forward_expand(result->gb_opt, opt_step);
        }
    }

    if (result->ctx_owned) {
        result->buf = ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);
        ggml_opt_new_reset(result, /*optimizer =*/ true);
    } else {
        result->buf = nullptr;
    }

    return result;
}

void ggml_opt_new_free(struct ggml_opt_new_context * opt_ctx) {
    if (opt_ctx->ctx_owned) {
        ggml_backend_buffer_free(opt_ctx->buf);
        ggml_free(opt_ctx->ctx);
    }
    delete opt_ctx;
}

void ggml_opt_new_reset(struct ggml_opt_new_context * opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

struct ggml_opt_new_dataset * ggml_opt_new_dataset_init(int64_t ne_datapoint, int64_t ne_label, int64_t ndata, int64_t ndata_shard) {
    ggml_opt_new_dataset * result = new ggml_opt_new_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        const size_t nbytes_data   = ndata*ne_datapoint*sizeof(float) + ggml_tensor_overhead();
        const size_t nbytes_labels = ndata*ne_label    *sizeof(float) + ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ nbytes_data + nbytes_labels,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        result->ctx = ggml_init(params);
    }

    result->data   = ggml_new_tensor_2d(result->ctx, GGML_TYPE_F32, ne_datapoint, ndata);
    result->labels = ggml_new_tensor_2d(result->ctx, GGML_TYPE_F32, ne_label,     ndata);

    result->nbs_data   = ggml_nbytes(result->data)   * ndata_shard/ndata;
    result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_new_dataset_free(struct ggml_opt_new_dataset * dataset) {
    ggml_free(dataset->ctx);
    delete dataset;
}

struct ggml_tensor * ggml_opt_new_dataset_data(struct ggml_opt_new_dataset * dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_new_dataset_labels(struct ggml_opt_new_dataset * dataset) {
    return dataset->labels;
}

void ggml_opt_new_dataset_shuffle(struct ggml_opt_new_context * opt_ctx, struct ggml_opt_new_dataset * dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_new_dataset_get_batch(struct ggml_opt_new_dataset * dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(ggml_is_contiguous(data_batch));
    GGML_ASSERT(ggml_is_contiguous(labels_batch));

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    const size_t nb_labels_batch = ggml_nbytes(labels_batch);
    GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data   = (const char *)   dataset->data->data + ishard*dataset->nbs_data;
        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(data_batch,   ptr_data,   ishard_batch*dataset->nbs_data,   dataset->nbs_data);
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

struct ggml_opt_new_result * ggml_opt_new_result_init() {
    return new ggml_opt_new_result;
}

void ggml_opt_new_result_free(struct ggml_opt_new_result * result) {
    delete result;
}

void ggml_opt_new_result_reset(struct ggml_opt_new_result * result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

struct ggml_tensor * ggml_opt_new_inputs(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_new_outputs(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->outputs;
}

struct ggml_tensor * ggml_opt_new_labels(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->labels;
}

struct ggml_tensor * ggml_opt_new_loss(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_new_pred(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_new_ncorrect(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->ncorrect;
}

static void ggml_opt_new_eval_graph(struct ggml_opt_new_context * opt_ctx, ggml_cgraph * graph, ggml_opt_new_result * result) {
    GGML_ASSERT(graph);
    ggml_backend_graph_compute(opt_ctx->backend, graph);

    if (!result) {
        return;
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
    std::vector<int32_t> pred(ndata);
    ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
    result->pred.insert(result->pred.end(), pred.begin(), pred.end());

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

void ggml_opt_new_forward(struct ggml_opt_new_context * opt_ctx, ggml_opt_new_result * result) {
    ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gf, result);
}

void ggml_opt_new_forward_backward(struct ggml_opt_new_context * opt_ctx, ggml_opt_new_result * result) {
    if (opt_ctx->opt_period == 1) {
        ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gf, result);
        return;
    }

    const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    if (opt_i_next == 0) {
        ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gb_opt, result);
        ggml_opt_new_reset(opt_ctx, /*optimizer =*/ false);
    } else {
        ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gb_grad, result);
    }
    opt_ctx->opt_i = opt_i_next;
}

void ggml_opt_new_result_ndata(struct ggml_opt_new_result * result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_new_result_loss(struct ggml_opt_new_result * result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size();

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        sum         += loss;
        sum_squared += loss*loss;
    }

    *loss = sum/nbatches;

    if (!unc) {
        return;
    }

    *unc = nbatches >= 2 ? sqrt((sum_squared/nbatches - (*loss)*(*loss)) / (nbatches - 1)) : NAN;
}

void ggml_opt_new_result_pred(struct ggml_opt_new_result * result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_new_result_accuracy(struct ggml_opt_new_result * result, double * accuracy, double * unc) {
    *accuracy = double(result->ncorrect) / double(result->ndata);

    if (!unc) {
        return;
    }

    *unc = sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1));
}

void ggml_opt_new_epoch(
        struct ggml_opt_new_context * opt_ctx,
        struct ggml_opt_new_dataset * dataset,
        struct ggml_opt_new_result  * result_train,
        struct ggml_opt_new_result  * result_eval,
        int64_t                       idata_split,
        ggml_opt_new_epoch_callback   callback_train,
        ggml_opt_new_epoch_callback   callback_eval) {
    struct ggml_tensor * inputs = ggml_opt_new_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_new_labels(opt_ctx);

    const int64_t ndata       = ggml_opt_new_dataset_data(dataset)->ne[1];
    const int64_t ndata_batch = inputs->ne[1];
    GGML_ASSERT(ndata % ndata_batch == 0);

    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_new_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_new_forward_backward(opt_ctx, result_train);
        if (callback_train) {
            callback_train("train: ", opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_new_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_new_forward(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval("val:   ", opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_new_epoch_callback_progress_bar(
        const char                  * prefix,
        struct ggml_opt_new_context * opt_ctx,
        struct ggml_opt_new_dataset * dataset,
        struct ggml_opt_new_result  * result,
        int64_t                       ibatch,
        int64_t                       ibatch_max,
        int64_t                       t_start_us) {
    fprintf(stderr, "%s[", prefix);

    constexpr int64_t bar_length = 40;
    for (int64_t j = 0; j < bar_length; ++j) {
        const int64_t ibatch_j = ibatch_max * j/bar_length;
        if (ibatch_j < ibatch) {
            fprintf(stderr, "=");
        } else if (ibatch_max * (j - 1)/bar_length < ibatch) {
            fprintf(stderr, ">");
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_new_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_new_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_new_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "| data=%06ld/%06ld, loss=%.6lf+-%.6lf, accuracy=%.2lf+-%.2lf%%, t=%02ld:%02ld:%02ld, ETA=%02ld:%02ld:%02ld]\r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_new_fit(
        ggml_backend_t         backend,
        ggml_tensor          * inputs,
        ggml_tensor          * outputs,
        ggml_opt_new_dataset * dataset,
        int64_t                nepoch,
        int64_t                nbatch_logical,
        float                  val_split) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_new_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    ggml_opt_new_params params = ggml_opt_new_default_params(backend, inputs, outputs);
    params.opt_period = opt_period;
    ggml_opt_new_context * opt_ctx = ggml_opt_new_init(params);

    ggml_opt_new_reset(opt_ctx, /*optimizer =*/ true);
    ggml_opt_new_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).

    struct ggml_opt_new_result * result_train = ggml_opt_new_result_init();
    struct ggml_opt_new_result * result_val   = ggml_opt_new_result_init();

    for (int epoch = 0; epoch < nepoch; ++epoch) {
        ggml_opt_new_dataset_shuffle(opt_ctx, dataset, idata_split);

        ggml_opt_new_result_reset(result_train);
        ggml_opt_new_result_reset(result_val);

        fprintf(stderr, "%s: epoch %04d:\n", __func__, epoch);
        ggml_opt_new_epoch(
            opt_ctx, dataset, result_train, result_val, idata_split,
            ggml_opt_new_epoch_callback_progress_bar,
            ggml_opt_new_epoch_callback_progress_bar);
        fprintf(stderr, "\n");
    }

    int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
    const int64_t t_total_h = t_total_s / 3600;
    t_total_s -= t_total_h * 3600;
    const int64_t t_total_m = t_total_s / 60;
    t_total_s -= t_total_m * 60;
    fprintf(stderr, "%s: training took %02ld:%02ld:%02ld\n", __func__, t_total_h, t_total_m, t_total_s);

    // FIXME
    // if (ggml_backend_is_cpu(model.backend)) {
    //     std::string fname = model.arch + "-f32.ggml";
    //     fprintf(stderr, "%s: saving the GGML graph for the forward pass to %s\n", __func__, fname.c_str());
    //     ggml_graph_export(gf, fname.c_str());
    // } else {
    //     fprintf(stderr, "%s: not saving the GGML graph for the forward pass because this is only supported for the CPU backend\n", __func__);
    // }

    ggml_opt_new_free(opt_ctx);
    ggml_opt_new_result_free(result_train);
    ggml_opt_new_result_free(result_val);
}
