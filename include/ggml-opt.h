// This file contains functionality for training models using GGML.
// It is not strictly needed vs. just vanilla GGML but it provides a more high-level interface for common needs such as datasets.
// At the bottom of this file especially there are relatively high-level functions that are suitable use or adaptation in user code.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_opt_new_params {
        ggml_backend_t backend;
        ggml_context * ctx;

        struct ggml_tensor * inputs;
        struct ggml_tensor * logits;

        bool forward_only;  // if true, don't build the graphs for the backward pass
        int32_t opt_period; // after how many gradient accumulation steps an optimizer step should be done

        // AdamW optimizer parameters
        struct {
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float wd;    // weight decay for AdamW, use 0.0f to disable
        } adamw;
    };

    struct ggml_opt_new_context;
    struct ggml_opt_new_dataset;
    struct ggml_opt_new_result;

    GGML_API ggml_opt_new_params ggml_opt_new_default_params(
            ggml_backend_t       backend,
            struct ggml_tensor * inputs,
            struct ggml_tensor * logits);

    GGML_API struct ggml_opt_new_context * ggml_opt_new_init(struct ggml_opt_new_params params);
    GGML_API void ggml_opt_new_free(struct ggml_opt_new_context * opt_ctx);

    // set gradients to zero, initilize loss, and optionally reset the optimizer
    GGML_API void ggml_opt_new_reset(struct ggml_opt_new_context * opt_ctx, bool optimizer);

    GGML_API struct ggml_opt_new_dataset * ggml_opt_new_dataset_init(
            int64_t ne_datapoint, // number of elements per datapoint
            int64_t ne_label,     // number of elements per label
            int64_t ndata,        // total number of datapoints/labels
            int64_t ndata_shard); // number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
    GGML_API void ggml_opt_new_dataset_free(struct ggml_opt_new_dataset * dataset);

    // get underlying tensors that store the data
    GGML_API struct ggml_tensor * ggml_opt_new_dataset_data(struct ggml_opt_new_dataset * dataset);
    GGML_API struct ggml_tensor * ggml_opt_new_dataset_labels(struct ggml_opt_new_dataset * dataset);

    // shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
    GGML_API void ggml_opt_new_dataset_shuffle(struct ggml_opt_new_context * opt_ctx, struct ggml_opt_new_dataset * dataset, int64_t idata);

    // get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
    GGML_API void ggml_opt_new_dataset_get_batch(
            struct ggml_opt_new_dataset * dataset,
            struct ggml_tensor          * data_batch,
            struct ggml_tensor          * labels_batch,
            int64_t                       ibatch);

    GGML_API struct ggml_opt_new_result * ggml_opt_new_result_init();
    GGML_API void ggml_opt_new_result_free(struct ggml_opt_new_result * result);
    GGML_API void ggml_opt_new_result_reset(struct ggml_opt_new_result * result);

    // get underlying tensors that store data
    GGML_API struct ggml_tensor * ggml_opt_new_inputs(  struct ggml_opt_new_context * opt_ctx); // forward graph input tensor
    GGML_API struct ggml_tensor * ggml_opt_new_outputs( struct ggml_opt_new_context * opt_ctx); // forward graph output tensor
    GGML_API struct ggml_tensor * ggml_opt_new_labels(  struct ggml_opt_new_context * opt_ctx); // labels to compare outputs against
    GGML_API struct ggml_tensor * ggml_opt_new_loss(    struct ggml_opt_new_context * opt_ctx); // scalar tensor that contains the loss
    GGML_API struct ggml_tensor * ggml_opt_new_pred(    struct ggml_opt_new_context * opt_ctx); // predictions made by outputs
    GGML_API struct ggml_tensor * ggml_opt_new_ncorrect(struct ggml_opt_new_context * opt_ctx); // number of matching predictions between outputs and labels

    // do forward pass, increment result if not NULL
    GGML_API void ggml_opt_new_forward(struct ggml_opt_new_context * opt_ctx, struct ggml_opt_new_result  * result);

    // do forward pass, increment result if not NULL, do backward pass
    GGML_API void ggml_opt_new_forward_backward(struct ggml_opt_new_context * opt_ctx, struct ggml_opt_new_result * result);

    // get data from result, uncertainties are optional and can be ignored by passing NULL
    GGML_API void ggml_opt_new_result_ndata(   struct ggml_opt_new_result * result, int64_t * ndata);                  // write 1 value, number of datapoints
    GGML_API void ggml_opt_new_result_loss(    struct ggml_opt_new_result * result, double  * loss,     double * unc); // write 1 value
    GGML_API void ggml_opt_new_result_pred(    struct ggml_opt_new_result * result, int32_t * pred);                   // write ndata values
    GGML_API void ggml_opt_new_result_accuracy(struct ggml_opt_new_result * result, double  * accuracy, double * unc); // write 1 value


    // the functions below this line are relatively high-level and can be copied and adapted for user code
    // ---------------------------------------------------------------------------------------------------


    // signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
    typedef void (*ggml_opt_new_epoch_callback)(
            const char                  * prefix,      // name of the dataset subsection being evaluated
            struct ggml_opt_new_context * opt_ctx,
            struct ggml_opt_new_dataset * dataset,
            struct ggml_opt_new_result  * result,      // result associated with the dataset subsection
            int64_t                       ibatch,      // number of batches that have been evaluated so far
            int64_t                       ibatch_max,  // total number of batches in this dataset subsection
            int64_t                       t_start_us); // time at which the evaluation on the dataset subsection was started

    // do training on front of dataset, do evaluation only on back of dataset
    GGML_API void ggml_opt_new_epoch(
            struct ggml_opt_new_context * opt_ctx,
            struct ggml_opt_new_dataset * dataset,
            struct ggml_opt_new_result  * result_train,   // result to increment during training, ignored if NULL
            struct ggml_opt_new_result  * result_eval,    // result to increment during evaluation, ignored if NULL
            int64_t                       idata_split,    // data index at which to split training and evaluation
            ggml_opt_new_epoch_callback   callback_train,
            ggml_opt_new_epoch_callback   callback_eval);

    // callback that prints a progress bar on stderr
    GGML_API void ggml_opt_new_epoch_callback_progress_bar(
            const char                  * prefix,
            struct ggml_opt_new_context * opt_ctx,
            struct ggml_opt_new_dataset * dataset,
            struct ggml_opt_new_result  * result,
            int64_t                       ibatch,
            int64_t                       ibatch_max,
            int64_t                       t_start_us);

    // fit model defined by inputs and outputs to dataset
    GGML_API void ggml_opt_new_fit(
            ggml_backend_t         backend,        // backend for allocating the backward graph
            ggml_tensor          * inputs,
            ggml_tensor          * outputs,
            ggml_opt_new_dataset * dataset,
            int64_t                nepoch,
            int64_t                nbatch_logical,
            float                  val_split);     // validation split, must be in [0.0f, 1.0f)

#ifdef  __cplusplus
}
#endif
