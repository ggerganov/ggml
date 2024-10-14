# auto-generated file
import ggml.ffi as ffi
import numpy as np
class lib:
  @property
  def GGML_BACKEND_CPU(self) -> int: ...
  @property
  def GGML_BACKEND_GPU(self) -> int: ...
  @property
  def GGML_BACKEND_GPU_SPLIT(self) -> int: ...
  @property
  def GGML_FTYPE_ALL_F32(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_F16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q2_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q3_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_0(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_1(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_1_SOME_F16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_0(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_1(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q6_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q8_0(self) -> int: ...
  @property
  def GGML_FTYPE_UNKNOWN(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_ARMIJO(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_WOLFE(self) -> int: ...
  @property
  def GGML_LINESEARCH_DEFAULT(self) -> int: ...
  @property
  def GGML_LINESEARCH_FAIL(self) -> int: ...
  @property
  def GGML_LINESEARCH_INVALID_PARAMETERS(self) -> int: ...
  @property
  def GGML_LINESEARCH_MAXIMUM_ITERATIONS(self) -> int: ...
  @property
  def GGML_LINESEARCH_MAXIMUM_STEP(self) -> int: ...
  @property
  def GGML_LINESEARCH_MINIMUM_STEP(self) -> int: ...
  @property
  def GGML_OBJECT_GRAPH(self) -> int: ...
  @property
  def GGML_OBJECT_TENSOR(self) -> int: ...
  @property
  def GGML_OBJECT_WORK_BUFFER(self) -> int: ...
  @property
  def GGML_OPT_ADAM(self) -> int: ...
  @property
  def GGML_OPT_DID_NOT_CONVERGE(self) -> int: ...
  @property
  def GGML_OPT_FAIL(self) -> int: ...
  @property
  def GGML_OPT_INVALID_WOLFE(self) -> int: ...
  @property
  def GGML_OPT_LBFGS(self) -> int: ...
  @property
  def GGML_OPT_NO_CONTEXT(self) -> int: ...
  @property
  def GGML_OPT_OK(self) -> int: ...
  @property
  def GGML_OP_ACC(self) -> int: ...
  @property
  def GGML_OP_ADD(self) -> int: ...
  @property
  def GGML_OP_ADD1(self) -> int: ...
  @property
  def GGML_OP_ALIBI(self) -> int: ...
  @property
  def GGML_OP_ARGMAX(self) -> int: ...
  @property
  def GGML_OP_CLAMP(self) -> int: ...
  @property
  def GGML_OP_CONT(self) -> int: ...
  @property
  def GGML_OP_CONV_1D(self) -> int: ...
  @property
  def GGML_OP_CONV_2D(self) -> int: ...
  @property
  def GGML_OP_COUNT(self) -> int: ...
  @property
  def GGML_OP_CPY(self) -> int: ...
  @property
  def GGML_OP_CROSS_ENTROPY_LOSS(self) -> int: ...
  @property
  def GGML_OP_CROSS_ENTROPY_LOSS_BACK(self) -> int: ...
  @property
  def GGML_OP_DIAG(self) -> int: ...
  @property
  def GGML_OP_DIAG_MASK_INF(self) -> int: ...
  @property
  def GGML_OP_DIAG_MASK_ZERO(self) -> int: ...
  @property
  def GGML_OP_DIV(self) -> int: ...
  @property
  def GGML_OP_DUP(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN_BACK(self) -> int: ...
  @property
  def GGML_OP_FLASH_FF(self) -> int: ...
  @property
  def GGML_OP_GET_ROWS(self) -> int: ...
  @property
  def GGML_OP_GET_ROWS_BACK(self) -> int: ...
  @property
  def GGML_OP_LOG(self) -> int: ...
  @property
  def GGML_OP_MAP_BINARY(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM1(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM1_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM2(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM2_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM3(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM3_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_UNARY(self) -> int: ...
  @property
  def GGML_OP_MEAN(self) -> int: ...
  @property
  def GGML_OP_MUL(self) -> int: ...
  @property
  def GGML_OP_MUL_MAT(self) -> int: ...
  @property
  def GGML_OP_NONE(self) -> int: ...
  @property
  def GGML_OP_NORM(self) -> int: ...
  @property
  def GGML_OP_OUT_PROD(self) -> int: ...
  @property
  def GGML_OP_PERMUTE(self) -> int: ...
  @property
  def GGML_OP_POOL_1D(self) -> int: ...
  @property
  def GGML_OP_POOL_2D(self) -> int: ...
  @property
  def GGML_OP_POOL_AVG(self) -> int: ...
  @property
  def GGML_OP_POOL_COUNT(self) -> int: ...
  @property
  def GGML_OP_POOL_MAX(self) -> int: ...
  @property
  def GGML_OP_REPEAT(self) -> int: ...
  @property
  def GGML_OP_REPEAT_BACK(self) -> int: ...
  @property
  def GGML_OP_RESHAPE(self) -> int: ...
  @property
  def GGML_OP_RMS_NORM(self) -> int: ...
  @property
  def GGML_OP_RMS_NORM_BACK(self) -> int: ...
  @property
  def GGML_OP_ROPE(self) -> int: ...
  @property
  def GGML_OP_ROPE_BACK(self) -> int: ...
  @property
  def GGML_OP_SCALE(self) -> int: ...
  @property
  def GGML_OP_SET(self) -> int: ...
  @property
  def GGML_OP_SILU_BACK(self) -> int: ...
  @property
  def GGML_OP_SOFT_MAX(self) -> int: ...
  @property
  def GGML_OP_SOFT_MAX_BACK(self) -> int: ...
  @property
  def GGML_OP_SQR(self) -> int: ...
  @property
  def GGML_OP_SQRT(self) -> int: ...
  @property
  def GGML_OP_SUB(self) -> int: ...
  @property
  def GGML_OP_SUM(self) -> int: ...
  @property
  def GGML_OP_SUM_ROWS(self) -> int: ...
  @property
  def GGML_OP_TRANSPOSE(self) -> int: ...
  @property
  def GGML_OP_UNARY(self) -> int: ...
  @property
  def GGML_OP_VIEW(self) -> int: ...
  @property
  def GGML_OP_WIN_PART(self) -> int: ...
  @property
  def GGML_OP_WIN_UNPART(self) -> int: ...
  @property
  def GGML_TASK_COMPUTE(self) -> int: ...
  @property
  def GGML_TASK_FINALIZE(self) -> int: ...
  @property
  def GGML_TASK_INIT(self) -> int: ...
  @property
  def GGML_TYPE_COUNT(self) -> int: ...
  @property
  def GGML_TYPE_F16(self) -> int: ...
  @property
  def GGML_TYPE_F32(self) -> int: ...
  @property
  def GGML_TYPE_I16(self) -> int: ...
  @property
  def GGML_TYPE_I32(self) -> int: ...
  @property
  def GGML_TYPE_I8(self) -> int: ...
  @property
  def GGML_TYPE_Q2_K(self) -> int: ...
  @property
  def GGML_TYPE_Q3_K(self) -> int: ...
  @property
  def GGML_TYPE_Q4_0(self) -> int: ...
  @property
  def GGML_TYPE_Q4_1(self) -> int: ...
  @property
  def GGML_TYPE_Q4_K(self) -> int: ...
  @property
  def GGML_TYPE_Q5_0(self) -> int: ...
  @property
  def GGML_TYPE_Q5_1(self) -> int: ...
  @property
  def GGML_TYPE_Q5_K(self) -> int: ...
  @property
  def GGML_TYPE_Q6_K(self) -> int: ...
  @property
  def GGML_TYPE_Q8_0(self) -> int: ...
  @property
  def GGML_TYPE_Q8_1(self) -> int: ...
  @property
  def GGML_TYPE_Q8_K(self) -> int: ...
  @property
  def GGML_UNARY_OP_ABS(self) -> int: ...
  @property
  def GGML_UNARY_OP_ELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_GELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_GELU_QUICK(self) -> int: ...
  @property
  def GGML_UNARY_OP_NEG(self) -> int: ...
  @property
  def GGML_UNARY_OP_RELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_SGN(self) -> int: ...
  @property
  def GGML_UNARY_OP_SILU(self) -> int: ...
  @property
  def GGML_UNARY_OP_STEP(self) -> int: ...
  @property
  def GGML_UNARY_OP_TANH(self) -> int: ...
  @property
  def GGUF_TYPE_ARRAY(self) -> int: ...
  @property
  def GGUF_TYPE_BOOL(self) -> int: ...
  @property
  def GGUF_TYPE_COUNT(self) -> int: ...
  @property
  def GGUF_TYPE_FLOAT32(self) -> int: ...
  @property
  def GGUF_TYPE_INT16(self) -> int: ...
  @property
  def GGUF_TYPE_INT32(self) -> int: ...
  @property
  def GGUF_TYPE_INT8(self) -> int: ...
  @property
  def GGUF_TYPE_STRING(self) -> int: ...
  @property
  def GGUF_TYPE_UINT16(self) -> int: ...
  @property
  def GGUF_TYPE_UINT32(self) -> int: ...
  @property
  def GGUF_TYPE_UINT8(self) -> int: ...
  def abort_callback(data: ffi.CData) -> bool:
    """
    abort ggml_graph_compute when true

            bool (*abort_callback)(void * data);
    """
    ...
  def dequantize_row_q2_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """
    Dequantization

    void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int k);
    """
    ...
  def dequantize_row_q3_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q4_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q5_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q6_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q8_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int k);"""
    ...
  def ggml_abs(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_abs(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_abs_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_abs_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_acc(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_acc(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_acc_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_acc_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_add(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add1(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add1(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add1_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add1_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_alibi(ctx: ffi.CData, a: ffi.CData, n_past: int, n_head: int, bias_max: float) -> ffi.CData:
    """
    alibi position embedding
    in-place, returns view(a)

        struct ggml_tensor * ggml_alibi(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past,
                int                   n_head,
                float                 bias_max);
    """
    ...
  def ggml_allocr_alloc(alloc: ffi.CData, tensor: ffi.CData) -> None:
    """GGML_API void   ggml_allocr_alloc(struct ggml_allocr * alloc, struct ggml_tensor * tensor);"""
    ...
  def ggml_allocr_alloc_graph(alloc: ffi.CData, graph: ffi.CData) -> int:
    """GGML_API size_t ggml_allocr_alloc_graph(struct ggml_allocr * alloc, struct ggml_cgraph * graph);"""
    ...
  def ggml_allocr_free(alloc: ffi.CData) -> None:
    """GGML_API void   ggml_allocr_free(struct ggml_allocr * alloc);"""
    ...
  def ggml_allocr_is_measure(alloc: ffi.CData) -> bool:
    """GGML_API bool   ggml_allocr_is_measure(struct ggml_allocr * alloc);"""
    ...
  def ggml_allocr_new(data: ffi.CData, size: int, alignment: int) -> ffi.CData:
    """GGML_API struct ggml_allocr * ggml_allocr_new(void * data, size_t size, size_t alignment);"""
    ...
  def ggml_allocr_new_measure(alignment: int) -> ffi.CData:
    """GGML_API struct ggml_allocr * ggml_allocr_new_measure(size_t alignment);"""
    ...
  def ggml_allocr_reset(alloc: ffi.CData) -> None:
    """GGML_API void   ggml_allocr_reset(struct ggml_allocr * alloc);"""
    ...
  def ggml_allocr_set_parse_seq(alloc: ffi.CData, list: ffi.CData, n: int) -> None:
    """
    tell the allocator to parse nodes following the order described in the list
    you should call this if your graph are optimized to execute out-of-order

    GGML_API void   ggml_allocr_set_parse_seq(struct ggml_allocr * alloc, int * list, int n);
    """
    ...
  def ggml_are_same_shape(t0: ffi.CData, t1: ffi.CData) -> bool:
    """    GGML_API bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);"""
    ...
  def ggml_argmax(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    argmax along rows

        GGML_API struct ggml_tensor * ggml_argmax(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_blck_size(type: int) -> int:
    """    GGML_API int     ggml_blck_size (enum ggml_type type);"""
    ...
  def ggml_build_backward(ctx: ffi.CData, gf: ffi.CData, keep: bool) -> ffi.CData:
    """    GGML_API struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);"""
    ...
  def ggml_build_forward(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);"""
    ...
  def ggml_build_forward_ctx(ctx: ffi.CData, tensor: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_cgraph * ggml_build_forward_ctx(struct ggml_context * ctx, struct ggml_tensor * tensor);"""
    ...
  def ggml_build_forward_expand(cgraph: ffi.CData, tensor: ffi.CData) -> None:
    """    GGML_API void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);"""
    ...
  def ggml_cl_can_mul_mat(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> bool:
    """bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cl_free_data(tensor: ffi.CData) -> None:
    """void ggml_cl_free_data(const struct ggml_tensor* tensor);"""
    ...
  def ggml_cl_host_free(ptr: ffi.CData) -> None:
    """void   ggml_cl_host_free(void * ptr);"""
    ...
  def ggml_cl_host_malloc(size: int) -> ffi.CData:
    """void * ggml_cl_host_malloc(size_t size);"""
    ...
  def ggml_cl_init() -> None:
    """void ggml_cl_init(void);"""
    ...
  def ggml_cl_mul(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> None:
    """void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cl_mul_mat(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData, wdata: ffi.CData, wsize: int) -> None:
    """void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);"""
    ...
  def ggml_cl_mul_mat_get_wsize(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> int:
    """size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cl_transform_tensor(data: ffi.CData, tensor: ffi.CData) -> None:
    """void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);"""
    ...
  def ggml_clamp(ctx: ffi.CData, a: ffi.CData, min: float, max: float) -> ffi.CData:
    """
    clamp
    in-place, returns view(a)

        struct ggml_tensor * ggml_clamp(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 min,
                float                 max);
    """
    ...
  def ggml_cont(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    make contiguous

        GGML_API struct ggml_tensor * ggml_cont(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_conv_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, p0: int, d0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s0,  // stride
                int                   p0,  // padding
                int                   d0); // dilation
    """
    ...
  def ggml_conv_1d_ph(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s: int, d: int) -> ffi.CData:
    """
    conv_1d with padding = half
    alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)

        GGML_API struct ggml_tensor * ggml_conv_1d_ph(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s,
                int                   d);
    """
    ...
  def ggml_conv_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s0,
                int                   s1,
                int                   p0,
                int                   p1,
                int                   d0,
                int                   d1);
    """
    ...
  def ggml_cpu_has_arm_fma() -> int:
    """    GGML_API int ggml_cpu_has_arm_fma    (void);"""
    ...
  def ggml_cpu_has_avx() -> int:
    """    GGML_API int ggml_cpu_has_avx        (void);"""
    ...
  def ggml_cpu_has_avx2() -> int:
    """    GGML_API int ggml_cpu_has_avx2       (void);"""
    ...
  def ggml_cpu_has_avx512() -> int:
    """    GGML_API int ggml_cpu_has_avx512     (void);"""
    ...
  def ggml_cpu_has_avx512_vbmi() -> int:
    """    GGML_API int ggml_cpu_has_avx512_vbmi(void);"""
    ...
  def ggml_cpu_has_avx512_vnni() -> int:
    """    GGML_API int ggml_cpu_has_avx512_vnni(void);"""
    ...
  def ggml_cpu_has_blas() -> int:
    """    GGML_API int ggml_cpu_has_blas       (void);"""
    ...
  def ggml_cpu_has_clblast() -> int:
    """    GGML_API int ggml_cpu_has_clblast    (void);"""
    ...
  def ggml_cpu_has_cuda() -> int:
    """    GGML_API int ggml_cpu_has_cuda       (void);"""
    ...
  def ggml_cpu_has_f16c() -> int:
    """    GGML_API int ggml_cpu_has_f16c       (void);"""
    ...
  def ggml_cpu_has_fma() -> int:
    """    GGML_API int ggml_cpu_has_fma        (void);"""
    ...
  def ggml_cpu_has_fp16_va() -> int:
    """    GGML_API int ggml_cpu_has_fp16_va    (void);"""
    ...
  def ggml_cpu_has_gpublas() -> int:
    """    GGML_API int ggml_cpu_has_gpublas    (void);"""
    ...
  def ggml_cpu_has_neon() -> int:
    """    GGML_API int ggml_cpu_has_neon       (void);"""
    ...
  def ggml_cpu_has_sse3() -> int:
    """    GGML_API int ggml_cpu_has_sse3       (void);"""
    ...
  def ggml_cpu_has_vsx() -> int:
    """    GGML_API int ggml_cpu_has_vsx        (void);"""
    ...
  def ggml_cpu_has_wasm_simd() -> int:
    """    GGML_API int ggml_cpu_has_wasm_simd  (void);"""
    ...
  def ggml_cpy(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    a -> b, return view(b)

        GGML_API struct ggml_tensor * ggml_cpy(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_cross_entropy_loss(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cross_entropy_loss(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b);
    """
    ...
  def ggml_cross_entropy_loss_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cross_entropy_loss_back(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b,
                struct ggml_tensor          * c);
    """
    ...
  def ggml_cuda_assign_buffers(tensor: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_assign_buffers(struct ggml_tensor * tensor);"""
    ...
  def ggml_cuda_assign_buffers_force_inplace(tensor: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor);"""
    ...
  def ggml_cuda_assign_buffers_no_scratch(tensor: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor);"""
    ...
  def ggml_cuda_can_mul_mat(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> bool:
    """GGML_API bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cuda_compute_forward(params: ffi.CData, tensor: ffi.CData) -> bool:
    """GGML_API bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);"""
    ...
  def ggml_cuda_free_data(tensor: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_free_data(struct ggml_tensor * tensor);"""
    ...
  def ggml_cuda_free_scratch() -> None:
    """GGML_API void   ggml_cuda_free_scratch(void);"""
    ...
  def ggml_cuda_get_device_count() -> int:
    """GGML_API int    ggml_cuda_get_device_count(void);"""
    ...
  def ggml_cuda_get_device_description(device: int, description: ffi.CData, description_size: int) -> None:
    """GGML_API void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);"""
    ...
  def ggml_cuda_host_free(ptr: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_host_free(void * ptr);"""
    ...
  def ggml_cuda_host_malloc(size: int) -> ffi.CData:
    """GGML_API void * ggml_cuda_host_malloc(size_t size);"""
    ...
  def ggml_cuda_set_main_device(main_device: int) -> None:
    """GGML_API void   ggml_cuda_set_main_device(int main_device);"""
    ...
  def ggml_cuda_set_mul_mat_q(mul_mat_q: bool) -> None:
    """GGML_API void   ggml_cuda_set_mul_mat_q(bool mul_mat_q);"""
    ...
  def ggml_cuda_set_scratch_size(scratch_size: int) -> None:
    """GGML_API void   ggml_cuda_set_scratch_size(size_t scratch_size);"""
    ...
  def ggml_cuda_set_tensor_split(tensor_split: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_set_tensor_split(const float * tensor_split);"""
    ...
  def ggml_cuda_transform_tensor(data: ffi.CData, tensor: ffi.CData) -> None:
    """GGML_API void   ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor);"""
    ...
  def ggml_cycles() -> int:
    """    GGML_API int64_t ggml_cycles(void);"""
    ...
  def ggml_cycles_per_ms() -> int:
    """    GGML_API int64_t ggml_cycles_per_ms(void);"""
    ...
  def ggml_diag(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_diag(
            struct ggml_context     * ctx,
            struct ggml_tensor      * a);
    """
    ...
  def ggml_diag_mask_inf(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    set elements above the diagonal to -INF

        GGML_API struct ggml_tensor * ggml_diag_mask_inf(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_diag_mask_inf_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_diag_mask_zero(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    set elements above the diagonal to 0

        GGML_API struct ggml_tensor * ggml_diag_mask_zero(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_diag_mask_zero_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_div(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_div(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_div_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_div_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_dup(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_dup(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_dup_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_dup_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_dup_tensor(ctx: ffi.CData, src: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);"""
    ...
  def ggml_element_size(tensor: ffi.CData) -> int:
    """    GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);"""
    ...
  def ggml_elu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_elu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_elu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_elu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_flash_attn(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, masked: bool) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_attn(
                struct ggml_context * ctx,
                struct ggml_tensor  * q,
                struct ggml_tensor  * k,
                struct ggml_tensor  * v,
                bool                  masked);
    """
    ...
  def ggml_flash_attn_back(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, d: ffi.CData, masked: bool) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_attn_back(
               struct ggml_context * ctx,
               struct ggml_tensor  * q,
               struct ggml_tensor  * k,
               struct ggml_tensor  * v,
               struct ggml_tensor  * d,
               bool                  masked);
    """
    ...
  def ggml_flash_ff(ctx: ffi.CData, a: ffi.CData, b0: ffi.CData, b1: ffi.CData, c0: ffi.CData, c1: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_ff(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b0,
                struct ggml_tensor  * b1,
                struct ggml_tensor  * c0,
                struct ggml_tensor  * c1);
    """
    ...
  def ggml_format_name(tensor: ffi.CData, fmt: ffi.CData, *args2) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);"""
    ...
  def ggml_fp16_to_fp32(x: np.float16) -> float:
    """
    convert FP16 <-> FP32

        GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t x);
    """
    ...
  def ggml_fp16_to_fp32_row(x: ffi.CData, y: ffi.CData, n: int) -> None:
    """    GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int n);"""
    ...
  def ggml_fp32_to_fp16(x: float) -> np.float16:
    """    GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);"""
    ...
  def ggml_fp32_to_fp16_row(x: ffi.CData, y: ffi.CData, n: int) -> None:
    """    GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int n);"""
    ...
  def ggml_free(ctx: ffi.CData) -> None:
    """    GGML_API void                  ggml_free(struct ggml_context * ctx);"""
    ...
  def ggml_ftype_to_ggml_type(ftype: int) -> int:
    """
    TODO: temporary until model loading of ggml examples is refactored

        GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
    """
    ...
  def ggml_gelu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    TODO: double-check this computation is correct

        GGML_API struct ggml_tensor * ggml_gelu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_gelu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_gelu_quick(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu_quick(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_gelu_quick_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_get_data(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_data_f32(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_f32_1d(tensor: ffi.CData, i: int) -> float:
    """    GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);"""
    ...
  def ggml_get_i32_1d(tensor: ffi.CData, i: int) -> int:
    """    GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);"""
    ...
  def ggml_get_max_tensor_size(ctx: ffi.CData) -> int:
    """    GGML_API size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);"""
    ...
  def ggml_get_mem_buffer(ctx: ffi.CData) -> ffi.CData:
    """    GGML_API void *  ggml_get_mem_buffer     (const struct ggml_context * ctx);"""
    ...
  def ggml_get_mem_size(ctx: ffi.CData) -> int:
    """    GGML_API size_t  ggml_get_mem_size       (const struct ggml_context * ctx);"""
    ...
  def ggml_get_name(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API const char *         ggml_get_name   (const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_no_alloc(ctx: ffi.CData) -> bool:
    """    GGML_API bool    ggml_get_no_alloc(struct ggml_context * ctx);"""
    ...
  def ggml_get_rows(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_get_rows(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_get_rows_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_get_rows_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c);
    """
    ...
  def ggml_get_tensor(ctx: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);"""
    ...
  def ggml_get_unary_op(tensor: ffi.CData) -> int:
    """    GGML_API enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);"""
    ...
  def ggml_graph_compute(cgraph: ffi.CData, cplan: ffi.CData) -> int:
    """    GGML_API               int ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);"""
    ...
  def ggml_graph_compute_with_ctx(ctx: ffi.CData, cgraph: ffi.CData, n_threads: int) -> None:
    """
    same as ggml_graph_compute() but the work data is allocated as a part of the context
    note: the drawback of this API is that you must have ensured that the context has enough memory for the work data

        GGML_API void ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
    """
    ...
  def ggml_graph_dump_dot(gb: ffi.CData, gf: ffi.CData, filename: ffi.CData) -> None:
    """
    dump the graph into a file using the dot format

        GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
    """
    ...
  def ggml_graph_export(cgraph: ffi.CData, fname: ffi.CData) -> None:
    """    GGML_API void               ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);"""
    ...
  def ggml_graph_get_tensor(cgraph: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);"""
    ...
  def ggml_graph_import(fname: ffi.CData, ctx_data: ffi.CData, ctx_eval: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_cgraph ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);"""
    ...
  def ggml_graph_overhead() -> int:
    """    GGML_API size_t ggml_graph_overhead(void);"""
    ...
  def ggml_graph_plan(cgraph: ffi.CData, n_threads: int) -> ffi.CData:
    """
    ggml_graph_plan() has to be called before ggml_graph_compute()
    when plan.work_size > 0, caller must allocate memory for plan.work_data

        GGML_API struct ggml_cplan ggml_graph_plan   (struct ggml_cgraph * cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);
    """
    ...
  def ggml_graph_print(cgraph: ffi.CData) -> None:
    """
    print info and performance information for the graph

        GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
    """
    ...
  def ggml_graph_reset(cgraph: ffi.CData) -> None:
    """    GGML_API              void ggml_graph_reset  (struct ggml_cgraph * cgraph);"""
    ...
  def ggml_init(params: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);"""
    ...
  def ggml_init_cuda() -> None:
    """GGML_API void   ggml_init_cuda(void);"""
    ...
  def ggml_internal_get_type_traits(type: int) -> ffi.CData:
    """    ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);"""
    ...
  def ggml_is_contiguous(tensor: ffi.CData) -> bool:
    """    GGML_API bool ggml_is_contiguous(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_numa() -> bool:
    """    GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node"""
    ...
  def ggml_is_permuted(tensor: ffi.CData) -> bool:
    """    GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_quantized(type: int) -> bool:
    """    GGML_API bool    ggml_is_quantized(enum ggml_type type);"""
    ...
  def ggml_is_transposed(tensor: ffi.CData) -> bool:
    """    GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);"""
    ...
  def ggml_log(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_log(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_log_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_log_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_map_binary_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_f32(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b,
                       ggml_binary_op_f32_t   fun),
            "use ggml_map_custom2 instead");
    """
    ...
  def ggml_map_binary_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_inplace_f32(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b,
                       ggml_binary_op_f32_t   fun),
            "use ggml_map_custom2_inplace instead");
    """
    ...
  def ggml_map_custom1(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom1(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                ggml_custom1_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom1_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                       ggml_custom1_op_f32_t   fun),
            "use ggml_map_custom1 instead");
    """
    ...
  def ggml_map_custom1_inplace(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom1_inplace(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                ggml_custom1_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom1_inplace_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_inplace_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                       ggml_custom1_op_f32_t   fun),
            "use ggml_map_custom1_inplace instead");
    """
    ...
  def ggml_map_custom2(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom2(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                ggml_custom2_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom2_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                       ggml_custom2_op_f32_t   fun),
            "use ggml_map_custom2 instead");
    """
    ...
  def ggml_map_custom2_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom2_inplace(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                ggml_custom2_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom2_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_inplace_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                       ggml_custom2_op_f32_t   fun),
            "use ggml_map_custom2_inplace instead");
    """
    ...
  def ggml_map_custom3(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom3(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                struct ggml_tensor    * c,
                ggml_custom3_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom3_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                struct ggml_tensor           * c,
                       ggml_custom3_op_f32_t   fun),
            "use ggml_map_custom3 instead");
    """
    ...
  def ggml_map_custom3_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom3_inplace(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                struct ggml_tensor    * c,
                ggml_custom3_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom3_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_inplace_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                struct ggml_tensor           * c,
                       ggml_custom3_op_f32_t   fun),
            "use ggml_map_custom3_inplace instead");
    """
    ...
  def ggml_map_unary_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_f32(
                struct ggml_context        * ctx,
                struct ggml_tensor         * a,
                       ggml_unary_op_f32_t   fun),
            "use ggml_map_custom1 instead");
    """
    ...
  def ggml_map_unary_inplace_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_inplace_f32(
                struct ggml_context        * ctx,
                struct ggml_tensor         * a,
                       ggml_unary_op_f32_t   fun),
            "use ggml_map_custom1_inplace instead");
    """
    ...
  def ggml_mean(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    mean along rows

        GGML_API struct ggml_tensor * ggml_mean(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_metal_add_buffer(ctx: ffi.CData, name: ffi.CData, data: ffi.CData, size: int, max_size: int) -> bool:
    """
    creates a mapping between a host memory buffer and a device memory buffer
    - make sure to map all buffers used in the graph before calling ggml_metal_graph_compute
    - the mapping is used during computation to determine the arguments of the compute kernels
    - you don't need to keep the host memory buffer allocated as it is never accessed by Metal
    - max_size specifies the maximum size of a tensor and is used to create shared views such
    that it is guaranteed that the tensor will fit in at least one of the views


    bool ggml_metal_add_buffer(
            struct ggml_metal_context * ctx,
                           const char * name,
                                 void * data,
                               size_t   size,
                               size_t   max_size);
    """
    ...
  def ggml_metal_free(ctx: ffi.CData) -> None:
    """void ggml_metal_free(struct ggml_metal_context * ctx);"""
    ...
  def ggml_metal_get_concur_list(ctx: ffi.CData) -> ffi.CData:
    """
    output the concur_list for ggml_alloc

    int * ggml_metal_get_concur_list(struct ggml_metal_context * ctx);
    """
    ...
  def ggml_metal_get_tensor(ctx: ffi.CData, t: ffi.CData) -> None:
    """
    get data from the device into host memory

    void ggml_metal_get_tensor(struct ggml_metal_context * ctx, struct ggml_tensor * t);
    """
    ...
  def ggml_metal_graph_compute(ctx: ffi.CData, gf: ffi.CData) -> None:
    """
    same as ggml_graph_compute but uses Metal
    creates gf->n_threads command buffers in parallel

    void ggml_metal_graph_compute(struct ggml_metal_context * ctx, struct ggml_cgraph * gf);
    """
    ...
  def ggml_metal_graph_find_concurrency(ctx: ffi.CData, gf: ffi.CData, check_mem: bool) -> None:
    """
    try to find operations that can be run concurrently in the graph
    you should run it again if the topology of your graph changes

    void ggml_metal_graph_find_concurrency(struct ggml_metal_context * ctx, struct ggml_cgraph * gf, bool check_mem);
    """
    ...
  def ggml_metal_host_free(data: ffi.CData) -> None:
    """void   ggml_metal_host_free  (void * data);"""
    ...
  def ggml_metal_host_malloc(n: int) -> ffi.CData:
    """void * ggml_metal_host_malloc(size_t n);"""
    ...
  def ggml_metal_if_optimized(ctx: ffi.CData) -> int:
    """
    if the graph has been optimized for concurrently dispatch, return length of the concur_list if optimized

    int ggml_metal_if_optimized(struct ggml_metal_context * ctx);
    """
    ...
  def ggml_metal_init(n_cb: int) -> ffi.CData:
    """
    number of command buffers to use

    struct ggml_metal_context * ggml_metal_init(int n_cb);
    """
    ...
  def ggml_metal_set_n_cb(ctx: ffi.CData, n_cb: int) -> None:
    """
    set the number of command buffers to use

    void ggml_metal_set_n_cb(struct ggml_metal_context * ctx, int n_cb);
    """
    ...
  def ggml_metal_set_tensor(ctx: ffi.CData, t: ffi.CData) -> None:
    """
    set data from host memory into the device

    void ggml_metal_set_tensor(struct ggml_metal_context * ctx, struct ggml_tensor * t);
    """
    ...
  def ggml_mpi_backend_free() -> None:
    """void ggml_mpi_backend_free(void);"""
    ...
  def ggml_mpi_backend_init() -> None:
    """void ggml_mpi_backend_init(void);"""
    ...
  def ggml_mpi_eval_init(ctx_mpi: ffi.CData, n_tokens: ffi.CData, n_past: ffi.CData, n_threads: ffi.CData) -> None:
    """
    void ggml_mpi_eval_init(
            struct ggml_mpi_context * ctx_mpi,
                                int * n_tokens,
                                int * n_past,
                                int * n_threads);
    """
    ...
  def ggml_mpi_free(ctx: ffi.CData) -> None:
    """void ggml_mpi_free(struct ggml_mpi_context * ctx);"""
    ...
  def ggml_mpi_graph_compute_post(ctx_mpi: ffi.CData, gf: ffi.CData, n_layers: int) -> None:
    """
    void ggml_mpi_graph_compute_post(
            struct ggml_mpi_context * ctx_mpi,
                 struct ggml_cgraph * gf,
                                int   n_layers);
    """
    ...
  def ggml_mpi_graph_compute_pre(ctx_mpi: ffi.CData, gf: ffi.CData, n_layers: int) -> None:
    """
    void ggml_mpi_graph_compute_pre(
            struct ggml_mpi_context * ctx_mpi,
                 struct ggml_cgraph * gf,
                                int   n_layers);
    """
    ...
  def ggml_mpi_init() -> ffi.CData:
    """struct ggml_mpi_context * ggml_mpi_init(void);"""
    ...
  def ggml_mpi_rank(ctx: ffi.CData) -> int:
    """int ggml_mpi_rank(struct ggml_mpi_context * ctx);"""
    ...
  def ggml_mul(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_mul(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_mul_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_mul_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_mul_mat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    A: n columns, m rows
    B: n columns, p rows  (i.e. we transpose it internally)
    result is m columns, p rows

        GGML_API struct ggml_tensor * ggml_mul_mat(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_nbytes(tensor: ffi.CData) -> int:
    """    GGML_API size_t  ggml_nbytes      (const struct ggml_tensor * tensor);"""
    ...
  def ggml_nbytes_pad(tensor: ffi.CData) -> int:
    """    GGML_API size_t  ggml_nbytes_pad  (const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN"""
    ...
  def ggml_nbytes_split(tensor: ffi.CData, nrows_split: int) -> int:
    """    GGML_API size_t  ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split);"""
    ...
  def ggml_neg(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_neg(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_neg_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_neg_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_nelements(tensor: ffi.CData) -> int:
    """    GGML_API int64_t ggml_nelements   (const struct ggml_tensor * tensor);"""
    ...
  def ggml_new_f32(ctx: ffi.CData, value: float) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);"""
    ...
  def ggml_new_graph(ctx: ffi.CData) -> ffi.CData:
    """
    graph allocation in a context

        GGML_API struct ggml_cgraph * ggml_new_graph        (struct ggml_context * ctx);
    """
    ...
  def ggml_new_i32(ctx: ffi.CData, value: int) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);"""
    ...
  def ggml_new_tensor(ctx: ffi.CData, type: int, n_dims: int, ne: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int    n_dims,
                const int64_t *ne);
    """
    ...
  def ggml_new_tensor_1d(ctx: ffi.CData, type: int, ne0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_1d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0);
    """
    ...
  def ggml_new_tensor_2d(ctx: ffi.CData, type: int, ne0: int, ne1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_2d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0,
                int64_t ne1);
    """
    ...
  def ggml_new_tensor_3d(ctx: ffi.CData, type: int, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_3d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2);
    """
    ...
  def ggml_new_tensor_4d(ctx: ffi.CData, type: int, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_4d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                int64_t ne3);
    """
    ...
  def ggml_norm(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    normalize along rows
    TODO: eps is hardcoded to 1e-5 for now

        GGML_API struct ggml_tensor * ggml_norm(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_norm_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_nrows(tensor: ffi.CData) -> int:
    """    GGML_API int64_t ggml_nrows       (const struct ggml_tensor * tensor);"""
    ...
  def ggml_numa_init() -> None:
    """    GGML_API void    ggml_numa_init(void); // call once for better performance on NUMA systems"""
    ...
  def ggml_op_name(op: int) -> ffi.CData:
    """    GGML_API const char * ggml_op_name  (enum ggml_op   op);"""
    ...
  def ggml_op_symbol(op: int) -> ffi.CData:
    """    GGML_API const char * ggml_op_symbol(enum ggml_op   op);"""
    ...
  def ggml_opt(ctx: ffi.CData, params: ffi.CData, f: ffi.CData) -> int:
    """
    optimize the function defined by the tensor f

        GGML_API enum ggml_opt_result ggml_opt(
                struct ggml_context * ctx,
                struct ggml_opt_params params,
                struct ggml_tensor * f);
    """
    ...
  def ggml_opt_default_params(type: int) -> ffi.CData:
    """    GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);"""
    ...
  def ggml_opt_init(ctx: ffi.CData, opt: ffi.CData, params: ffi.CData, nx: int) -> None:
    """
    initialize optimizer context

        GGML_API void ggml_opt_init(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_opt_params params,
                int64_t nx);
    """
    ...
  def ggml_opt_resume(ctx: ffi.CData, opt: ffi.CData, f: ffi.CData) -> int:
    """
    continue optimizing the function defined by the tensor f

        GGML_API enum ggml_opt_result ggml_opt_resume(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_tensor * f);
    """
    ...
  def ggml_opt_resume_g(ctx: ffi.CData, opt: ffi.CData, f: ffi.CData, gf: ffi.CData, gb: ffi.CData) -> int:
    """
    continue optimizing the function defined by the tensor f

        GGML_API enum ggml_opt_result ggml_opt_resume_g(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_tensor * f,
                struct ggml_cgraph * gf,
                struct ggml_cgraph * gb);
    """
    ...
  def ggml_out_prod(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    A: m columns, n rows,
    B: p columns, n rows,
    result is m columns, p rows

        GGML_API struct ggml_tensor * ggml_out_prod(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_permute(ctx: ffi.CData, a: ffi.CData, axis0: int, axis1: int, axis2: int, axis3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_permute(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   axis0,
                int                   axis1,
                int                   axis2,
                int                   axis3);
    """
    ...
  def ggml_pool_1d(ctx: ffi.CData, a: ffi.CData, op: int, k0: int, s0: int, p0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_pool_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_op_pool     op,
                int                   k0, // kernel size
                int                   s0, // stride
                int                   p0); // padding
    """
    ...
  def ggml_pool_2d(ctx: ffi.CData, a: ffi.CData, op: int, k0: int, k1: int, s0: int, s1: int, p0: int, p1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_pool_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_op_pool     op,
                int                   k0,
                int                   k1,
                int                   s0,
                int                   s1,
                int                   p0,
                int                   p1);
    """
    ...
  def ggml_print_object(obj: ffi.CData) -> None:
    """    GGML_API void    ggml_print_object (const struct ggml_object * obj);"""
    ...
  def ggml_print_objects(ctx: ffi.CData) -> None:
    """    GGML_API void    ggml_print_objects(const struct ggml_context * ctx);"""
    ...
  def ggml_quantize_chunk(type: int, src: ffi.CData, dst: ffi.CData, start: int, n: int, hist: ffi.CData) -> int:
    """    GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);"""
    ...
  def ggml_quantize_q2_K(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """
    Quantization with histogram collection

    size_t ggml_quantize_q2_K(const float * src, void * dst, int n, int k, int64_t * hist);
    """
    ...
  def ggml_quantize_q3_K(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """size_t ggml_quantize_q3_K(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q4_0(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """    GGML_API size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q4_1(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """    GGML_API size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q4_K(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """size_t ggml_quantize_q4_K(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q5_0(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """    GGML_API size_t ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q5_1(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """    GGML_API size_t ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q5_K(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """size_t ggml_quantize_q5_K(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q6_K(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """size_t ggml_quantize_q6_K(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_quantize_q8_0(src: ffi.CData, dst: ffi.CData, n: int, k: int, hist: ffi.CData) -> int:
    """    GGML_API size_t ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);"""
    ...
  def ggml_relu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_relu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_relu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_relu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_repeat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    if a is the same shape as b, and a is not parameter, return a
    otherwise, return a new tensor: repeat(a) to fit in b

        GGML_API struct ggml_tensor * ggml_repeat(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_repeat_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_repeat_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_reshape(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    return view(a), b specifies the new shape
    TODO: when we start computing gradient, make a copy instead of view

        GGML_API struct ggml_tensor * ggml_reshape(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_reshape_1d(ctx: ffi.CData, a: ffi.CData, ne0: int) -> ffi.CData:
    """
    return view(a)
    TODO: when we start computing gradient, make a copy instead of view

        GGML_API struct ggml_tensor * ggml_reshape_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0);
    """
    ...
  def ggml_reshape_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_reshape_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1);
    """
    ...
  def ggml_reshape_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
    return view(a)
    TODO: when we start computing gradient, make a copy instead of view

        GGML_API struct ggml_tensor * ggml_reshape_3d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2);
    """
    ...
  def ggml_reshape_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_reshape_4d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                int64_t               ne3);
    """
    ...
  def ggml_rms_norm(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_rms_norm(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 eps);
    """
    ...
  def ggml_rms_norm_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    a - x
    b - dy
    TODO: update with configurable eps

        GGML_API struct ggml_tensor * ggml_rms_norm_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_rms_norm_inplace(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_rms_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 eps);
    """
    ...
  def ggml_rope(ctx: ffi.CData, a: ffi.CData, n_past: int, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
    rotary position embedding
    if mode & 1 == 1, skip n_past elements
    if mode & 2 == 1, GPT-NeoX style
    if mode & 4 == 1, ChatGLM style
    TODO: avoid creating a new tensor every time

        GGML_API struct ggml_tensor * ggml_rope(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past,
                int                   n_dims,
                int                   mode,
                int                   n_ctx);
    """
    ...
  def ggml_rope_back(ctx: ffi.CData, a: ffi.CData, n_past: int, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
    rotary position embedding backward, i.e compute dx from dy
    a - dy

        GGML_API struct ggml_tensor * ggml_rope_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past,
                int                   n_dims,
                int                   mode,
                int                   n_ctx);
    """
    ...
  def ggml_rope_custom(ctx: ffi.CData, a: ffi.CData, n_past: int, n_dims: int, mode: int, n_ctx: int, freq_base: float, freq_scale: float) -> ffi.CData:
    """
    custom RoPE

        GGML_API struct ggml_tensor * ggml_rope_custom(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past,
                int                   n_dims,
                int                   mode,
                int                   n_ctx,
                float                 freq_base,
                float                 freq_scale);
    """
    ...
  def ggml_rope_custom_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int, n_dims: int, mode: int, n_ctx: int, freq_base: float, freq_scale: float) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_rope_custom_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past,
                int                   n_dims,
                int                   mode,
                int                   n_ctx,
                float                 freq_base,
                float                 freq_scale);
    """
    ...
  def ggml_rope_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_rope_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past,
                int                   n_dims,
                int                   mode,
                int                   n_ctx);
    """
    ...
  def ggml_scale(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_scale(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_scale_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_scale_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_set(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return modified a

        GGML_API struct ggml_tensor * ggml_set(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_set_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_set_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                offset);
    """
    ...
  def ggml_set_1d_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_set_1d_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                offset);
    """
    ...
  def ggml_set_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return modified a

        GGML_API struct ggml_tensor * ggml_set_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                offset);
    """
    ...
  def ggml_set_2d_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return view(a)

        GGML_API struct ggml_tensor * ggml_set_2d_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                offset);
    """
    ...
  def ggml_set_f32(tensor: ffi.CData, value: float) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);"""
    ...
  def ggml_set_f32_1d(tensor: ffi.CData, i: int, value: float) -> None:
    """    GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);"""
    ...
  def ggml_set_i32(tensor: ffi.CData, value: int) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);"""
    ...
  def ggml_set_i32_1d(tensor: ffi.CData, i: int, value: int) -> None:
    """    GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);"""
    ...
  def ggml_set_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return view(a)

        GGML_API struct ggml_tensor * ggml_set_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_set_name(tensor: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_name   (      struct ggml_tensor * tensor, const char * name);"""
    ...
  def ggml_set_no_alloc(ctx: ffi.CData, no_alloc: bool) -> None:
    """    GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);"""
    ...
  def ggml_set_param(ctx: ffi.CData, tensor: ffi.CData) -> None:
    """
        GGML_API void ggml_set_param(
                struct ggml_context * ctx,
                struct ggml_tensor  * tensor);
    """
    ...
  def ggml_set_scratch(ctx: ffi.CData, scratch: ffi.CData) -> int:
    """    GGML_API size_t  ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);"""
    ...
  def ggml_set_zero(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);"""
    ...
  def ggml_sgn(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sgn(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sgn_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sgn_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_silu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_silu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_silu_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    a - x
    b - dy

        GGML_API struct ggml_tensor * ggml_silu_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_silu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_silu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_soft_max(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_soft_max(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_soft_max_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_soft_max_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_soft_max_back_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_soft_max_back_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_soft_max_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_soft_max_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqr(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqr(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqr_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqr_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqrt(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqrt(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqrt_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqrt_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_step(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_step(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_step_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_step_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sub(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sub(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_sub_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sub_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_sum(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    return scalar

        GGML_API struct ggml_tensor * ggml_sum(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sum_rows(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]

        GGML_API struct ggml_tensor * ggml_sum_rows(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_tanh(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_tanh(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_tanh_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_tanh_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_tensor_overhead() -> int:
    """
    use this to compute the memory overhead of a tensor

        GGML_API size_t ggml_tensor_overhead(void);
    """
    ...
  def ggml_time_init() -> None:
    """    GGML_API void    ggml_time_init(void); // call this once at the beginning of the program"""
    ...
  def ggml_time_ms() -> int:
    """    GGML_API int64_t ggml_time_ms(void);"""
    ...
  def ggml_time_us() -> int:
    """    GGML_API int64_t ggml_time_us(void);"""
    ...
  def ggml_transpose(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    alias for ggml_permute(ctx, a, 1, 0, 2, 3)

        GGML_API struct ggml_tensor * ggml_transpose(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_type_name(type: int) -> ffi.CData:
    """    GGML_API const char * ggml_type_name(enum ggml_type type);"""
    ...
  def ggml_type_size(type: int) -> int:
    """    GGML_API size_t  ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block"""
    ...
  def ggml_type_sizef(type: int) -> float:
    """    GGML_API float   ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float"""
    ...
  def ggml_unary(ctx: ffi.CData, a: ffi.CData, op: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_unary(
                struct ggml_context * ctx,
                 struct ggml_tensor * a,
                 enum ggml_unary_op op);
    """
    ...
  def ggml_unary_inplace(ctx: ffi.CData, a: ffi.CData, op: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_unary_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_unary_op op);
    """
    ...
  def ggml_used_mem(ctx: ffi.CData) -> int:
    """    GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);"""
    ...
  def ggml_vec_dot_q2_K_q8_K(n: int, s: ffi.CData, vx: ffi.CData, vy: ffi.CData) -> None:
    """
    Dot product

    void ggml_vec_dot_q2_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);
    """
    ...
  def ggml_vec_dot_q3_K_q8_K(n: int, s: ffi.CData, vx: ffi.CData, vy: ffi.CData) -> None:
    """void ggml_vec_dot_q3_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);"""
    ...
  def ggml_vec_dot_q4_K_q8_K(n: int, s: ffi.CData, vx: ffi.CData, vy: ffi.CData) -> None:
    """void ggml_vec_dot_q4_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);"""
    ...
  def ggml_vec_dot_q5_K_q8_K(n: int, s: ffi.CData, vx: ffi.CData, vy: ffi.CData) -> None:
    """void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);"""
    ...
  def ggml_vec_dot_q6_K_q8_K(n: int, s: ffi.CData, vx: ffi.CData, vy: ffi.CData) -> None:
    """void ggml_vec_dot_q6_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);"""
    ...
  def ggml_view_1d(ctx: ffi.CData, a: ffi.CData, ne0: int, offset: int) -> ffi.CData:
    """
    offset in bytes

        GGML_API struct ggml_tensor * ggml_view_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                size_t                offset);
    """
    ...
  def ggml_view_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, nb1: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_view_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                size_t                nb1, // row stride in bytes
                size_t                offset);
    """
    ...
  def ggml_view_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_view_3d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                size_t                nb1, // row   stride in bytes
                size_t                nb2, // slice stride in bytes
                size_t                offset);
    """
    ...
  def ggml_view_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_view_4d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                int64_t               ne3,
                size_t                nb1, // row   stride in bytes
                size_t                nb2, // slice stride in bytes
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_view_tensor(ctx: ffi.CData, src: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);"""
    ...
  def ggml_win_part(ctx: ffi.CData, a: ffi.CData, w: int) -> ffi.CData:
    """
    partition into non-overlapping windows with padding if needed
    example:
    a:   768   64   64    1
    w:    14
    res: 768   14   14    25
    used in sam

        GGML_API struct ggml_tensor * ggml_win_part(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   w);
    """
    ...
  def ggml_win_unpart(ctx: ffi.CData, a: ffi.CData, w0: int, h0: int, w: int) -> ffi.CData:
    """
    reverse of ggml_win_part
    used in sam

        GGML_API struct ggml_tensor * ggml_win_unpart(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   w0,
                int                   h0,
                int                   w);
    """
    ...
  def gguf_add_tensor(ctx: ffi.CData, tensor: ffi.CData) -> None:
    """
    manage tensor info

        GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
    """
    ...
  def gguf_find_key(ctx: ffi.CData, key: ffi.CData) -> int:
    """    GGML_API int          gguf_find_key(struct gguf_context * ctx, const char * key);"""
    ...
  def gguf_find_tensor(ctx: ffi.CData, name: ffi.CData) -> int:
    """    GGML_API int    gguf_find_tensor      (struct gguf_context * ctx, const char * name);"""
    ...
  def gguf_free(ctx: ffi.CData) -> None:
    """    GGML_API void gguf_free(struct gguf_context * ctx);"""
    ...
  def gguf_get_alignment(ctx: ffi.CData) -> int:
    """    GGML_API size_t gguf_get_alignment  (struct gguf_context * ctx);"""
    ...
  def gguf_get_arr_data(ctx: ffi.CData, i: int) -> ffi.CData:
    """    GGML_API const void * gguf_get_arr_data(struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_arr_n(ctx: ffi.CData, i: int) -> int:
    """    GGML_API int          gguf_get_arr_n   (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_arr_str(ctx: ffi.CData, key_id: int, i: int) -> ffi.CData:
    """    GGML_API const char * gguf_get_arr_str (struct gguf_context * ctx, int key_id, int i);"""
    ...
  def gguf_get_arr_type(ctx: ffi.CData, i: int) -> int:
    """    GGML_API enum gguf_type gguf_get_arr_type(struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_data(ctx: ffi.CData) -> ffi.CData:
    """    GGML_API void * gguf_get_data       (struct gguf_context * ctx);"""
    ...
  def gguf_get_data_offset(ctx: ffi.CData) -> int:
    """    GGML_API size_t gguf_get_data_offset(struct gguf_context * ctx);"""
    ...
  def gguf_get_key(ctx: ffi.CData, i: int) -> ffi.CData:
    """    GGML_API const char * gguf_get_key (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_kv_type(ctx: ffi.CData, i: int) -> int:
    """    GGML_API enum gguf_type gguf_get_kv_type (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_meta_data(ctx: ffi.CData, data: ffi.CData) -> None:
    """    GGML_API void   gguf_get_meta_data(struct gguf_context * ctx, void * data);"""
    ...
  def gguf_get_meta_size(ctx: ffi.CData) -> int:
    """
    get the size in bytes of the meta data (header, kv pairs, tensor info) including padding

        GGML_API size_t gguf_get_meta_size(struct gguf_context * ctx);
    """
    ...
  def gguf_get_n_kv(ctx: ffi.CData) -> int:
    """    GGML_API int          gguf_get_n_kv(struct gguf_context * ctx);"""
    ...
  def gguf_get_n_tensors(ctx: ffi.CData) -> int:
    """    GGML_API int    gguf_get_n_tensors    (struct gguf_context * ctx);"""
    ...
  def gguf_get_tensor_name(ctx: ffi.CData, i: int) -> ffi.CData:
    """    GGML_API char * gguf_get_tensor_name  (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_tensor_offset(ctx: ffi.CData, i: int) -> int:
    """    GGML_API size_t gguf_get_tensor_offset(struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_bool(ctx: ffi.CData, i: int) -> bool:
    """    GGML_API bool         gguf_get_val_bool(struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_f32(ctx: ffi.CData, i: int) -> float:
    """    GGML_API float        gguf_get_val_f32 (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_i16(ctx: ffi.CData, i: int) -> int:
    """    GGML_API int16_t      gguf_get_val_i16 (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_i32(ctx: ffi.CData, i: int) -> int:
    """    GGML_API int32_t      gguf_get_val_i32 (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_i8(ctx: ffi.CData, i: int) -> int:
    """    GGML_API int8_t       gguf_get_val_i8  (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_str(ctx: ffi.CData, i: int) -> ffi.CData:
    """    GGML_API const char * gguf_get_val_str (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_u16(ctx: ffi.CData, i: int) -> int:
    """    GGML_API uint16_t     gguf_get_val_u16 (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_u32(ctx: ffi.CData, i: int) -> int:
    """    GGML_API uint32_t     gguf_get_val_u32 (struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_u8(ctx: ffi.CData, i: int) -> int:
    """
    results are undefined if the wrong type is used for the key

        GGML_API uint8_t      gguf_get_val_u8  (struct gguf_context * ctx, int i);
    """
    ...
  def gguf_get_version(ctx: ffi.CData) -> int:
    """    GGML_API int    gguf_get_version    (struct gguf_context * ctx);"""
    ...
  def gguf_init_empty() -> ffi.CData:
    """    GGML_API struct gguf_context * gguf_init_empty(void);"""
    ...
  def gguf_init_from_file(fname: ffi.CData, params: ffi.CData) -> ffi.CData:
    """    GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);"""
    ...
  def gguf_set_arr_data(ctx: ffi.CData, key: ffi.CData, type: int, data: ffi.CData, n: int) -> None:
    """    GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);"""
    ...
  def gguf_set_arr_str(ctx: ffi.CData, key: ffi.CData, data: ffi.CData, n: int) -> None:
    """    GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);"""
    ...
  def gguf_set_kv(ctx: ffi.CData, src: ffi.CData) -> None:
    """
    set or add KV pairs from another context

        GGML_API void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);
    """
    ...
  def gguf_set_tensor_data(ctx: ffi.CData, name: ffi.CData, data: ffi.CData, size: int) -> None:
    """    GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);"""
    ...
  def gguf_set_tensor_type(ctx: ffi.CData, name: ffi.CData, type: int) -> None:
    """    GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);"""
    ...
  def gguf_set_val_bool(ctx: ffi.CData, key: ffi.CData, val: bool) -> None:
    """    GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);"""
    ...
  def gguf_set_val_f32(ctx: ffi.CData, key: ffi.CData, val: float) -> None:
    """    GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);"""
    ...
  def gguf_set_val_i16(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);"""
    ...
  def gguf_set_val_i32(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);"""
    ...
  def gguf_set_val_i8(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);"""
    ...
  def gguf_set_val_str(ctx: ffi.CData, key: ffi.CData, val: ffi.CData) -> None:
    """    GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);"""
    ...
  def gguf_set_val_u16(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);"""
    ...
  def gguf_set_val_u32(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);"""
    ...
  def gguf_set_val_u8(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """
    overrides existing values or adds a new one

        GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
    """
    ...
  def gguf_type_name(type: int) -> ffi.CData:
    """    GGML_API const char * gguf_type_name(enum gguf_type type);"""
    ...
  def gguf_write_to_file(ctx: ffi.CData, fname: ffi.CData, only_meta: bool) -> None:
    """
    write the entire context to a binary file

        GGML_API void gguf_write_to_file(struct gguf_context * ctx, const char * fname, bool only_meta);
    """
    ...
  def quantize_row_q2_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q2_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q2_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """
    Quantization

    void quantize_row_q2_K_reference(const float * restrict x, block_q2_K * restrict y, int k);
    """
    ...
  def quantize_row_q3_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q3_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q3_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q3_K_reference(const float * restrict x, block_q3_K * restrict y, int k);"""
    ...
  def quantize_row_q4_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q4_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_K_reference(const float * restrict x, block_q4_K * restrict y, int k);"""
    ...
  def quantize_row_q5_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q5_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_K_reference(const float * restrict x, block_q5_K * restrict y, int k);"""
    ...
  def quantize_row_q6_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q6_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q6_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int k);"""
    ...
  def quantize_row_q8_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q8_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_K_reference(const float * restrict x, block_q8_K * restrict y, int k);"""
    ...
