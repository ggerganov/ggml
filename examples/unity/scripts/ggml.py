"""This module is the core of the ggml-python library, it exposes a low-level [ctypes](https://docs.python.org/3/library/ctypes.html)-based interface for ggml.

Structures and functions in the `ggml.ggml` module map directly to the original ggml C library and
they operate at a fairly low level.
No additional runtime checks checks are performed nor is memory management handled automatically.
You've been warned :).

With that in mind here are some useful things to keep in mind

- Functions accept both ctypes types (c_int, c_bool, c_float, etc.) and Python types (int, bool, float, etc.) as parameters.
- Functions return Python types for simple values (int, bool, float, etc.) and ctypes types for complex values ([ggml_context_p][ggml.ggml_context_p], [ggml_tensor_p][ggml.ggml_tensor_p], etc.).
- Memory management is the responsibility of the user. The user must call [ggml.ggml_free][] on the context after calling [ggml.ggml_init][].

Example

```python
import ggml
import ctypes

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
ctx = ggml.ggml_init(params=params)

# Instantiate tensors
x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

# Use ggml operations to build a computational graph
x2 = ggml.ggml_mul(ctx, x, x)
f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

gf = ggml.ggml_new_graph(ctx)
ggml.ggml_build_forward_expand(gf, f)

# Set the input values
ggml.ggml_set_f32(x, 2.0)
ggml.ggml_set_f32(a, 3.0)
ggml.ggml_set_f32(b, 4.0)

# Compute the graph
ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)

# Get the output value
output = ggml.ggml_get_f32_1d(f, 0)
assert output == 16.0

# Free the context
ggml.ggml_free(ctx)
```

"""
import os
import sys
import ctypes
import pathlib
import importlib.resources
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable
from typing_extensions import TypeAlias


# Load the library
def load_shared_library(base_path: Path, lib_base_name: str):
    # Construct the paths to the possible shared library names
    # Searching for the library in the current directory under the name "libggml" (default name
    # for ggml) and "ggml" (default name for this repo)
    lib_names: List[str] = [
        f"lib{lib_base_name}.so",
        f"lib{lib_base_name}.dylib",
        f"{lib_base_name}.dll",
    ]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(base_path))
        cdll_args["winmode"] = 0

    for lib_name in lib_names:
        # Try to load the shared library, handling potential errors
        path = base_path / lib_name
        if not path.exists():
            continue
        try:
            return ctypes.CDLL(str(path), **cdll_args)
        except Exception as e:
            raise RuntimeError(f"Failed to load shared library '{path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found in {base_path}"
    )


base_path = pathlib.Path(__file__).parent.resolve() / "../../../build/examples/unity"
lib_base_name = "fairseq2_cpp"
lib = load_shared_library(base_path, lib_base_name)

#####################################################
# GGML Utility Types
#####################################################

CFloatArray: TypeAlias = "ctypes.Array[ctypes.c_float]"
CInt64Array: TypeAlias = "ctypes.Array[ctypes.c_int64]"
CIntPointer: TypeAlias = "ctypes._Pointer[ctypes.c_int]"  # type: ignore
CCharPointer: TypeAlias = "ctypes._Pointer[ctypes.c_char]"  # type: ignore


#####################################################
# GGML API
# source: ggml.h
#####################################################


# define GGML_FILE_MAGIC   0x67676d6c // "ggml"
# define GGML_FILE_VERSION 1
GGML_FILE_MAGIC = 0x67676D6C
GGML_FILE_VERSION = 1

# define GGML_QNT_VERSION        2    // bump this on quantization format changes
# define GGML_QNT_VERSION_FACTOR 1000 // do not change this
GGML_QNT_VERSION = 2
GGML_QNT_VERSION_FACTOR = 1000

# define GGML_MAX_DIMS           4
# define GGML_MAX_PARAMS         2048
# define GGML_MAX_CONTEXTS       64
# define GGML_MAX_SRC            10
# define GGML_MAX_NAME           64
# define GGML_MAX_OP_PARAMS      64
# define GGML_DEFAULT_N_THREADS  4
# define GGML_DEFAULT_GRAPH_SIZE 2048
GGML_MAX_DIMS = 4
GGML_MAX_PARAMS = 2048
GGML_MAX_CONTEXTS = 64
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64
GGML_MAX_OP_PARAMS = 64
GGML_DEFAULT_N_THREADS = 4
GGML_DEFAULT_GRAPH_SIZE = 2048

# #if UINTPTR_MAX == 0XFFFFFFFF
#     #define GGML_MEMALIGN 4
# #else
#     # define GGML_MEMALIGN 16
# #endif
GGML_MEMALIGN = (
    16 if ctypes.sizeof(ctypes.c_void_p) == 4 else 32
)  # FIXME: Check if this is correct

# #define GGML_EXIT_SUCCESS 0
GGML_EXIT_SUCCESS = 0
# #define GGML_EXIT_ABORTED 1
GGML_EXIT_ABORTED = 1

# define GGUF_MAGIC "GGUF"
GGUF_MAGIC = "GGUF"

# define GGUF_VERSION 3
GGUF_VERSION = 3

# #define GGUF_DEFAULT_ALIGNMENT 32
GGUF_DEFAULT_ALIGNMENT = 32

# TODO: Check if this is correct
# typedef uint16_t ggml_fp16_t;
ggml_fp16_t = ctypes.c_uint16

CFP16Array: TypeAlias = "ctypes.Array[ggml_fp16_t]"


# GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t x);
def ggml_fp16_to_fp32(x: ggml_fp16_t) -> float:
    return lib.ggml_fp16_to_fp32(x)


lib.ggml_fp16_to_fp32.argtypes = [ggml_fp16_t]
lib.ggml_fp16_to_fp32.restype = ctypes.c_float


# GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);
def ggml_fp32_to_fp16(x: ctypes.c_float) -> int:
    return lib.ggml_fp32_to_fp16(x)


lib.ggml_fp32_to_fp16.argtypes = [ctypes.c_float]
lib.ggml_fp32_to_fp16.restype = ggml_fp16_t


# GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, size_t n);
def ggml_fp16_to_fp32_row(
    x: CFP16Array,
    y: CFloatArray,
    n: Union[ctypes.c_int, int],
) -> None:
    return lib.ggml_fp16_to_fp32_row(x, y, n)


lib.ggml_fp16_to_fp32_row.argtypes = [
    ctypes.POINTER(ggml_fp16_t),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
lib.ggml_fp16_to_fp32_row.restype = None


# GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, size_t n);
def ggml_fp32_to_fp16_row(
    x: CFloatArray,
    y: CFP16Array,
    n: Union[ctypes.c_int, int],
) -> None:
    return lib.ggml_fp32_to_fp16_row(x, y, n)


lib.ggml_fp32_to_fp16_row.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ggml_fp16_t),
    ctypes.c_int,
]
lib.ggml_fp32_to_fp16_row.restype = None

# struct ggml_context;
ggml_context_p = ctypes.c_void_p
"""Opaque pointer to a ggml_context.

ggml_context structs are not accessed directly instead they must be created using [ggml_init](ggml.ggml_init) and freed using [ggml_free](ggml.ggml_free)."""


# enum ggml_type {
#     GGML_TYPE_F32  = 0,
#     GGML_TYPE_F16  = 1,
#     GGML_TYPE_Q4_0 = 2,
#     GGML_TYPE_Q4_1 = 3,
#     // GGML_TYPE_Q4_2 = 4, support has been removed
#     // GGML_TYPE_Q4_3 (5) support has been removed
#     GGML_TYPE_Q5_0 = 6,
#     GGML_TYPE_Q5_1 = 7,
#     GGML_TYPE_Q8_0 = 8,
#     GGML_TYPE_Q8_1 = 9,
#     GGML_TYPE_Q2_K = 10,
#     GGML_TYPE_Q3_K = 11,
#     GGML_TYPE_Q4_K = 12,
#     GGML_TYPE_Q5_K = 13,
#     GGML_TYPE_Q6_K = 14,
#     GGML_TYPE_Q8_K = 15,
#     GGML_TYPE_IQ2_XXS = 16,
#     GGML_TYPE_IQ2_XS  = 17,
#     GGML_TYPE_I8,
#     GGML_TYPE_I16,
#     GGML_TYPE_I32,
#     GGML_TYPE_COUNT,
# };
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_I8 = 18
GGML_TYPE_I16 = 19
GGML_TYPE_I32 = 20
GGML_TYPE_COUNT = 21

# // precision
# enum ggml_prec {
#     GGML_PREC_DEFAULT,
#     GGML_PREC_F32,
# };
GGML_PREC_DEFAULT = 0
GGML_PREC_F32 = 1

# enum ggml_backend_type {
#     GGML_BACKEND_CPU = 0,
#     GGML_BACKEND_GPU = 10,
#     GGML_BACKEND_GPU_SPLIT = 20,
# };
GGML_BACKEND_CPU = 0
GGML_BACKEND_GPU = 10
GGML_BACKEND_GPU_SPLIT = 20


# // model file types
# enum ggml_ftype {
#     GGML_FTYPE_UNKNOWN     = -1,
#     GGML_FTYPE_ALL_F32     = 0,
#     GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
# };
GGML_FTYPE_UNKNOWN = -1
GGML_FTYPE_ALL_F32 = 0
GGML_FTYPE_MOSTLY_F16 = 1
GGML_FTYPE_MOSTLY_Q4_0 = 2
GGML_FTYPE_MOSTLY_Q4_1 = 3
GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
GGML_FTYPE_MOSTLY_Q8_0 = 7
GGML_FTYPE_MOSTLY_Q5_0 = 8
GGML_FTYPE_MOSTLY_Q5_1 = 9
GGML_FTYPE_MOSTLY_Q2_K = 10
GGML_FTYPE_MOSTLY_Q3_K = 11
GGML_FTYPE_MOSTLY_Q4_K = 12
GGML_FTYPE_MOSTLY_Q5_K = 13
GGML_FTYPE_MOSTLY_Q6_K = 14
GGML_FTYPE_MOSTLY_IQ2_XXS = 15
GGML_FTYPE_MOSTLY_IQ2_XS = 16


# // available tensor operations:
# enum ggml_op {
#     GGML_OP_NONE = 0,

#     GGML_OP_DUP,
#     GGML_OP_ADD,
#     GGML_OP_ADD1,
#     GGML_OP_ACC,
#     GGML_OP_SUB,
#     GGML_OP_MUL,
#     GGML_OP_DIV,
#     GGML_OP_SQR,
#     GGML_OP_SQRT,
#     GGML_OP_LOG,
#     GGML_OP_SUM,
#     GGML_OP_SUM_ROWS,
#     GGML_OP_MEAN,
#     GGML_OP_ARGMAX,
#     GGML_OP_REPEAT,
#     GGML_OP_REPEAT_BACK,
#     GGML_OP_CONCAT,
#     GGML_OP_SILU_BACK,
#     GGML_OP_NORM, // normalize
#     GGML_OP_RMS_NORM,
#     GGML_OP_RMS_NORM_BACK,
#     GGML_OP_GROUP_NORM,

#     GGML_OP_MUL_MAT,
#     GGML_OP_MUL_MAT_ID,
#     GGML_OP_OUT_PROD,

#     GGML_OP_SCALE,
#     GGML_OP_SET,
#     GGML_OP_CPY,
#     GGML_OP_CONT,
#     GGML_OP_RESHAPE,
#     GGML_OP_VIEW,
#     GGML_OP_PERMUTE,
#     GGML_OP_TRANSPOSE,
#     GGML_OP_GET_ROWS,
#     GGML_OP_GET_ROWS_BACK,
#     GGML_OP_DIAG,
#     GGML_OP_DIAG_MASK_INF,
#     GGML_OP_DIAG_MASK_ZERO,
#     GGML_OP_SOFT_MAX,
#     GGML_OP_SOFT_MAX_BACK,
#     GGML_OP_ROPE,
#     GGML_OP_ROPE_BACK,
#     GGML_OP_ALIBI,
#     GGML_OP_CLAMP,
#     GGML_OP_CONV_TRANSPOSE_1D,
#     GGML_OP_IM2COL,
#     GGML_OP_CONV_TRANSPOSE_2D,
#     GGML_OP_POOL_1D,
#     GGML_OP_POOL_2D,
#     GGML_OP_UPSCALE, // nearest interpolate
#     GGML_OP_PAD,
#     GGML_OP_ARGSORT,
#     GGML_OP_LEAKY_RELU,

#     GGML_OP_FLASH_ATTN,
#     GGML_OP_FLASH_FF,
#     GGML_OP_FLASH_ATTN_BACK,
#     GGML_OP_WIN_PART,
#     GGML_OP_WIN_UNPART,
#     GGML_OP_GET_REL_POS,
#     GGML_OP_ADD_REL_POS,

#     GGML_OP_UNARY,

#     GGML_OP_MAP_UNARY,
#     GGML_OP_MAP_BINARY,

#     GGML_OP_MAP_CUSTOM1_F32,
#     GGML_OP_MAP_CUSTOM2_F32,
#     GGML_OP_MAP_CUSTOM3_F32,

#     GGML_OP_MAP_CUSTOM1,
#     GGML_OP_MAP_CUSTOM2,
#     GGML_OP_MAP_CUSTOM3,

#     GGML_OP_CROSS_ENTROPY_LOSS,
#     GGML_OP_CROSS_ENTROPY_LOSS_BACK,

#     GGML_OP_COUNT,
# };
GGML_OP_NONE = 0
GGML_OP_DUP = 1
GGML_OP_ADD = 2
GGML_OP_ADD1 = 3
GGML_OP_ACC = 4
GGML_OP_SUB = 5
GGML_OP_MUL = 6
GGML_OP_DIV = 7
GGML_OP_SQR = 8
GGML_OP_SQRT = 9
GGML_OP_LOG = 10
GGML_OP_SUM = 11
GGML_OP_SUM_ROWS = 12
GGML_OP_MEAN = 13
GGML_OP_ARGMAX = 14
GGML_OP_REPEAT = 15
GGML_OP_REPEAT_BACK = 16
GGML_OP_CONCAT = 17
GGML_OP_SILU_BACK = 18
GGML_OP_NORM = 19
GGML_OP_RMS_NORM = 20
GGML_OP_RMS_NORM_BACK = 21
GGML_OP_GROUP_NORM = 22
GGML_OP_MUL_MAT = 23
GGML_OP_MUL_MAT_ID = 24
GGML_OP_OUT_PROD = 25
GGML_OP_SCALE = 26
GGML_OP_SET = 27
GGML_OP_CPY = 28
GGML_OP_CONT = 29
GGML_OP_RESHAPE = 30
GGML_OP_VIEW = 31
GGML_OP_PERMUTE = 32
GGML_OP_TRANSPOSE = 33
GGML_OP_GET_ROWS = 34
GGML_OP_GET_ROWS_BACK = 35
GGML_OP_DIAG = 36
GGML_OP_DIAG_MASK_INF = 37
GGML_OP_DIAG_MASK_ZERO = 38
GGML_OP_SOFT_MAX = 39
GGML_OP_SOFT_MAX_BACK = 40
GGML_OP_ROPE = 41
GGML_OP_ROPE_BACK = 42
GGML_OP_ALIBI = 43
GGML_OP_CLAMP = 44
GGML_OP_CONV_TRANSPOSE_1D = 45
GGML_OP_IM2COL = 46
GGML_OP_CONV_TRANSPOSE_2D = 47
GGML_OP_POOL_1D = 48
GGML_OP_POOL_2D = 49
GGML_OP_UPSCALE = 50
GGML_OP_PAD = 51
GGML_OP_ARGSORT = 52
GGML_OP_LEAKY_RELU = 53
GGML_OP_FLASH_ATTN = 54
GGML_OP_FLASH_FF = 55
GGML_OP_FLASH_ATTN_BACK = 56
GGML_OP_WIN_PART = 57
GGML_OP_WIN_UNPART = 58
GGML_OP_GET_REL_POS = 59
GGML_OP_ADD_REL_POS = 60
GGML_OP_UNARY = 61
GGML_OP_MAP_UNARY = 62
GGML_OP_MAP_BINARY = 63
GGML_OP_MAP_CUSTOM1_F32 = 64
GGML_OP_MAP_CUSTOM2_F32 = 65
GGML_OP_MAP_CUSTOM3_F32 = 66
GGML_OP_MAP_CUSTOM1 = 67
GGML_OP_MAP_CUSTOM2 = 68
GGML_OP_MAP_CUSTOM3 = 69
GGML_OP_CROSS_ENTROPY_LOSS = 70
GGML_OP_CROSS_ENTROPY_LOSS_BACK = 71
GGML_OP_COUNT = 72

# enum ggml_unary_op {
#     GGML_UNARY_OP_ABS,
#     GGML_UNARY_OP_SGN,
#     GGML_UNARY_OP_NEG,
#     GGML_UNARY_OP_STEP,
#     GGML_UNARY_OP_TANH,
#     GGML_UNARY_OP_ELU,
#     GGML_UNARY_OP_RELU,
#     GGML_UNARY_OP_GELU,
#     GGML_UNARY_OP_GELU_QUICK,
#     GGML_UNARY_OP_SILU,
#     GGML_UNARY_OP_LEAKY

#     GGML_UNARY_OP_COUNT,
# };
GGML_UNARY_OP_ABS = 0
GGML_UNARY_OP_SGN = 1
GGML_UNARY_OP_NEG = 2
GGML_UNARY_OP_STEP = 3
GGML_UNARY_OP_TANH = 4
GGML_UNARY_OP_ELU = 5
GGML_UNARY_OP_RELU = 6
GGML_UNARY_OP_GELU = 7
GGML_UNARY_OP_GELU_QUICK = 8
GGML_UNARY_OP_SILU = 9
GGML_UNARY_OP_LEAKY = 10
GGML_UNARY_OP_COUNT = 11

# enum ggml_object_type {
#     GGML_OBJECT_TENSOR,
#     GGML_OBJECT_GRAPH,
#     GGML_OBJECT_WORK_BUFFER
# };
GGML_OBJECT_TENSOR = 0
GGML_OBJECT_GRAPH = 1
GGML_OBJECT_WORK_BUFFER = 2

# enum ggml_log_level {
#     GGML_LOG_LEVEL_ERROR = 2,
#     GGML_LOG_LEVEL_WARN = 3,
#     GGML_LOG_LEVEL_INFO = 4,
#     GGML_LOG_LEVEL_DEBUG = 5
# };
GGML_LOG_LEVEL_ERROR = 2
GGML_LOG_LEVEL_WARN = 3
GGML_LOG_LEVEL_INFO = 4
GGML_LOG_LEVEL_DEBUG = 5

# // ggml object
# struct ggml_object {
#     size_t offs;
#     size_t size;

#     struct ggml_object * next;

#     enum ggml_object_type type;


#     char padding[4];
# };
class ggml_object(ctypes.Structure):
    pass


ggml_object._fields_ = [
    ("offs", ctypes.c_size_t),
    ("size", ctypes.c_size_t),
    ("next", ctypes.POINTER(ggml_object)),
    ("type", ctypes.c_int),
    ("padding", ctypes.c_char * 4),
]

ggml_object_p: TypeAlias = "ctypes._Pointer[ggml_object]"  # type: ignore

GGML_OBJECT_SIZE = ctypes.sizeof(ggml_object)


# // n-dimensional tensor
# struct ggml_tensor {
#     enum ggml_type         type;
#     enum ggml_backend_type backend;

#     struct ggml_backend_buffer * buffer;

#     int64_t ne[GGML_MAX_DIMS]; // number of elements
#     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
#                                // nb[0] = ggml_type_size(type)
#                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
#                                // nb[i] = nb[i-1] * ne[i-1]

#     // compute data
#     enum ggml_op op;

#     // op params - allocated as int32_t for alignment
#     int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

#     bool is_param;

#     struct ggml_tensor * grad;
#     struct ggml_tensor * src[GGML_MAX_SRC];

#     // performance
#     int     perf_runs;
#     int64_t perf_cycles;
#     int64_t perf_time_us;

#     struct ggml_tensor * view_src;
#     size_t               view_offs;

#     void * data;

#     char name[GGML_MAX_NAME];

#     void * extra; // extra things e.g. for ggml-cuda.cu


#     char padding[8];
# };
class ggml_tensor(ctypes.Structure):
    """n-dimensional tensor

    Attributes:
        type (int): ggml_type
        backend (int): ggml_backend
        buffer (ctypes.pointer[ggml_backend_buffer]): pointer to backend buffer
        ne (ctypes.Array[ctypes.c_int64]): number of elements in each dimension
        nb (ctypes.Array[ctypes.c_size_t]): stride in bytes for each dimension
        op (int): ggml operation
        op_params (ctypes.Array[ctypes.c_int32]): `GGML_MAX_OP_PARAMS`-length array of operation parameters
        is_param (bool): is this a parameter tensor
        grad (ggml_tensor_p): reference to gradient tensor
        src (ctypes.Array[ggml_tensor_p]): `GGML_MAX_SRC`-length array of source tensors
        perf_runs (int): number of performance runs
        perf_cycles (int): number of cycles
        perf_time_us (int): time in microseconds
        view_src (ggml_tensor_p): pointer to tensor if this tensor is a view, None if the tensor is not a view
        view_offs (ctypes.c_size_t): offset into the data pointer of the view tensor
        data (ctypes.c_void_p): reference to raw tensor data
        name (bytes): name of tensor
        extra (ctypes.c_void_p): extra data (e.g. for CUDA)
    """

    pass


ggml_tensor._fields_ = [
    ("type", ctypes.c_int),
    ("backend", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    (
        "op_params",
        ctypes.c_int32 * (GGML_MAX_OP_PARAMS // ctypes.sizeof(ctypes.c_int32)),
    ),
    ("is_param", ctypes.c_bool),
    ("grad", ctypes.POINTER(ggml_tensor)),
    ("src", ctypes.POINTER(ggml_tensor) * GGML_MAX_SRC),
    ("perf_runs", ctypes.c_int),
    ("perf_cycles", ctypes.c_int64),
    ("perf_time_us", ctypes.c_int64),
    ("view_src", ctypes.POINTER(ggml_tensor)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]

GGML_TENSOR_SIZE = ctypes.sizeof(ggml_tensor)

ggml_tensor_p: TypeAlias = "ctypes._Pointer[ggml_tensor]"  # type: ignore
"""ctypes pointer to a [ggml_tensor][ggml.ggml_tensor]

Can be dereferenced to a [ggml_tensor][ggml.ggml_tensor] object using
the `.contents` attribute."""

abort_callback_t = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

# // the compute plan that needs to be prepared for ggml_graph_compute()
# // since https://github.com/ggerganov/ggml/issues/287
# struct ggml_cplan {
#     size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
#     uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

#     int n_threads;


#     // abort ggml_graph_compute when true
#     bool (*abort_callback)(void * data);
#     void * abort_callback_data;
# };
class ggml_cplan(ctypes.Structure):
    """Compute plan for a ggml computation graph

    Attributes:
        work_size (int): size of work buffer
        work_data (ctypes.pointer[ctypes.c_uint8]): work buffer
        n_threads (int): number of threads
        abort_callback (abort_callback_t): abort callback
        abort_callback_data (ctypes.c_void_p): abort callback data
    """

    _fields_ = [
        ("work_size", ctypes.c_size_t),
        ("work_data", ctypes.POINTER(ctypes.c_uint8)),
        ("n_threads", ctypes.c_int),
        (
            "abort_callback",
            abort_callback_t,
        ),
        ("abort_callback_data", ctypes.c_void_p),
    ]


GGML_CPLAN_SIZE = ctypes.sizeof(ggml_cplan)

ggml_cplan_p: TypeAlias = "ctypes._Pointer[ggml_cplan]"  # type: ignore
"""ctypes pointer to a [ggml_cplan][ggml.ggml_cplan]

Can be dereferenced to a [ggml_cplan][ggml.ggml_cplan] object using
the `.contents` attribute."""

# enum ggml_cgraph_eval_order {
#     GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
#     GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
#     GGML_CGRAPH_EVAL_ORDER_COUNT
# };
GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0
GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1
GGML_CGRAPH_EVAL_ORDER_COUNT = 2


# struct ggml_hash_set {
#     size_t size;
#     struct ggml_tensor ** keys;
# };
class ggml_hash_set(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("keys", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
    ]


# // computation graph
# struct ggml_cgraph {
#     int size;
#     int n_nodes;
#     int n_leafs;

#     struct ggml_tensor ** nodes;
#     struct ggml_tensor ** grads;
#     struct ggml_tensor ** leafs;

#     struct ggml_hash_set visited_hash_table;

#     enum ggml_cgraph_eval_order order;


#     // performance
#     int     perf_runs;
#     int64_t perf_cycles;
#     int64_t perf_time_us;
# };
class ggml_cgraph(ctypes.Structure):
    """ggml computation graph

    Attributes:
        n_nodes (int): number of nodes
        n_leafs (int): number of leafs
        nodes (ctypes.Array[ggml_tensor_p]): `n_nodes`-length array of compute tensors
        grads (ctypes.Array[ggml_tensor_p]): `n_nodes`-length array of gradient tensors
        leafs (ctypes.Array[ggml_tensor_p]): `n_leafs`-length array of parameter tensors
        visited_hash_table (ctypes.Array[ctypes.POINTER(ggml_tensor)]): hash table of visited tensors
        order (int): evaluation order
        perf_runs (int): number of runs
        perf_cycles (int): number of cycles
        perf_time_us (int): computation time in microseconds"""

    _fields_ = [
        ("size", ctypes.c_int),
        ("n_nodes", ctypes.c_int),
        ("n_leafs", ctypes.c_int),
        ("nodes", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("grads", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("leafs", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("visited_hash_table", ggml_hash_set),
        ("order", ctypes.c_int),
        ("perf_runs", ctypes.c_int),
        ("perf_cycles", ctypes.c_int64),
        ("perf_time_us", ctypes.c_int64),
    ]


ggml_cgraph_p: TypeAlias = "ctypes._Pointer[ggml_cgraph]"  # type: ignore
"""ctypes pointer to a [ggml_cgraph][ggml.ggml_cgraph]

Can be dereferenced to a [ggml_cgraph][ggml.ggml_cgraph] object using
the `.contents` attribute."""


# struct ggml_scratch {
#     size_t offs;
#     size_t size;
#     void * data;
# };
class ggml_scratch(ctypes.Structure):
    _fields_ = [
        ("offs", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    ]


# struct ggml_init_params {
#     // memory pool
#     size_t mem_size;   // bytes
#     void * mem_buffer; // if NULL, memory will be allocated internally
#     bool   no_alloc;   // don't allocate memory for the tensor data
# };
class ggml_init_params(ctypes.Structure):
    """Initialization parameters for a ggml context

    **NOTE**: Reference counting does not cross into ggml, if you allocate a memory buffer
    in python using ctypes Arrays or a numpy array, you must keep a reference to it until
    you free the ggml context otherwise you will encounter a segmentation fault.

    Attributes:
        mem_size (int): size of memory pool in bytes
        mem_buffer (ctypes.c_void_p): pointer to memory pool, if None, memory will be allocated internally
        no_alloc (bool): don't allocate memory for tensor data
    """

    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


# // compute types

# // NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
# // This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.
# enum ggml_task_type {
#     GGML_TASK_INIT = 0,
#     GGML_TASK_COMPUTE,
#     GGML_TASK_FINALIZE,
# };
GGML_TASK_INIT = 0
GGML_TASK_COMPUTE = 1
GGML_TASK_FINALIZE = 2

# struct ggml_compute_params {
#     enum ggml_task_type type;

#     // ith = thread index, nth = number of threads
#     int ith, nth;


#     // work buffer for all threads
#     size_t wsize;
#     void * wdata;
# };
class ggml_compute_params(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("ith", ctypes.c_int),
        ("nth", ctypes.c_int),
        ("wsize", ctypes.c_size_t),
        ("wdata", ctypes.c_void_p),
    ]


ggml_compute_params_p: TypeAlias = "ctypes._Pointer[ggml_compute_params]"  # type: ignore

# // misc


# GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
def ggml_time_init():
    return lib.ggml_time_init()


lib.ggml_time_init.argtypes = []
lib.ggml_time_init.restype = None


# GGML_API int64_t ggml_time_ms(void);
def ggml_time_ms() -> int:
    return lib.ggml_time_ms()


lib.ggml_time_ms.argtypes = []
lib.ggml_time_ms.restype = ctypes.c_int64


# GGML_API int64_t ggml_time_us(void);
def ggml_time_us() -> int:
    return lib.ggml_time_us()


lib.ggml_time_us.argtypes = []
lib.ggml_time_us.restype = ctypes.c_int64


# GGML_API int64_t ggml_cycles(void);
def ggml_cycles() -> int:
    return lib.ggml_cycles()


lib.ggml_cycles.argtypes = []
lib.ggml_cycles.restype = ctypes.c_int64


# GGML_API int64_t ggml_cycles_per_ms(void);
def ggml_cycles_per_ms() -> int:
    return lib.ggml_cycles_per_ms()


lib.ggml_cycles_per_ms.argtypes = []
lib.ggml_cycles_per_ms.restype = ctypes.c_int64


# GGML_API void    ggml_print_backtrace(void);
def ggml_print_backtrace():
    return lib.ggml_print_backtrace()


lib.ggml_print_backtrace.argtypes = []
lib.ggml_print_backtrace.restype = None


# GGML_API void    ggml_numa_init(void); // call once for better performance on NUMA systems
def ggml_numa_init():
    return lib.ggml_numa_init()


lib.ggml_numa_init.argtypes = []
lib.ggml_numa_init.restype = None


# GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node
def ggml_is_numa() -> bool:
    return lib.ggml_is_numa()


lib.ggml_is_numa.argtypes = []
lib.ggml_is_numa.restype = ctypes.c_bool


# GGML_API void    ggml_print_object (const struct ggml_object * obj);
def ggml_print_object(obj: ggml_object_p):
    return lib.ggml_print_object(obj)


lib.ggml_print_object.argtypes = [ctypes.POINTER(ggml_object)]
lib.ggml_print_object.restype = None


# GGML_API void    ggml_print_objects(const struct ggml_context * ctx);
def ggml_print_objects(ctx: ggml_context_p):
    return lib.ggml_print_objects(ctx)


lib.ggml_print_objects.argtypes = [ggml_context_p]
lib.ggml_print_objects.restype = None


# GGML_API int64_t ggml_nelements   (const struct ggml_tensor * tensor);
def ggml_nelements(
    tensor: ggml_tensor_p,
) -> int:
    """Get the number of elements in a tensor

    Parameters:
        tensor: tensor

    Returns:
        number of elements"""
    return lib.ggml_nelements(tensor)


lib.ggml_nelements.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_nelements.restype = ctypes.c_int64


# GGML_API int64_t ggml_nrows       (const struct ggml_tensor * tensor);
def ggml_nrows(
    tensor: ggml_tensor_p,
) -> int:
    """Get the number of rows in a tensor

    Parameters:
        tensor: tensor

    Returns:
        number of rows"""
    return lib.ggml_nrows(tensor)


lib.ggml_nrows.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_nrows.restype = ctypes.c_int64


# GGML_API size_t  ggml_nbytes      (const struct ggml_tensor * tensor);
def ggml_nbytes(
    tensor: ggml_tensor_p,
) -> int:
    """Get the number of bytes required to store tensor data

    Parameters:
        tensor: tensor

    Returns:
        number of bytes"""
    return lib.ggml_nbytes(tensor)


lib.ggml_nbytes.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_nbytes.restype = ctypes.c_size_t


# GGML_API size_t  ggml_nbytes_pad  (const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
def ggml_nbytes_pad(
    tensor: ggml_tensor_p,
) -> int:
    """Get the number of bytes required to store tensor data, padded to GGML_MEM_ALIGN

    Parameters:
        tensor: tensor

    Returns:
        number of bytes"""
    return lib.ggml_nbytes_pad(tensor)


lib.ggml_nbytes_pad.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_nbytes_pad.restype = ctypes.c_size_t


# GGML_API int    ggml_blck_size(enum ggml_type type);
def ggml_blck_size(type: Union[ctypes.c_int, int]) -> int:
    return lib.ggml_blck_size(type)


lib.ggml_blck_size.argtypes = [ctypes.c_int]
lib.ggml_blck_size.restype = ctypes.c_int


# GGML_API size_t ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
def ggml_type_size(type: Union[ctypes.c_int, int]) -> int:
    return lib.ggml_type_size(type)


lib.ggml_type_size.argtypes = [ctypes.c_int]
lib.ggml_type_size.restype = ctypes.c_size_t


# GGML_API size_t ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
def ggml_row_size(
    type: Union[ctypes.c_int, int],
    ne: int,
) -> int:
    return lib.ggml_row_size(type, ne)


lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
lib.ggml_row_size.restype = ctypes.c_size_t


# GGML_DEPRECATED(
# GGML_API double ggml_type_sizef(enum ggml_type type), // ggml_type_size()/ggml_blck_size() as float
# "use ggml_row_size() instead");
def ggml_type_sizef(type: Union[ctypes.c_int, int]) -> float:
    return lib.ggml_type_sizef(type)


lib.ggml_type_sizef.argtypes = [ctypes.c_int]
lib.ggml_type_sizef.restype = ctypes.c_double


# GGML_API const char * ggml_type_name(enum ggml_type type);
def ggml_type_name(type: Union[ctypes.c_int, int]) -> bytes:
    return lib.ggml_type_name(type)


lib.ggml_type_name.argtypes = [ctypes.c_int]
lib.ggml_type_name.restype = ctypes.c_char_p


# GGML_API const char * ggml_op_name  (enum ggml_op   op);
def ggml_op_name(op: Union[ctypes.c_int, int]) -> bytes:
    return lib.ggml_op_name(op)


lib.ggml_op_name.argtypes = [ctypes.c_int]
lib.ggml_op_name.restype = ctypes.c_char_p


# GGML_API const char * ggml_op_symbol(enum ggml_op   op);
def ggml_op_symbol(op: Union[ctypes.c_int, int]) -> bytes:
    return lib.ggml_op_symbol(op)


lib.ggml_op_symbol.argtypes = [ctypes.c_int]
lib.ggml_op_symbol.restype = ctypes.c_char_p


# GGML_API const char * ggml_unary_op_name(enum ggml_unary_op op);
def ggml_unary_op_name(op: Union[ctypes.c_int, int]) -> bytes:
    return lib.ggml_unary_op_name(op)


lib.ggml_unary_op_name.argtypes = [ctypes.c_int]
lib.ggml_unary_op_name.restype = ctypes.c_char_p


# GGML_API const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name
def ggml_op_desc(
    t: ggml_tensor_p,
) -> bytes:
    return lib.ggml_op_desc(t)


lib.ggml_op_desc.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_op_desc.restype = ctypes.c_char_p


# GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
def ggml_element_size(
    tensor: ggml_tensor_p,
) -> int:
    return lib.ggml_element_size(tensor)


lib.ggml_element_size.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_element_size.restype = ctypes.c_size_t


# GGML_API bool    ggml_is_quantized(enum ggml_type type);
def ggml_is_quantized(type: Union[ctypes.c_int, int]) -> bool:
    return lib.ggml_is_quantized(type)


lib.ggml_is_quantized.argtypes = [ctypes.c_int]
lib.ggml_is_quantized.restype = ctypes.c_bool


# // TODO: temporary until model loading of ggml examples is refactored
# GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
def ggml_ftype_to_ggml_type(ftype: Union[ctypes.c_int, int]) -> int:
    return lib.ggml_ftype_to_ggml_type(ftype)


lib.ggml_ftype_to_ggml_type.argtypes = [ctypes.c_int]
lib.ggml_ftype_to_ggml_type.restype = ctypes.c_int


# GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
def ggml_is_transposed(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is transposed

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is transposed else False"""
    return lib.ggml_is_transposed(tensor)


lib.ggml_is_transposed.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_transposed.restype = ctypes.c_bool


# GGML_API bool ggml_is_contiguous(const struct ggml_tensor * tensor);
def ggml_is_contiguous(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is contiguous

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is contiguous else False"""
    return lib.ggml_is_contiguous(tensor)


lib.ggml_is_contiguous.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_contiguous.restype = ctypes.c_bool


# GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
def ggml_is_permuted(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is permuted

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is permuted else False"""
    return lib.ggml_is_permuted(tensor)


lib.ggml_is_permuted.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_permuted.restype = ctypes.c_bool


# GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
def ggml_is_scalar(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is a scalar"""
    return lib.ggml_is_scalar(tensor)


lib.ggml_is_scalar.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_scalar.restype = ctypes.c_bool


# GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
def ggml_is_vector(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is a vector"""
    return lib.ggml_is_vector(tensor)


lib.ggml_is_vector.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_vector.restype = ctypes.c_bool


# GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
def ggml_is_matrix(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is a matrix"""
    return lib.ggml_is_matrix(tensor)


lib.ggml_is_matrix.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_matrix.restype = ctypes.c_bool


# GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
def ggml_is_3d(
    tensor: ggml_tensor_p,
) -> bool:
    """Check if a tensor is 3d"""
    return lib.ggml_is_3d(tensor)


lib.ggml_is_3d.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_is_3d.restype = ctypes.c_bool


# GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
def ggml_n_dims(
    tensor: ggml_tensor_p,
) -> int:
    """Get the number of dimensions in a tensor"""
    return lib.ggml_n_dims(tensor)


lib.ggml_n_dims.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_n_dims.restype = ctypes.c_int


# GGML_API bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
def ggml_are_same_shape(
    t0: ggml_tensor_p,
    t1: ggml_tensor_p,
) -> bool:
    """Check if two tensors have the same shape

    Parameters:
        t0: tensor 0
        t1: tensor 1

    Returns:
        True if tensors have the same shape else False"""
    return lib.ggml_are_same_shape(t0, t1)


lib.ggml_are_same_shape.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_are_same_shape.restype = ctypes.c_bool


# // use this to compute the memory overhead of a tensor
# GGML_API size_t ggml_tensor_overhead(void);
def ggml_tensor_overhead() -> int:
    """Overhead required for a tensor struct in bytes

    Returns:
        size of tensor struct in bytes"""
    return lib.ggml_tensor_overhead()


lib.ggml_tensor_overhead.argtypes = []
lib.ggml_tensor_overhead.restype = ctypes.c_size_t

# // main


# GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
def ggml_init(
    params: ggml_init_params,
) -> ggml_context_p:
    """Instantiate a new ggml context with params.

    You must call `ggml_free()` to free the context.

    Parameters:
        params: ggml init params

    Returns:
        Pointer to ggml_context"""
    return lib.ggml_init(params)


lib.ggml_init.argtypes = [ggml_init_params]
lib.ggml_init.restype = ggml_context_p


# GGML_API void                  ggml_free(struct ggml_context * ctx);
def ggml_free(ctx: ggml_context_p):
    """Free the ggml context.

    Parameters:
        ctx: ggml context"""
    return lib.ggml_free(ctx)


lib.ggml_free.argtypes = [ggml_context_p]
lib.ggml_free.restype = None


# GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);
def ggml_used_mem(ctx: ggml_context_p) -> int:
    """Return the amount of memory used by the ggml context in bytes.

    Parameters:
        ctx: ggml context

    Returns:
        amount of memory used in bytes"""
    return lib.ggml_used_mem(ctx)


lib.ggml_used_mem.argtypes = [ggml_context_p]
lib.ggml_used_mem.restype = ctypes.c_size_t


# GGML_API size_t  ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);
def ggml_set_scratch(ctx: ggml_context_p, scratch: ggml_scratch) -> int:
    """Set the scratch buffer for the ggml context."""
    return lib.ggml_set_scratch(ctx, scratch)


lib.ggml_set_scratch.argtypes = [ggml_context_p, ggml_scratch]
lib.ggml_set_scratch.restype = ctypes.c_size_t


# GGML_API bool    ggml_get_no_alloc(struct ggml_context * ctx);
def ggml_get_no_alloc(ctx: ggml_context_p) -> bool:
    """Return the no_alloc flag for the ggml context."""
    return lib.ggml_get_no_alloc(ctx)


lib.ggml_get_no_alloc.argtypes = [ggml_context_p]
lib.ggml_get_no_alloc.restype = ctypes.c_bool


# GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
def ggml_set_no_alloc(ctx: ggml_context_p, no_alloc: Union[ctypes.c_bool, bool]):
    """Set the no_alloc flag for the ggml context."""
    return lib.ggml_set_no_alloc(ctx, no_alloc)


lib.ggml_set_no_alloc.argtypes = [ggml_context_p, ctypes.c_bool]
lib.ggml_set_no_alloc.restype = None


# GGML_API void *  ggml_get_mem_buffer     (struct ggml_context * ctx);
def ggml_get_mem_buffer(ctx: ggml_context_p) -> Optional[ctypes.c_void_p]:
    """Return the memory buffer for the ggml context."""
    return lib.ggml_get_mem_buffer(ctx)


lib.ggml_get_mem_buffer.argtypes = [ggml_context_p]
lib.ggml_get_mem_buffer.restype = ctypes.c_void_p


# GGML_API size_t  ggml_get_mem_size       (struct ggml_context * ctx);
def ggml_get_mem_size(ctx: ggml_context_p) -> int:
    """Return the size of the memory buffer for the ggml context in bytes."""
    return lib.ggml_get_mem_size(ctx)


lib.ggml_get_mem_size.argtypes = [ggml_context_p]
lib.ggml_get_mem_size.restype = ctypes.c_size_t


# GGML_API size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);
def ggml_get_max_tensor_size(ctx: ggml_context_p) -> int:
    """Return the maximum size of a tensor in bytes."""
    return lib.ggml_get_max_tensor_size(ctx)


lib.ggml_get_max_tensor_size.argtypes = [ggml_context_p]
lib.ggml_get_max_tensor_size.restype = ctypes.c_size_t


# GGML_API struct ggml_tensor * ggml_new_tensor(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int    n_dims,
#         const int64_t *ne);
def ggml_new_tensor(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    n_dims: Union[ctypes.c_int, int],
    ne: CInt64Array,
) -> ggml_tensor_p:
    """Create a new tensor with the given type, number of dimensions, and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        n_dims: number of dimensions
        ne (ctypes.Array[ctypes.c_int64]): number of elements in each dimension (array of length n_dims)

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_tensor(ctx, type, n_dims, ne)


lib.ggml_new_tensor.argtypes = [
    ggml_context_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_new_tensor.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_new_tensor_1d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0);
def ggml_new_tensor_1d(
    ctx: ggml_context_p, type: Union[ctypes.c_int, int], ne0: Union[ctypes.c_int64, int]
) -> ggml_tensor_p:
    """Create a new 1-dimensional tensor with the given type and number of elements.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_tensor_1d(ctx, type, ne0)


lib.ggml_new_tensor_1d.argtypes = [ggml_context_p, ctypes.c_int, ctypes.c_int64]
lib.ggml_new_tensor_1d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_new_tensor_2d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1);
def ggml_new_tensor_2d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    """Create a new 2-dimensional tensor with the given type and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0
        ne1: number of elements in dimension 1

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_tensor_2d(ctx, type, ne0, ne1)


lib.ggml_new_tensor_2d.argtypes = [
    ggml_context_p,
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_new_tensor_2d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_new_tensor_3d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1,
#         int64_t ne2);
def ggml_new_tensor_3d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    """Create a new 3-dimensional tensor with the given type and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0
        ne1: number of elements in dimension 1
        ne2: number of elements in dimension 2

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)


lib.ggml_new_tensor_3d.argtypes = [
    ggml_context_p,
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_new_tensor_3d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_new_tensor_4d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1,
#         int64_t ne2,
#         int64_t ne3);
def ggml_new_tensor_4d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    """Create a new 4-dimensional tensor with the given type and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0
        ne1: number of elements in dimension 1
        ne2: number of elements in dimension 2

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)


lib.ggml_new_tensor_4d.argtypes = [
    ggml_context_p,
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_new_tensor_4d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
def ggml_new_i32(
    ctx: ggml_context_p, value: Union[ctypes.c_int32, int]
) -> ggml_tensor_p:
    """Create a 1 element tensor with the given integer value.

    Parameters:
        ctx: ggml context
        value: integer value

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_i32(ctx, value)


lib.ggml_new_i32.argtypes = [ggml_context_p, ctypes.c_int32]
lib.ggml_new_i32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
def ggml_new_f32(
    ctx: ggml_context_p,
    value: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Create a 1 element tensor with the given float value.

    Parameters:
        ctx: ggml context
        value: float value

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_new_f32(ctx, value)


lib.ggml_new_f32.argtypes = [ggml_context_p, ctypes.c_float]
lib.ggml_new_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
def ggml_dup_tensor(ctx: ggml_context_p, src: ggml_tensor_p) -> ggml_tensor_p:
    """Create a new tensor with the same type and dimensions as the source tensor.

    Parameters:
        ctx: ggml context
        src: source tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_dup_tensor(ctx, src)


lib.ggml_dup_tensor.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_dup_tensor.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);
def ggml_view_tensor(ctx: ggml_context_p, src: ggml_tensor_p) -> ggml_tensor_p:
    """Create a new tensor with the same type, dimensions and data as the source tensor.

    Parameters:
        ctx: ggml context
        src: source tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_view_tensor(ctx, src)


lib.ggml_view_tensor.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_view_tensor.restype = ctypes.POINTER(ggml_tensor)


# // Context tensor enumeration and lookup
# GGML_API struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
def ggml_get_first_tensor(ctx: ggml_context_p) -> ggml_tensor_p:
    """Get the first tensor from the ggml context.

    Parameters:
        ctx: ggml context

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_get_first_tensor(ctx)


lib.ggml_get_first_tensor.argtypes = [ggml_context_p]
lib.ggml_get_first_tensor.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);
def ggml_get_next_tensor(ctx: ggml_context_p, tensor: ggml_tensor_p) -> ggml_tensor_p:
    """Get the next tensor from the ggml context.

    Parameters:
        ctx: ggml context
        tensor: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_get_next_tensor(ctx, tensor)


lib.ggml_get_next_tensor.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_get_next_tensor.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
def ggml_get_tensor(ctx: ggml_context_p, name: bytes) -> ggml_tensor_p:
    """Get a tensor from the ggml context by name.

    Parameters:
        ctx: ggml context
        name: name of tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_get_tensor(ctx, name)


lib.ggml_get_tensor.argtypes = [ggml_context_p, ctypes.c_char_p]
lib.ggml_get_tensor.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
def ggml_set_zero(
    tensor: ggml_tensor_p,
) -> ggml_tensor_p:
    """Zero all elements in a tensor.

    Parameters:
        tensor: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_set_zero(tensor)


lib.ggml_set_zero.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_set_zero.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
def ggml_set_i32(
    tensor: ggml_tensor_p,
    value: Union[ctypes.c_int32, int],
) -> ggml_tensor_p:
    """Set all elements in a tensor to the given integer value.

    Parameters:
        tensor: tensor
        value: integer value

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_set_i32(tensor, value)


lib.ggml_set_i32.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int32]
lib.ggml_set_i32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
def ggml_set_f32(
    tensor: ggml_tensor_p,
    value: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Set all elements in a tensor to the given float value.

    Parameters:
        tensor: tensor
        value: float value

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_set_f32(tensor, value)


lib.ggml_set_f32.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_float]
lib.ggml_set_f32.restype = ctypes.POINTER(ggml_tensor)


# // Converts a flat index into coordinates
# GGML_API void    ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);
def ggml_unravel_index(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int64, int],
    i0,  # type: "ctypes._Pointer(ctypes.c_int64)" # type: ignore
    i1,  # type: "ctypes._Pointer(ctypes.c_int64)" # type: ignore
    i2,  # type: "ctypes._Pointer(ctypes.c_int64)" # type: ignore
    i3,  # type: "ctypes._Pointer(ctypes.c_int64)" # type: ignore
):
    """Convert a flat index into coordinates.

    Parameters:
        tensor: tensor
        i: flat index
        i0: pointer to index 0
        i1: pointer to index 1
        i2: pointer to index 2
        i3: pointer to index 3"""
    return lib.ggml_unravel_index(tensor, i, i0, i1, i2, i3)


lib.ggml_unravel_index.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_unravel_index.restype = None


# GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
def ggml_get_i32_1d(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int, int],
) -> int:
    """Get the integer value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor
        i: index of element

    Returns:
        integer value of element at index i"""
    return lib.ggml_get_i32_1d(tensor, i)


lib.ggml_get_i32_1d.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_get_i32_1d.restype = ctypes.c_int32


# GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
def ggml_set_i32_1d(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int, int],
    value: Union[ctypes.c_int32, int],
):
    """Set the integer value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor
        i: index of element
        value: integer value to set element to"""
    return lib.ggml_set_i32_1d(tensor, i, value)


lib.ggml_set_i32_1d.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int32,
]
lib.ggml_set_i32_1d.restype = None


# GGML_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
def ggml_get_i32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
) -> int:
    """Get the integer value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3

    Returns:
        integer value of element at coordinates"""
    return lib.ggml_get_i32_nd(tensor, i0, i1, i2, i3)


lib.ggml_get_i32_nd.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_get_i32_nd.restype = ctypes.c_int32


# GGML_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
def ggml_set_i32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
    value: Union[ctypes.c_int32, int],
):
    """Set the integer value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3
        value: integer value to set element to"""
    return lib.ggml_set_i32_nd(tensor, i0, i1, i2, i3, value)


lib.ggml_set_i32_nd.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int32,
]
lib.ggml_set_i32_nd.restype = None


# GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
def ggml_get_f32_1d(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int, int],
) -> float:
    """Get the float value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor

    Returns:
        float value of element at index i"""
    return lib.ggml_get_f32_1d(tensor, i)


lib.ggml_get_f32_1d.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_get_f32_1d.restype = ctypes.c_float


# GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
def ggml_set_f32_1d(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int, int],
    value: Union[ctypes.c_float, float],
):
    """Set the float value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor
        i: index of element
        value: float value to set element to"""
    return lib.ggml_set_f32_1d(tensor, i, value)


lib.ggml_set_f32_1d.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_float,
]
lib.ggml_set_f32_1d.restype = None


# GGML_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
def ggml_get_f32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
) -> float:
    """Get the float value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3

    Returns:
        float value of element at coordinates"""
    return lib.ggml_get_f32_nd(tensor, i0, i1, i2, i3)


lib.ggml_get_f32_nd.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_get_f32_nd.restype = ctypes.c_float


# GGML_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
def ggml_set_f32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
    value: Union[ctypes.c_float, float],
):
    """Set the float value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3
        value: float value to set element to"""
    return lib.ggml_set_f32_nd(tensor, i0, i1, i2, i3, value)


lib.ggml_set_f32_nd.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.ggml_set_f32_nd.restype = None


# GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
def ggml_get_data(
    tensor: ggml_tensor_p,
) -> Optional[ctypes.c_void_p]:
    """Get the data pointer of a tensor.

    Parameters:
        tensor: tensor

    Returns:
        Pointer to data, or None if tensor has no data"""
    return lib.ggml_get_data(tensor)


lib.ggml_get_data.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_data.restype = ctypes.c_void_p


# GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);
def ggml_get_data_f32(
    tensor: ggml_tensor_p,
) -> Optional[CFloatArray]:
    """Get the data pointer of a tensor as a float array.

    Parameters:
        tensor: tensor

    Returns:
        (Optional[ctypes.Array[ctypes.c_float]]): array of float to data, or None if tensor has no data
    """
    return lib.ggml_get_data_f32(tensor)


lib.ggml_get_data_f32.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_data_f32.restype = ctypes.POINTER(ctypes.c_float)


# GGML_API enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
def ggml_get_unary_op(
    tensor: ggml_tensor_p,
) -> int:
    """Get the unary operation of a tensor.

    Parameters:
        tensor: tensor

    Returns:
        unary operation"""
    return lib.ggml_get_unary_op(tensor)


lib.ggml_get_unary_op.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_unary_op.restype = ctypes.c_int


# GGML_API const char *         ggml_get_name(const struct ggml_tensor * tensor);
def ggml_get_name(
    tensor: ggml_tensor_p,
) -> bytes:
    """Get the name of a tensor.

    Parameters:
        tensor: tensor

    Returns:
        name of tensor"""
    return lib.ggml_get_name(tensor)


lib.ggml_get_name.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_name.restype = ctypes.c_char_p


# GGML_API struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name);
def ggml_set_name(
    tensor: ggml_tensor_p,
    name: bytes,
) -> ggml_tensor_p:
    """Set the name of a tensor.

    Parameters:
        tensor: tensor
        name: name to set tensor to

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_set_name(tensor, name)


lib.ggml_set_name.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_char_p]
lib.ggml_set_name.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt, ...);
def ggml_format_name(
    tensor: ggml_tensor_p,
    fmt: bytes,
    *args: Sequence[Union[bool, int, float, str]],
) -> ggml_tensor_p:
    """Format the name of a tensor using the given format c string and arguments.

    Parameters:
        tensor: tensor
        fmt: format c string
        args: arguments to format string

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_format_name(tensor, fmt, *args)


lib.ggml_format_name.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_char_p]
lib.ggml_format_name.restype = ctypes.POINTER(ggml_tensor)

# //
# // operations on tensors with backpropagation
# //


# GGML_API struct ggml_tensor * ggml_dup(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_dup(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    return lib.ggml_dup(ctx, a)


lib.ggml_dup.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_dup.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_dup_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_dup_inplace(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    return lib.ggml_dup_inplace(ctx, a)


lib.ggml_dup_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_dup_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_add(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Add two tensors together and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_add(ctx, a, b)


lib.ggml_add.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_add.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_add_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Add two tensors together and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_add_inplace(ctx, a, b)


lib.ggml_add_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_add_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_add_cast(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         enum   ggml_type      type);
def ggml_add_cast(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    type: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Add two tensors together and cast the result to the given type.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor
        type: type to cast result to

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_add_cast(ctx, a, b, type)


lib.ggml_add_cast.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_add_cast.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_add1(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add1(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_add1(ctx, a, b)


lib.ggml_add1.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_add1.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_add1_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add1_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_add1_inplace(ctx, a, b)


lib.ggml_add1_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_add1_inplace.restype = ctypes.POINTER(ggml_tensor)


# // dst = a
# // view(dst, nb1, nb2, nb3, offset) += b
# // return dst
# GGML_API struct ggml_tensor * ggml_acc(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_acc(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)


lib.ggml_acc.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_acc.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_acc_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_acc_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)


lib.ggml_acc_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_acc_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sub(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_sub(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Subtract two tensors and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sub(ctx, a, b)


lib.ggml_sub.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_sub.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sub_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_sub_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Subtract two tensors and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sub_inplace(ctx, a, b)


lib.ggml_sub_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_sub_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_mul(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_mul(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Element-wise multiply two tensors and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_mul(ctx, a, b)


lib.ggml_mul.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_mul.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_mul_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_mul_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Element-wise multiply two tensors and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_mul_inplace(ctx, a, b)


lib.ggml_mul_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_mul_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_div(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_div(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Element-wise divide two tensors and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_div(ctx, a, b)


lib.ggml_div.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_div.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_div_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_div_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Element-wise divide two tensors and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_div_inplace(ctx, a, b)


lib.ggml_div_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_div_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sqr(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sqr(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Square all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sqr(ctx, a)


lib.ggml_sqr.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sqr.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sqr_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sqr_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Square all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sqr_inplace(ctx, a)


lib.ggml_sqr_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sqr_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sqrt(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sqrt(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Square root all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sqrt(ctx, a)


lib.ggml_sqrt.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sqrt.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sqrt_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sqrt_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Square root all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sqrt_inplace(ctx, a)


lib.ggml_sqrt_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sqrt_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_log(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_log(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Take the natural logarithm of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_log(ctx, a)


lib.ggml_log.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_log.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_log_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_log_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Take the natural logarithm of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_log_inplace(ctx, a)


lib.ggml_log_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_log_inplace.restype = ctypes.POINTER(ggml_tensor)


# // return scalar
# GGML_API struct ggml_tensor * ggml_sum(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sum(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Sum all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sum(ctx, a)


lib.ggml_sum.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sum.restype = ctypes.POINTER(ggml_tensor)


# // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
# GGML_API struct ggml_tensor * ggml_sum_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sum_rows(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Sum all elements in a tensor along the first axis and return the result.

    sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sum_rows(ctx, a)


lib.ggml_sum_rows.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sum_rows.restype = ctypes.POINTER(ggml_tensor)


# // mean along rows
# GGML_API struct ggml_tensor * ggml_mean(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_mean(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Take the mean of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_mean(ctx, a)


lib.ggml_mean.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_mean.restype = ctypes.POINTER(ggml_tensor)


# // argmax along rows
# GGML_API struct ggml_tensor * ggml_argmax(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_argmax(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Take the argmax of all elements in a tensor and return the result.

    argmax along rows

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_argmax(ctx, a)


lib.ggml_argmax.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_argmax.restype = ctypes.POINTER(ggml_tensor)


# // if a is the same shape as b, and a is not parameter, return a
# // otherwise, return a new tensor: repeat(a) to fit in b
# GGML_API struct ggml_tensor * ggml_repeat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_repeat(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Repeat a tensor to fit the shape of another tensor.

    If a is the same shape as b, and a is not parameter, return a

    Parameters:
        ctx: ggml context
        a: tensor to repeat
        b: tensor to fit

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_repeat(ctx, a, b)


lib.ggml_repeat.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_repeat.restype = ctypes.POINTER(ggml_tensor)


# // sums repetitions in a into shape of b
# GGML_API struct ggml_tensor * ggml_repeat_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_repeat_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_repeat_back(ctx, a, b)


lib.ggml_repeat_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_repeat_back.restype = ctypes.POINTER(ggml_tensor)


# // concat a and b on dim 2
# // used in stable-diffusion
# GGML_API struct ggml_tensor * ggml_concat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_concat(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Concatenate two tensors along the second axis and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_concat(ctx, a, b)


lib.ggml_concat.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_concat.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_abs(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_abs(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Take the absolute value of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_abs(ctx, a)


lib.ggml_abs.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_abs.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_abs_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_abs_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Take the absolute value of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_abs_inplace(ctx, a)


lib.ggml_abs_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_abs_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sgn(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sgn(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Get the sign of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sgn(ctx, a)


lib.ggml_sgn.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sgn.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_sgn_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sgn_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Get the sign of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_sgn_inplace(ctx, a)


lib.ggml_sgn_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sgn_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_neg(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_neg(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Negate all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_neg(ctx, a)


lib.ggml_neg.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_neg.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_neg_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_neg_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Negate all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_neg_inplace(ctx, a)


lib.ggml_neg_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_neg_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_step(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_step(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    return lib.ggml_step(ctx, a)


lib.ggml_step.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_step.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_tanh(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_tanh(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Apply the tanh activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_tanh(ctx, a)


lib.ggml_tanh.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_tanh.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_tanh_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_tanh_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Apply the tanh activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_tanh_inplace(ctx, a)


lib.ggml_tanh_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_tanh_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_elu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_elu(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Apply the ELU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_elu(ctx, a)


lib.ggml_elu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_elu.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_elu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_elu_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Apply the ELU activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_elu_inplace(ctx, a)


lib.ggml_elu_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_elu_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_relu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_relu(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Apply the ReLU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_relu(ctx, a)


lib.ggml_relu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_relu.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_leaky_relu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a, float negative_slope, bool inplace);
def ggml_leaky_relu(
    ctx: ggml_context_p, a: ggml_tensor_p, negative_slope: float, inplace: bool
) -> ggml_tensor_p:
    """Apply the Leaky ReLU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        negative_slope: negative slope
        inplace: whether to store the result in the first tensor

    Returns:
        Pointer to ggml_tensor"""

    return lib.ggml_leaky_relu(ctx, a, negative_slope, inplace)


lib.ggml_leaky_relu.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
    ctypes.c_bool,
]
lib.ggml_leaky_relu.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_relu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_relu_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Apply the ReLU activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_relu_inplace(ctx, a)


lib.ggml_relu_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_relu_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_gelu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_gelu(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_gelu(ctx, a)


lib.ggml_gelu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_gelu.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_gelu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_gelu_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_gelu_inplace(ctx, a)


lib.ggml_gelu_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_gelu_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_gelu_quick(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_gelu_quick(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_gelu_quick(ctx, a)


lib.ggml_gelu_quick.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_gelu_quick.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_gelu_quick_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_gelu_quick_inplace(ctx, a)


lib.ggml_gelu_quick_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_gelu_quick_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_silu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_silu(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Apply the Sigmoid Linear Unit activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_silu(ctx, a)


lib.ggml_silu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_silu.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_silu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_silu_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
) -> ggml_tensor_p:
    """Apply the Sigmoid Linear Unit activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_silu_inplace(ctx, a)


lib.ggml_silu_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_silu_inplace.restype = ctypes.POINTER(ggml_tensor)


# // a - x
# // b - dy
# GGML_API struct ggml_tensor * ggml_silu_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_silu_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_silu_back(ctx, a, b)


lib.ggml_silu_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_silu_back.restype = ctypes.POINTER(ggml_tensor)


# // normalize along rows
# GGML_API struct ggml_tensor * ggml_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a
#         float                eps);
def ggml_norm(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    eps: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Normalize all elements in a tensor along the first axis and return the result.

    normalize along rows.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_norm(ctx, a, eps)


lib.ggml_norm.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_float]
lib.ggml_norm.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a
#         float                eps);
def ggml_norm_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    eps: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Normalize all elements in a tensor along the first axis and store the result in the first tensor.

    normalize along rows.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_norm_inplace(ctx, a, eps)


lib.ggml_norm_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
]
lib.ggml_norm_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_rms_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 eps);
def ggml_rms_norm(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    eps: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Compute the RMS norm of a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: float

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_rms_norm(ctx, a, eps)


lib.ggml_rms_norm.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
]
lib.ggml_rms_norm.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_rms_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 eps);
def ggml_rms_norm_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    eps: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    return lib.ggml_rms_norm_inplace(ctx, a, eps)


lib.ggml_rms_norm_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
]
lib.ggml_rms_norm_inplace.restype = ctypes.POINTER(ggml_tensor)


# // group normalize along ne0*ne1*n_groups
# // used in stable-diffusion
# // TODO: eps is hardcoded to 1e-6 for now
# GGML_API struct ggml_tensor * ggml_group_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_groups);
def ggml_group_norm(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_groups: int,
) -> ggml_tensor_p:
    """Group normalize a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        n_groups: int

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_group_norm(ctx, a, n_groups)


lib.ggml_group_norm.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_group_norm.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_group_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_groups);
def ggml_group_norm_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_groups: int,
) -> ggml_tensor_p:
    """Group normalize a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
        n_groups: int

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_group_norm_inplace(ctx, a, n_groups)


lib.ggml_group_norm_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_group_norm_inplace.restype = ctypes.POINTER(ggml_tensor)


# // a - x
# // b - dy
# GGML_API struct ggml_tensor * ggml_rms_norm_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b
#         float                 eps);
def ggml_rms_norm_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    eps: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    return lib.ggml_rms_norm_back(ctx, a, b, eps)


lib.ggml_rms_norm_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
]
lib.ggml_rms_norm_back.restype = ctypes.POINTER(ggml_tensor)


# // A: k columns, n rows => [ne03, ne02, n, k]
# // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
# // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
# GGML_API struct ggml_tensor * ggml_mul_mat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_mul_mat(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Multiply two matrices and return the result.

    A: k columns, n rows => [ne03, ne02, n, k]
    B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
    result is n columns, m rows => [ne03 * x, ne02 * y, m, n]

    Parameters:
        ctx: ggml context
        a: tensor
        b: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_mul_mat(ctx, a, b)


lib.ggml_mul_mat.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_mul_mat.restype = ctypes.POINTER(ggml_tensor)


# // change the precision of a matrix multiplication
# // set to GGML_PREC_F32 for higher precision (useful for phi-2)
# GGML_API void ggml_mul_mat_set_prec(
#         struct ggml_tensor * a,
#         enum ggml_prec       prec);
def ggml_mul_mat_set_prec(a: ggml_tensor_p, prec: Union[ctypes.c_int, int]) -> None:
    """Change the precision of a matrix multiplication.

    set to GGML_PREC_F32 for higher precision (useful for phi-2)

    Parameters:
        a: tensor
        prec: precision"""
    return lib.ggml_mul_mat_set_prec(a, prec)


lib.ggml_mul_mat_set_prec.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_mul_mat_set_prec.restype = None


# // indirect matrix multiplication
# //  ggml_mul_mat_id(ctx, as, ids, id, b) ~= ggml_mul_mat(as[ids[id]], b)
# GGML_API struct ggml_tensor * ggml_mul_mat_id(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * const as[],
#         int                   n_as,
#         struct ggml_tensor  * ids,
#         int                   id,
#         struct ggml_tensor  * b);
def ggml_mul_mat_id(
    ctx: ggml_context_p,
    as_,  # type: ctypes.POINTER(ctypes.POINTER(ggml_tensor)) # type: ignore
    n_as: int,
    ids: ggml_tensor_p,
    id_: int,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Multiply two matrices and return the result.

    indirect matrix multiplication

    ggml_mul_mat_id(ctx, as, ids, id, b) ~= ggml_mul_mat(as[ids[id]], b)

    Parameters:
        ctx: ggml context
        as_: array of tensor pointers
        n_as: int
        ids: tensor
        id_: int
        b: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_mul_mat_id(ctx, as_, n_as, ids, id_, b)


lib.ggml_mul_mat_id.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
    ctypes.c_int,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_mul_mat_id.restype = ctypes.POINTER(ggml_tensor)


# // A: m columns, n rows,
# // B: p columns, n rows,
# // result is m columns, p rows
# GGML_API struct ggml_tensor * ggml_out_prod(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_out_prod(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Compute the outer product of two matrices and return the result.

    A: m columns, n rows,
    B: p columns, n rows,
    result is m columns, p rows

    Parameters:
        ctx: ggml context
        a: tensor
        b: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_out_prod(ctx, a, b)


lib.ggml_out_prod.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_out_prod.restype = ctypes.POINTER(ggml_tensor)

# //
# // operations on tensors without backpropagation
# //


# GGML_API struct ggml_tensor * ggml_scale(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 s);
def ggml_scale(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    s: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Scale a tensor by another tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        s: float

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_scale(ctx, a, s)


lib.ggml_scale.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_float]
lib.ggml_scale.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_scale_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 s);
def ggml_scale_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    s: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Scale a tensor by another tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
        s: float

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_scale_inplace(ctx, a, b)


lib.ggml_scale_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
]
lib.ggml_scale_inplace.restype = ctypes.POINTER(ggml_tensor)


# // b -> view(a,offset,nb1,nb2,3), return modified a
# GGML_API struct ggml_tensor * ggml_set(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_set(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_set(ctx, a, b, nb1, nb2, nb3, offset)


lib.ggml_set.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_set.restype = ctypes.POINTER(ggml_tensor)


# // b -> view(a,offset,nb1,nb2,3), return view(a)
# GGML_API struct ggml_tensor * ggml_set_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_set_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)


lib.ggml_set_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_set_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_set_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                offset);
def ggml_set_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_set_1d(ctx, a, b, offset)


lib.ggml_set_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
]
lib.ggml_set_1d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_set_1d_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                offset);
def ggml_set_1d_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_set_1d_inplace(ctx, a, b, offset)


lib.ggml_set_1d_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
]
lib.ggml_set_1d_inplace.restype = ctypes.POINTER(ggml_tensor)


# // b -> view(a,offset,nb1,nb2,3), return modified a
# GGML_API struct ggml_tensor * ggml_set_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                offset);
def ggml_set_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_set_2d(ctx, a, b, nb1, offset)


lib.ggml_set_2d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_set_2d.restype = ctypes.POINTER(ggml_tensor)


# // b -> view(a,offset,nb1,nb2,3), return view(a)
# GGML_API struct ggml_tensor * ggml_set_2d_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                offset);
def ggml_set_2d_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_set_2d_inplace(ctx, a, b, nb1, offset)


lib.ggml_set_2d_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_set_2d_inplace.restype = ctypes.POINTER(ggml_tensor)


# // a -> b, return view(b)
# GGML_API struct ggml_tensor * ggml_cpy(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_cpy(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_cpy(ctx, a, b)


lib.ggml_cpy.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_cpy.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_cast(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum   ggml_type      type);
def ggml_cast(
    ctx: ggml_context_p, a: ggml_tensor_p, type_: Union[ctypes.c_int, int]
) -> ggml_tensor_p:
    return lib.ggml_cast(ctx, a, type_)


lib.ggml_cast.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_cast.restype = ctypes.POINTER(ggml_tensor)


# // make contiguous
# GGML_API struct ggml_tensor * ggml_cont(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_cont(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Make a tensor contiguous and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_cont(ctx, a)


lib.ggml_cont.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_cont.restype = ctypes.POINTER(ggml_tensor)


# // make contiguous, with new shape
# GGML_API struct ggml_tensor * ggml_cont_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0);
def ggml_cont_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_cont_1d(ctx, a, ne0)


lib.ggml_cont_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
]
lib.ggml_cont_1d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_cont_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1);
def ggml_cont_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_cont_2d(ctx, a, ne0, ne1)


lib.ggml_cont_2d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_cont_2d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_cont_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2);
def ggml_cont_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_cont_3d(ctx, a, ne0, ne1, ne2)


lib.ggml_cont_3d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_cont_3d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_cont_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3);
def ggml_cont_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3)


lib.ggml_cont_4d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_cont_4d.restype = ctypes.POINTER(ggml_tensor)


# // return view(a), b specifies the new shape
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_reshape(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_reshape(ctx, a, b)


lib.ggml_reshape.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_reshape.restype = ctypes.POINTER(ggml_tensor)


# // return view(a)
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0);
def ggml_reshape_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_reshape_1d(ctx, a, ne0)


lib.ggml_reshape_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
]
lib.ggml_reshape_1d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_reshape_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1);
def ggml_reshape_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_reshape_2d(ctx, a, ne0, ne1)


lib.ggml_reshape_2d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_reshape_2d.restype = ctypes.POINTER(ggml_tensor)


# // return view(a)
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2);
def ggml_reshape_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_reshape_3d(ctx, a, ne0, ne1, ne2)


lib.ggml_reshape_3d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_reshape_3d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_reshape_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3);
def ggml_reshape_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
) -> ggml_tensor_p:
    return lib.ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)


lib.ggml_reshape_4d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.ggml_reshape_4d.restype = ctypes.POINTER(ggml_tensor)


# // offset in bytes
# GGML_API struct ggml_tensor * ggml_view_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         size_t                offset);
def ggml_view_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_view_1d(ctx, a, ne0, offset)


lib.ggml_view_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_size_t,
]
lib.ggml_view_1d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_view_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         size_t                nb1, // row stride in bytes
#         size_t                offset);
def ggml_view_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    nb1: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)


lib.ggml_view_2d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_view_2d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_view_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         size_t                nb1, // row   stride in bytes
#         size_t                nb2, // slice stride in bytes
#         size_t                offset);
def ggml_view_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)


lib.ggml_view_3d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_view_3d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_view_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3,
#         size_t                nb1, // row   stride in bytes
#         size_t                nb2, // slice stride in bytes
#         size_t                nb3,
#         size_t                offset);
def ggml_view_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
) -> ggml_tensor_p:
    return lib.ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)


lib.ggml_view_4d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_view_4d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_permute(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   axis0,
#         int                   axis1,
#         int                   axis2,
#         int                   axis3);
def ggml_permute(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    axis0: Union[ctypes.c_int, int],
    axis1: Union[ctypes.c_int, int],
    axis2: Union[ctypes.c_int, int],
    axis3: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_permute(ctx, a, axis0, axis1, axis2, axis3)


lib.ggml_permute.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_permute.restype = ctypes.POINTER(ggml_tensor)


# // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
# GGML_API struct ggml_tensor * ggml_transpose(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_transpose(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    """Transpose *the first two dimensions* of a tensor and return the result.

    alias for `ggml_permute(ctx, a, 1, 0, 2, 3)`

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_transpose(ctx, a)


lib.ggml_transpose.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_transpose.restype = ctypes.POINTER(ggml_tensor)


# // supports 3D: a->ne[2] == b->ne[1]
# GGML_API struct ggml_tensor * ggml_get_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_get_rows(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_get_rows(ctx, a, b)


lib.ggml_get_rows.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_get_rows.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_get_rows_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c);
def ggml_get_rows_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_get_rows_back(ctx, a, b, c)


lib.ggml_get_rows_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_get_rows_back.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_diag(
#     struct ggml_context     * ctx,
#     struct ggml_tensor      * a);
def ggml_diag(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    return lib.ggml_diag(ctx, a)


lib.ggml_diag.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_diag.restype = ctypes.POINTER(ggml_tensor)


# // set elements above the diagonal to -INF
# GGML_API struct ggml_tensor * ggml_diag_mask_inf(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_inf(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_past: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_diag_mask_inf(ctx, a, n_past)


lib.ggml_diag_mask_inf.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_diag_mask_inf.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_inf_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_past: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_diag_mask_inf_inplace(ctx, a, n_past)


lib.ggml_diag_mask_inf_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_diag_mask_inf_inplace.restype = ctypes.POINTER(ggml_tensor)


# // set elements above the diagonal to 0
# GGML_API struct ggml_tensor * ggml_diag_mask_zero(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_zero(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_past: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_diag_mask_zero(ctx, a, n_past)


lib.ggml_diag_mask_zero.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_diag_mask_zero.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_zero_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_past: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_diag_mask_zero_inplace(ctx, a, n_past)


lib.ggml_diag_mask_zero_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_diag_mask_zero_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_soft_max(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_soft_max(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    return lib.ggml_soft_max(ctx, a)


lib.ggml_soft_max.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_soft_max.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_soft_max_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_soft_max_inplace(ctx: ggml_context_p, a: ggml_tensor_p) -> ggml_tensor_p:
    return lib.ggml_soft_max_inplace(ctx, a)


lib.ggml_soft_max_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_soft_max_inplace.restype = ctypes.POINTER(ggml_tensor)


# // fused soft_max(a*scale + mask)
# // mask is optional
# GGML_API struct ggml_tensor * ggml_soft_max_ext(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * mask,
#         float                 scale);
def ggml_soft_max_ext(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    mask: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    return lib.ggml_soft_max_ext(ctx, a, mask, scale)


lib.ggml_soft_max_ext.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
]
lib.ggml_soft_max_ext.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_soft_max_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_soft_max_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_soft_max_back(ctx, a, b)


lib.ggml_soft_max_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_soft_max_back.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_soft_max_back_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_soft_max_back_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_soft_max_back_inplace(ctx, a, b)


lib.ggml_soft_max_back_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_soft_max_back_inplace.restype = ctypes.POINTER(ggml_tensor)


# // rotary position embedding
# // if mode & 1 == 1, skip n_past elements (DEPRECATED)
# // if mode & 2 == 1, GPT-NeoX style
# // if mode & 4 == 1, ChatGLM style
# //
# // b is an int32 vector with size a->ne[2], it contains the positions
# GGML_API struct ggml_tensor * ggml_rope(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx);
def ggml_rope(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Rotary position embedding

    Parameters:
        ctx: ggml context
        a: tensor
        b: int32 vector with size a->ne[2], it contains the positions
        n_dims: number of dimensions
        mode: if mode & 1 == 1, skip n_past elements (DEPRECATED)
                if mode & 2 == 1, GPT-NeoX style
                if mode & 4 == 1, ChatGLM style
        n_ctx: context size

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_rope(ctx, a, b, n_dims, mode, n_ctx)


lib.ggml_rope.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_rope.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_rope_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx);
def ggml_rope_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Rotary position embedding inplace

    Parameters:
        ctx: ggml context
        a: tensor
        b: int32 vector with size a->ne[2], it contains the positions
        n_dims: number of dimensions
        mode: if mode & 1 == 1, skip n_past elements (DEPRECATED)
                if mode & 2 == 1, GPT-NeoX style
                if mode & 4 == 1, ChatGLM style
        n_ctx: context size

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_rope_inplace(ctx, a, b, n_dims, mode, n_ctx)


lib.ggml_rope_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_rope_inplace.restype = ctypes.POINTER(ggml_tensor)


# // custom RoPE
# GGML_API struct ggml_tensor * ggml_rope_custom(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx,
#         int                   n_orig_ctx,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
def ggml_rope_custom(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx: Union[ctypes.c_int, int],
    n_orig_ctx: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Custom rotary position embedding"""
    return lib.ggml_rope_custom(
        ctx,
        a,
        b,
        n_dims,
        mode,
        n_ctx,
        n_orig_ctx,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
    )


lib.ggml_rope_custom.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
lib.ggml_rope_custom.restype = ctypes.POINTER(ggml_tensor)


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_rope_custom_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx,
#         int                   n_orig_ctx,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
def ggml_rope_custom_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx: Union[ctypes.c_int, int],
    n_orig_ctx: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Custom rotary position embedding inplace"""
    return lib.ggml_rope_custom_inplace(
        ctx,
        a,
        b,
        n_dims,
        mode,
        n_ctx,
        n_orig_ctx,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
    )


lib.ggml_rope_custom_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
lib.ggml_rope_custom_inplace.restype = ctypes.POINTER(ggml_tensor)


# // compute correction dims for YaRN RoPE scaling
# void ggml_rope_yarn_corr_dims(
#     int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]);
def ggml_rope_yarn_corr_dims(
    n_dims: Union[ctypes.c_int, int],
    n_orig_ctx: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    dims: CFloatArray,
) -> None:
    """Compute correction dims for YaRN RoPE scaling"""
    return lib.ggml_rope_yarn_corr_dims(
        n_dims,
        n_orig_ctx,
        freq_base,
        beta_fast,
        beta_slow,
        dims,
    )


lib.ggml_rope_yarn_corr_dims.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
]
lib.ggml_rope_yarn_corr_dims.restype = None


# // xPos RoPE, in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_rope_xpos_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         float                 base,
#         bool                  down);
def ggml_rope_xpos_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    base: Union[ctypes.c_float, float],
    down: Union[ctypes.c_bool, bool],
) -> ggml_tensor_p:
    """xPos RoPE, in-place, returns view(a)"""
    return lib.ggml_rope_xpos_inplace(ctx, a, b, n_dims, base, down)


lib.ggml_rope_xpos_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_bool,
]
lib.ggml_rope_xpos_inplace.restype = ctypes.POINTER(ggml_tensor)


# // rotary position embedding backward, i.e compute dx from dy
# // a - dy
# GGML_API struct ggml_tensor * ggml_rope_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx,
#         int                   n_orig_ctx,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow,
#         float                 xpos_base,
#         bool                  xpos_down);
def ggml_rope_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx: Union[ctypes.c_int, int],
    n_orig_ctx: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    xpos_base: Union[ctypes.c_float, float],
    xpos_down: Union[ctypes.c_bool, bool],
) -> ggml_tensor_p:
    """Rotary position embedding backward pass"""
    return lib.ggml_rope_back(
        ctx,
        a,
        b,
        n_dims,
        mode,
        n_ctx,
        n_orig_ctx,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
        xpos_base,
        xpos_down,
    )


lib.ggml_rope_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_bool,
]
lib.ggml_rope_back.restype = ctypes.POINTER(ggml_tensor)


# // alibi position embedding
# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_alibi(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past,
#         int                   n_head,
#         float                 bias_max);
def ggml_alibi(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_past: Union[ctypes.c_int, int],
    n_head: Union[ctypes.c_int, int],
    bias_max: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    return lib.ggml_alibi(ctx, a, n_past, n_head, bias_max)


lib.ggml_alibi.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.ggml_alibi.restype = ctypes.POINTER(ggml_tensor)


# // clamp
# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_clamp(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 min,
#         float                 max);
def ggml_clamp(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    min: Union[ctypes.c_float, float],
    max: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """Clamp tensor values between min and max

    Parameters:
        ctx: ggml context
        a: tensor
        min: minimum value
        max: maximum value

    Returns:
        Pointer to ggml_tensor"""
    return lib.ggml_clamp(ctx, a, min, max)


lib.ggml_clamp.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_float,
    ctypes.c_float,
]
lib.ggml_clamp.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_im2col(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                  s0,
#         int                  s1,
#         int                  p0,
#         int                  p1,
#         int                  d0,
#         int                  d1,
#         bool                 is_2D);
def ggml_im2col(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    is_2D: Union[ctypes.c_bool, bool],
) -> ggml_tensor_p:
    return lib.ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D)


lib.ggml_im2col.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
]
lib.ggml_im2col.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_conv_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,  // stride
#         int                   p0,  // padding
#         int                   d0); // dilation
def ggml_conv_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Convolution 1D

    Parameters:
        a: input tensor
        b: filter tensor
        s0: stride
        p0: padding
        d0: dilation

    Returns:
        output tensor"""
    return lib.ggml_conv_1d(ctx, a, b, s0, p0, d0)


lib.ggml_conv_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_conv_1d.restype = ctypes.POINTER(ggml_tensor)


# // conv_1d with padding = half
# // alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
# GGML_API struct ggml_tensor* ggml_conv_1d_ph(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s,
#         int                   d);
def ggml_conv_1d_ph(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s: Union[ctypes.c_int, int],
    d: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Convolution 1D with padding = half

    Parameters:
        a: input tensor
        b: filter tensor
        s: stride
        d: dilation

    Returns:
        output tensor"""
    return lib.ggml_conv_1d_ph(ctx, a, b, s, d)


lib.ggml_conv_1d_ph.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_conv_1d_ph.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_conv_transpose_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   p0,
#         int                   d0);
def ggml_conv_transpose_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Convolution transpose 1D

    Parameters:
        a: input tensor
        b: filter tensor
        s0: stride
        p0: padding
        d0: dilation

    Returns:
        output tensor"""
    return lib.ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0)


lib.ggml_conv_transpose_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_conv_transpose_1d.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_conv_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   s1,
#         int                   p0,
#         int                   p1,
#         int                   d0,
#         int                   d1);
def ggml_conv_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Convolution 2D

    Parameters:
        a: input tensor
        b: filter tensor
        s0: stride
        s1: stride
        p0: padding
        p1: padding
        d0: dilation
        d1: dilation

    Returns:
        output tensor"""
    return lib.ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)


lib.ggml_conv_2d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_conv_2d.restype = ctypes.POINTER(ggml_tensor)


# // kernel size is a->ne[0] x a->ne[1]
# // stride is equal to kernel size
# // padding is zero
# // example:
# // a:     16   16    3  768
# // b:   1024 1024    3    1
# // res:   64   64  768    1
# // used in sam
# GGML_API struct ggml_tensor * ggml_conv_2d_sk_p0(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_conv_2d_sk_p0(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Convolution 2D

    Parameters:
        a: input tensor
        b: filter tensor

    Returns:
        output tensor"""
    return lib.ggml_conv_2d_sk_p0(ctx, a, b)


lib.ggml_conv_2d_sk_p0.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_conv_2d_sk_p0.restype = ctypes.POINTER(ggml_tensor)


# // kernel size is a->ne[0] x a->ne[1]
# // stride is 1
# // padding is half
# // example:
# // a:      3    3    256  256
# // b:     64   64    256    1
# // res:   64   64    256    1
# // used in sam
# GGML_API struct ggml_tensor * ggml_conv_2d_s1_ph(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_conv_2d_s1_ph(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    """Convolution 2D with stride = 1 and padding = half

    Parameters:
        a: input tensor
        b: filter tensor

    Returns:
        output tensor"""
    return lib.ggml_conv_2d_s1_ph(ctx, a, b)


lib.ggml_conv_2d_s1_ph.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_conv_2d_s1_ph.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_conv_transpose_2d_p0(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   stride);
def ggml_conv_transpose_2d_p0(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    stride: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Convolution Transpose 2D with padding = zero

    Parameters:
        a: input tensor
        b: filter tensor
        stride: stride

    Returns:
        output tensor"""
    return lib.ggml_conv_transpose_2d_p0(ctx, a, b, stride)


lib.ggml_conv_transpose_2d_p0.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_conv_transpose_2d_p0.restype = ctypes.POINTER(ggml_tensor)

# enum ggml_op_pool {
#     GGML_OP_POOL_MAX,
#     GGML_OP_POOL_AVG,
#     GGML_OP_POOL_COUNT,
# };
GGML_OP_POOL_MAX = 0
GGML_OP_POOL_AVG = 1
GGML_OP_POOL_COUNT = 2


# GGML_API struct ggml_tensor * ggml_pool_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_op_pool     op,
#         int                   k0, // kernel size
#         int                   s0, // stride
#         int                   p0); // padding
def ggml_pool_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    k0: Union[ctypes.c_int, int],
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """1D Pooling

    Parameters:
        a: input tensor
        op: pooling operation
        k0: kernel size
        s0: stride
        p0: padding

    Returns:
        output tensor"""
    return lib.ggml_pool_1d(ctx, a, op, k0, s0, p0)


lib.ggml_pool_1d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_pool_1d.restype = ctypes.POINTER(ggml_tensor)


# // the result will have 2*p0 padding for the first dimension
# // and 2*p1 padding for the second dimension
# GGML_API struct ggml_tensor * ggml_pool_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_op_pool     op,
#         int                   k0,
#         int                   k1,
#         int                   s0,
#         int                   s1,
#         float                 p0,
#         float                 p1);
def ggml_pool_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    k0: Union[ctypes.c_int, int],
    k1: Union[ctypes.c_int, int],
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_float, float],
    p1: Union[ctypes.c_float, float],
) -> ggml_tensor_p:
    """2D Pooling

    Parameters:
        a: input tensor
        op: pooling operation
        k0: kernel size
        k1: kernel size
        s0: stride
        s1: stride
        p0: padding
        p1: padding

    Returns:
        output tensor"""
    return lib.ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)


lib.ggml_pool_2d.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
]
lib.ggml_pool_2d.restype = ctypes.POINTER(ggml_tensor)


# // nearest interpolate
# // used in stable-diffusion
# GGML_API struct ggml_tensor * ggml_upscale(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   scale_factor);
def ggml_upscale(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    scale_factor: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Upscale

    Parameters:
        a: input tensor
        scale_factor: scale factor

    Returns:
        output tensor"""
    return lib.ggml_upscale(ctx, a, scale_factor)


lib.ggml_upscale.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_upscale.restype = ctypes.POINTER(ggml_tensor)


# // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
# GGML_API struct ggml_tensor * ggml_pad(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                  p0,
#         int                  p1,
#         int                  p2,
#         int                  p3);
def ggml_pad(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    p2: Union[ctypes.c_int, int],
    p3: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Pad tensor with zeros

    Parameters:
        a: input tensor
        p0: padding
        p1: padding
        p2: padding
        p3: padding

    Returns:
        output tensor"""
    return lib.ggml_pad(ctx, a, p0, p1, p2, p3)


lib.ggml_pad.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_pad.restype = ctypes.POINTER(ggml_tensor)

# // sort rows
# enum ggml_sort_order {
#     GGML_SORT_ASC,
#     GGML_SORT_DESC,
# };
GGML_SORT_ASC = 0
GGML_SORT_DESC = 1


# GGML_API struct ggml_tensor * ggml_argsort(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_sort_order  order);
def ggml_argsort(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    order: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Argsort

    Parameters:
        a: input tensor
        order: sort order

    Returns:
        output tensor"""
    return lib.ggml_argsort(ctx, a, order)


lib.ggml_argsort.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_argsort.restype = ctypes.POINTER(ggml_tensor)


# // top k elements per row
# GGML_API struct ggml_tensor * ggml_top_k(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   k);
def ggml_top_k(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    k: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    """Top k elements per row

    Parameters:
        a: input tensor
        k: number of elements

    Returns:
        output tensor"""
    return lib.ggml_top_k(ctx, a, k)


# GGML_API struct ggml_tensor * ggml_flash_attn(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         bool                  masked);
def ggml_flash_attn(
    ctx: ggml_context_p,
    q: ggml_tensor_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    masked: Union[ctypes.c_bool, bool],
) -> ggml_tensor_p:
    return lib.ggml_flash_attn(ctx, q, k, v, masked)


lib.ggml_flash_attn.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_bool,
]
lib.ggml_flash_attn.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_flash_attn_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * d,
#         bool                  masked);
def ggml_flash_attn_back(
    ctx: ggml_context_p,
    q: ggml_tensor_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    d: ggml_tensor_p,
    masked: Union[ctypes.c_bool, bool],
) -> ggml_tensor_p:
    return lib.ggml_flash_attn_back(ctx, q, k, v, d, masked)


lib.ggml_flash_attn_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_bool,
]
lib.ggml_flash_attn_back.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_flash_ff(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b0,
#         struct ggml_tensor  * b1,
#         struct ggml_tensor  * c0,
#         struct ggml_tensor  * c1);
def ggml_flash_ff(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b0: ggml_tensor_p,
    b1: ggml_tensor_p,
    c0: ggml_tensor_p,
    c1: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_flash_ff(ctx, a, b0, b1, c0, c1)


lib.ggml_flash_ff.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_flash_ff.restype = ctypes.POINTER(ggml_tensor)


# // partition into non-overlapping windows with padding if needed
# // example:
# // a:   768   64   64    1
# // w:    14
# // res: 768   14   14    25
# // used in sam
# GGML_API struct ggml_tensor * ggml_win_part(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   w);
def ggml_win_part(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    w: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_win_part(ctx, a, w)


lib.ggml_win_part.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_win_part.restype = ctypes.POINTER(ggml_tensor)


# // reverse of ggml_win_part
# // used in sam
# GGML_API struct ggml_tensor * ggml_win_unpart(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   w0,
#         int                   h0,
#         int                   w);
def ggml_win_unpart(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    w0: Union[ctypes.c_int, int],
    h0: Union[ctypes.c_int, int],
    w: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_win_unpart(ctx, a, w0, h0, w)


lib.ggml_win_unpart.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_win_unpart.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_unary(
#         struct ggml_context * ctx,
#             struct ggml_tensor * a,
#             enum ggml_unary_op op);
def ggml_unary(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_unary(ctx, a, op)


lib.ggml_unary.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_unary.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_unary_inplace(
#     struct ggml_context * ctx,
#     struct ggml_tensor  * a,
#     enum ggml_unary_op op);
def ggml_unary_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_unary_inplace(ctx, a, op)


lib.ggml_unary_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.ggml_unary_inplace.restype = ctypes.POINTER(ggml_tensor)


# // used in sam
# GGML_API struct ggml_tensor * ggml_get_rel_pos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   qh,
#         int                   kh);
def ggml_get_rel_pos(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    qh: Union[ctypes.c_int, int],
    kh: Union[ctypes.c_int, int],
) -> ggml_tensor_p:
    return lib.ggml_get_rel_pos(ctx, a, qh, kh)


lib.ggml_get_rel_pos.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
]
lib.ggml_get_rel_pos.restype = ctypes.POINTER(ggml_tensor)


# // used in sam
# GGML_API struct ggml_tensor * ggml_add_rel_pos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * pw,
#         struct ggml_tensor  * ph);
def ggml_add_rel_pos(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    pw: ggml_tensor_p,
    ph: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_add_rel_pos(ctx, a, pw, ph)


lib.ggml_add_rel_pos.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_add_rel_pos.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_add_rel_pos_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * pw,
#         struct ggml_tensor  * ph);
def ggml_add_rel_pos_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    pw: ggml_tensor_p,
    ph: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_add_rel_pos_inplace(ctx, a, pw, ph)


lib.ggml_add_rel_pos_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_add_rel_pos_inplace.restype = ctypes.POINTER(ggml_tensor)

# // custom operators (DEPRECATED)

# typedef void (*ggml_unary_op_f32_t)(const int, float *, const float *);
ggml_unary_op_f32_t = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
)

# typedef void (*ggml_binary_op_f32_t)(const int, float *, const float *, const float *);
ggml_binary_op_f32_t = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
)

# typedef void (*ggml_custom1_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *);
ggml_custom1_op_f32_t = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)
)
"""Unary operator function type"""

# typedef void (*ggml_custom2_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *);
ggml_custom2_op_f32_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
"""Binary operator function type"""

# typedef void (*ggml_custom3_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *);
ggml_custom3_op_f32_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
"""Ternary operator function type"""


# GGML_API struct ggml_tensor * ggml_map_unary_f32(
#         struct ggml_context        * ctx,
#         struct ggml_tensor         * a,
#                ggml_unary_op_f32_t   fun);
def ggml_map_unary_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: "ctypes._FuncPointer"  # type: ignore
) -> ggml_tensor_p:
    return lib.ggml_map_unary_f32(ctx, a, fun)


lib.ggml_map_unary_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ggml_unary_op_f32_t,
]
lib.ggml_map_unary_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_unary_inplace_f32(
#         struct ggml_context        * ctx,
#         struct ggml_tensor         * a,
#                 ggml_unary_op_f32_t   fun);
def ggml_map_unary_inplace_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: "ctypes._FuncPointer"  # type: ignore
) -> ggml_tensor_p:
    return lib.ggml_map_unary_inplace_f32(ctx, a, fun)


lib.ggml_map_unary_inplace_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ggml_unary_op_f32_t,
]
lib.ggml_map_unary_inplace_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_binary_f32(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#                ggml_binary_op_f32_t   fun);
def ggml_map_binary_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
) -> ggml_tensor_p:
    return lib.ggml_map_binary_f32(ctx, a, b, fun)


lib.ggml_map_binary_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_binary_op_f32_t,
]
lib.ggml_map_binary_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_binary_inplace_f32(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#                 ggml_binary_op_f32_t   fun);
def ggml_map_binary_inplace_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
) -> ggml_tensor_p:
    return lib.ggml_map_binary_inplace_f32(ctx, a, b, fun)


lib.ggml_map_binary_inplace_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_binary_op_f32_t,
]
lib.ggml_map_binary_inplace_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom1_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#                 ggml_custom1_op_f32_t   fun);
def ggml_map_custom1_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: "ctypes._FuncPointer"  # type: ignore
) -> ggml_tensor_p:
    """Custom unary operator on a tensor.

    Example:
        ```python
        import ggml

        @ggml.ggml_custom1_op_f32_t
        def custom_op(b: ggml.tensor_p, a: ggml.tensor_p):
            # do something with a and copy to b
            return

        ...

        b = ggml.ggml_map_custom1_f32(ctx, a, custom_op)
        ```

    Parameters:
        a: input tensor
        fun (ggml.ggml_custom1_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    return lib.ggml_map_custom1_f32(ctx, a, fun)


lib.ggml_map_custom1_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ggml_custom1_op_f32_t,
]
lib.ggml_map_custom1_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom1_inplace_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#                 ggml_custom1_op_f32_t   fun);
def ggml_map_custom1_inplace_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: "ctypes._CFuncPtr"  # type: ignore
) -> ggml_tensor_p:
    """Custom unary operator on a tensor inplace.

    Parameters:
        a: input tensor
        fun (ggml.ggml_custom1_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    return lib.ggml_map_custom1_inplace_f32(ctx, a, fun)


lib.ggml_map_custom1_inplace_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ggml_custom1_op_f32_t,
]
lib.ggml_map_custom1_inplace_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom2_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#                 ggml_custom2_op_f32_t   fun);
def ggml_map_custom2_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
) -> ggml_tensor_p:
    """Custom binary operator on two tensors.

    Parameters:
        a: input tensor
        b: input tensor
        fun (ggml.ggml_custom2_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    return lib.ggml_map_custom2_f32(ctx, a, b, fun)


lib.ggml_map_custom2_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom2_op_f32_t,
]
lib.ggml_map_custom2_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom2_inplace_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#                 ggml_custom2_op_f32_t   fun);
def ggml_map_custom2_inplace_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
) -> ggml_tensor_p:
    """Custom binary operator on two tensors inplace.

    Parameters:
        a: input tensor
        b: input tensor
        fun (ggml.ggml_custom2_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    return lib.ggml_map_custom2_inplace_f32(ctx, a, b, fun)


lib.ggml_map_custom2_inplace_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom2_op_f32_t,
]
lib.ggml_map_custom2_inplace_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom3_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#         struct ggml_tensor           * c,
#                 ggml_custom3_op_f32_t   fun);
def ggml_map_custom3_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
) -> ggml_tensor_p:
    """Custom ternary operator on three tensors.

    Parameters:
        a: input tensor
        b: input tensor
        c: input tensor
        fun (ggml.ggml_custom3_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    return lib.ggml_map_custom3_f32(ctx, a, b, c, fun)


lib.ggml_map_custom3_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom3_op_f32_t,
]
lib.ggml_map_custom3_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom3_inplace_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#         struct ggml_tensor           * c,
#                 ggml_custom3_op_f32_t   fun);
def ggml_map_custom3_inplace_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
) -> ggml_tensor_p:
    """Custom ternary operator on three tensors inplace.

    Parameters:
        a: input tensor
        b: input tensor
        c: input tensor
        fun (ggml.ggml_custom3_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    return lib.ggml_map_custom3_inplace_f32(ctx, a, b, c, fun)


lib.ggml_map_custom3_inplace_f32.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom3_op_f32_t,
]
lib.ggml_map_custom3_inplace_f32.restype = ctypes.POINTER(ggml_tensor)

# // custom operators v2

# typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata);
ggml_custom1_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom unary operator on a tensor."""

# typedef void (*ggml_custom2_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);
ggml_custom2_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom binary operator on two tensors."""

# typedef void (*ggml_custom3_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);
ggml_custom3_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom ternary operator on three tensors."""

# #define GGML_N_TASKS_MAX -1
GGML_N_TASKS_MAX = -1


# GGML_API struct ggml_tensor * ggml_map_custom1(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         ggml_custom1_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
def ggml_map_custom1(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Optional[ctypes.c_void_p],
) -> ggml_tensor_p:
    return lib.ggml_map_custom1(ctx, a, fun, n_tasks, userdata)


lib.ggml_map_custom1.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ggml_custom1_op_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.ggml_map_custom1.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom1_inplace(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         ggml_custom1_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
def ggml_map_custom1_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Optional[ctypes.c_void_p],
) -> ggml_tensor_p:
    return lib.ggml_map_custom1_inplace(ctx, a, fun, n_tasks, userdata)


lib.ggml_map_custom1_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ggml_custom1_op_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.ggml_map_custom1_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom2(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         ggml_custom2_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
def ggml_map_custom2(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Optional[ctypes.c_void_p],
) -> ggml_tensor_p:
    return lib.ggml_map_custom2(ctx, a, b, fun, n_tasks, userdata)


lib.ggml_map_custom2.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom2_op_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.ggml_map_custom2.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom2_inplace(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         ggml_custom2_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
def ggml_map_custom2_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Optional[ctypes.c_void_p],
) -> ggml_tensor_p:
    return lib.ggml_map_custom2_inplace(ctx, a, b, fun, n_tasks, userdata)


lib.ggml_map_custom2_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom2_op_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.ggml_map_custom2_inplace.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom3(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         struct ggml_tensor    * c,
#         ggml_custom3_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
def ggml_map_custom3(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Optional[ctypes.c_void_p],
) -> ggml_tensor_p:
    return lib.ggml_map_custom3(ctx, a, b, c, fun, n_tasks, userdata)


lib.ggml_map_custom3.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom3_op_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.ggml_map_custom3.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_map_custom3_inplace(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         struct ggml_tensor    * c,
#         ggml_custom3_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
def ggml_map_custom3_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: "ctypes._FuncPointer",  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Optional[ctypes.c_void_p],
) -> ggml_tensor_p:
    return lib.ggml_map_custom3_inplace(ctx, a, b, c, fun, n_tasks, userdata)


lib.ggml_map_custom3_inplace.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ggml_custom3_op_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.ggml_map_custom3_inplace.restype = ctypes.POINTER(ggml_tensor)

# // loss function


# GGML_API struct ggml_tensor * ggml_cross_entropy_loss(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b);
def ggml_cross_entropy_loss(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_cross_entropy_loss(ctx, a, b)


lib.ggml_cross_entropy_loss.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_cross_entropy_loss.restype = ctypes.POINTER(ggml_tensor)


# GGML_API struct ggml_tensor * ggml_cross_entropy_loss_back(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#         struct ggml_tensor          * c);
def ggml_cross_entropy_loss_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
) -> ggml_tensor_p:
    return lib.ggml_cross_entropy_loss_back(ctx, a, b, c)


lib.ggml_cross_entropy_loss_back.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_cross_entropy_loss_back.restype = ctypes.POINTER(ggml_tensor)

# //
# // automatic differentiation
# //


# GGML_API void ggml_set_param(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * tensor);
def ggml_set_param(ctx: ggml_context_p, tensor: ggml_tensor_p):
    return lib.ggml_set_param(ctx, tensor)


lib.ggml_set_param.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_set_param.restype = None


# GGML_API void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
def ggml_build_forward_expand(
    cgraph: ggml_cgraph_p,
    tensor: ggml_tensor_p,
):
    """Add a tensor to the forward computation graph. This is used to
    compute and save the value of the tensor.

    Parameters:
        cgraph: The graph.
        tensor: The tensor."""
    return lib.ggml_build_forward_expand(cgraph, tensor)


lib.ggml_build_forward_expand.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_build_forward_expand.restype = None


# GGML_API void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep);
def ggml_build_backward_expand(
    ctx: ggml_context_p,
    gf: ggml_cgraph_p,
    gb: ggml_cgraph_p,
    keep: Union[ctypes.c_bool, bool],
):
    """Add a tensor to the backward computation graph. This is used to
    compute the gradient of the tensor.

    Parameters:
        ctx: The context.
        gf: The forward graph.
        gb: The backward graph.
        keep: Whether to keep the tensor."""
    return lib.ggml_build_backward_expand(ctx, gf, gb, keep)


lib.ggml_build_backward_expand.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_bool,
]
lib.ggml_build_backward_expand.restype = None


# // graph allocation in a context
# GGML_API struct ggml_cgraph * ggml_new_graph         (struct ggml_context * ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
def ggml_new_graph(ctx: ggml_context_p) -> ggml_cgraph_p:
    """Create a new graph.

    Parameters:
        ctx: The context.

    Returns:
        The graph."""
    return lib.ggml_new_graph(ctx)


lib.ggml_new_graph.argtypes = [ggml_context_p]
lib.ggml_new_graph.restype = ctypes.POINTER(ggml_cgraph)


# GGML_API struct ggml_cgraph * ggml_new_graph_custom  (struct ggml_context * ctx, size_t size, bool grads);
def ggml_new_graph_custom(
    ctx: ggml_context_p,
    size: Union[ctypes.c_size_t, int],
    grads: Union[ctypes.c_bool, bool],
) -> ggml_cgraph_p:
    """Create a new graph with custom size and grads.

    Parameters:
        ctx: The context.
        size: The size of the graph.
        grads: Whether to keep the gradients.

    Returns:
        The graph."""
    return lib.ggml_new_graph_custom(ctx, size, grads)


lib.ggml_new_graph_custom.argtypes = [ggml_context_p, ctypes.c_size_t, ctypes.c_bool]
lib.ggml_new_graph_custom.restype = ctypes.POINTER(ggml_cgraph)


# GGML_API struct ggml_cgraph * ggml_graph_dup         (struct ggml_context * ctx, struct ggml_cgraph * cgraph);
def ggml_graph_dup(
    ctx: ggml_context_p,
    cgraph: ggml_cgraph_p,
) -> ggml_cgraph_p:
    """Duplicate a graph.

    Parameters:
        ctx: The context.
        cgraph: The graph.

    Returns:
        The graph."""
    return lib.ggml_graph_dup(ctx, cgraph)


lib.ggml_graph_dup.argtypes = [ggml_context_p, ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_dup.restype = ctypes.POINTER(ggml_cgraph)


# GGML_API struct ggml_cgraph   ggml_graph_view        (struct ggml_cgraph * cgraph, int i0, int i1);
def ggml_graph_view(
    cgraph: ggml_cgraph_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
) -> ggml_cgraph:
    """View a graph.

    Parameters:
        cgraph: The graph.
        i0: The start index.
        i1: The end index.

    Returns:
        The graph."""
    return lib.ggml_graph_view(cgraph, i0, i1)


lib.ggml_graph_view.argtypes = [ctypes.POINTER(ggml_cgraph), ctypes.c_int, ctypes.c_int]
lib.ggml_graph_view.restype = ggml_cgraph


# GGML_API void                 ggml_graph_cpy         (struct ggml_cgraph * src, struct ggml_cgraph * dst);
def ggml_graph_cpy(
    src: ggml_cgraph_p,
    dst: ggml_cgraph_p,
):
    """Copy a graph.

    Parameters:
        src: The source graph.
        dst: The destination graph."""
    return lib.ggml_graph_cpy(src, dst)


lib.ggml_graph_cpy.argtypes = [ctypes.POINTER(ggml_cgraph), ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_cpy.restype = None


# GGML_API void                 ggml_graph_reset       (struct ggml_cgraph * cgraph);  // zero grads
def ggml_graph_reset(
    cgraph: ggml_cgraph_p,
):
    """Reset a graph.

    Parameters:
        cgraph: The graph."""
    return lib.ggml_graph_reset(cgraph)


lib.ggml_graph_reset.argtypes = [ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_reset.restype = None


# GGML_API void                 ggml_graph_clear       (struct ggml_cgraph * cgraph);
def ggml_graph_clear(
    cgraph: ggml_cgraph_p,
):
    """Clear a graph.

    Parameters:
        cgraph: The graph."""
    return lib.ggml_graph_clear(cgraph)


lib.ggml_graph_clear.argtypes = [ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_clear.restype = None


# GGML_API size_t ggml_graph_overhead(void);
def ggml_graph_overhead() -> int:
    """Get the overhead of the graph."""
    return lib.ggml_graph_overhead()


lib.ggml_graph_overhead.argtypes = []
lib.ggml_graph_overhead.restype = ctypes.c_size_t


# GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);
def ggml_graph_overhead_custom(
    size: Union[ctypes.c_size_t, int],
    grads: Union[ctypes.c_bool, bool],
) -> int:
    return lib.ggml_graph_overhead_custom(size, grads)


lib.ggml_graph_overhead_custom.argtypes = [ctypes.c_size_t, ctypes.c_bool]
lib.ggml_graph_overhead_custom.restype = ctypes.c_size_t


# // ggml_graph_plan() has to be called before ggml_graph_compute()
# // when plan.work_size > 0, caller must allocate memory for plan.work_data
# GGML_API struct ggml_cplan ggml_graph_plan   (const struct ggml_cgraph * cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);
def ggml_graph_plan(
    cgraph: ggml_cgraph_p,
    n_threads: Union[ctypes.c_int, int] = GGML_DEFAULT_N_THREADS,
) -> ggml_cplan:
    """Plan the computation graph.

    Parameters:
        cgraph: The graph.
        n_threads: The number of threads to use.

    Returns:
        The plan."""
    return lib.ggml_graph_plan(cgraph, n_threads)


lib.ggml_graph_plan.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_int,
]
lib.ggml_graph_plan.restype = ggml_cplan


# GGML_API int               ggml_graph_compute(      struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
def ggml_graph_compute(
    cgraph: ggml_cgraph_p,
    cplan: ggml_cplan_p,
) -> int:
    return lib.ggml_graph_compute(cgraph, cplan)


lib.ggml_graph_compute.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_cplan),
]
lib.ggml_graph_compute.restype = ctypes.c_int


# // same as ggml_graph_compute() but the work data is allocated as a part of the context
# // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
# GGML_API void ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
def ggml_graph_compute_with_ctx(
    ctx: ggml_context_p,
    cgraph: ggml_cgraph_p,
    n_threads: Union[ctypes.c_int, int],
):
    """Compute the graph with a context.

    Parameters:
        ctx: The context.
        cgraph: The graph.
        n_threads: The number of threads to use."""
    return lib.ggml_graph_compute_with_ctx(ctx, cgraph, n_threads)


lib.ggml_graph_compute_with_ctx.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_int,
]
lib.ggml_graph_compute_with_ctx.restype = None


# GGML_API struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);
def ggml_graph_get_tensor(
    cgraph: ggml_cgraph_p,
    name: bytes,
) -> ggml_tensor_p:
    """Get a tensor from the graph by name.

    Parameters:
        cgraph: The graph.
        name: The name of the tensor.

    Returns:
        The tensor."""
    return lib.ggml_graph_get_tensor(cgraph, name)


lib.ggml_graph_get_tensor.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_char_p,
]
lib.ggml_graph_get_tensor.restype = ctypes.POINTER(ggml_tensor)


# GGML_API void                 ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
def ggml_graph_export(
    cgraph: ggml_cgraph_p,
    fname: bytes,
):
    return lib.ggml_graph_export(cgraph, fname)


lib.ggml_graph_export.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_char_p,
]
lib.ggml_graph_export.restype = None


# GGML_API struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
def ggml_graph_import(
    fname: bytes,
    ctx_data: "ctypes._Pointer[ggml_context_p]",  # type: ignore
    ctx_eval: "ctypes._Pointer[ggml_context_p]",  # type: ignore
) -> ggml_cgraph_p:
    return lib.ggml_graph_import(fname, ctx_data, ctx_eval)


lib.ggml_graph_import.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ggml_context_p),
    ctypes.POINTER(ggml_context_p),
]
lib.ggml_graph_import.restype = ctypes.POINTER(ggml_cgraph)


# // print info and performance information for the graph
# GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
def ggml_graph_print(
    cgraph: ggml_cgraph_p,
):
    return lib.ggml_graph_print(cgraph)


lib.ggml_graph_print.argtypes = [ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_print.restype = None


# // dump the graph into a file using the dot format
# GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
def ggml_graph_dump_dot(
    gb: ggml_cgraph_p,
    gf: ggml_cgraph_p,
    filename: bytes,
):
    return lib.ggml_graph_dump_dot(gb, gf, filename)


lib.ggml_graph_dump_dot.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_char_p,
]
lib.ggml_graph_dump_dot.restype = None


# // build gradient checkpointing backward graph gb for gf using provided checkpoints
# // gb_tmp will contain original backward graph with rewritten backward process nodes,
# // but without the second forward pass nodes.
# GGML_API void ggml_build_backward_gradient_checkpointing(
#         struct ggml_context   * ctx,
#         struct ggml_cgraph    * gf,
#         struct ggml_cgraph    * gb,
#         struct ggml_cgraph    * gb_tmp,
#         struct ggml_tensor  * * checkpoints,
#         int                     n_checkpoints);
def ggml_build_backward_gradient_checkpointing(
    ctx: ggml_context_p,
    gf: ggml_cgraph_p,
    gb: ggml_cgraph_p,
    gb_tmp: ggml_cgraph_p,
    checkpoints: "ctypes._Pointer[ggml_tensor_p]",  # type: ignore
    n_checkpoints: Union[ctypes.c_int, int],
):
    return lib.ggml_build_backward_gradient_checkpointing(
        ctx, gf, gb, gb_tmp, checkpoints, n_checkpoints
    )


lib.ggml_build_backward_gradient_checkpointing.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
    ctypes.c_int,
]
lib.ggml_build_backward_gradient_checkpointing.restype = None


# //
# // optimization
# //

# // optimization methods
# enum ggml_opt_type {
#     GGML_OPT_ADAM,
#     GGML_OPT_LBFGS,
# };
GGML_OPT_ADAM = 0
GGML_OPT_LBFGS = 1

# // linesearch methods
# enum ggml_linesearch {
#     GGML_LINESEARCH_DEFAULT = 1,

#     GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
#     GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
#     GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
# };
GGML_LINESEARCH_DEFAULT = 1
GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0
GGML_LINESEARCH_BACKTRACKING_WOLFE = 1
GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2

# // optimization return values
# enum ggml_opt_result {
#     GGML_OPT_OK = 0,
#     GGML_OPT_DID_NOT_CONVERGE,
#     GGML_OPT_NO_CONTEXT,
#     GGML_OPT_INVALID_WOLFE,
#     GGML_OPT_FAIL,
#     GGML_OPT_CANCEL,

#     GGML_LINESEARCH_FAIL = -128,
#     GGML_LINESEARCH_MINIMUM_STEP,
#     GGML_LINESEARCH_MAXIMUM_STEP,
#     GGML_LINESEARCH_MAXIMUM_ITERATIONS,
#     GGML_LINESEARCH_INVALID_PARAMETERS,
# };
GGML_OPT_OK = 0
GGML_OPT_DID_NOT_CONVERGE = 1
GGML_OPT_NO_CONTEXT = 2
GGML_OPT_INVALID_WOLFE = 3
GGML_OPT_FAIL = 4
GGML_OPT_CANCEL = 5
GGML_LINESEARCH_FAIL = -128
GGML_LINESEARCH_MINIMUM_STEP = -127
GGML_LINESEARCH_MAXIMUM_STEP = -126
GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125
GGML_LINESEARCH_INVALID_PARAMETERS = -124

# typedef void (*ggml_opt_callback)(void * data, int accum_step, float * sched, bool * cancel);
ggml_opt_callback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_bool),
)

# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
ggml_log_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)

# // optimization parameters
# //
# //   see ggml.c (ggml_opt_default_params) for default values
# //
# struct ggml_opt_params {
#     enum ggml_opt_type type;

#     size_t graph_size;

#     int n_threads;

#     // delta-based convergence test
#     //
#     //   if past == 0 - disabled
#     //   if past > 0:
#     //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
#     //
#     int past;
#     float delta;

#     // maximum number of iterations without improvement
#     //
#     //   if 0 - disabled
#     //   if > 0:
#     //     assume convergence if no cost improvement in this number of iterations
#     //
#     int max_no_improvement;

#     bool print_forward_graph;
#     bool print_backward_graph;

#     int n_gradient_accumulation;

#     // ADAM parameters
#     struct {
#         int n_iter;

#         float sched; // schedule multiplier (fixed, decay or warmup)
#         float decay; // weight decay for AdamW, use 0.0f to disable
#         int   decay_min_ndim; // minimum number of tensor dimension to apply weight decay
#         float alpha; // learning rate
#         float beta1;
#         float beta2;
#         float eps;   // epsilon for numerical stability
#         float eps_f; // epsilon for convergence test
#         float eps_g; // epsilon for convergence test
#         float gclip; // gradient clipping
#     } adam;

#     // LBFGS parameters
#     struct {
#         int m; // number of corrections to approximate the inv. Hessian
#         int n_iter;
#         int max_linesearch;

#         float eps;      // convergence tolerance
#         float ftol;     // line search tolerance
#         float wolfe;
#         float min_step;
#         float max_step;

#         enum ggml_linesearch linesearch;
#     } lbfgs;
# };


class ggml_opt_params_adam(ctypes.Structure):
    _fields_ = [
        ("n_iter", ctypes.c_int),
        ("sched", ctypes.c_float),
        ("decay", ctypes.c_float),
        ("decay_min_ndim", ctypes.c_int),
        ("alpha", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("eps_f", ctypes.c_float),
        ("eps_g", ctypes.c_float),
        ("gclip", ctypes.c_float),
    ]


class ggml_opt_params_lbfgs(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n_iter", ctypes.c_int),
        ("max_linesearch", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("ftol", ctypes.c_float),
        ("wolfe", ctypes.c_float),
        ("min_step", ctypes.c_float),
        ("max_step", ctypes.c_float),
        ("linesearch", ctypes.c_int),
    ]


class ggml_opt_params(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("graph_size", ctypes.c_size_t),
        ("n_threads", ctypes.c_int),
        ("past", ctypes.c_int),
        ("delta", ctypes.c_float),
        ("max_no_improvement", ctypes.c_int),
        ("print_forward_graph", ctypes.c_bool),
        ("print_backward_graph", ctypes.c_bool),
        ("n_gradient_accumulation", ctypes.c_int),
        ("adam", ggml_opt_params_adam),
        ("lbfgs", ggml_opt_params_lbfgs),
    ]


# struct ggml_opt_context {
#     struct ggml_context * ctx;
#     struct ggml_opt_params params;

#     int iter;
#     int64_t nx; // number of parameter elements

#     bool just_initialized;

#     float loss_before;
#     float loss_after;

#     struct {
#         struct ggml_tensor * g;  // current gradient
#         struct ggml_tensor * m;  // first moment
#         struct ggml_tensor * v;  // second moment
#         struct ggml_tensor * pf; // past function values
#         float fx_best;
#         float fx_prev;
#         int n_no_improvement;
#     } adam;

#     struct {
#         struct ggml_tensor * x;    // current parameters
#         struct ggml_tensor * xp;   // previous parameters
#         struct ggml_tensor * g;    // current gradient
#         struct ggml_tensor * gp;   // previous gradient
#         struct ggml_tensor * d;    // search direction
#         struct ggml_tensor * pf;   // past function values
#         struct ggml_tensor * lmal; // the L-BFGS memory alpha
#         struct ggml_tensor * lmys; // the L-BFGS memory ys
#         struct ggml_tensor * lms;  // the L-BFGS memory s
#         struct ggml_tensor * lmy;  // the L-BFGS memory y
#         float fx_best;
#         float step;
#         int j;
#         int k;
#         int end;
#         int n_no_improvement;
#     } lbfgs;
# };


class ggml_opt_context_adam(ctypes.Structure):
    _fields_ = [
        ("g", ctypes.POINTER(ggml_tensor)),
        ("m", ctypes.POINTER(ggml_tensor)),
        ("v", ctypes.POINTER(ggml_tensor)),
        ("pf", ctypes.POINTER(ggml_tensor)),
        ("fx_best", ctypes.c_float),
        ("fx_prev", ctypes.c_float),
        ("n_no_improvement", ctypes.c_int),
    ]


class ggml_opt_context_lbfgs(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.POINTER(ggml_tensor)),
        ("xp", ctypes.POINTER(ggml_tensor)),
        ("g", ctypes.POINTER(ggml_tensor)),
        ("gp", ctypes.POINTER(ggml_tensor)),
        ("d", ctypes.POINTER(ggml_tensor)),
        ("pf", ctypes.POINTER(ggml_tensor)),
        ("lmal", ctypes.POINTER(ggml_tensor)),
        ("lmys", ctypes.POINTER(ggml_tensor)),
        ("lms", ctypes.POINTER(ggml_tensor)),
        ("lmy", ctypes.POINTER(ggml_tensor)),
        ("fx_best", ctypes.c_float),
        ("step", ctypes.c_float),
        ("j", ctypes.c_int),
        ("k", ctypes.c_int),
        ("end", ctypes.c_int),
        ("n_no_improvement", ctypes.c_int),
    ]


class ggml_opt_context(ctypes.Structure):
    _fields_ = [
        ("ctx", ggml_context_p),
        ("params", ggml_opt_params),
        ("iter", ctypes.c_int),
        ("nx", ctypes.c_int64),
        ("just_initialized", ctypes.c_bool),
        ("loss_before", ctypes.c_float),
        ("loss_after", ctypes.c_float),
        ("adam", ggml_opt_context_adam),
        ("lbfgs", ggml_opt_context_lbfgs),
    ]


ggml_opt_context_p = ctypes.POINTER(ggml_opt_context)


# GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);
def ggml_opt_default_params(type: Union[ctypes.c_int, bool]) -> ggml_opt_params:
    return lib.ggml_opt_default_params(type)


lib.ggml_opt_default_params.argtypes = [ctypes.c_int]
lib.ggml_opt_default_params.restype = ggml_opt_params


# // optimize the function defined by the tensor f
# GGML_API enum ggml_opt_result ggml_opt(
#         struct ggml_context * ctx,
#         struct ggml_opt_params params,
#         struct ggml_tensor * f);
def ggml_opt(
    ctx: ggml_context_p,
    params: ggml_opt_params,
    f: ggml_tensor_p,
) -> int:
    return lib.ggml_opt(ctx, params, f)


lib.ggml_opt.argtypes = [ggml_context_p, ggml_opt_params, ctypes.POINTER(ggml_tensor)]
lib.ggml_opt.restype = ctypes.c_int


# // initialize optimizer context
# GGML_API void ggml_opt_init(
#         struct ggml_context     * ctx,
#         struct ggml_opt_context * opt,
#         struct ggml_opt_params    params,
#         int64_t                   nx);
def ggml_opt_init(
    ctx: ggml_context_p,
    opt: "ctypes._Pointer[ggml_opt_context]",  # type: ignore
    params: ggml_opt_params,
    nx: Union[ctypes.c_int64, int],
):
    return lib.ggml_opt_init(ctx, opt, params, nx)


lib.ggml_opt_init.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_opt_context),
    ggml_opt_params,
    ctypes.c_int64,
]
lib.ggml_opt_init.restype = None


# // continue optimizing the function defined by the tensor f
# GGML_API enum ggml_opt_result ggml_opt_resume(
#         struct ggml_context * ctx,
#         struct ggml_opt_context * opt,
#         struct ggml_tensor * f);
def ggml_opt_resume(
    ctx: ggml_context_p,
    opt: "ctypes._Pointer[ggml_opt_context]",  # type: ignore
    f: ggml_tensor_p,
) -> int:
    return lib.ggml_opt_resume(ctx, opt, f)


lib.ggml_opt_resume.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_opt_context),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_opt_resume.restype = ctypes.c_int


# // continue optimizing the function defined by the tensor f
# GGML_API enum ggml_opt_result ggml_opt_resume_g(
#         struct ggml_context * ctx,
#         struct ggml_opt_context * opt,
#         struct ggml_tensor * f,
#         struct ggml_cgraph * gf,
#         struct ggml_cgraph * gb,
#         ggml_opt_callback callback,
#         void * callback_data);
def ggml_opt_resume_g(
    ctx: ggml_context_p,
    opt: "ctypes._Pointer[ggml_opt_context]",  # type: ignore
    f: ggml_tensor_p,
    gf: ggml_cgraph_p,
    gb: ggml_cgraph_p,
    callback: "ctypes._CFuncPtr[None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_bool)]",  # type: ignore
    callback_data: ctypes.c_void_p,
) -> int:
    return lib.ggml_opt_resume_g(ctx, opt, f, gf, gb, callback, callback_data)


lib.ggml_opt_resume_g.argtypes = [
    ggml_context_p,
    ctypes.POINTER(ggml_opt_context),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ggml_cgraph),
    ggml_opt_callback,
    ctypes.c_void_p,
]
lib.ggml_opt_resume_g.restype = ctypes.c_int

# //
# // quantization
# //


# // TODO: these would probably get removed in favor of the more general ggml_quantize_chunk
# GGML_API size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q4_0(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q4_0(src, dst, n, k, hist)


lib.ggml_quantize_q4_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q4_0.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q4_1(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q4_1(src, dst, n, k, hist)


lib.ggml_quantize_q4_1.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q4_1.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q5_0(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q5_0(src, dst, n, k, hist)


lib.ggml_quantize_q5_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q5_0.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q5_1(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q5_1(src, dst, n, k, hist)


lib.ggml_quantize_q5_1.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q5_1.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q8_0(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q8_0(src, dst, n, k, hist)


lib.ggml_quantize_q8_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q8_0.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q2_K(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q2_K(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q2_K(src, dst, n, k, hist)


lib.ggml_quantize_q2_K.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q2_K.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q3_K(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q3_K(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q3_K(src, dst, n, k, hist)


lib.ggml_quantize_q3_K.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q3_K.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q4_K(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q4_K(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q4_K(src, dst, n, k, hist)


lib.ggml_quantize_q4_K.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q4_K.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q5_K(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q5_K(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q5_K(src, dst, n, k, hist)


lib.ggml_quantize_q5_K.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q5_K.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_q6_K(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q6_K(
    src: CFloatArray,
    dst: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
    k: Union[ctypes.c_int, int],
    hist: CInt64Array,
) -> int:
    return lib.ggml_quantize_q6_K(src, dst, n, k, hist)


lib.ggml_quantize_q6_K.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
]
lib.ggml_quantize_q6_K.restype = ctypes.c_size_t


# GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst,
#         int start, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
def ggml_quantize_chunk(
    type: Union[ctypes.c_int, int],
    src: CFloatArray,
    dst: ctypes.c_void_p,
    start: Union[ctypes.c_int, int],
    nrows: Union[ctypes.c_int, int],
    n_per_row: Union[ctypes.c_int, int],
    hist: CInt64Array,
    imatrix: CFloatArray,
) -> int:
    return lib.ggml_quantize_chunk(
        type, src, dst, start, nrows, n_per_row, hist, imatrix
    )


lib.ggml_quantize_chunk.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_float),
]
lib.ggml_quantize_chunk.restype = ctypes.c_size_t


# // These are needed for IQ2_XS and IQ2_XXS quantizations
# GGML_API void ggml_init_iq2_quantization(enum ggml_type type);
# def ggml_init_iq2_quantization(
#     type: Union[ctypes.c_int, int],
# ):
#     return lib.ggml_init_iq2_quantization(type)


# lib.ggml_init_iq2_quantization.argtypes = [
#     ctypes.c_int,
# ]
# lib.ggml_init_iq2_quantization.restype = None


# GGML_API void ggml_deinit_iq2_quantization(enum ggml_type type);
# def ggml_deinit_iq2_quantization(
#     type: Union[ctypes.c_int, int],
# ):
#     return lib.ggml_deinit_iq2_quantization(type)


# lib.ggml_deinit_iq2_quantization.argtypes = [
#     ctypes.c_int,
# ]
# lib.ggml_deinit_iq2_quantization.restype = None

# //
# // Importance matrix
# //
# typedef void(*ggml_collect_imatrix_t)(const struct ggml_tensor * src0, const struct ggml_tensor * src1);
ggml_collect_imatrix_t = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)
)


# GGML_API void ggml_set_imatrix_collection(ggml_collect_imatrix_t imatrix_collect);
# def ggml_set_imatrix_collection(
#     imatrix_collect: Callable[
#         [ggml_tensor_p, ggml_tensor_p], None
#     ]  # TODO: Fix type signature here
# ):
#     return lib.ggml_set_imatrix_collection(imatrix_collect)


# lib.ggml_set_imatrix_collection.argtypes = [
#     ggml_collect_imatrix_t,
# ]
# lib.ggml_set_imatrix_collection.restype = None

# //
# // gguf
# //

# enum gguf_type {
#     GGUF_TYPE_UINT8   = 0,
#     GGUF_TYPE_INT8    = 1,
#     GGUF_TYPE_UINT16  = 2,
#     GGUF_TYPE_INT16   = 3,
#     GGUF_TYPE_UINT32  = 4,
#     GGUF_TYPE_INT32   = 5,
#     GGUF_TYPE_FLOAT32 = 6,
#     GGUF_TYPE_BOOL    = 7,
#     GGUF_TYPE_STRING  = 8,
#     GGUF_TYPE_ARRAY   = 9,
#     GGUF_TYPE_UINT64  = 10,
#     GGUF_TYPE_INT64   = 11,
#     GGUF_TYPE_FLOAT64 = 12,
#     GGUF_TYPE_COUNT,       // marks the end of the enum
# };
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_COUNT = 10

# struct gguf_context;
gguf_context_p = ctypes.c_void_p

# struct gguf_init_params {
#     bool no_alloc;


#     // if not NULL, create a ggml_context and allocate the tensor data in it
#     struct ggml_context ** ctx;
# };
class gguf_init_params(ctypes.Structure):
    _fields_ = [
        ("no_alloc", ctypes.c_bool),
        ("ctx", ctypes.POINTER(ggml_context_p)),
    ]


# GGML_API struct gguf_context * gguf_init_empty(void);
def gguf_init_empty() -> gguf_context_p:
    return lib.gguf_init_empty()


lib.gguf_init_empty.argtypes = []
lib.gguf_init_empty.restype = gguf_context_p


# GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
def gguf_init_from_file(
    fname: bytes,
    params: gguf_init_params,
) -> gguf_context_p:
    return lib.gguf_init_from_file(fname, params)


lib.gguf_init_from_file.argtypes = [
    ctypes.c_char_p,
    gguf_init_params,
]
lib.gguf_init_from_file.restype = gguf_context_p

# //GGML_API struct gguf_context * gguf_init_from_buffer(..);


# GGML_API void gguf_free(struct gguf_context * ctx);
def gguf_free(
    ctx: gguf_context_p,
):
    return lib.gguf_free(ctx)


lib.gguf_free.argtypes = [
    gguf_context_p,
]
lib.gguf_free.restype = None


# GGML_API const char * gguf_type_name(enum gguf_type type);
def gguf_type_name(
    type: Union[ctypes.c_int, int],
) -> bytes:
    return lib.gguf_type_name(type)


lib.gguf_type_name.argtypes = [
    ctypes.c_int,
]
lib.gguf_type_name.restype = ctypes.c_char_p


# GGML_API int    gguf_get_version    (const struct gguf_context * ctx);
def gguf_get_version(
    ctx: gguf_context_p,
) -> int:
    return lib.gguf_get_version(ctx)


lib.gguf_get_version.argtypes = [
    gguf_context_p,
]
lib.gguf_get_version.restype = ctypes.c_int


# GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);
def gguf_get_alignment(
    ctx: gguf_context_p,
) -> int:
    return lib.gguf_get_alignment(ctx)


lib.gguf_get_alignment.argtypes = [
    gguf_context_p,
]
lib.gguf_get_alignment.restype = ctypes.c_size_t


# GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
def gguf_get_data_offset(
    ctx: gguf_context_p,
) -> int:
    return lib.gguf_get_data_offset(ctx)


lib.gguf_get_data_offset.argtypes = [
    gguf_context_p,
]
lib.gguf_get_data_offset.restype = ctypes.c_size_t


# GGML_API void * gguf_get_data       (const struct gguf_context * ctx);
def gguf_get_data(
    ctx: gguf_context_p,
) -> ctypes.c_void_p:
    return lib.gguf_get_data(ctx)


lib.gguf_get_data.argtypes = [
    gguf_context_p,
]
lib.gguf_get_data.restype = ctypes.c_void_p


# GGML_API int          gguf_get_n_kv(const struct gguf_context * ctx);
def gguf_get_n_kv(
    ctx: gguf_context_p,
) -> int:
    return lib.gguf_get_n_kv(ctx)


lib.gguf_get_n_kv.argtypes = [
    gguf_context_p,
]
lib.gguf_get_n_kv.restype = ctypes.c_int


# GGML_API int          gguf_find_key(const struct gguf_context * ctx, const char * key);
def gguf_find_key(
    ctx: gguf_context_p,
    key: bytes,
) -> int:
    return lib.gguf_find_key(ctx, key)


lib.gguf_find_key.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
]
lib.gguf_find_key.restype = ctypes.c_int


# GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int key_id);
def gguf_get_key(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> bytes:
    return lib.gguf_get_key(ctx, key_id)


lib.gguf_get_key.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_key.restype = ctypes.c_char_p


# GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);
def gguf_get_kv_type(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_kv_type(ctx, key_id)


lib.gguf_get_kv_type.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_kv_type.restype = ctypes.c_int


# GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);
def gguf_get_arr_type(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_arr_type(ctx, key_id)


lib.gguf_get_arr_type.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_arr_type.restype = ctypes.c_int


# // results are undefined if the wrong type is used for the key
# GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int key_id);
def gguf_get_val_u8(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_u8(ctx, key_id)


lib.gguf_get_val_u8.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_u8.restype = ctypes.c_uint8


# GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int key_id);
def gguf_get_val_i8(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_i8(ctx, key_id)


lib.gguf_get_val_i8.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_i8.restype = ctypes.c_int8


# GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_u16(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_u16(ctx, key_id)


lib.gguf_get_val_u16.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_u16.restype = ctypes.c_uint16


# GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_i16(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_i16(ctx, key_id)


lib.gguf_get_val_i16.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_i16.restype = ctypes.c_int16


# GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_u32(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_u32(ctx, key_id)


lib.gguf_get_val_u32.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_u32.restype = ctypes.c_uint32


# GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_i32(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_i32(ctx, key_id)


lib.gguf_get_val_i32.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_i32.restype = ctypes.c_int32


# GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_f32(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> float:
    return lib.gguf_get_val_f32(ctx, key_id)


lib.gguf_get_val_f32.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_f32.restype = ctypes.c_float


# GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_u64(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_u64(ctx, key_id)


lib.gguf_get_val_u64.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_u64.restype = ctypes.c_uint64


# GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_i64(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_val_i64(ctx, key_id)


lib.gguf_get_val_i64.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_i64.restype = ctypes.c_int64


# GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);
def gguf_get_val_f64(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> float:
    return lib.gguf_get_val_f64(ctx, key_id)


lib.gguf_get_val_f64.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_f64.restype = ctypes.c_double


# GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int key_id);
def gguf_get_val_bool(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> bool:
    return lib.gguf_get_val_bool(ctx, key_id)


lib.gguf_get_val_bool.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_bool.restype = ctypes.c_bool


# GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);
def gguf_get_val_str(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> bytes:
    return lib.gguf_get_val_str(ctx, key_id)


lib.gguf_get_val_str.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_str.restype = ctypes.c_char_p


# GGML_API const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id);
def gguf_get_val_data(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> ctypes.c_void_p:
    return lib.gguf_get_val_data(ctx, key_id)


lib.gguf_get_val_data.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_val_data.restype = ctypes.c_void_p


# GGML_API int          gguf_get_arr_n   (const struct gguf_context * ctx, int key_id);
def gguf_get_arr_n(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_arr_n(ctx, key_id)


lib.gguf_get_arr_n.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_arr_n.restype = ctypes.c_int


# GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);
def gguf_get_arr_data(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
) -> ctypes.c_void_p:
    return lib.gguf_get_arr_data(ctx, key_id)


lib.gguf_get_arr_data.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_arr_data.restype = ctypes.c_void_p


# GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);
def gguf_get_arr_str(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int, int],
    i: Union[ctypes.c_int, int],
) -> bytes:
    return lib.gguf_get_arr_str(ctx, key_id, i)


lib.gguf_get_arr_str.argtypes = [
    gguf_context_p,
    ctypes.c_int,
    ctypes.c_int,
]
lib.gguf_get_arr_str.restype = ctypes.c_char_p


# GGML_API int            gguf_get_n_tensors    (const struct gguf_context * ctx);
def gguf_get_n_tensors(
    ctx: gguf_context_p,
) -> int:
    return lib.gguf_get_n_tensors(ctx)


lib.gguf_get_n_tensors.argtypes = [
    gguf_context_p,
]
lib.gguf_get_n_tensors.restype = ctypes.c_int


# GGML_API int            gguf_find_tensor      (const struct gguf_context * ctx, const char * name);
def gguf_find_tensor(
    ctx: gguf_context_p,
    name: bytes,
) -> int:
    return lib.gguf_find_tensor(ctx, name)


lib.gguf_find_tensor.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
]
lib.gguf_find_tensor.restype = ctypes.c_int


# GGML_API size_t         gguf_get_tensor_offset(const struct gguf_context * ctx, int i);
def gguf_get_tensor_offset(
    ctx: gguf_context_p,
    i: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_tensor_offset(ctx, i)


lib.gguf_get_tensor_offset.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_tensor_offset.restype = ctypes.c_size_t


# GGML_API char *         gguf_get_tensor_name  (const struct gguf_context * ctx, int i);
def gguf_get_tensor_name(
    ctx: gguf_context_p,
    i: Union[ctypes.c_int, int],
) -> bytes:
    return lib.gguf_get_tensor_name(ctx, i)


lib.gguf_get_tensor_name.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_tensor_name.restype = ctypes.c_char_p


# GGML_API enum ggml_type gguf_get_tensor_type  (const struct gguf_context * ctx, int i);
def gguf_get_tensor_type(
    ctx: gguf_context_p,
    i: Union[ctypes.c_int, int],
) -> int:
    return lib.gguf_get_tensor_type(ctx, i)


lib.gguf_get_tensor_type.argtypes = [
    gguf_context_p,
    ctypes.c_int,
]
lib.gguf_get_tensor_type.restype = ctypes.c_int


# // overrides existing values or adds a new one
# GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
def gguf_set_val_u8(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint8, int],
):
    return lib.gguf_set_val_u8(ctx, key, val)


lib.gguf_set_val_u8.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_uint8,
]
lib.gguf_set_val_u8.restype = None


# GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);
def gguf_set_val_i8(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int8, int],
):
    return lib.gguf_set_val_i8(ctx, key, val)


lib.gguf_set_val_i8.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_int8,
]
lib.gguf_set_val_i8.restype = None


# GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
def gguf_set_val_u16(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint16, int],
):
    return lib.gguf_set_val_u16(ctx, key, val)


lib.gguf_set_val_u16.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_uint16,
]
lib.gguf_set_val_u16.restype = None


# GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);
def gguf_set_val_i16(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int16, int],
):
    return lib.gguf_set_val_i16(ctx, key, val)


lib.gguf_set_val_i16.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_int16,
]
lib.gguf_set_val_i16.restype = None


# GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
def gguf_set_val_u32(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint32, int],
):
    return lib.gguf_set_val_u32(ctx, key, val)


lib.gguf_set_val_u32.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_uint32,
]
lib.gguf_set_val_u32.restype = None


# GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);
def gguf_set_val_i32(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int32, int],
):
    return lib.gguf_set_val_i32(ctx, key, val)


lib.gguf_set_val_i32.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_int32,
]
lib.gguf_set_val_i32.restype = None


# GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);
def gguf_set_val_f32(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_float, float],
):
    return lib.gguf_set_val_f32(ctx, key, val)


lib.gguf_set_val_f32.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_float,
]
lib.gguf_set_val_f32.restype = None


# GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
def gguf_set_val_u64(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint64, int],
):
    return lib.gguf_set_val_u64(ctx, key, val)


lib.gguf_set_val_u64.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_uint64,
]
lib.gguf_set_val_u64.restype = None


# GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t  val);
def gguf_set_val_i64(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int64, int],
):
    return lib.gguf_set_val_i64(ctx, key, val)


lib.gguf_set_val_i64.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_int64,
]
lib.gguf_set_val_i64.restype = None


# GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double   val);
def gguf_set_val_f64(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_double, float],
):
    return lib.gguf_set_val_f64(ctx, key, val)


lib.gguf_set_val_f64.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_double,
]
lib.gguf_set_val_f64.restype = None


# GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);
def gguf_set_val_bool(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_bool, bool],
):
    return lib.gguf_set_val_bool(ctx, key, val)


lib.gguf_set_val_bool.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_bool,
]
lib.gguf_set_val_bool.restype = None


# GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
def gguf_set_val_str(
    ctx: gguf_context_p,
    key: bytes,
    val: bytes,
):
    return lib.gguf_set_val_str(ctx, key, val)


lib.gguf_set_val_str.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
]
lib.gguf_set_val_str.restype = None


# GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);
def gguf_set_arr_data(
    ctx: gguf_context_p,
    key: bytes,
    type: Union[ctypes.c_int, int],
    data: ctypes.c_void_p,
    n: Union[ctypes.c_int, int],
):
    return lib.gguf_set_arr_data(ctx, key, type, data, n)


lib.gguf_set_arr_data.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]
lib.gguf_set_arr_data.restype = None


# GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);
def gguf_set_arr_str(
    ctx: gguf_context_p,
    key: bytes,
    data: CCharPointer,
    n: Union[ctypes.c_int, int],
):
    return lib.gguf_set_arr_str(ctx, key, data, n)


lib.gguf_set_arr_str.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
]
lib.gguf_set_arr_str.restype = None


# // set or add KV pairs from another context
# GGML_API void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);
def gguf_set_kv(
    ctx: gguf_context_p,
    src: gguf_context_p,
):
    return lib.gguf_set_kv(ctx, src)


lib.gguf_set_kv.argtypes = [
    gguf_context_p,
    gguf_context_p,
]
lib.gguf_set_kv.restype = None


# // manage tensor info
# GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
def gguf_add_tensor(
    ctx: gguf_context_p,
    tensor: ggml_tensor_p,
):
    return lib.gguf_add_tensor(ctx, tensor)


lib.gguf_add_tensor.argtypes = [
    gguf_context_p,
    ctypes.POINTER(ggml_tensor),
]
lib.gguf_add_tensor.restype = None


# GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
def gguf_set_tensor_type(
    ctx: gguf_context_p,
    name: bytes,
    type: Union[ctypes.c_int, int],
):
    return lib.gguf_set_tensor_type(ctx, name, type)


lib.gguf_set_tensor_type.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_int,
]
lib.gguf_set_tensor_type.restype = None


# GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);
def gguf_set_tensor_data(
    ctx: gguf_context_p,
    name: bytes,
    data: ctypes.c_void_p,
    size: Union[ctypes.c_size_t, int],
):
    return lib.gguf_set_tensor_data(ctx, name, data, size)


lib.gguf_set_tensor_data.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
lib.gguf_set_tensor_data.restype = None

# // writing gguf files can be done in 2 ways:
# //
# // - write the entire gguf_context to a binary file in a single pass:
# //
# //   gguf_write_to_file(ctx, fname);
# //
# // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
# //
# //   FILE * f = fopen(fname, "wb");
# //   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
# //   fwrite(f, ...);
# //   void * data = gguf_meta_get_meta_data(ctx);
# //   fseek(f, 0, SEEK_SET);
# //   fwrite(f, data, gguf_get_meta_size(ctx));
# //   free(data);
# //   fclose(f);
# //


# // write the entire context to a binary file
# GGML_API void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
def gguf_write_to_file(
    ctx: gguf_context_p,
    fname: bytes,
    only_meta: Union[ctypes.c_bool, bool],
):
    return lib.gguf_write_to_file(ctx, fname, only_meta)


lib.gguf_write_to_file.argtypes = [
    gguf_context_p,
    ctypes.c_char_p,
    ctypes.c_bool,
]
lib.gguf_write_to_file.restype = None


# // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
# GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
def gguf_get_meta_size(
    ctx: gguf_context_p,
) -> int:
    return lib.gguf_get_meta_size(ctx)


lib.gguf_get_meta_size.argtypes = [
    gguf_context_p,
]
lib.gguf_get_meta_size.restype = ctypes.c_size_t


# GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);
def gguf_get_meta_data(
    ctx: gguf_context_p,
    data: ctypes.c_void_p,
):
    return lib.gguf_get_meta_data(ctx, data)


lib.gguf_get_meta_data.argtypes = [
    gguf_context_p,
    ctypes.c_void_p,
]
lib.gguf_get_meta_data.restype = None


# //
# // system info
# //


# GGML_API int ggml_cpu_has_avx        (void);
def ggml_cpu_has_avx() -> int:
    return lib.ggml_cpu_has_avx()


lib.ggml_cpu_has_avx.argtypes = []
lib.ggml_cpu_has_avx.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_avx_vnni   (void);
def ggml_cpu_has_avx_vnni() -> int:
    return lib.ggml_cpu_has_avx_vnni()


lib.ggml_cpu_has_avx_vnni.argtypes = []
lib.ggml_cpu_has_avx_vnni.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_avx2       (void);
def ggml_cpu_has_avx2() -> int:
    return lib.ggml_cpu_has_avx2()


lib.ggml_cpu_has_avx2.argtypes = []
lib.ggml_cpu_has_avx2.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_avx512     (void);
def ggml_cpu_has_avx512() -> int:
    return lib.ggml_cpu_has_avx512()


lib.ggml_cpu_has_avx512.argtypes = []
lib.ggml_cpu_has_avx512.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_avx512_vbmi(void);
def ggml_cpu_has_avx512_vbmi() -> int:
    return lib.ggml_cpu_has_avx512_vbmi()


lib.ggml_cpu_has_avx512_vbmi.argtypes = []
lib.ggml_cpu_has_avx512_vbmi.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_avx512_vnni(void);
def ggml_cpu_has_avx512_vnni() -> int:
    return lib.ggml_cpu_has_avx512_vnni()


lib.ggml_cpu_has_avx512_vnni.argtypes = []
lib.ggml_cpu_has_avx512_vnni.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_fma        (void);
def ggml_cpu_has_fma() -> int:
    return lib.ggml_cpu_has_fma()


lib.ggml_cpu_has_fma.argtypes = []
lib.ggml_cpu_has_fma.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_neon       (void);
def ggml_cpu_has_neon() -> int:
    return lib.ggml_cpu_has_neon()


lib.ggml_cpu_has_neon.argtypes = []
lib.ggml_cpu_has_neon.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_arm_fma    (void);
def ggml_cpu_has_arm_fma() -> int:
    return lib.ggml_cpu_has_arm_fma()


lib.ggml_cpu_has_arm_fma.argtypes = []
lib.ggml_cpu_has_arm_fma.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_metal      (void);
def ggml_cpu_has_metal() -> int:
    return lib.ggml_cpu_has_metal()


lib.ggml_cpu_has_metal.argtypes = []
lib.ggml_cpu_has_metal.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_f16c       (void);
def ggml_cpu_has_f16c() -> int:
    return lib.ggml_cpu_has_f16c()


lib.ggml_cpu_has_f16c.argtypes = []
lib.ggml_cpu_has_f16c.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_fp16_va    (void);
def ggml_cpu_has_fp16_va() -> int:
    return lib.ggml_cpu_has_fp16_va()


lib.ggml_cpu_has_fp16_va.argtypes = []
lib.ggml_cpu_has_fp16_va.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_wasm_simd  (void);
def ggml_cpu_has_wasm_simd() -> int:
    return lib.ggml_cpu_has_wasm_simd()


lib.ggml_cpu_has_wasm_simd.argtypes = []
lib.ggml_cpu_has_wasm_simd.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_blas       (void);
def ggml_cpu_has_blas() -> int:
    return lib.ggml_cpu_has_blas()


lib.ggml_cpu_has_blas.argtypes = []
lib.ggml_cpu_has_blas.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_cublas     (void);
def ggml_cpu_has_cublas() -> int:
    return lib.ggml_cpu_has_cublas()


lib.ggml_cpu_has_cublas.argtypes = []
lib.ggml_cpu_has_cublas.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_clblast    (void);
def ggml_cpu_has_clblast() -> int:
    return lib.ggml_cpu_has_clblast()


lib.ggml_cpu_has_clblast.argtypes = []
lib.ggml_cpu_has_clblast.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_gpublas    (void);
def ggml_cpu_has_gpublas() -> int:
    return lib.ggml_cpu_has_gpublas()


lib.ggml_cpu_has_gpublas.argtypes = []
lib.ggml_cpu_has_gpublas.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_sse3       (void);
def ggml_cpu_has_sse3() -> int:
    return lib.ggml_cpu_has_sse3()


lib.ggml_cpu_has_sse3.argtypes = []
lib.ggml_cpu_has_sse3.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_ssse3      (void);
def ggml_cpu_has_ssse3() -> int:
    return lib.ggml_cpu_has_ssse3()


lib.ggml_cpu_has_ssse3.argtypes = []
lib.ggml_cpu_has_ssse3.restype = ctypes.c_int


# GGML_API int ggml_cpu_has_vsx        (void);
def ggml_cpu_has_vsx() -> int:
    return lib.ggml_cpu_has_vsx()


lib.ggml_cpu_has_vsx.argtypes = []
lib.ggml_cpu_has_vsx.restype = ctypes.c_int


# //
# // Internal types and functions exposed for tests and benchmarks
# //

# typedef void (*ggml_to_float_t)(const void * x, float * y, int k);
ggml_to_float_t = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
)

# typedef void (*ggml_from_float_t)(const float * x, void * y, int k);
ggml_from_float_t = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int
)

# typedef void (*ggml_vec_dot_t)(const int n, float * s, const void * x, const void * y);
ggml_vec_dot_t = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p
)


# typedef struct {
#     const char      * type_name;
#     int               blck_size;
#     size_t            type_size;
#     bool              is_quantized;
#     ggml_to_float_t   to_float;
#     ggml_from_float_t from_float;
#     ggml_from_float_t from_float_reference;
#     ggml_vec_dot_t    vec_dot;
#     enum ggml_type    vec_dot_type;
# } ggml_type_traits_t;
class ggml_type_traits_t(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ggml_to_float_t),
        ("from_float", ggml_from_float_t),
        ("from_float_reference", ggml_from_float_t),
        ("vec_dot", ggml_vec_dot_t),
        ("vec_dot_type", ctypes.c_int),
    ]


# GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);
def ggml_internal_get_type_traits(type: Union[ctypes.c_int, int]) -> ggml_type_traits_t:
    return lib.ggml_internal_get_type_traits(type)


lib.ggml_internal_get_type_traits.argtypes = [ctypes.c_int]
lib.ggml_internal_get_type_traits.restype = ggml_type_traits_t

#####################################################
# GGML ALLOC API
# source: ggml-alloc.h
#####################################################

# struct ggml_backend;
# struct ggml_backend_buffer;
# struct ggml_backend_buffer_type;
ggml_backend_t = ctypes.c_void_p
ggml_backend_buffer_p = ctypes.c_void_p
ggml_backend_buffer_type_p = ctypes.c_void_p

# //
# // Legacy API
# //

# typedef struct ggml_allocr * ggml_allocr_t;
ggml_allocr_t = ctypes.c_void_p


# // initialize allocator for use with CPU backend only
# GGML_API ggml_allocr_t ggml_allocr_new(void * data, size_t size, size_t alignment);
def ggml_allocr_new(
    data: ctypes.c_void_p,
    size: Union[ctypes.c_size_t, int],
    alignment: Union[ctypes.c_size_t, int],
) -> ggml_allocr_t:
    return lib.ggml_allocr_new(data, size, alignment)


lib.ggml_allocr_new.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_allocr_new.restype = ggml_allocr_t


# GGML_API ggml_allocr_t ggml_allocr_new_measure(size_t alignment);
def ggml_allocr_new_measure(alignment: Union[ctypes.c_size_t, int]) -> ggml_allocr_t:
    return lib.ggml_allocr_new_measure(alignment)


lib.ggml_allocr_new_measure.argtypes = [ctypes.c_size_t]
lib.ggml_allocr_new_measure.restype = ggml_allocr_t


# // initialize allocator for use with ggml-backend
# GGML_API ggml_allocr_t ggml_allocr_new_from_buffer(struct ggml_backend_buffer * buffer);
def ggml_allocr_new_from_buffer(buffer: ggml_backend_buffer_p) -> ggml_allocr_t:
    return lib.ggml_allocr_new_from_buffer(buffer)


lib.ggml_allocr_new_from_buffer.argtypes = [ggml_backend_buffer_p]
lib.ggml_allocr_new_from_buffer.restype = ggml_allocr_t


# GGML_API ggml_allocr_t ggml_allocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
def ggml_allocr_new_from_backend(
    backend: ggml_backend_t, size: Union[ctypes.c_size_t, int]
) -> ggml_allocr_t:
    return lib.ggml_allocr_new_from_backend(backend, size)


lib.ggml_allocr_new_from_backend.argtypes = [ggml_backend_t, ctypes.c_size_t]
lib.ggml_allocr_new_from_backend.restype = ggml_allocr_t


# GGML_API ggml_allocr_t ggml_allocr_new_measure_from_backend(struct ggml_backend * backend);
def ggml_allocr_new_measure_from_backend(backend: ggml_backend_t) -> ggml_allocr_t:
    return lib.ggml_allocr_new_measure_from_backend(backend)


lib.ggml_allocr_new_measure_from_backend.argtypes = [ggml_backend_t]
lib.ggml_allocr_new_measure_from_backend.restype = ggml_allocr_t


# GGML_API struct ggml_backend_buffer * ggml_allocr_get_buffer(ggml_allocr_t alloc);
def ggml_allocr_get_buffer(alloc: ggml_allocr_t) -> ggml_backend_buffer_p:
    return lib.ggml_allocr_get_buffer(alloc)


lib.ggml_allocr_get_buffer.argtypes = [ggml_allocr_t]
lib.ggml_allocr_get_buffer.restype = ggml_backend_buffer_p


# // tell the allocator to parse nodes following the order described in the list
# // you should call this if your graph are optimized to execute out-of-order
# GGML_API void   ggml_allocr_set_parse_seq(ggml_allocr_t alloc, const int * list, int n);
def ggml_allocr_set_parse_seq(
    alloc: ggml_allocr_t,
    list: "ctypes._Pointer(ctypes.c_int)",  # type: ignore
    n: Union[ctypes.c_int, int],
) -> None:
    return lib.ggml_allocr_set_parse_seq(alloc, list, n)


lib.ggml_allocr_set_parse_seq.argtypes = [
    ggml_allocr_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.ggml_allocr_set_parse_seq.restype = None


# GGML_API void   ggml_allocr_free       (ggml_allocr_t alloc);
def ggml_allocr_free(alloc: ggml_allocr_t) -> None:
    return lib.ggml_allocr_free(alloc)


lib.ggml_allocr_free.argtypes = [ggml_allocr_t]
lib.ggml_allocr_free.restype = None


# GGML_API bool   ggml_allocr_is_measure (ggml_allocr_t alloc);
def ggml_allocr_is_measure(alloc: ggml_allocr_t) -> ctypes.c_bool:
    return lib.ggml_allocr_is_measure(alloc)


lib.ggml_allocr_is_measure.argtypes = [ggml_allocr_t]
lib.ggml_allocr_is_measure.restype = ctypes.c_bool


# GGML_API void   ggml_allocr_reset      (ggml_allocr_t alloc);
def ggml_allocr_reset(alloc: ggml_allocr_t) -> None:
    return lib.ggml_allocr_reset(alloc)


lib.ggml_allocr_reset.argtypes = [ggml_allocr_t]
lib.ggml_allocr_reset.restype = None


# GGML_API void   ggml_allocr_alloc      (ggml_allocr_t alloc, struct ggml_tensor * tensor);
def ggml_allocr_alloc(alloc: ggml_allocr_t, tensor: ggml_tensor_p) -> None:
    return lib.ggml_allocr_alloc(alloc, tensor)


lib.ggml_allocr_alloc.argtypes = [ggml_allocr_t, ctypes.POINTER(ggml_tensor)]
lib.ggml_allocr_alloc.restype = None


# GGML_API size_t ggml_allocr_max_size   (ggml_allocr_t alloc);
def ggml_allocr_max_size(alloc: ggml_allocr_t) -> Union[ctypes.c_size_t, int]:
    return lib.ggml_allocr_max_size(alloc)


lib.ggml_allocr_max_size.argtypes = [ggml_allocr_t]
lib.ggml_allocr_max_size.restype = ctypes.c_size_t


# GGML_API size_t ggml_allocr_alloc_graph(ggml_allocr_t alloc, struct ggml_cgraph * graph);
def ggml_allocr_alloc_graph(alloc: ggml_allocr_t, graph: ggml_cgraph_p) -> int:
    return lib.ggml_allocr_alloc_graph(alloc, graph)


lib.ggml_allocr_alloc_graph.argtypes = [ggml_allocr_t, ctypes.POINTER(ggml_cgraph)]
lib.ggml_allocr_alloc_graph.restype = ctypes.c_size_t

# //
# // ggml-backend v2 API
# //

# // Separate tensor and graph allocator objects
# // This is necessary for multi-backend allocation because the graph allocator needs to use multiple tensor allocators
# // The original API is kept as a wrapper around the new API

# // Tensor allocator
# typedef struct ggml_tallocr * ggml_tallocr_t;
ggml_tallocr_t = ctypes.c_void_p


# GGML_API ggml_tallocr_t ggml_tallocr_new(void * data, size_t size, size_t alignment);
def ggml_tallocr_new(
    data: ctypes.c_void_p,
    size: Union[ctypes.c_size_t, int],
    alignment: Union[ctypes.c_size_t, int],
) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new(data, size, alignment)


lib.ggml_tallocr_new.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_tallocr_new.restype = ggml_tallocr_t


# GGML_API ggml_tallocr_t ggml_tallocr_new_measure(size_t alignment);
def ggml_tallocr_new_measure(alignment: Union[ctypes.c_size_t, int]) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new_measure(alignment)


lib.ggml_tallocr_new_measure.argtypes = [ctypes.c_size_t]
lib.ggml_tallocr_new_measure.restype = ggml_tallocr_t


# GGML_API ggml_tallocr_t ggml_tallocr_new_from_buft(struct ggml_backend_buffer_type * buft, size_t size);
def ggml_tallocr_new_from_buft(
    buft: ggml_backend_buffer_type_p, size: Union[ctypes.c_size_t, int]
) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new_from_buft(buft, size)


lib.ggml_tallocr_new_from_buft.argtypes = [ggml_backend_buffer_type_p, ctypes.c_size_t]
lib.ggml_tallocr_new_from_buft.restype = ggml_tallocr_t


# GGML_API ggml_tallocr_t ggml_tallocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
def ggml_tallocr_new_from_backend(
    backend: ggml_backend_t, size: Union[ctypes.c_size_t, int]
) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new_from_backend(backend, size)


lib.ggml_tallocr_new_from_backend.argtypes = [ggml_backend_t, ctypes.c_size_t]
lib.ggml_tallocr_new_from_backend.restype = ggml_tallocr_t


# GGML_API ggml_tallocr_t ggml_tallocr_new_from_buffer(struct ggml_backend_buffer * buffer);
def ggml_tallocr_new_from_buffer(buffer: ggml_backend_buffer_p) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new_from_buffer(buffer)


lib.ggml_tallocr_new_from_buffer.argtypes = [ggml_backend_buffer_p]
lib.ggml_tallocr_new_from_buffer.restype = ggml_tallocr_t


# GGML_API ggml_tallocr_t ggml_tallocr_new_measure_from_buft(struct ggml_backend_buffer_type * buft);
def ggml_tallocr_new_measure_from_buft(
    buft: ggml_backend_buffer_type_p,
) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new_measure_from_buft(buft)


lib.ggml_tallocr_new_measure_from_buft.argtypes = [ggml_backend_buffer_type_p]
lib.ggml_tallocr_new_measure_from_buft.restype = ggml_tallocr_t


# GGML_API ggml_tallocr_t ggml_tallocr_new_measure_from_backend(struct ggml_backend * backend);
def ggml_tallocr_new_measure_from_backend(backend: ggml_backend_t) -> ggml_tallocr_t:
    return lib.ggml_tallocr_new_measure_from_backend(backend)


lib.ggml_tallocr_new_measure_from_backend.argtypes = [ggml_backend_t]
lib.ggml_tallocr_new_measure_from_backend.restype = ggml_tallocr_t


# GGML_API struct ggml_backend_buffer * ggml_tallocr_get_buffer(ggml_tallocr_t talloc);
def ggml_tallocr_get_buffer(talloc: ggml_tallocr_t) -> ggml_backend_buffer_p:
    return lib.ggml_tallocr_get_buffer(talloc)


lib.ggml_tallocr_get_buffer.argtypes = [ggml_tallocr_t]
lib.ggml_tallocr_get_buffer.restype = ggml_backend_buffer_p


# GGML_API void   ggml_tallocr_free       (ggml_tallocr_t talloc);
def ggml_tallocr_free(talloc: ggml_tallocr_t) -> None:
    return lib.ggml_tallocr_free(talloc)


lib.ggml_tallocr_free.argtypes = [ggml_tallocr_t]
lib.ggml_tallocr_free.restype = None


# GGML_API bool   ggml_tallocr_is_measure (ggml_tallocr_t talloc);
def ggml_tallocr_is_measure(talloc: ggml_tallocr_t) -> bool:
    return lib.ggml_tallocr_is_measure(talloc)


lib.ggml_tallocr_is_measure.argtypes = [ggml_tallocr_t]
lib.ggml_tallocr_is_measure.restype = ctypes.c_bool


# GGML_API void   ggml_tallocr_reset      (ggml_tallocr_t talloc);
def ggml_tallocr_reset(talloc: ggml_tallocr_t) -> None:
    return lib.ggml_tallocr_reset(talloc)


lib.ggml_tallocr_reset.argtypes = [ggml_tallocr_t]
lib.ggml_tallocr_reset.restype = None


# GGML_API void   ggml_tallocr_alloc      (ggml_tallocr_t talloc, struct ggml_tensor * tensor);
def ggml_tallocr_alloc(talloc: ggml_tallocr_t, tensor: ggml_tensor_p) -> None:
    return lib.ggml_tallocr_alloc(talloc, tensor)


lib.ggml_tallocr_alloc.argtypes = [ggml_tallocr_t, ctypes.POINTER(ggml_tensor)]
lib.ggml_tallocr_alloc.restype = None


# GGML_API size_t ggml_tallocr_max_size   (ggml_tallocr_t talloc);
def ggml_tallocr_max_size(talloc: ggml_tallocr_t) -> Union[ctypes.c_size_t, int]:
    return lib.ggml_tallocr_max_size(talloc)


lib.ggml_tallocr_max_size.argtypes = [ggml_tallocr_t]
lib.ggml_tallocr_max_size.restype = ctypes.c_size_t


# // Graph allocator
# typedef struct ggml_gallocr * ggml_gallocr_t;
ggml_gallocr_t = ctypes.c_void_p


# GGML_API ggml_gallocr_t ggml_gallocr_new(void);
def ggml_gallocr_new() -> ggml_gallocr_t:
    return lib.ggml_gallocr_new()


lib.ggml_gallocr_new.argtypes = []
lib.ggml_gallocr_new.restype = ggml_gallocr_t


# GGML_API void   ggml_gallocr_free(ggml_gallocr_t galloc);
def ggml_gallocr_free(galloc: ggml_gallocr_t) -> None:
    return lib.ggml_gallocr_free(galloc)


lib.ggml_gallocr_free.argtypes = [ggml_gallocr_t]
lib.ggml_gallocr_free.restype = None


# GGML_API void   ggml_gallocr_set_parse_seq(ggml_gallocr_t galloc, const int * list, int n);
def ggml_gallocr_set_parse_seq(
    galloc: ggml_gallocr_t,
    list: "ctypes._Pointer(ctypes.c_int)",  # type: ignore
    n: Union[ctypes.c_int, int],
) -> None:
    return lib.ggml_gallocr_set_parse_seq(galloc, list, n)


lib.ggml_gallocr_set_parse_seq.argtypes = [
    ggml_gallocr_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.ggml_gallocr_set_parse_seq.restype = None


# GGML_API size_t ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ggml_tallocr_t talloc, struct ggml_cgraph * graph);
def ggml_gallocr_alloc_graph(
    galloc: ggml_gallocr_t, talloc: ggml_tallocr_t, graph: ggml_cgraph_p
) -> Union[ctypes.c_size_t, int]:
    return lib.ggml_gallocr_alloc_graph(galloc, talloc, graph)


lib.ggml_gallocr_alloc_graph.argtypes = [
    ggml_gallocr_t,
    ggml_tallocr_t,
    ctypes.POINTER(ggml_cgraph),
]
lib.ggml_gallocr_alloc_graph.restype = ctypes.c_size_t


# // Allocate tensors from the allocators given by the hash table
# GGML_API void   ggml_gallocr_alloc_graph_n(
#                     ggml_gallocr_t galloc,
#                     struct ggml_cgraph * graph,
#                     struct ggml_hash_set hash_set,
#                     ggml_tallocr_t * hash_node_talloc);
def ggml_gallocr_alloc_graph_n(
    galloc: ggml_gallocr_t,
    graph: ggml_cgraph_p,
    hash_set: ggml_hash_set,
    hash_node_talloc: "ctypes._Pointer(ggml_tallocr_t)",  # type: ignore
) -> None:
    return lib.ggml_gallocr_alloc_graph_n(galloc, graph, hash_set, hash_node_talloc)


lib.ggml_gallocr_alloc_graph_n.argtypes = [
    ggml_gallocr_t,
    ctypes.POINTER(ggml_cgraph),
    ggml_hash_set,
    ctypes.POINTER(ggml_tallocr_t),
]
lib.ggml_gallocr_alloc_graph_n.restype = None


# // Utils
# // Create a buffer and allocate all the tensors in a ggml_context
# GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, struct ggml_backend_buffer_type * buft);
def ggml_backend_alloc_ctx_tensors_from_buft(
    ctx: ggml_context_p, buft: ggml_backend_buffer_type_p
) -> ggml_backend_buffer_p:
    return lib.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)


lib.ggml_backend_alloc_ctx_tensors_from_buft.argtypes = [
    ggml_context_p,
    ggml_backend_buffer_type_p,
]
lib.ggml_backend_alloc_ctx_tensors_from_buft.restype = ggml_backend_buffer_p


# GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, struct ggml_backend * backend);
def ggml_backend_alloc_ctx_tensors(
    ctx: ggml_context_p, backend: ggml_backend_t
) -> ggml_backend_buffer_p:
    return lib.ggml_backend_alloc_ctx_tensors(ctx, backend)


lib.ggml_backend_alloc_ctx_tensors.argtypes = [
    ggml_context_p,
    ggml_backend_t,
]
lib.ggml_backend_alloc_ctx_tensors.restype = ggml_backend_buffer_p

#####################################################
# GGML Backend API
# source: ggml-backend.h
#####################################################

# typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
# typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
# typedef struct ggml_backend * ggml_backend_t;
# typedef void * ggml_backend_graph_plan_t;
ggml_backend_buffer_type_t = ctypes.c_void_p
ggml_backend_buffer_t = ctypes.c_void_p
ggml_backend_t = ctypes.c_void_p
ggml_backend_graph_plan_t = ctypes.c_void_p

# //
# // Backend buffer
# //


# // buffer type
# GGML_API const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
def ggml_backend_buft_name(
    buft: ggml_backend_buffer_type_t,
) -> bytes:
    return lib.ggml_backend_buft_name(buft)


lib.ggml_backend_buft_name.argtypes = [ggml_backend_buffer_type_t]
lib.ggml_backend_buft_name.restype = ctypes.c_char_p


# GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer    (ggml_backend_buffer_type_t buft, size_t size);
def ggml_backend_buft_alloc_buffer(
    buft: ggml_backend_buffer_type_t, size: Union[ctypes.c_size_t, int]
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_buft_alloc_buffer(buft, size)


lib.ggml_backend_buft_alloc_buffer.argtypes = [
    ggml_backend_buffer_type_t,
    ctypes.c_size_t,
]
lib.ggml_backend_buft_alloc_buffer.restype = ggml_backend_buffer_t


# GGML_API size_t                ggml_backend_buft_get_alignment   (ggml_backend_buffer_type_t buft);
def ggml_backend_buft_get_alignment(
    buft: ggml_backend_buffer_type_t,
) -> int:
    return lib.ggml_backend_buft_get_alignment(buft)


lib.ggml_backend_buft_get_alignment.argtypes = [ggml_backend_buffer_type_t]
lib.ggml_backend_buft_get_alignment.restype = ctypes.c_size_t


# GGML_API size_t                ggml_backend_buft_get_alloc_size  (ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor);
def ggml_backend_buft_get_alloc_size(
    buft: ggml_backend_buffer_type_t, tensor: ggml_tensor_p
) -> int:
    return lib.ggml_backend_buft_get_alloc_size(buft, tensor)


lib.ggml_backend_buft_get_alloc_size.argtypes = [
    ggml_backend_buffer_type_t,
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_buft_get_alloc_size.restype = ctypes.c_size_t


# GGML_API bool                  ggml_backend_buft_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend);
def ggml_backend_buft_supports_backend(
    buft: ggml_backend_buffer_type_t, backend: ggml_backend_t
) -> bool:
    return lib.ggml_backend_buft_supports_backend(buft, backend)


lib.ggml_backend_buft_supports_backend.argtypes = [
    ggml_backend_buffer_type_t,
    ggml_backend_t,
]
lib.ggml_backend_buft_supports_backend.restype = ctypes.c_bool


# GGML_API bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);
def ggml_backend_buft_is_host(
    buft: ggml_backend_buffer_type_t,
) -> bool:
    return lib.ggml_backend_buft_is_host(buft)


lib.ggml_backend_buft_is_host.argtypes = [ggml_backend_buffer_type_t]
lib.ggml_backend_buft_is_host.restype = ctypes.c_bool


# // buffer
# enum ggml_backend_buffer_usage {
#     GGML_BACKEND_BUFFER_USAGE_ANY = 0,
#     GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
# };
GGML_BACKEND_BUFFER_USAGE_ANY = 0
GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1


# GGML_API const char *               ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_name(
    buffer: ggml_backend_buffer_t,
) -> bytes:
    return lib.ggml_backend_buffer_name(buffer)


lib.ggml_backend_buffer_name.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_name.restype = ctypes.c_char_p


# GGML_API void                       ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_free(
    buffer: ggml_backend_buffer_t,
):
    return lib.ggml_backend_buffer_free(buffer)


lib.ggml_backend_buffer_free.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_free.restype = None


# GGML_API void *                     ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_get_base(
    buffer: ggml_backend_buffer_t,
) -> ctypes.c_void_p:
    return lib.ggml_backend_buffer_get_base(buffer)


lib.ggml_backend_buffer_get_base.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_get_base.restype = ctypes.c_void_p


# GGML_API size_t                     ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_get_size(
    buffer: ggml_backend_buffer_t,
) -> int:
    return lib.ggml_backend_buffer_get_size(buffer)


lib.ggml_backend_buffer_get_size.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_get_size.restype = ctypes.c_size_t


# GGML_API void                       ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
def ggml_backend_buffer_init_tensor(
    buffer: ggml_backend_buffer_t,
    tensor: ggml_tensor_p,
):
    return lib.ggml_backend_buffer_init_tensor(buffer, tensor)


lib.ggml_backend_buffer_init_tensor.argtypes = [
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_buffer_init_tensor.restype = None


# GGML_API size_t                     ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_get_alignment(
    buffer: ggml_backend_buffer_t,
) -> int:
    return lib.ggml_backend_buffer_get_alignment(buffer)


lib.ggml_backend_buffer_get_alignment.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_get_alignment.restype = ctypes.c_size_t


# GGML_API size_t                     ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
def ggml_backend_buffer_get_alloc_size(
    buffer: ggml_backend_buffer_t, tensor: ggml_tensor_p
) -> int:
    return lib.ggml_backend_buffer_get_alloc_size(buffer, tensor)


lib.ggml_backend_buffer_get_alloc_size.argtypes = [
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_buffer_get_alloc_size.restype = ctypes.c_size_t


# GGML_API void                       ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
def ggml_backend_buffer_clear(buffer: ggml_backend_buffer_t, value: ctypes.c_uint8):
    return lib.ggml_backend_buffer_clear(buffer, value)


lib.ggml_backend_buffer_clear.argtypes = [ggml_backend_buffer_t, ctypes.c_uint8]
lib.ggml_backend_buffer_clear.restype = None


# GGML_API bool                       ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_is_host(
    buffer: ggml_backend_buffer_t,
) -> bool:
    return lib.ggml_backend_buffer_is_host(buffer)


lib.ggml_backend_buffer_is_host.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_is_host.restype = ctypes.c_bool


# GGML_API void                       ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
def ggml_backend_buffer_set_usage(
    buffer: ggml_backend_buffer_t, usage: Union[ctypes.c_int, int]
):
    return lib.ggml_backend_buffer_set_usage(buffer, usage)


lib.ggml_backend_buffer_set_usage.argtypes = [ggml_backend_buffer_t, ctypes.c_int]
lib.ggml_backend_buffer_set_usage.restype = None


# GGML_API ggml_backend_buffer_type_t ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_get_type(
    buffer: ggml_backend_buffer_t,
) -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_buffer_get_type(buffer)


lib.ggml_backend_buffer_get_type.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_get_type.restype = ggml_backend_buffer_type_t


# GGML_API void                       ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);
def ggml_backend_buffer_reset(
    buffer: ggml_backend_buffer_t,
):
    return lib.ggml_backend_buffer_reset(buffer)


lib.ggml_backend_buffer_reset.argtypes = [ggml_backend_buffer_t]
lib.ggml_backend_buffer_reset.restype = None

# //
# // Backend
# //


# GGML_API const char * ggml_backend_name(ggml_backend_t backend);
def ggml_backend_name(
    backend: ggml_backend_t,
) -> bytes:
    return lib.ggml_backend_name(backend)


lib.ggml_backend_name.argtypes = [ggml_backend_t]
lib.ggml_backend_name.restype = ctypes.c_char_p


# GGML_API void         ggml_backend_free(ggml_backend_t backend);
def ggml_backend_free(
    backend: ggml_backend_t,
):
    return lib.ggml_backend_free(backend)


lib.ggml_backend_free.argtypes = [ggml_backend_t]
lib.ggml_backend_free.restype = None


# GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
def ggml_backend_get_default_buffer_type(
    backend: ggml_backend_t,
) -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_get_default_buffer_type(backend)


lib.ggml_backend_get_default_buffer_type.argtypes = [ggml_backend_t]
lib.ggml_backend_get_default_buffer_type.restype = ggml_backend_buffer_type_t


# GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
def ggml_backend_alloc_buffer(
    backend: ggml_backend_t,
    size: Union[ctypes.c_size_t, int],
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_alloc_buffer(backend, size)


lib.ggml_backend_alloc_buffer.argtypes = [ggml_backend_t, ctypes.c_size_t]
lib.ggml_backend_alloc_buffer.restype = ggml_backend_buffer_t


# GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
def ggml_backend_get_alignment(
    backend: ggml_backend_t,
) -> int:
    return lib.ggml_backend_get_alignment(backend)


lib.ggml_backend_get_alignment.argtypes = [ggml_backend_t]
lib.ggml_backend_get_alignment.restype = ctypes.c_size_t


# GGML_API void ggml_backend_tensor_set_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
def ggml_backend_tensor_set_async(
    backend: ggml_backend_t,
    tensor: ggml_tensor_p,
    data: ctypes.c_void_p,
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
):
    return lib.ggml_backend_tensor_set_async(backend, tensor, data, offset, size)


lib.ggml_backend_tensor_set_async.argtypes = [
    ggml_backend_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_backend_tensor_set_async.restype = None


# GGML_API void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
def ggml_backend_tensor_get_async(
    backend: ggml_backend_t,
    tensor: ggml_tensor_p,
    data: ctypes.c_void_p,
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
):
    return lib.ggml_backend_tensor_get_async(backend, tensor, data, offset, size)


lib.ggml_backend_tensor_get_async.argtypes = [
    ggml_backend_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_backend_tensor_get_async.restype = None


# GGML_API void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
def ggml_backend_tensor_set(
    tensor: ggml_tensor_p,
    data: ctypes.c_void_p,
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
):
    return lib.ggml_backend_tensor_set(tensor, data, offset, size)


lib.ggml_backend_tensor_set.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_backend_tensor_set.restype = None


# GGML_API void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
def ggml_backend_tensor_get(
    tensor: ggml_tensor_p,
    data: ctypes.c_void_p,
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
):
    return lib.ggml_backend_tensor_get(tensor, data, offset, size)


lib.ggml_backend_tensor_get.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.ggml_backend_tensor_get.restype = None


# GGML_API void ggml_backend_synchronize(ggml_backend_t backend);
def ggml_backend_synchronize(
    backend: ggml_backend_t,
):
    return lib.ggml_backend_synchronize(backend)


lib.ggml_backend_synchronize.argtypes = [ggml_backend_t]
lib.ggml_backend_synchronize.restype = None


# GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create (ggml_backend_t backend, struct ggml_cgraph * cgraph);
def ggml_backend_graph_plan_create(
    backend: ggml_backend_t,
    cgraph: ggml_cgraph_p,
) -> ggml_backend_graph_plan_t:
    return lib.ggml_backend_graph_plan_create(backend, cgraph)


lib.ggml_backend_graph_plan_create.argtypes = [
    ggml_backend_t,
    ctypes.POINTER(ggml_cgraph),
]
lib.ggml_backend_graph_plan_create.restype = ggml_backend_graph_plan_t


# GGML_API void ggml_backend_graph_plan_free   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
def ggml_backend_graph_plan_free(
    backend: ggml_backend_t,
    plan: ggml_backend_graph_plan_t,
):
    return lib.ggml_backend_graph_plan_free(backend, plan)


lib.ggml_backend_graph_plan_free.argtypes = [ggml_backend_t, ggml_backend_graph_plan_t]
lib.ggml_backend_graph_plan_free.restype = None


# GGML_API void ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
def ggml_backend_graph_plan_compute(
    backend: ggml_backend_t,
    plan: ggml_backend_graph_plan_t,
):
    return lib.ggml_backend_graph_plan_compute(backend, plan)


lib.ggml_backend_graph_plan_compute.argtypes = [
    ggml_backend_t,
    ggml_backend_graph_plan_t,
]
lib.ggml_backend_graph_plan_compute.restype = None


# GGML_API bool ggml_backend_graph_compute     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
def ggml_backend_graph_compute(
    backend: ggml_backend_t,
    cgraph: ggml_cgraph_p,
) -> bool:
    return lib.ggml_backend_graph_compute(backend, cgraph)


lib.ggml_backend_graph_compute.argtypes = [ggml_backend_t, ctypes.POINTER(ggml_cgraph)]
lib.ggml_backend_graph_compute.restype = ctypes.c_bool


# GGML_API bool ggml_backend_supports_op       (ggml_backend_t backend, const struct ggml_tensor * op);
def ggml_backend_supports_op(
    backend: ggml_backend_t,
    op: ggml_tensor_p,
) -> Union[ctypes.c_bool, bool]:
    return lib.ggml_backend_supports_op(backend, op)


lib.ggml_backend_supports_op.argtypes = [ggml_backend_t, ctypes.POINTER(ggml_tensor)]
lib.ggml_backend_supports_op.restype = ctypes.c_bool


# // tensor copy between different backends
# GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);
def ggml_backend_tensor_copy(
    src: ggml_tensor_p,
    dst: ggml_tensor_p,
):
    return lib.ggml_backend_tensor_copy(src, dst)


lib.ggml_backend_tensor_copy.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_tensor_copy.restype = None


# GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst); // automatic fallback to sync copy
def ggml_backend_tensor_copy_async(
    backend: ggml_backend_t,
    src: ggml_tensor_p,
    dst: ggml_tensor_p,
):
    return lib.ggml_backend_tensor_copy_async(backend, src, dst)


# lib.ggml_backend_tensor_copy_async.argtypes = [
#     ggml_backend_t,
#     ctypes.POINTER(ggml_tensor),
#     ctypes.POINTER(ggml_tensor),
# ]
# lib.ggml_backend_tensor_copy_async.restype = None

# //
# // CPU backend
# //


# GGML_API ggml_backend_t ggml_backend_cpu_init(void);
def ggml_backend_cpu_init() -> ggml_backend_t:
    return lib.ggml_backend_cpu_init()


lib.ggml_backend_cpu_init.argtypes = []
lib.ggml_backend_cpu_init.restype = ggml_backend_t


# GGML_API bool ggml_backend_is_cpu(ggml_backend_t backend);
def ggml_backend_is_cpu(
    backend: ggml_backend_t,
) -> bool:
    return lib.ggml_backend_is_cpu(backend)


lib.ggml_backend_is_cpu.argtypes = [ggml_backend_t]
lib.ggml_backend_is_cpu.restype = ctypes.c_bool


# GGML_API void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);
def ggml_backend_cpu_set_n_threads(
    backend_cpu: ggml_backend_t,
    n_threads: Union[ctypes.c_int, int],
):
    return lib.ggml_backend_cpu_set_n_threads(backend_cpu, n_threads)


lib.ggml_backend_cpu_set_n_threads.argtypes = [ggml_backend_t, ctypes.c_int]
lib.ggml_backend_cpu_set_n_threads.restype = None


# // Create a backend buffer from an existing pointer
# GGML_API ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
def ggml_backend_cpu_buffer_from_ptr(
    ptr: ctypes.c_void_p,
    size: Union[ctypes.c_size_t, int],
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_cpu_buffer_from_ptr(ptr, size)


lib.ggml_backend_cpu_buffer_from_ptr.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.ggml_backend_cpu_buffer_from_ptr.restype = ggml_backend_buffer_t


# GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);
def ggml_backend_cpu_buffer_type() -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_cpu_buffer_type()


lib.ggml_backend_cpu_buffer_type.argtypes = []
lib.ggml_backend_cpu_buffer_type.restype = ggml_backend_buffer_type_t


# #ifdef GGML_USE_CPU_HBM
#     GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void);
# #endif
def ggml_backend_cpu_hbm_buffer_type() -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_cpu_hbm_buffer_type()


if hasattr(lib, "ggml_backend_cpu_hbm_buffer_type"):
    lib.ggml_backend_cpu_hbm_buffer_type.argtypes = []
    lib.ggml_backend_cpu_hbm_buffer_type.restype = ggml_backend_buffer_type_t

# //
# // Backend registry
# //

# // The backend registry is a registry of all the available backends, and allows initializing backends in a generic way


# GGML_API size_t                     ggml_backend_reg_get_count(void);
def ggml_backend_reg_get_count() -> int:
    return lib.ggml_backend_reg_get_count()


lib.ggml_backend_reg_get_count.argtypes = []
lib.ggml_backend_reg_get_count.restype = ctypes.c_size_t


# GGML_API size_t                     ggml_backend_reg_find_by_name(const char * name);
def ggml_backend_reg_find_by_name(
    name: bytes,
) -> int:
    return lib.ggml_backend_reg_find_by_name(name)


lib.ggml_backend_reg_find_by_name.argtypes = [ctypes.c_char_p]
lib.ggml_backend_reg_find_by_name.restype = ctypes.c_size_t


# GGML_API ggml_backend_t             ggml_backend_reg_init_backend_from_str(const char * backend_str); // str is name[:params]
def ggml_backend_reg_init_backend_from_str(
    backend_str: bytes,
) -> ggml_backend_t:
    return lib.ggml_backend_reg_init_backend_from_str(backend_str)


lib.ggml_backend_reg_init_backend_from_str.argtypes = [ctypes.c_char_p]
lib.ggml_backend_reg_init_backend_from_str.restype = ggml_backend_t


# GGML_API const char *               ggml_backend_reg_get_name(size_t i);
def ggml_backend_reg_get_name(
    i: Union[ctypes.c_size_t, int],
) -> bytes:
    return lib.ggml_backend_reg_get_name(i)


lib.ggml_backend_reg_get_name.argtypes = [ctypes.c_size_t]
lib.ggml_backend_reg_get_name.restype = ctypes.c_char_p


# GGML_API ggml_backend_t             ggml_backend_reg_init_backend(size_t i, const char * params); // params is backend-specific
def ggml_backend_reg_init_backend(
    i: Union[ctypes.c_size_t, int],
    params: bytes,
) -> ggml_backend_t:
    return lib.ggml_backend_reg_init_backend(i, params)


lib.ggml_backend_reg_init_backend.argtypes = [ctypes.c_size_t, ctypes.c_char_p]
lib.ggml_backend_reg_init_backend.restype = ggml_backend_t


# GGML_API ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i);
def ggml_backend_reg_get_default_buffer_type(
    i: Union[ctypes.c_size_t, int],
) -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_reg_get_default_buffer_type(i)


lib.ggml_backend_reg_get_default_buffer_type.argtypes = [ctypes.c_size_t]
lib.ggml_backend_reg_get_default_buffer_type.restype = ggml_backend_buffer_type_t


# GGML_API ggml_backend_buffer_t      ggml_backend_reg_alloc_buffer(size_t i, size_t size);
def ggml_backend_reg_alloc_buffer(
    i: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_reg_alloc_buffer(i, size)


lib.ggml_backend_reg_alloc_buffer.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_backend_reg_alloc_buffer.restype = ggml_backend_buffer_t

# //
# // Backend scheduler
# //

# // The backend scheduler allows for multiple backends to be used together
# // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
# // The backends are selected based on:
# // - the backend that supports the operation
# // - the location of the pre-allocated tensors (e.g. the weights)
# /*
#   Example usage:

#     sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, num_backends);
#     // sched is initialized with measure allocators and cannot be used until allocated with a measure graph

#     // initialize buffers from a measure graph
#     measure_graph = build_graph(sched); // use the allocr to allocate inputs as needed

#     // in build_graph:
#     build_graph(...) {
#         // allocating tensors in a specific backend (optional, recommended: pre-allocate inputs in a different buffer)
#         alloc_cpu = ggml_backend_sched_get_allocr(sched, backend_cpu);
#         ggml_allocr_alloc(alloc_cpu, tensor);

#         // manually assigning nodes to a backend (optional, shouldn't be needed in most cases)
#         struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
#         ggml_backend_sched_set_node_backend(sched, node, backend_gpu);
#     }

#     // allocate backend buffers from measure graph
#     ggml_backend_sched_init_measure(sched, measure_graph);

#     // the scheduler is now ready to compute graphs

#     // compute
#     graph = build_graph(sched);
#     ggml_backend_sched_graph_compute(sched, graph);
# */

# struct ggml_backend_sched;
# typedef struct ggml_backend_sched * ggml_backend_sched_t;
ggml_backend_sched_t = ctypes.c_void_p


# // Initialize a backend scheduler
# GGML_API ggml_backend_sched_t  ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size);
def ggml_backend_sched_new(
    backends: "ctypes._Pointer(ggml_backend_t)",  # type: ignore
    bufts: "ctypes._Pointer(ggml_backend_buffer_type_t)",  # type: ignore
    n_backends: Union[ctypes.c_int, int],
    graph_size: Union[ctypes.c_size_t, int],
) -> ggml_backend_sched_t:
    return lib.ggml_backend_sched_new(backends, bufts, n_backends, graph_size)


lib.ggml_backend_sched_new.argtypes = [
    ctypes.POINTER(ggml_backend_t),
    ctypes.POINTER(ggml_backend_buffer_type_t),
    ctypes.c_int,
    ctypes.c_size_t,
]
lib.ggml_backend_sched_new.restype = ggml_backend_sched_t


# GGML_API void                  ggml_backend_sched_free(ggml_backend_sched_t sched);
def ggml_backend_sched_free(
    sched: ggml_backend_sched_t,
):
    return lib.ggml_backend_sched_free(sched)


lib.ggml_backend_sched_free.argtypes = [ggml_backend_sched_t]
lib.ggml_backend_sched_free.restype = None


# // Initialize backend buffers from a measure graph
# GGML_API void                  ggml_backend_sched_init_measure(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
def ggml_backend_sched_init_measure(
    sched: ggml_backend_sched_t,
    measure_graph: ggml_cgraph_p,
):
    return lib.ggml_backend_sched_init_measure(sched, measure_graph)


lib.ggml_backend_sched_init_measure.argtypes = [
    ggml_backend_sched_t,
    ctypes.POINTER(ggml_cgraph),
]
lib.ggml_backend_sched_init_measure.restype = None


# // Get the number of splits of the last graph
# GGML_API int                   ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
def ggml_backend_sched_get_n_splits(
    sched: ggml_backend_sched_t,
) -> int:
    return lib.ggml_backend_sched_get_n_splits(sched)


lib.ggml_backend_sched_get_n_splits.argtypes = [ggml_backend_sched_t]
lib.ggml_backend_sched_get_n_splits.restype = ctypes.c_int


# GGML_API ggml_tallocr_t        ggml_backend_sched_get_tallocr(ggml_backend_sched_t sched, ggml_backend_t backend);
def ggml_backend_sched_get_tallocr(
    sched: ggml_backend_sched_t,
    backend: ggml_backend_t,
) -> ggml_tallocr_t:
    return lib.ggml_backend_sched_get_tallocr(sched, backend)


lib.ggml_backend_sched_get_tallocr.argtypes = [ggml_backend_sched_t, ggml_backend_t]
lib.ggml_backend_sched_get_tallocr.restype = ggml_tallocr_t


# GGML_API ggml_backend_buffer_t ggml_backend_sched_get_buffer (ggml_backend_sched_t sched, ggml_backend_t backend);
def ggml_backend_sched_get_buffer(
    sched: ggml_backend_sched_t,
    backend: ggml_backend_t,
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_sched_get_buffer(sched, backend)


lib.ggml_backend_sched_get_buffer.argtypes = [ggml_backend_sched_t, ggml_backend_t]
lib.ggml_backend_sched_get_buffer.restype = ggml_backend_buffer_t


# GGML_API void                  ggml_backend_sched_set_node_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
def ggml_backend_sched_set_node_backend(
    sched: ggml_backend_sched_t,
    node: ggml_tensor_p,
    backend: ggml_backend_t,
):
    return lib.ggml_backend_sched_set_node_backend(sched, node, backend)


lib.ggml_backend_sched_set_node_backend.argtypes = [
    ggml_backend_sched_t,
    ctypes.POINTER(ggml_tensor),
    ggml_backend_t,
]
lib.ggml_backend_sched_set_node_backend.restype = None


# GGML_API ggml_backend_t        ggml_backend_sched_get_node_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);
def ggml_backend_sched_get_node_backend(
    sched: ggml_backend_sched_t,
    node: ggml_tensor_p,
) -> ggml_backend_t:
    return lib.ggml_backend_sched_get_node_backend(sched, node)


lib.ggml_backend_sched_get_node_backend.argtypes = [
    ggml_backend_sched_t,
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_sched_get_node_backend.restype = ggml_backend_t


# // Allocate and compute graph on the backend scheduler
# GGML_API void                  ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
def ggml_backend_sched_graph_compute(
    sched: ggml_backend_sched_t,
    graph: ggml_cgraph_p,
):
    return lib.ggml_backend_sched_graph_compute(sched, graph)


lib.ggml_backend_sched_graph_compute.argtypes = [
    ggml_backend_sched_t,
    ctypes.POINTER(ggml_cgraph),
]
lib.ggml_backend_sched_graph_compute.restype = None


# // Reset all assignments and allocators - must be called before using the sched allocators to allocate inputs
# GGML_API void                  ggml_backend_sched_reset(ggml_backend_sched_t sched);
def ggml_backend_sched_reset(
    sched: ggml_backend_sched_t,
):
    return lib.ggml_backend_sched_reset(sched)


lib.ggml_backend_sched_reset.argtypes = [ggml_backend_sched_t]
lib.ggml_backend_sched_reset.restype = None

# //
# // Utils
# //


# struct ggml_backend_graph_copy {
#     ggml_backend_buffer_t buffer;
#     struct ggml_context * ctx_allocated;
#     struct ggml_context * ctx_unallocated;
#     struct ggml_cgraph * graph;
# };
class ggml_backend_graph_copy(ctypes.Structure):
    _fields_ = [
        ("buffer", ggml_backend_buffer_t),
        ("ctx_allocated", ggml_context_p),
        ("ctx_unallocated", ggml_context_p),
        ("graph", ctypes.POINTER(ggml_cgraph)),
    ]


ggml_backend_graph_copy_t = ggml_backend_graph_copy


# // Copy a graph to a different backend
# GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
def ggml_backend_graph_copy(
    backend: ggml_backend_t,
    graph: ggml_cgraph_p,
) -> ggml_backend_graph_copy_t:
    return lib.ggml_backend_graph_copy(backend, graph)


lib.ggml_backend_graph_copy.argtypes = [
    ggml_backend_t,
    ctypes.POINTER(ggml_cgraph),
]
lib.ggml_backend_graph_copy.restype = ggml_backend_graph_copy_t


# GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);
def ggml_backend_graph_copy_free(
    copy: ggml_backend_graph_copy_t,
):
    return lib.ggml_backend_graph_copy_free(copy)


lib.ggml_backend_graph_copy_free.argtypes = [ggml_backend_graph_copy_t]
lib.ggml_backend_graph_copy_free.restype = None

# typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);
ggml_backend_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
)


# // Compare the output of two backends
# GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data);
def ggml_backend_compare_graph_backend(
    backend1: ggml_backend_t,
    backend2: ggml_backend_t,
    graph: ggml_cgraph_p,
    callback,
    user_data: ctypes.c_void_p,
) -> bool:
    return lib.ggml_backend_compare_graph_backend(
        backend1, backend2, graph, callback, user_data
    )


lib.ggml_backend_compare_graph_backend.argtypes = [
    ggml_backend_t,
    ggml_backend_t,
    ctypes.POINTER(ggml_cgraph),
    ggml_backend_eval_callback,
    ctypes.c_void_p,
]
lib.ggml_backend_compare_graph_backend.restype = ctypes.c_bool


# // Tensor initialization
# GGML_API void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
def ggml_backend_tensor_alloc(
    buffer: ggml_backend_buffer_t,
    tensor: ggml_tensor_p,
    addr: ctypes.c_void_p,
):
    return lib.ggml_backend_tensor_alloc(buffer, tensor, addr)


lib.ggml_backend_tensor_alloc.argtypes = [
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
]
lib.ggml_backend_tensor_alloc.restype = None


# GGML_API void ggml_backend_view_init(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
def ggml_backend_view_init(
    buffer: ggml_backend_buffer_t,
    tensor: ggml_tensor_p,
):
    return lib.ggml_backend_view_init(buffer, tensor)


lib.ggml_backend_view_init.argtypes = [
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_view_init.restype = None


#####################################################
# GGML Backend Implementation API
# source: ggml-backend-impl.h
#####################################################

# //
# // Backend buffer
# //

# // buffer type
# typedef void * ggml_backend_buffer_type_context_t;
ggml_backend_buffer_type_context_t = ctypes.c_void_p

# struct ggml_backend_buffer_type_i {
#     const char *          (*get_name)        (ggml_backend_buffer_type_t buft);
#     ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
#     size_t                (*get_alignment)   (ggml_backend_buffer_type_t buft); // tensor alignment
#     size_t                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
#     bool                  (*supports_backend)(ggml_backend_buffer_type_t buft, ggml_backend_t backend); // check if the buffer type is usable by the backend
#     // check if tensor data is in host memory
#     // should be equivalent to supports_backend(buft, ggml_backend_cpu_init())
#     bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
# };
ggml_backend_buffer_type_i_get_name = ctypes.CFUNCTYPE(
    ctypes.c_char_p, ggml_backend_buffer_type_t
)
ggml_backend_buffer_i_alloc_buffer = ctypes.CFUNCTYPE(
    ggml_backend_buffer_t, ggml_backend_buffer_type_t, ctypes.c_size_t
)
ggml_backend_buffer_i_get_alignment = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ggml_backend_buffer_type_t
)
ggml_backend_buffer_i_get_alloc_size = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ggml_backend_buffer_type_t, ctypes.POINTER(ggml_tensor)
)
ggml_backend_buffer_i_supports_backend = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_buffer_type_t, ggml_backend_t
)
ggml_backend_buffer_i_is_host = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_buffer_type_t
)


class ggml_backend_buffer_type_i(ctypes.Structure):
    _fields_ = [
        ("get_name", ggml_backend_buffer_type_i_get_name),
        ("alloc_buffer", ggml_backend_buffer_i_alloc_buffer),
        ("get_alignment", ggml_backend_buffer_i_get_alignment),
        ("get_alloc_size", ggml_backend_buffer_i_get_alloc_size),
        ("supports_backend", ggml_backend_buffer_i_supports_backend),
        ("is_host", ggml_backend_buffer_i_is_host),
    ]


# struct ggml_backend_buffer_type {
#     struct ggml_backend_buffer_type_i  iface;
#     ggml_backend_buffer_type_context_t context;
# };
class ggml_backend_buffer_type(ctypes.Structure):
    _fields_ = [
        ("iface", ggml_backend_buffer_type_i),
        ("context", ggml_backend_buffer_type_context_t),
    ]


# typedef void * ggml_backend_buffer_context_t;
ggml_backend_buffer_context_t = ctypes.c_void_p


# struct ggml_backend_buffer_i {
#     const char * (*get_name)   (ggml_backend_buffer_t buffer);
#     void         (*free_buffer)(ggml_backend_buffer_t buffer);
#     void *       (*get_base)   (ggml_backend_buffer_t buffer);
#     void         (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
#     void         (*set_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
#     void         (*get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
#     bool         (*cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
#     void         (*clear)      (ggml_backend_buffer_t buffer, uint8_t value);
#     void         (*reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
# };
ggml_backend_buffer_i_get_name = ctypes.CFUNCTYPE(
    ctypes.c_char_p, ggml_backend_buffer_t
)
ggml_backend_buffer_i_free_buffer = ctypes.CFUNCTYPE(None, ggml_backend_buffer_t)
ggml_backend_buffer_i_get_base = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ggml_backend_buffer_t
)
ggml_backend_buffer_i_init_tensor = ctypes.CFUNCTYPE(
    None, ggml_backend_buffer_t, ctypes.POINTER(ggml_tensor)
)
ggml_backend_buffer_i_set_tensor = ctypes.CFUNCTYPE(
    None,
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_buffer_i_get_tensor = ctypes.CFUNCTYPE(
    None,
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_buffer_i_cpy_tensor = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ggml_backend_buffer_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
ggml_backend_buffer_i_clear = ctypes.CFUNCTYPE(
    None, ggml_backend_buffer_t, ctypes.c_uint8
)
ggml_backend_buffer_i_reset = ctypes.CFUNCTYPE(None, ggml_backend_buffer_t)


class ggml_backend_buffer_i(ctypes.Structure):
    _fields_ = [
        ("get_name", ggml_backend_buffer_i_get_name),
        ("free_buffer", ggml_backend_buffer_i_free_buffer),
        ("get_base", ggml_backend_buffer_i_get_base),
        ("init_tensor", ggml_backend_buffer_i_init_tensor),
        ("set_tensor", ggml_backend_buffer_i_set_tensor),
        ("get_tensor", ggml_backend_buffer_i_get_tensor),
        ("cpy_tensor", ggml_backend_buffer_i_cpy_tensor),
        ("clear", ggml_backend_buffer_i_clear),
        ("reset", ggml_backend_buffer_i_reset),
    ]


# struct ggml_backend_buffer {
#     struct ggml_backend_buffer_i  iface;
#     ggml_backend_buffer_type_t    buft;
#     ggml_backend_buffer_context_t context;
#     size_t size;
#     enum ggml_backend_buffer_usage usage;
# };
class ggml_backend_buffer(ctypes.Structure):
    _fields_ = [
        ("iface", ggml_backend_buffer_i),
        ("buft", ggml_backend_buffer_type_t),
        ("context", ggml_backend_buffer_context_t),
        ("size", ctypes.c_size_t),
    ]


# ggml_backend_buffer_t ggml_backend_buffer_init(
#                ggml_backend_buffer_type_t      buft,
#         struct ggml_backend_buffer_i           iface,
#                ggml_backend_buffer_context_t   context,
#                size_t                          size);
def ggml_backend_buffer_init(
    buft: ggml_backend_buffer_type_t,
    iface: ggml_backend_buffer_i,
    context: ggml_backend_buffer_context_t,
    size: Union[ctypes.c_size_t, int],
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_buffer_init(buft, iface, context, size)


lib.ggml_backend_buffer_init.argtypes = [
    ggml_backend_buffer_type_t,
    ggml_backend_buffer_i,
    ggml_backend_buffer_context_t,
    ctypes.c_size_t,
]
lib.ggml_backend_buffer_init.restype = ggml_backend_buffer_t


# // do not use directly, use ggml_backend_tensor_copy instead
# bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);
def ggml_backend_buffer_copy_tensor(
    src: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> bool:
    return lib.ggml_backend_buffer_copy_tensor(src, dst)


lib.ggml_backend_buffer_copy_tensor.argtypes = [
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
]
lib.ggml_backend_buffer_copy_tensor.restype = ctypes.c_bool

# //
# // Backend
# //

# typedef void * ggml_backend_context_t;
ggml_backend_context_t = ctypes.c_void_p


# struct ggml_backend_i {
#     const char * (*get_name)(ggml_backend_t backend);

#     void (*free)(ggml_backend_t backend);

#     // buffer allocation
#     ggml_backend_buffer_type_t (*get_default_buffer_type)(ggml_backend_t backend);

#     // (optional) asynchronous tensor data access
#     void (*set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
#     void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
#     bool (*cpy_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst);

#     // (optional) complete all pending operations
#     void (*synchronize)(ggml_backend_t backend);

#     // compute graph with a plan
#     ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
#     void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
#     void                      (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

#     // compute graph without a plan (async)
#     bool (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);

#     // check if the backend supports an operation
#     bool (*supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);
# };
ggml_backend_i_get_name = ctypes.CFUNCTYPE(ctypes.c_char_p, ggml_backend_t)
ggml_backend_i_free = ctypes.CFUNCTYPE(None, ggml_backend_t)
ggml_backend_i_get_default_buffer_type = ctypes.CFUNCTYPE(
    ggml_backend_buffer_type_t, ggml_backend_t
)
ggml_backend_i_set_tensor_async = ctypes.CFUNCTYPE(
    None,
    ggml_backend_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_i_get_tensor_async = ctypes.CFUNCTYPE(
    None,
    ggml_backend_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_i_cpy_tensor_async = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ggml_backend_t,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
ggml_backend_i_synchronize = ctypes.CFUNCTYPE(None, ggml_backend_t)
ggml_backend_i_graph_plan_create = ctypes.CFUNCTYPE(
    ggml_backend_graph_plan_t, ggml_backend_t, ctypes.POINTER(ggml_cgraph)
)
ggml_backend_i_graph_plan_free = ctypes.CFUNCTYPE(
    None, ggml_backend_t, ggml_backend_graph_plan_t
)
ggml_backend_i_graph_plan_compute = ctypes.CFUNCTYPE(
    None, ggml_backend_t, ggml_backend_graph_plan_t
)
ggml_backend_i_graph_compute = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_t, ctypes.POINTER(ggml_cgraph)
)
ggml_backend_i_supports_op = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_t, ctypes.POINTER(ggml_tensor)
)


class ggml_backend_i(ctypes.Structure):
    _fields_ = [
        ("get_name", ggml_backend_i_get_name),
        ("free", ggml_backend_i_free),
        ("get_default_buffer_type", ggml_backend_i_get_default_buffer_type),
        ("set_tensor_async", ggml_backend_i_set_tensor_async),
        ("get_tensor_async", ggml_backend_i_get_tensor_async),
        ("cpy_tensor_async", ggml_backend_i_cpy_tensor_async),
        ("synchronize", ggml_backend_i_synchronize),
        ("graph_plan_create", ggml_backend_i_graph_plan_create),
        ("graph_plan_free", ggml_backend_i_graph_plan_free),
        ("graph_plan_compute", ggml_backend_i_graph_plan_compute),
        ("graph_compute", ggml_backend_i_graph_compute),
        ("supports_op", ggml_backend_i_supports_op),
    ]


# struct ggml_backend {
#     struct ggml_backend_i iface;


#     ggml_backend_context_t context;
# };
class ggml_backend(ctypes.Structure):
    _fields_ = [
        ("iface", ggml_backend_i),
        ("context", ggml_backend_context_t),
    ]


# //
# // Backend registry
# //

# typedef ggml_backend_t (*ggml_backend_init_fn)(const char * params, void * user_data);
ggml_backend_init_fn = ctypes.CFUNCTYPE(
    ggml_backend_t, ctypes.c_char_p, ctypes.c_void_p
)


# void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data);
def ggml_backend_register(
    name: bytes,
    init_fn,
    default_buffer_type: ggml_backend_buffer_type_t,
    user_data: ctypes.c_void_p,
):
    return lib.ggml_backend_register(name, init_fn, default_buffer_type, user_data)


lib.ggml_backend_register.argtypes = [
    ctypes.c_char_p,
    ggml_backend_init_fn,
    ggml_backend_buffer_type_t,
    ctypes.c_void_p,
]
lib.ggml_backend_register.restype = None

#####################################################
# GGML CUDA API
# source: ggml-cuda.h
#####################################################


GGML_USE_CUBLAS = hasattr(lib, "ggml_init_cublas")


GGML_CUDA_MAX_DEVICES = 16


# // Always success. To check if CUDA is actually loaded, use `ggml_cublas_loaded`.
# GGML_API void   ggml_init_cublas(void);
def ggml_init_cublas():
    return lib.ggml_init_cublas()


if GGML_USE_CUBLAS:
    lib.ggml_init_cublas.argtypes = []
    lib.ggml_init_cublas.restype = None


# // Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
# GGML_API bool   ggml_cublas_loaded(void);
def ggml_cublas_loaded() -> bool:
    return lib.ggml_cublas_loaded()


if GGML_USE_CUBLAS:
    lib.ggml_cublas_loaded.argtypes = []
    lib.ggml_cublas_loaded.restype = ctypes.c_bool


# void * ggml_cuda_host_malloc(size_t size);
def ggml_cuda_host_malloc(
    size: Union[ctypes.c_size_t, int],
) -> Optional[ctypes.c_void_p]:
    return lib.ggml_cuda_host_malloc(size)


if GGML_USE_CUBLAS:
    lib.ggml_cuda_host_malloc.argtypes = [ctypes.c_size_t]
    lib.ggml_cuda_host_malloc.restype = ctypes.c_void_p


# void   ggml_cuda_host_free(void * ptr);
def ggml_cuda_host_free(
    ptr: ctypes.c_void_p,
):
    return lib.ggml_cuda_host_free(ptr)


if GGML_USE_CUBLAS:
    lib.ggml_cuda_host_free.argtypes = [ctypes.c_void_p]
    lib.ggml_cuda_host_free.restype = None


# GGML_API bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
def ggml_cuda_can_mul_mat(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> bool:
    return lib.ggml_cuda_can_mul_mat(src0, src1, dst)


if GGML_USE_CUBLAS:
    lib.ggml_cuda_can_mul_mat.argtypes = [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cuda_can_mul_mat.restype = ctypes.c_bool


# GGML_API bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);
def ggml_cuda_compute_forward(
    params: ggml_compute_params_p,
    tensor: ggml_tensor_p,
) -> bool:
    return lib.ggml_cuda_compute_forward(params, tensor)


if GGML_USE_CUBLAS:
    lib.ggml_cuda_compute_forward.argtypes = [
        ctypes.POINTER(ggml_compute_params),
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cuda_compute_forward.restype = ctypes.c_bool


# GGML_API int    ggml_cuda_get_device_count(void);
def ggml_cuda_get_device_count() -> int:
    return lib.ggml_cuda_get_device_count()


if GGML_USE_CUBLAS:
    lib.ggml_cuda_get_device_count.argtypes = []
    lib.ggml_cuda_get_device_count.restype = ctypes.c_int


# GGML_API void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);
def ggml_cuda_get_device_description(
    device: Union[ctypes.c_int, int],
    description: bytes,
    description_size: Union[ctypes.c_size_t, int],
):
    return lib.ggml_cuda_get_device_description(device, description, description_size)


if GGML_USE_CUBLAS:
    lib.ggml_cuda_get_device_description.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ggml_cuda_get_device_description.restype = None


# // backend API
# GGML_API ggml_backend_t ggml_backend_cuda_init(void); // TODO: take a list of devices to use
def ggml_backend_cuda_init() -> ggml_backend_t:
    return lib.ggml_backend_cuda_init()


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_init.argtypes = []
    lib.ggml_backend_cuda_init.restype = ggml_backend_t


# GGML_API bool ggml_backend_is_cuda(ggml_backend_t backend);
def ggml_backend_is_cuda(
    backend: ggml_backend_t,
) -> bool:
    return lib.ggml_backend_is_cuda(backend)


if GGML_USE_CUBLAS:
    lib.ggml_backend_is_cuda.argtypes = [ggml_backend_t]
    lib.ggml_backend_is_cuda.restype = ctypes.c_bool


# GGML_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);
def ggml_backend_cuda_buffer_type(
    device: Union[ctypes.c_int, int],
) -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_cuda_buffer_type(device)


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_buffer_type.argtypes = [ctypes.c_int]
    lib.ggml_backend_cuda_buffer_type.restype = ggml_backend_buffer_type_t


# // split tensor buffer that splits matrices by rows across multiple devices
# GGML_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);
def ggml_backend_cuda_split_buffer_type(
    tensor_split: CFloatArray,
) -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_cuda_split_buffer_type(tensor_split)


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_split_buffer_type.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ggml_backend_cuda_split_buffer_type.restype = ggml_backend_buffer_type_t


# // pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
# GGML_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);
def ggml_backend_cuda_host_buffer_type() -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_cuda_host_buffer_type()


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_host_buffer_type.argtypes = []
    lib.ggml_backend_cuda_host_buffer_type.restype = ggml_backend_buffer_type_t


# GGML_API int  ggml_backend_cuda_get_device_count(void);
def ggml_backend_cuda_get_device_count() -> int:
    return lib.ggml_backend_cuda_get_device_count()


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_get_device_count.argtypes = []
    lib.ggml_backend_cuda_get_device_count.restype = ctypes.c_int


# GGML_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
def ggml_backend_cuda_get_device_description(
    device: Union[ctypes.c_int, int],
    description: ctypes.c_char_p,
    description_size: Union[ctypes.c_size_t, int],
):
    return lib.ggml_backend_cuda_get_device_description(
        device, description, description_size
    )


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_get_device_description.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ggml_backend_cuda_get_device_description.restype = None


# GGML_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
def ggml_backend_cuda_get_device_memory(
    device: Union[ctypes.c_int, int],
    free: "ctypes._Pointer[ctypes.c_size_t]",  # type: ignore
    total: "ctypes._Pointer[ctypes.c_size_t]",  # type: ignore
):
    return lib.ggml_backend_cuda_get_device_memory(device, free, total)


if GGML_USE_CUBLAS:
    lib.ggml_backend_cuda_get_device_memory.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ggml_backend_cuda_get_device_memory.restype = None

#####################################################
# GGML METAL API
# source: ggml-metal.h
#####################################################


GGML_USE_METAL = hasattr(lib, "ggml_metal_init")


# // max memory buffers that can be mapped to the device
# #define GGML_METAL_MAX_BUFFERS 64
GGML_METAL_MAX_BUFFERS = 64
# #define GGML_METAL_MAX_COMMAND_BUFFERS 32
GGML_METAL_MAX_COMMAND_BUFFERS = 32

# //
# // backend API
# // user-code should use only these functions
# //


# GGML_API void ggml_backend_metal_log_set_callback(ggml_log_callback log_callback, void * user_data);
def ggml_backend_metal_log_set_callback(
    log_callback,
    user_data: ctypes.c_void_p,
):
    return lib.ggml_backend_metal_log_set_callback(log_callback, user_data)


if GGML_USE_METAL:
    lib.ggml_backend_metal_log_set_callback.argtypes = [
        ggml_log_callback,
        ctypes.c_void_p,
    ]
    lib.ggml_backend_metal_log_set_callback.restype = None


# GGML_API ggml_backend_t ggml_backend_metal_init(void);
def ggml_backend_metal_init() -> ggml_backend_t:
    return lib.ggml_backend_metal_init()


if GGML_USE_METAL:
    lib.ggml_backend_metal_init.argtypes = []
    lib.ggml_backend_metal_init.restype = ggml_backend_t


# GGML_API bool ggml_backend_is_metal(ggml_backend_t backend);
def ggml_backend_is_metal(
    backend: ggml_backend_t,
) -> bool:
    return lib.ggml_backend_is_metal(backend)


if GGML_USE_METAL:
    lib.ggml_backend_is_metal.argtypes = [ggml_backend_t]
    lib.ggml_backend_is_metal.restype = ctypes.c_bool


# GGML_API ggml_backend_buffer_t ggml_backend_metal_buffer_from_ptr(void * data, size_t size, size_t max_size);
def ggml_backend_metal_buffer_from_ptr(
    data: ctypes.c_void_p,
    size: Union[ctypes.c_size_t, int],
    max_size: Union[ctypes.c_size_t, int],
) -> ggml_backend_buffer_t:
    return lib.ggml_backend_metal_buffer_from_ptr(data, size, max_size)


if GGML_USE_METAL:
    lib.ggml_backend_metal_buffer_from_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.ggml_backend_metal_buffer_from_ptr.restype = ggml_backend_buffer_t


# GGML_API void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb);
def ggml_backend_metal_set_n_cb(
    backend: ggml_backend_t,
    n_cb: Union[ctypes.c_int, int],
):
    return lib.ggml_backend_metal_set_n_cb(backend, n_cb)


if GGML_USE_METAL:
    lib.ggml_backend_metal_set_n_cb.argtypes = [ggml_backend_t, ctypes.c_int]
    lib.ggml_backend_metal_set_n_cb.restype = None


# GGML_API ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void);
def ggml_backend_metal_buffer_type() -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_metal_buffer_type()


if GGML_USE_METAL:
    lib.ggml_backend_metal_buffer_type.argtypes = []
    lib.ggml_backend_metal_buffer_type.restype = ggml_backend_buffer_type_t


# // helper to check if the device supports a specific family
# // ideally, the user code should be doing these checks
# // ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
# GGML_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);
def ggml_backend_metal_supports_family(
    backend: ggml_backend_t,
    family: Union[ctypes.c_int, int],
) -> bool:
    return lib.ggml_backend_metal_supports_family(backend, family)


if GGML_USE_METAL:
    lib.ggml_backend_metal_supports_family.argtypes = [ggml_backend_t, ctypes.c_int]
    lib.ggml_backend_metal_supports_family.restype = ctypes.c_bool


#####################################################
# GGML OPENCL API
# source: ggml-opencl.h
#####################################################


GGML_USE_CLBLAST = hasattr(lib, "ggml_cl_init")


# GGML_API void ggml_cl_init(void);
def ggml_cl_init():
    return lib.ggml_cl_init()


if GGML_USE_CLBLAST:
    lib.ggml_cl_init.argtypes = []
    lib.ggml_cl_init.restype = None


# GGML_API void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
def ggml_cl_mul(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
):
    return lib.ggml_cl_mul(src0, src1, dst)


if GGML_USE_CLBLAST:
    lib.ggml_cl_mul.argtypes = [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cl_mul.restype = None


# GGML_API bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
def ggml_cl_can_mul_mat(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> bool:
    return lib.ggml_cl_can_mul_mat(src0, src1, dst)


if GGML_USE_CLBLAST:
    lib.ggml_cl_can_mul_mat.argtypes = [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cl_can_mul_mat.restype = ctypes.c_bool


# GGML_API size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
def ggml_cl_mul_mat_get_wsize(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> int:
    return lib.ggml_cl_mul_mat_get_wsize(src0, src1, dst)


if GGML_USE_CLBLAST:
    lib.ggml_cl_mul_mat_get_wsize.argtypes = [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cl_mul_mat_get_wsize.restype = ctypes.c_size_t


# GGML_API void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);
def ggml_cl_mul_mat(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
    wdata: ctypes.c_void_p,
    wsize: Union[ctypes.c_size_t, int],
):
    return lib.ggml_cl_mul_mat(src0, src1, dst, wdata, wsize)


if GGML_USE_CLBLAST:
    lib.ggml_cl_mul_mat.argtypes = [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.ggml_cl_mul_mat.restype = None


# GGML_API void ggml_cl_free_data(const struct ggml_tensor* tensor);
def ggml_cl_free_data(
    tensor: ggml_tensor_p,
):
    return lib.ggml_cl_free_data(tensor)


if GGML_USE_CLBLAST:
    lib.ggml_cl_free_data.argtypes = [
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cl_free_data.restype = None


# GGML_API void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);
def ggml_cl_transform_tensor(
    data: ctypes.c_void_p,
    tensor: ggml_tensor_p,
):
    return lib.ggml_cl_transform_tensor(data, tensor)


if GGML_USE_CLBLAST:
    lib.ggml_cl_transform_tensor.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ggml_tensor),
    ]
    lib.ggml_cl_transform_tensor.restype = None

# // backend API

# // GGML_API ggml_backend_t ggml_backend_opencl_init(void);

# // GGML_API bool ggml_backend_is_opencl(ggml_backend_t backend);


# GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
# // GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);
def ggml_backend_opencl_host_buffer_type() -> ggml_backend_buffer_type_t:
    return lib.ggml_backend_opencl_host_buffer_type()


if GGML_USE_CLBLAST:
    lib.ggml_backend_opencl_host_buffer_type.argtypes = []
    lib.ggml_backend_opencl_host_buffer_type.restype = ggml_backend_buffer_type_t


# TODO: Add ggml-quants.h