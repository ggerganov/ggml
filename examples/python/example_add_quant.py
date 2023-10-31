from ggml import lib, ffi
from ggml.utils import init, copy, numpy
import numpy as np

ctx = init(mem_size=12*1024*1024) # automatically freed when pointer is GC'd
n = 256
n_threads = 4

a = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
b = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n) # can't both be quantized
sum = lib.ggml_add(ctx, a, b) # all zeroes for now. Will be quantized too!

# See cffi's doc on how to allocate native memory: it's very simple!
# https://cffi.readthedocs.io/en/latest/ref.html#ffi-interface
gf = ffi.new('struct ggml_cgraph*')
lib.ggml_build_forward_expand(gf, sum)

copy(np.array([i for i in range(n)], np.float32), a)
copy(np.array([i*100 for i in range(n)], np.float32), b)

lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

print(numpy(a, allow_copy=True))
print(numpy(b))
print(numpy(sum, allow_copy=True))