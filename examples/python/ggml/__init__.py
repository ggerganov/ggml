"""
  Python bindings for the ggml library.

  Usage example:

      from ggml import lib, ffi
      from ggml.utils import init, copy, numpy
      import numpy as np

      ctx = init(mem_size=10*1024*1024)
      n = 1024
      n_threads = 4

      a = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
      b = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
      sum = lib.ggml_add(ctx, a, b)

      gf = ffi.new('struct ggml_cgraph*')
      lib.ggml_build_forward_expand(gf, sum)

      copy(np.array([i for i in range(n)], np.float32), a)
      copy(np.array([i*100 for i in range(n)], np.float32), b)
      lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

      print(numpy(sum, allow_copy=True))

  See https://cffi.readthedocs.io/en/latest/cdef.html for more on cffi.
"""

try:
    from ggml.cffi import ffi as ffi
except ImportError as e:
    raise ImportError(f"Couldn't find ggml bindings ({e}). Run `python regenerate.py` or check your PYTHONPATH.")

import os, platform

__exact_library = os.environ.get("GGML_LIBRARY")
if __exact_library:
    __candidates = [__exact_library]
elif platform.system() == "Windows":
    __candidates = ["ggml_shared.dll", "llama.dll"]
else:
    __candidates = ["libggml_shared.so", "libllama.so"]
    if platform.system() == "Darwin":
        __candidates += ["libggml_shared.dylib", "libllama.dylib"]

for i, name in enumerate(__candidates):
    try:
        # This is where all the functions, enums and constants are defined
        lib = ffi.dlopen(name)
    except OSError:
        if i < len(__candidates) - 1:
            continue
        raise OSError(f"Couldn't find ggml's shared library (tried names: {__candidates}). Add its directory to DYLD_LIBRARY_PATH (on Mac) or LD_LIBRARY_PATH, or define GGML_LIBRARY.")

# This contains the cffi helpers such as new, cast, string, etc.
# https://cffi.readthedocs.io/en/latest/ref.html#ffi-interface
ffi = ffi
