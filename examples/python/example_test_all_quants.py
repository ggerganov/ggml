from ggml import ffi, lib
from ggml.utils import init, numpy, copy
import numpy as np
from math import pi, cos, sin, ceil

import matplotlib.pyplot as plt

ctx = init(mem_size=100*1024*1024) # Will be auto-GC'd
n = 256

orig = np.array([
    [
        cos(j * 2 * pi / n) * (sin(i * 2 * pi / n))
        for j in range(n)
    ]
    for i in range(n)
], np.float32)
orig_tensor = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F32, n, n)
copy(orig, orig_tensor)

quants = [
    type for type in range(lib.GGML_TYPE_COUNT)
    if lib.ggml_is_quantized(type) and
       type not in [lib.GGML_TYPE_Q8_1, lib.GGML_TYPE_Q8_K] # Apparently not supported
]
# quants = [lib.GGML_TYPE_Q2_K] # Test a single one

def get_name(type):
    name = lib.ggml_type_name(type)
    return ffi.string(name).decode('utf-8') if name else '?'

quants.sort(key=get_name)
quants.insert(0, None)
print(quants)

ncols=4
nrows = ceil(len(quants) / ncols)

plt.figure(figsize=(ncols * 5, nrows * 5), layout='tight')

for i, type in enumerate(quants):
    plt.subplot(nrows, ncols, i + 1)
    try:
        if type == None:
            plt.title('Original')
            plt.imshow(orig)
        else:
            quantized_tensor = lib.ggml_new_tensor_2d(ctx, type, n, n)
            copy(orig_tensor, quantized_tensor)
            quantized = numpy(quantized_tensor, allow_copy=True)
            d = quantized - orig
            results = {
                "l2": np.linalg.norm(d, 2),
                "linf": np.linalg.norm(d, np.inf),
                "compression":
                    round(lib.ggml_nbytes(orig_tensor) /
                          lib.ggml_nbytes(quantized_tensor), 1)
            }
            name = get_name(type)
            print(f'{name}: {results}')

            plt.title(f'{name} ({results["compression"]}x smaller)')
            plt.imshow(quantized, interpolation='nearest')
        
    except Exception as e:
        print(f'Error: {e}')

plt.show()