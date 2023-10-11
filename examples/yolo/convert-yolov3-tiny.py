#!/usr/bin/env python3
import sys
import gguf
import numpy as np

def save_conv2d_layer(f, gguf_writer, prefix, inp_c, filters, size, batch_normalize=True):
    biases = np.fromfile(f, dtype=np.float32, count=filters)
    gguf_writer.add_tensor(prefix + "_biases", biases, raw_shape=(1, filters, 1, 1))

    if batch_normalize:
        scales = np.fromfile(f, dtype=np.float32, count=filters)
        gguf_writer.add_tensor(prefix + "_scales", scales, raw_shape=(1, filters, 1, 1))
        rolling_mean = np.fromfile(f, dtype=np.float32, count=filters)
        gguf_writer.add_tensor(prefix + "_rolling_mean", rolling_mean, raw_shape=(1, filters, 1, 1))
        rolling_variance = np.fromfile(f, dtype=np.float32, count=filters)
        gguf_writer.add_tensor(prefix + "_rolling_variance", rolling_variance, raw_shape=(1, filters, 1, 1))

    weights_count = filters * inp_c * size * size
    l0_weights = np.fromfile(f, dtype=np.float32, count=weights_count)
    ## ggml doesn't support f32 convolution yet, use f16 instead
    l0_weights = l0_weights.astype(np.float16)
    gguf_writer.add_tensor(prefix + "_weights", l0_weights, raw_shape=(filters, inp_c, size, size))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <yolov3-tiny.weights>" % sys.argv[0])
        sys.exit(1)
    outfile = 'yolov3-tiny.gguf'
    gguf_writer = gguf.GGUFWriter(outfile, 'yolov3-tiny')

    f = open(sys.argv[1], 'rb')
    f.read(20) # skip header
    save_conv2d_layer(f, gguf_writer, "l0", 3, 16, 3)
    save_conv2d_layer(f, gguf_writer, "l1", 16, 32, 3)
    save_conv2d_layer(f, gguf_writer, "l2", 32, 64, 3)
    save_conv2d_layer(f, gguf_writer, "l3", 64, 128, 3)
    save_conv2d_layer(f, gguf_writer, "l4", 128, 256, 3)
    save_conv2d_layer(f, gguf_writer, "l5", 256, 512, 3)
    save_conv2d_layer(f, gguf_writer, "l6", 512, 1024, 3)
    save_conv2d_layer(f, gguf_writer, "l7", 1024, 256, 1)
    save_conv2d_layer(f, gguf_writer, "l8", 256, 512, 3)
    save_conv2d_layer(f, gguf_writer, "l9", 512, 255, 1, batch_normalize=False)
    save_conv2d_layer(f, gguf_writer, "l10", 256, 128, 1)
    save_conv2d_layer(f, gguf_writer, "l11", 384, 256, 3)
    save_conv2d_layer(f, gguf_writer, "l12", 256, 255, 1, batch_normalize=False)
    f.close()
    
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("{} converted to {}".format(sys.argv[1], outfile))
