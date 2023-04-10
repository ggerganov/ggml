# Convert a model checkpoint to a ggml compatible file
#
# Load the model using TensorFlow.
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import sys
import json
import struct
import numpy as np
import tensorflow as tf

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# helper method to convert a numpy array to different float types
def convert_to_ftype(data, ftype):
    # fp16
    if ftype == 1:
        return data.astype(np.float16)

    assert False, "Invalid ftype: " + str(ftype)

if len(sys.argv) < 3:
    print("Usage: convert-ckpt-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/encoder.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

with open(dir_model + "/hparams.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"

list_vars = tf.train.list_variables(dir_model)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["n_vocab"]))
fout.write(struct.pack("i", hparams["n_ctx"]))
fout.write(struct.pack("i", hparams["n_embd"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", ftype))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(encoder)))

for key in encoder:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name, shape in list_vars:
    print("Processing variable: " + name + " with shape: ", shape)

    data = tf.train.load_variable(dir_model, name).squeeze()
    n_dims = len(data.shape);

    # for efficiency - transpose the projection matrices
    # "model/h.*/attn/c_attn/w"
    # "model/h.*/attn/c_proj/w"
    # "model/h.*/mlp/c_fc/w"
    # "model/h.*/mlp/c_proj/w"
    if name[-14:] == "/attn/c_attn/w" or \
       name[-14:] == "/attn/c_proj/w" or \
       name[-11:] == "/mlp/c_fc/w" or \
       name[-13:] == "/mlp/c_proj/w":
        print("  Transposing")
        data = data.transpose()

    dshape = data.shape

    ftype_cur = 0
    if ftype != 0:
        # match name:
        #  "model/wte"
        #  "model/h.*/attn/c_attn/w"
        #  "model/h.*/attn/c_proj/w"
        #  "model/h.*/mlp/c_fc/w"
        #  "model/h.*/mlp/c_proj/w"
        if name == "model/wte" or name[-2:] == "/w":
            print("  Converting to " + ftype_str[ftype])
            data = convert_to_ftype(data, ftype)
            ftype_cur = ftype
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
