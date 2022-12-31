import io
import sys
import torch
import json
import struct
import numpy

import code # tmp

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: convert-flan-t5-pt-to-ggml.py path-to-pt-model dir-output [use-f32]\n")
    sys.exit(1)

dir_inp = sys.argv[1]
dir_out = sys.argv[2]

fname_inp = dir_inp + "/pytorch_model.bin"
fname_out = dir_out + "/ggml-t5-model.bin"

fname_config = dir_inp + "/config.json"

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 3:
    use_f16 = False
    fname_out = dir_out + "/ggml-t5-model-f32.bin"

# load torch model
try:
    model_bytes = open(fname_inp, "rb").read()
    with io.BytesIO(model_bytes) as fp:
        checkpoint = torch.load(fp, map_location="cpu")
except:
    print("Error: failed to load PyTorch model file: %s" % fname_inp)
    sys.exit(1)

# load config (json)
config = json.load(open(fname_config, "r"))

# list all keys
for k in checkpoint.keys():
    print(k)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# list methods of tokenizer
for m in dir(tokenizer):
    print(m)

print(config)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", config["vocab_size"]))
fout.write(struct.pack("i", config["d_ff"]))
fout.write(struct.pack("i", config["d_kv"]))
fout.write(struct.pack("i", config["d_model"]))
fout.write(struct.pack("i", config["n_positions"]))
fout.write(struct.pack("i", config["num_heads"]))
fout.write(struct.pack("i", config["num_layers"]))

# sort tokenizer.vocab by value
tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
fout.write(struct.pack("i", len(tokens)))

print("tokens: %d" % len(tokens))

for key in tokens:
    # TODO: this probably is wrong, but it should work for english at least
    token = key[0].replace("▁", " ")
    text = bytearray(token, "utf-8")
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# tokenize "hello world"
#print(tokenizer.encode("Hello hello world.Hello-Hello"))
#print(tokenizer("добър ден", return_tensors="pt"))

# dump weights
for k in checkpoint.keys():
    data = checkpoint[k].squeeze().numpy()

    name = k
    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    ftype = 1;
    if use_f16:
        if n_dims < 2:
            print("  Converting to float32")
            ftype = 0
        else:
            print("  Converting to float16")
            data = data.astype(numpy.float16)
            ftype = 1
    else:
        ftype = 0

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")

#code.interact(local=locals())
