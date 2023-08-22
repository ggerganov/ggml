# Convert HF models to ggml format
#

import sys
import struct
import json
import torch
import numpy as np
import re
import os
import argparse

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BloomForCausalLM

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

parser = argparse.ArgumentParser(description='Convert starcoder HF model to GGML')
parser.add_argument('model_name_or_path', type=str, help='Name of model on HF hub, or local model folder')
parser.add_argument('--outfile', type=str, default='ggml-model.bin', help='Path of GGML file to write.')
parser.add_argument('--use_f32', action="store_true", help='Save GGML file in fp32')

args = parser.parse_args()

# use 16-bit or 32-bit floats
use_f16 = not args.use_f32

fname_out = args.outfile
fname_dir = os.path.dirname(fname_out)
if fname_dir:
    os.makedirs(fname_dir, exist_ok=True)

print("Loading model: ", args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
hparams = config.to_dict()
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.float16 if use_f16 else torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
print("Model loaded: ", args.model_name_or_path)

list_vars = model.state_dict()

encoder = tokenizer.vocab
# Add added_tokens (special tokens) to the encoder
encoder.update(tokenizer.get_added_vocab())
print(hparams)

print("Saving ggml model to: ", fname_out)
fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
vocab_size = hparams["vocab_size"]
fout.write(struct.pack("i", vocab_size))
# fout.write(struct.pack("i", len(encoder)))
fout.write(struct.pack("i", hparams["n_positions"]))
fout.write(struct.pack("i", hparams["n_embd"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", use_f16))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", vocab_size))

counter = 0
# sort by value
for key in sorted(encoder, key=encoder.get):
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

# TODO: Repeat last token until vocab_size
while counter < vocab_size:
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1
# assert counter == config.vocab_size

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # rename headers to keep compatibility
    if name == "transformer.ln_f.weight":
        name = "model/ln_f/g"
    elif name == "transformer.ln_f.bias":
        name = "model/ln_f/b"
    elif name == "transformer.wte.weight":
        name = "model/wte"
    elif name == "transformer.wpe.weight":
        name = "model/wpe"
    elif name == "lm_head.weight":
        name = "model/lm_head"
    elif re.match(r"transformer.h\.\d+\.ln_1\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/g"
    elif re.match(r"transformer.h\.\d+\.ln_1\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/b"
    elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/w"
    elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/b"
    elif re.match(r"transformer.h\.\d+\.attn\.c_proj\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/w"
    elif re.match(r"transformer.h.\d+.attn.c_proj.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/b"
    elif re.match(r"transformer.h.\d+.ln_2.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/g"
    elif re.match(r"transformer.h.\d+.ln_2.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/b"
    elif re.match(r"transformer.h.\d+.mlp.c_fc.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/w"
    elif re.match(r"transformer.h.\d+.mlp.c_fc.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/b"
    elif re.match(r"transformer.h.\d+.mlp.c_proj.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/w"
    elif re.match(r"transformer.h.\d+.mlp.c_proj.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/b"
    else:
        print("Unrecognized variable name. %s", name)

    # we don't need these
    if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0;
    if use_f16:
        if (name == "model/wte" or name == "model/lm_head" or name[-2:] == "/g" or name[-2:] == "/w") and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0

    "model/h.*/attn/c_attn/w"
    "model/h.*/attn/c_proj/w"
    "model/h.*/mlp/c_fc/w"
    "model/h.*/mlp/c_proj/w"
    if name[-14:] == "/attn/c_attn/w" or name[-14:] == "/attn/c_attn/b":
        print("  Duplicate K,V heads to use MHA instead of MQA")

        embed_dim = hparams["n_embd"]
        head_dim = embed_dim // hparams["n_head"]

        # ((n_heads + 2) * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
        q, k ,v = np.split(data, (hparams["n_head"] * head_dim, (hparams["n_head"] + 1) * head_dim), axis=0)
        # duplicate k, v along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
        if len(k.shape) == 2:
            k = np.tile(k, (hparams["n_head"], 1))
            v = np.tile(v, (hparams["n_head"], 1))
        elif len(k.shape) == 1:
            k = np.tile(k, (hparams["n_head"]))
            v = np.tile(v, (hparams["n_head"]))
        # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
        data = np.concatenate((q, k, v), axis=0)

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
