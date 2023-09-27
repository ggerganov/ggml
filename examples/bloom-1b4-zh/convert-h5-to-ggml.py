# Convert bloom-1b4-zh h5 transformer model to ggml format
#
# Load the model using BloomForCausalLM.
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
import struct
import json
import numpy as np
import re

from transformers import BloomForCausalLM


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


# Download model from https://huggingface.co/Langboat/bloom-1b4-zh
if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model dir-output [use-f32]\n")
    sys.exit(1)


# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[2] + "/ggml-model-f16.bin"

with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
    vocab = vocab['model']['vocab']

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 3:
    use_f16 = False
    fname_out = sys.argv[2] + "/ggml-model-f32.bin"

model = BloomForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)

list_vars = model.state_dict()
# bloom-1b4-zh models share the WTE tensor as the LM head
assert np.allclose(list_vars["transformer.word_embeddings.weight"].numpy(),
                   list_vars["lm_head.weight"].numpy())
del list_vars["lm_head.weight"]

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["seq_length"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", use_f16))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(vocab)))

for key in vocab:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0
    if use_f16:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0

    # "transformer.h.*.mlp.dense_4h_to_h.weight" ==> "h.*.mlp.dense_4h_to_h.weight"  # noqa
    if name.startswith("transformer."):
        name = name[12:]

    # rename headers to keep compatibility
    if name == "word_embeddings_layernorm.weight":
        name = "model/ln_wte/g"
    elif name == "word_embeddings_layernorm.bias":
        name = "model/ln_wte/b"
    elif name == "word_embeddings.weight":
        name = "model/wte"
    elif re.match(r"h\.\d+\.input_layernorm\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/g"
    elif re.match(r"h\.\d+\.input_layernorm\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/b"
    elif re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
        # Map bloom-style qkv_linear to gpt-style qkv_linear
        # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
        # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
        qkv_weights = data.reshape(
            (hparams["n_head"], 3, hparams["hidden_size"] // hparams["n_head"],
             hparams["hidden_size"])
        )
        data = np.concatenate(
            (qkv_weights[:, 0, :, :].reshape((-1, hparams["hidden_size"])),
             qkv_weights[:, 1, :, :].reshape((-1, hparams["hidden_size"])),
             qkv_weights[:, 2, :, :].reshape((-1, hparams["hidden_size"]))),
            axis=0
        )
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/w"
    elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
        qkv_bias = data.reshape(
            (hparams["n_head"], 3,
             hparams["hidden_size"] // hparams["n_head"])
        )
        data = np.concatenate(
            (qkv_bias[:, 0, :].reshape((hparams["hidden_size"],)),
             qkv_bias[:, 1, :].reshape((hparams["hidden_size"],)),
             qkv_bias[:, 2, :].reshape((hparams["hidden_size"],))),
            axis=0
        )
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/b"
    elif re.match(r"h\.\d+\.self_attention\.dense\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/w"
    elif re.match(r"h.\d+\.self_attention\.dense\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/b"
    elif re.match(r"h.\d+\.post_attention_layernorm\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/g"
    elif re.match(r"h.\d+\.post_attention_layernorm\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/b"
    elif re.match(r"h.\d+\.mlp\.dense_h_to_4h\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/w"
    elif re.match(r"h.\d+\.mlp\.dense_h_to_4h\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/b"
    elif re.match(r"h.\d+\.mlp\.dense_4h_to_h\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/w"
    elif re.match(r"h.\d+\.mlp\.dense_4h_to_h\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/b"
    elif name == "ln_f.weight":
        name = "model/ln_f/g"
    elif name == "ln_f.bias":
        name = "model/ln_f/b"
    else:
        print("Unrecognized variable name. %s", name)

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
