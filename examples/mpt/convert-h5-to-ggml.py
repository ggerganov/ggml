import os
import struct
import sys

import torch
from transformers import AutoConfig, AutoTokenizer


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
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


def count_model_parts(dir_model: str) -> int:
    """Returns the number of model parts in the model directory."""
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print(f"Found {num_parts} model parts in {dir_model}")
    return num_parts


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)


# output in the same directory as the model
dir_model = sys.argv[1]
# get number of model parts
num_parts = count_model_parts(dir_model)

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
    fname_out = dir_model + "/ggml-model-" + ftype_str[ftype] + ".bin"


tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
config = AutoConfig.from_pretrained(dir_model, trust_remote_code=True)
hparams = config.to_dict()

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["max_seq_len"]))
fout.write(struct.pack("i", hparams["n_heads"]))
fout.write(struct.pack("i", hparams["n_layers"]))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("f", hparams["attn_config"]["alibi_bias_max"]))
fout.write(struct.pack("f", hparams["attn_config"]["clip_qkv"] or 0.0))
fout.write(struct.pack("i", ftype))

vocab_size = hparams["vocab_size"]

encoder = tokenizer.vocab
# Add added_tokens (special tokens) to the encoder
encoder.update(tokenizer.get_added_vocab())

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

counter = 0
# sort by value
for key in sorted(encoder, key=encoder.get):
    # workaround for key error when c not found
    text = ""
    for c in key:
        if c not in byte_decoder:
            text += c
        else:
            text += chr(byte_decoder[c])
    text = bytearray(text, encoding="utf-8")
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

# Repeat last token until vocab_size
while counter < vocab_size:
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

if num_parts == 0:
    part_names = ("pytorch_model.bin",)
else:
    part_names = (
        f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    print(f"\n* Loading part: {part_name}")
    model_part = torch.load(f"{dir_model}/{part_name}", map_location="cpu")

    for name in model_part.keys():
        data = model_part[name].squeeze()
        n_dims = len(data.shape)

        # ftype == 0 -> float32, ftype == 1 -> float16
        # default type is fp32
        ftype_cur = 0
        if ftype == 1 and name[-7:] == ".weight" and n_dims > 1:
            ftype_cur = 1
        data = data.to(dtype=torch.float16 if ftype_cur == 1 else torch.float32).numpy()

        print(
            "Processing variable: " + name + " with shape: ",
            data.shape,
            "->",
            data.dtype,
        )

        # header
        str = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # data
        data.tofile(fout)

    # release memory
    del model_part

fout.close()

print("Done. Output file: " + fname_out)
print("")
