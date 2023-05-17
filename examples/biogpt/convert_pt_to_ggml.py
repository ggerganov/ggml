"""Convert the BioGPT checkpoints into the GGML format.

The GGML format is organized as follows. Each tensor must be registered in the following
order:
    - Number of dimensions (int)
    - Name length (int)
    - Dimensions (int[n_dims])
    - Name (char[name_length])
    - Data (float[n_dims])
.
The information are converted into binary format and packed into a struct.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--use-f16", action="store_true")


def parse_vocab(dir_model, outfile):
    with open(dir_model / "vocab.json", "r", encoding="utf-8") as infile:
        vocab = json.load(infile)

    tokens = sorted(vocab.items(), key=lambda x: x[1])
    outfile.write(struct.pack("i", len(tokens)))
    print("Vocab size:", len(tokens))

    for token, _ in tokens:
        text = bytearray(token, "utf-8")
        outfile.write(struct.pack("i", len(text)))
        outfile.write(text)


def parse_bpe_merges(dir_model, outfile):
    with open(dir_model / "merges.txt", "r", encoding="utf-8") as infile:
        bpe_merges = infile.read().split("\n")[:-1]
    bpe_merges = [tuple(merge.split()[:2]) for merge in bpe_merges]
    outfile.write(struct.pack("i", len(bpe_merges)))
    print("BPE merges size:", len(bpe_merges))
    for merge in bpe_merges:
        text = bytearray(" ".join(merge), "utf-8")
        outfile.write(struct.pack("i", len(text)))
        outfile.write(text)


def parse_model(checkpoint, outfile, use_f16):
    for name in checkpoint.keys():
        var_data = checkpoint[name].squeeze().numpy()
        print(f"Processing variable: {name} with shape: {var_data.shape}")

        n_dims = len(var_data.shape)

        ftype_cur = 0
        if use_f16:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                var_data = var_data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
        else:
            if var_data.dtype != np.float32:
                print("  Converting to float32")
                var_data = var_data.astype(np.float32)
                ftype_cur = 0

        encoded_name = name.encode("utf-8")
        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))
        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)


def parse_hparams(dir_model, outfile, use_f16):
    with open(dir_model / "config.json", "r", encoding="utf-8") as infile:
        hparams = json.load(infile)

    outfile.write(struct.pack("i", 0x67676d6c))
    outfile.write(struct.pack("i", hparams["vocab_size"]))
    outfile.write(struct.pack("i", hparams["num_hidden_layers"]))
    outfile.write(struct.pack("i", hparams["num_attention_heads"]))
    outfile.write(struct.pack("i", hparams["max_position_embeddings"]))
    outfile.write(struct.pack("i", hparams["intermediate_size"]))
    outfile.write(struct.pack("i", hparams["hidden_size"]))
    outfile.write(struct.pack("i", int(use_f16)))


if __name__ == "__main__":
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    outfile = open(out_dir / "ggml-model.bin", "wb")

    dir_model = Path(args.dir_model)

    with open(dir_model / "pytorch_model.bin", "rb") as infile:
        checkpoint = torch.load(infile, map_location="cpu")

    parse_hparams(dir_model, outfile, args.use_f16)
    parse_vocab(dir_model, outfile)
    parse_bpe_merges(dir_model, outfile)
    parse_model(checkpoint, outfile, args.use_f16)

    outfile.close()

    print("Done.")
