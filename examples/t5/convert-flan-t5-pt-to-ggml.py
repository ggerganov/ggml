import io
import sys
import torch

import code

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: convert-flan-t5-pt-to-ggml.py path-to-pt-model dir-output [use-f32]\n")
    sys.exit(1)

fname_inp=sys.argv[1] + "/pytorch_model.bin"

try:
    model_bytes = open(fname_inp, "rb").read()
    with io.BytesIO(model_bytes) as fp:
        checkpoint = torch.load(fp, map_location="cpu")
except:
    print("Error: failed to load PyTorch model file: %s" % fname_inp)
    sys.exit(1)

# list all keys
for k in checkpoint.keys():
    print(k)
