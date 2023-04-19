import sys
import struct
import json
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
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


tokenizer = AutoTokenizer.from_pretrained(dir_model)
model = AutoModelForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
#print (model)

#print(tokenizer.encode('I believe the meaning of life is'))

list_vars = model.state_dict()
for name in list_vars.keys():
    print(name, list_vars[name].shape, list_vars[name].dtype)

fout = open(fname_out, "wb")

print(hparams)

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_position_embeddings"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
fout.write(struct.pack("i", int(hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"]))))
fout.write(struct.pack("i", ftype))

# TODO: temporary hack to not deal with implementing the tokenizer
dot_token = tokenizer.encode('.')[0]
for i in range(hparams["vocab_size"]):
    text = tokenizer.decode([dot_token, i]).encode('utf-8')
    # remove the first byte (it's always '.')
    text = text[1:]
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# Taken from:
#   https://github.com/NolanoOrg/cformers/blob/master/cformers/cpp/converters/convert_gptneox_to_ggml.py
#
# All the `gpt_neox.layers.<LAYER_ID>.attention.query_key_value.weight` layers
# Should be split into 3 layers:
#  gpt_neox.layers.<LAYER_ID>.attention.query.weight
#  gpt_neox.layers.<LAYER_ID>.attention.key.weight
#  gpt_neox.layers.<LAYER_ID>.attention.value.weight
# Similarly split `gpt_neox.layers.<LAYER_ID>.attention.query_key_value.bias`.
new_list_vars = list_vars.copy()
with torch.no_grad():
    for layer in range(hparams["num_hidden_layers"]):
        weight_key = "gpt_neox.layers." + str(layer) + ".attention.query_key_value.weight"
        bias_key = "gpt_neox.layers." + str(layer) + ".attention.query_key_value.bias"

        # Reverse engineering: https://github.com/huggingface/transformers/blob/c07a02a4b7892edfee22cbe57d3cdd9e10ae7a4d/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL115-L125
        # "View" in pytorch makes it hard to simply extract q, k, v from the matrix.
        qkv_matrix = list_vars[weight_key]
        qkv_bias = list_vars[bias_key]
        qkv_linear = torch.nn.Linear(hparams["hidden_size"], hparams["hidden_size"] * 3)
        qkv_linear.weight.data = qkv_matrix.float()
        qkv_linear.bias.data = qkv_bias.float()
        head_size = hparams["hidden_size"] // hparams["num_attention_heads"]

        # Get Wq_x_plus_bq, Wk_x_plus_bk, Wv_x_plus_bv
        identityMatrix = torch.eye(hparams["hidden_size"]) # pylint: disable=no-member
        qkv = qkv_linear(identityMatrix).unsqueeze(0)
        new_qkv_shape = qkv.size()[:-1] + (hparams["num_attention_heads"], 3 * head_size)
        qkv = qkv.view(*new_qkv_shape)
        Wq_x_plus_bq = qkv[..., :head_size]
        Wk_x_plus_bk = qkv[..., head_size:2*head_size]
        Wv_x_plus_bv = qkv[..., 2*head_size:]

        #   [batch, seq_len, num_heads, 3 * head_size] -> [batch, seq_len, (num_heads * 3 * head_size)]
        new_shape = Wq_x_plus_bq.size()[:-2] + (Wq_x_plus_bq.size(-2) * Wq_x_plus_bq.size(-1),)
        Wq_x_plus_bq = Wq_x_plus_bq.contiguous().view(*new_shape).squeeze(0)
        Wk_x_plus_bk = Wk_x_plus_bk.contiguous().view(*new_shape).squeeze(0)
        Wv_x_plus_bv = Wv_x_plus_bv.contiguous().view(*new_shape).squeeze(0)

        # Get bq, bk, bv
        zeroMatrix = torch.zeros(hparams["hidden_size"], hparams["hidden_size"]) # pylint: disable=no-member
        qkv = qkv_linear(zeroMatrix).unsqueeze(0)
        new_qkv_shape = qkv.size()[:-1] + (hparams["num_attention_heads"], 3 * head_size)
        qkv = qkv.view(*new_qkv_shape)
        bq = qkv[..., :head_size]
        bk = qkv[..., head_size:2*head_size]
        bv = qkv[..., 2*head_size:]

        #   [batch, seq_len, num_heads, 3 * head_size] -> [batch, seq_len, (num_heads * 3 * head_size)]
        new_shape = bq.size()[:-2] + (bq.size(-2) * bq.size(-1),)
        bq = bq.contiguous().view(*new_shape)[0, 0, :]
        bk = bk.contiguous().view(*new_shape)[0, 0, :]
        bv = bv.contiguous().view(*new_shape)[0, 0, :]

        # Get Wq_x, Wk_x, Wv_x
        Wq_x = (Wq_x_plus_bq - bq).T
        Wk_x = (Wk_x_plus_bk - bk).T
        Wv_x = (Wv_x_plus_bv - bv).T

        # Sanity check that the split is correct
        dummy_linear = torch.nn.Linear(hparams["hidden_size"], hparams["hidden_size"])

        dummy_linear.weight.data = Wq_x.float()
        dummy_linear.bias.data = bq.float()
        assert torch.allclose(Wq_x_plus_bq, dummy_linear(identityMatrix).unsqueeze(0))

        dummy_linear.weight.data = Wk_x.float()
        dummy_linear.bias.data = bk.float()
        assert torch.allclose(Wk_x_plus_bk, dummy_linear(identityMatrix).unsqueeze(0))

        dummy_linear.weight.data = Wv_x.float()
        dummy_linear.bias.data = bv.float()
        assert torch.allclose(Wv_x_plus_bv, dummy_linear(identityMatrix).unsqueeze(0))

        # Save the new weights and biases
        new_list_vars["gpt_neox.layers." + str(layer) + ".attention.query.weight"] = Wq_x
        new_list_vars["gpt_neox.layers." + str(layer) + ".attention.key.weight"] = Wk_x
        new_list_vars["gpt_neox.layers." + str(layer) + ".attention.value.weight"] = Wv_x
        new_list_vars["gpt_neox.layers." + str(layer) + ".attention.query.bias"] = bq
        new_list_vars["gpt_neox.layers." + str(layer) + ".attention.key.bias"] = bk
        new_list_vars["gpt_neox.layers." + str(layer) + ".attention.value.bias"] = bv

        # Delete the old weights and biases
        del new_list_vars[weight_key]
        del new_list_vars[bias_key]

list_vars = new_list_vars

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith(".attention.masked_bias") or     \
       name.endswith(".attention.bias") or \
       name.endswith(".attention.rotary_emb.inv_freq"):
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0;
    if ftype != 0:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
