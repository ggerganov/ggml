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

    # qint4_0
    # C code:
    #    {
    #        for (int l = 0; l < QK; l++) {
    #            const float v = src[i*QK + l];
    #            amax = MAX(amax, fabsf(v));
    #        }
    #
    #        const float d = amax / ((1 << (QB - 1)) - 1);
    #        const float id = d ? 1.0/d : 0.0;
    #
    #        pd[i] = GGML_FP32_TO_GQ(d);
    #
    #        for (int l = 0; l < QK; l++) {
    #            const float v = src[i*QK + l]*id;
    #            const int8_t vi = ((int8_t) (round(v))) + 8;
    #            assert(vi >= 0 && vi < 16);
    #            pp[l/2] |= (vi & 0xf) << (4*(l & 1));
    #        }
    #
    #        memcpy(pb + i*QK/2, pp, sizeof(pp));
    #    }
    if ftype == 2:
        assert data.dtype == np.float32
        assert data.shape[-1] % 64 == 0

        # create 2 new arrays:
        #  - pd: float32 (lowest dimension is data.shape[-1] // 64)
        #  - pb: int8
        pd = np.zeros(data.shape[:-1] + (data.shape[-1] // 64,), dtype=np.float32)
        pb = np.zeros(data.shape[:-1] + (data.shape[-1],      ), dtype=np.int8)

        # the quantized data goes here
        dst = np.zeros((data.size // 64) * (4 + 32), dtype=np.uint8)

        print("data:", data.shape, data.size)
        print("pd:  ", pd.shape, pd.size)
        print("pb:  ", pb.shape, pb.size)
        print("dst: ", dst.shape, dst.size)

        for i in range(0, data.shape[-1], 64):
            max_abs = np.max(np.abs(data[..., i:i+64]))
            max_q = (1 << 3) - 1
            d = max_abs / max_q
            id = 1.0 / d if d != 0 else 0.0
            pd[..., i//64] = d

            for j in range(64):
                v = data[..., i+j] * id
                vi = np.round(v).astype(np.int8) + 8
                assert np.all(vi >= 0) and np.all(vi < 16)

                #ve = vi[...,(j & 1) == 0].reshape(-1, 1)

                #print("ve:", ve.shape, ve)
                #print("vo:", vo.shape, vo)
                #print("pb:", pb[..., (i+j)//2].shape, pb[..., (i+j)//2])

                pb[..., i+j] = vi

        # convert to 1D array
        pd = pd.reshape(-1, 1)
        pb = pb.reshape(-1, 1)

        # populate the destination array
        n = data.size
        nr = data.shape[-1]
        nn = nr//64
        for i in range(0, n, nr):
            for j in range(0, nr, 64):
                d = pd[(i//nr)*nn + j//64][0]
                b = pb[i+j:i+j+64].reshape(-1)

                db = struct.unpack("4B", struct.pack("f", d))
                dst[(i//nr)*nn*36 + (j//64)*4 + 0] = db[0]
                dst[(i//nr)*nn*36 + (j//64)*4 + 1] = db[1]
                dst[(i//nr)*nn*36 + (j//64)*4 + 2] = db[2]
                dst[(i//nr)*nn*36 + (j//64)*4 + 3] = db[3]
                for k in range(32):
                    dst[(i//nr)*nn*36 + nn*4 + (j//64)*32 + k] = b[2*k] | (b[2*k+1] << 4)

        return dst

    # qint4_1
    # C code:
    #    {
    #        for (int l = 0; l < QK; l++) {
    #            const float v = src[i*QK + l];
    #            if (v < min) min = v;
    #            if (v > max) max = v;
    #        }

    #        const float d = (max - min) / ((1 << QB) - 1);
    #        const float id = d ? 1.0/d : 0.0;

    #        pm[i] = GGML_FP32_TO_GQ(min);
    #        pd[i] = GGML_FP32_TO_GQ(d);

    #        for (int l = 0; l < QK; l++) {
    #            const float v = (src[i*QK + l] - min) * id;
    #            const uint8_t vi = (uint8_t) (v + frand());
    #            pp[l/2] |= (vi & 0xf) << (4*(l & 1));
    #        }

    #        memcpy(pb + i*QK/2, pp, sizeof(pp));
    #    }
    if ftype == 3:
        assert data.dtype == np.float32
        assert data.shape[-1] % 64 == 0

        # create 2 new arrays:
        #  - pd: float32 (lowest dimension is data.shape[-1] // 64)
        #  - pb: int8
        pm = np.zeros(data.shape[:-1] + (data.shape[-1] // 64,), dtype=np.float32)
        pd = np.zeros(data.shape[:-1] + (data.shape[-1] // 64,), dtype=np.float32)
        pb = np.zeros(data.shape[:-1] + (data.shape[-1],      ), dtype=np.int8)

        # the quantized data goes here
        dst = np.zeros((data.size // 64) * (4 + 4 + 32), dtype=np.uint8)

        print("data:", data.shape, data.size)
        print("pm:  ", pm.shape, pm.size)
        print("pd:  ", pd.shape, pd.size)
        print("pb:  ", pb.shape, pb.size)
        print("dst: ", dst.shape, dst.size)

        for i in range(0, data.shape[-1], 64):
            mmin = np.min(data[..., i:i+64])
            mmax = np.max(data[..., i:i+64])
            max_q = (1 << 4) - 1
            d = (mmax - mmin) / max_q
            id = 1.0 / d if d != 0 else 0.0

            pm[..., i//64] = mmin
            pd[..., i//64] = d

            for j in range(64):
                v = (data[..., i+j] - mmin) * id
                vi = np.round(v).astype(np.uint8)
                assert np.all(vi >= 0) and np.all(vi < 16)

                pb[..., i+j] = vi

        # convert to 1D array
        pm = pm.reshape(-1, 1)
        pd = pd.reshape(-1, 1)
        pb = pb.reshape(-1, 1)

        # populate the destination array
        n = data.size
        nr = data.shape[-1]
        nn = nr//64
        for i in range(0, n, nr):
            for j in range(0, nr, 64):
                m = pm[(i//nr)*nn + j//64][0]

                idx = (i//nr)*nn*40 + (j//64)*4

                mb = struct.unpack("4B", struct.pack("f", m))
                dst[idx + 0] = mb[0]
                dst[idx + 1] = mb[1]
                dst[idx + 2] = mb[2]
                dst[idx + 3] = mb[3]

            for j in range(0, nr, 64):
                d = pd[(i//nr)*nn + j//64][0]

                idx = (i//nr)*nn*40 + 4*nn + (j//64)*4

                db = struct.unpack("4B", struct.pack("f", d))
                dst[idx + 0] = db[0]
                dst[idx + 1] = db[1]
                dst[idx + 2] = db[2]
                dst[idx + 3] = db[3]

            for j in range(0, nr, 64):
                b = pb[i+j:i+j+64].reshape(-1)

                idx = (i//nr)*nn*40 + nn*8 + (j//64)*32
                for k in range(32):
                    dst[idx + k] = b[2*k] | (b[2*k+1] << 4)

        return dst

    assert False, "Invalid ftype: " + str(ftype)

if len(sys.argv) < 2:
    print("Usage: convert-ckpt-to-ggml.py dir-model [use-f32]\n")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/encoder.json", "r") as f:
    encoder = json.load(f)

with open(dir_model + "/hparams.json", "r") as f:
    hparams = json.load(f)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#   ftype == 2 -> qint4_0
#   ftype == 3 -> qint4_1
#
# map from ftype to string
ftype_str = ["f32", "f16", "q4_0", "q4_1"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 3:
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
    if name[-13:] == "/mlp/c_proj/w":
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
        #if name[-6:] == "attn/w":
        #if name == "model/wte":
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
