# SAM.cpp

Inference of Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/) in pure C/C++

## Description

The example currently supports only the [ViT-B SAM model checkpoint](https://huggingface.co/facebook/sam-vit-base).

## Next steps

- [ ] Reduce memory usage by utilizing the new ggml-alloc
- [ ] Remove redundant graph nodes
- [ ] Make inference faster
- [ ] Fix the difference in output masks compared to the PyTorch implementation
- [X] Filter masks based on stability score
- [ ] Add support for user input
- [ ] Support F16 for heavy F32 ops
- [ ] Test quantization
- [ ] Support bigger model checkpoints
- [ ] GPU support

## Quick start
```bash
git clone https://github.com/ggerganov/ggml
cd ggml

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Convert PTH model to ggml
python convert-pth-to-ggml.py examples/sam/sam_vit_b_01ec64.pth 1

# Build ggml + examples
mkdir build && cd build
cmake .. && make -j4

# run inference
./bin/sam -t 16 -i ../img.jpg -m ../examples/sam/ggml-model-f16.bin
```

## Downloading and converting the model checkpoints

You can download a [model checkpoint](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints) and convert it to `ggml` format using the script `convert-pth-to-ggml.py`:

```
# Convert PTH model to ggml
python convert-pth-to-ggml.py examples/sam/sam_vit_b_01ec64.pth 1
```

## Example output
```
$ ./bin/sam -t 16 -i ../img.jpg -m ../examples/sam/ggml-model-f16.bin
main: seed = 1692347524
main: loaded image '../img.jpg' (680 x 453)
sam_image_preprocess: scale = 0.664062
main: preprocessed image (1024 x 1024)
sam_model_load: loading model from '../examples/sam/ggml-model-f16.bin' - please wait ...
sam_model_load: n_enc_state      = 768
sam_model_load: n_enc_layer      = 12
sam_model_load: n_enc_head       = 12
sam_model_load: n_enc_out_chans  = 256
sam_model_load: n_pt_embd        = 4
sam_model_load: ftype            = 1
sam_model_load: qntvr            = 0
operator(): ggml ctx size = 202.32 MB
sam_model_load: ...................................... done
sam_model_load: model size =   185.05 MB / num tensors = 304
point: 624.500000 245.593750


main:     load time =    88.36 ms
main:    total time =  5697.57 ms
```

Input point is (414.375, 162.796875) (currently hardcoded)

Input image:

![llamas](https://user-images.githubusercontent.com/8558655/261301565-37b7bf4b-bf91-40cf-8ec1-1532316e1612.jpg)

Output mask:

![mask_glasses](https://user-images.githubusercontent.com/8558655/261301844-9fc2dbbc-5fd6-42ce-af69-643df9e6fad1.png)

## References

- [ggml](https://github.com/ggerganov/ggml)
- [SAM](https://segment-anything.com/)
- [SAM demo](https://segment-anything.com/demo)
