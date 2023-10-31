# SAM.cpp

Inference of Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/) in pure C/C++

## Description

The example currently supports only the [ViT-B SAM model checkpoint](https://huggingface.co/facebook/sam-vit-base).

## Next steps

- [X] Reduce memory usage by utilizing the new ggml-alloc
- [X] Remove redundant graph nodes
- [ ] Make inference faster
- [X] Fix the difference in output masks compared to the PyTorch implementation
- [X] Filter masks based on stability score
- [ ] Add support for user input
- [ ] Support F16 for heavy F32 ops
- [ ] Test quantization
- [X] Support bigger model checkpoints
- [ ] GPU support

## Quick start
```bash
git clone https://github.com/ggerganov/ggml
cd ggml

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Convert PTH model to ggml
python convert-pth-to-ggml.py examples/sam/sam_vit_b_01ec64.pth . 1

# Build ggml + examples
mkdir build && cd build
cmake .. && make -j4

# run inference
./bin/sam -t 16 -i ../img.jpg -m examples/sam/ggml-model-f16.bin
```

## Downloading and converting the model checkpoints

You can download a [model checkpoint](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints) and convert it to `ggml` format using the script `convert-pth-to-ggml.py`:

```
# Convert PTH model to ggml
python convert-pth-to-ggml.py examples/sam/sam_vit_b_01ec64.pth . 1
```

## Example output on M2 Ultra
```
 $ ▶ make -j sam && time ./bin/sam -t 8 -i img.jpg
[ 28%] Built target common
[ 71%] Built target ggml
[100%] Built target sam
main: seed = 1693224265
main: loaded image 'img.jpg' (680 x 453)
sam_image_preprocess: scale = 0.664062
main: preprocessed image (1024 x 1024)
sam_model_load: loading model from 'models/sam-vit-b/ggml-model-f16.bin' - please wait ...
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
embd_img
dims: 64 64 256 1 f32
First & Last 10 elements:
-0.05117 -0.06408 -0.07154 -0.06991 -0.07212 -0.07690 -0.07508 -0.07281 -0.07383 -0.06779
0.01589 0.01775 0.02250 0.01675 0.01766 0.01661 0.01811 0.02051 0.02103 0.03382
sum:  12736.272313

Skipping mask 0 with iou 0.705935 below threshold 0.880000
Skipping mask 1 with iou 0.762136 below threshold 0.880000
Mask 2: iou = 0.947081, stability_score = 0.955437, bbox (371, 436), (144, 168)


main:     load time =    51.28 ms
main:    total time =  2047.49 ms

real	0m2.068s
user	0m16.343s
sys	0m0.214s
```

Input point is (414.375, 162.796875) (currently hardcoded)

Input image:

![llamas](https://user-images.githubusercontent.com/8558655/261301565-37b7bf4b-bf91-40cf-8ec1-1532316e1612.jpg)

Output mask (mask_out_2.png in build folder):

![mask_glasses](https://user-images.githubusercontent.com/8558655/263706800-47eeea30-1457-4c87-938b-8f11536c5aa7.png)

## References

- [ggml](https://github.com/ggerganov/ggml)
- [SAM](https://segment-anything.com/)
- [SAM demo](https://segment-anything.com/demo)
