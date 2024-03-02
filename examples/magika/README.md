# Google Magika inference

Simple example that shows how to use GGML for inference with the [Google Magika](https://github.com/google/magika) file type detection model.

### Usage

- Obtain the Magika model in H5 format
  - Pinned version: https://github.com/google/magika/blob/4460acb5d3f86807c3b53223229dee2afa50c025/assets_generation/models/standard_v1/model.h5
- Use `convert.py` to convert the model to gguf format:
```sh
  $ python examples/magika/convert.py /path/to/model.h5
```
- Invoke the program with the model file and a list of files to identify:
```sh
  $ build/bin/magika model.h5.gguf examples/sam/example.jpg examples/magika/convert.py README.md src/ggml.c /bin/gcc write.exe jfk.wav
  examples/sam/example.jpg      : jpeg (100.00%) pptx (0.00%) smali (0.00%) shell (0.00%) sevenzip (0.00%)
  examples/magika/convert.py    : python (99.99%) javascript (0.00%) txt (0.00%) asm (0.00%) scala (0.00%)
  README.md                     : markdown (100.00%) txt (0.00%) yaml (0.00%) ppt (0.00%) shell (0.00%)
  src/ggml.c                    : c (99.95%) txt (0.04%) asm (0.01%) yaml (0.00%) html (0.00%)
  /bin/gcc                      : elf (99.98%) odex (0.02%) pptx (0.00%) smali (0.00%) shell (0.00%)
  write.exe                     : pebin (100.00%) ppt (0.00%) smali (0.00%) shell (0.00%) sevenzip (0.00%)
  jfk.wav                       : wav (100.00%) ppt (0.00%) shell (0.00%) sevenzip (0.00%) scala (0.00%)
```
