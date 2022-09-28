# whisper

Port of [OpenAI's Whisper](https://github.com/openai/whisper) ASR model in C/C++ using
[ggml](https://github.com/ggerganov/ggml)

## More info

Checkout https://github.com/ggerganov/whisper.cpp

## Memory usage

| Model | Mem |
| ---   | --- |
| tiny.en | ~460 MB |
| base.en | ~620 MB |
| small.en | ~1.3 GB |
| medium.en | ~2.8 GB |
| large | ~4.9 GB |

## ggml format

The original models are converted to a custom binary format. This allows to pack everything needed into a single file:

- model parameters
- mel filters
- vocabulary
- weights

For more details, see the conversion script [convert-pt-to-ggml.py](convert-pt-to-ggml.py)
