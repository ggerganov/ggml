# whisper

Port of [OpenAI's Whisper](https://github.com/openai/whisper) ASR model in C/C++ using
[ggml](https://github.com/ggerganov/ggml)

## More info

Checkout https://github.com/ggerganov/whisper.cpp

## Memory usage

| Model  | Disk   | Mem     |
| ---    | ---    | ---     |
| tiny   |  75 MB | ~280 MB |
| base   | 142 MB | ~430 MB |
| small  | 466 MB | ~1.0 GB |
| medium | 1.5 GB | ~2.6 GB |
| large  | 2.9 GB | ~4.7 GB |

## ggml format

The original models are converted to a custom binary format. This allows to pack everything needed into a single file:

- model parameters
- mel filters
- vocabulary
- weights

For more details, see the conversion script [convert-pt-to-ggml.py](convert-pt-to-ggml.py)
