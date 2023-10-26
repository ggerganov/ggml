---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "ggml"
  text: "Tensor library for machine learning"
  tagline: ggml is a tensor library for machine learning to enable large models and high performance on commodity hardware. It is used by llama.cpp and whisper.cpp
  actions:
    - theme: brand
      text: Get Start
      link: /get-start
    - theme: alt
      text: API Examples
      link: /api-examples

features:
  - title: Written in C
  - title: 16-bit float support
  - title: Integer quantization support (4-bit, 5-bit, 8-bit, etc.)
  - title: ADAM and L-BFGS optimizers
  - title: Optimized for Apple Silicon
  - title: On x86 architectures utilizes AVX / AVX2 intrinsics
  - title: On ppc64 architectures utilizes VSX intrinsics
  - title: No third-party dependencies
  - title: Zero memory allocations during runtime
---

