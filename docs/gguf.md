# GGUF

GGUF is a file format for storing models for inference with GGML and executors based on GGML. GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML.

It is a successor file format to GGML, GGMF and GGJT, and is designed to be unambiguous by containing all the information needed to load a model. It is also designed to be extensible, so that new features can be added to GGML without breaking compatibility with older models.

For more information about the motivation behind GGUF, see [Current State of Affairs](#current-state-of-affairs).

## Specification

GGUF is a format based on the existing GGJT, but makes a few changes to the format to make it more extensible and easier to use. The following features are desired:

- Single-file deployment: they can be easily distributed and loaded, and do not require any external files for additional information.
- Extensible: new features can be added to GGML without breaking compatibility with existing models.
- `mmap` compatibility: models can be loaded using `mmap` for fast loading and saving.
- Easy to use: models can be easily loaded and saved using a small amount of code, with no need for external libraries, regardless of the language used.
- Full information: all information needed to load a model is contained in the model file, and no additional information needs to be provided by the user.

The key difference between GGJT and GGUF is the use of a key-value structure for the hyperparameters (now referred to as metadata), rather than a list of untyped values. This allows for new metadata to be added without breaking compatibility with existing models, and to annotate the model with additional information that may be useful for inference or for identifying the model.

### File Structure

GGUF files are structured as follows. They assume the use of a global `ALIGNMENT` constant, which is the alignment of the model data. This is currently 64 bytes, but may change in the future. [^1] To achieve this, where relevant, the file is padded with `0x00` bytes to the next multiple of `ALIGNMENT`.

Fields, including arrays, are written sequentially without alignment unless otherwise specified.

[^1]: This may be moved to a per-model key-value pair in the future.

```c
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
}

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint32_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
}

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    bool bool_;
    gguf_string_t string;
    struct {
        // Number of elements, not bytes
        uint32_t len;
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // A standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The length of the value, in bytes
    uint32_t value_len;
    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `'GGUF'`/`0x47475546`.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `1` for version described in this spec.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint32_t tensor_count;
    // The number of metadata key-value pairs.
    uint32_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

struct gguf_tensor_info_t {
    // The name of the tensor.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint32_t dimensions[n_dimensions];
    // The number of elements in the tensor.
    uint32_t n_elements;
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    // Must be a multiple of `ALIGNMENT`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Padding to the nearest multiple of `ALIGNMENT`.
    uint8_t _padding[ALIGNMENT - (sizeof(header) % ALIGNMENT)];

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```

## Standardized key-value pairs

The following key-value pairs are standardized. This list may grow in the future as more use cases are discovered. Where possible, names are shared with the original model definitions to make it easier to map between the two.

Not all of these are required, but they are all recommended. Keys that are required are bolded. For omitted pairs, the reader should assume that the value is unknown and either default or error as appropriate.

The community can develop their own key-value pairs to carry additional data. However, these should be namespaced with the relevant community name to avoid collisions. For example, the `rustformers` community might use `rustformers.` as a prefix for all of their keys.

If a particular community key is widely used, it may be promoted to a standardized key.

### General

#### Required

- **`general.architecture: string`**: describes what architecture this model implements. All lowercase ASCII, with only `[a-z0-9]+` characters allowed. Known values include:
  - `llama`
  - `mpt`
  - `gptneox`
  - `gptj`
  - `gpt2`
  - `bloom`
  - `falcon`
  - `rwkv`
- **`general.quantization_version: u32`**: version of quantization scheme. Not required if the model is not quantized (i.e. no tensors are quantized). If any tensors are quantized, this _must_ be present.

#### General metadata

- `general.name`: The name of the model. This should be a human-readable name that can be used to identify the model. It should be unique within the community that the model is defined in.
- `general.author`: The author of the model.
- `general.url`: URL to the model's homepage. This can be a GitHub repo, a paper, etc.
- `general.description: string`: free-form description of the model including anything that isn't covered by the other fields
- `general.file_type: string`: type of the majority of the tensors in the file. This shouldn't have any semantic meaning and should be purely informational, hence the use of `string`.
- `general.license: string`: SPDX license of the model

#### Source metadata

Information about where this model came from. This is useful for tracking the provenance of the model, and for finding the original source if the model is modified. For a model that was converted from GGML, for example, these keys would point to the model that was converted from.

- `general.source.url: string`: URL to the source of the model. Can be a GitHub repo, a paper, etc.
- `general.source.huggingface.repository: string`: Hugging Face model repository that this model is either hosted on or based on

### LLM

In the following, `[llm]` is used to fill in for the name of a specific LLM architecture. They will be used in each architecture's section.

- `[llm].context_length: u32`: Also known as `n_ctx`. length of the context (in tokens) that the model was trained on. For most architectures, this is the hard limit on the length of the input. Architectures, like RWKV, that are not reliant on transformer-style attention may be able to handle larger inputs, but this is not guaranteed.
- `[llm].embedding_length: u32`: Also known as `n_embd`. Embedding layer size.
- `[llm].layer_count: u32`: Also known as `n_layers`. The number of attention+feedforward layers (i.e. the bulk of the LLM). Does not include the input or embedding layers.
- `[llm].feedforward_length: u32`: Also known as `n_ff`. The length of the feedforward layer.
- `[llm].use_parallel_residual: bool`: Whether or not the parallel residual logic should be used.
- `[llm].tensor_data_layout: string`: When a model is converted to GGUF, tensors may be rearranged to improve performance. This key describes the layout of the tensor data. This is not required; if not present, it is assumed to be `reference`.
  - `reference`: tensors are laid out in the same order as the original model
  - further options can be found for each architecture in their respective sections

#### Attention

- `[llm].attention.head_count: u32`: Also known as `n_head`. Number of attention heads.
- `[llm].attention.head_count_kv: u32`: The number of heads per group used in Grouped-Query-Attention. If not present, the model does not use GQA.
- `[llm].attention.max_alibi_bias: f32`: The maximum bias to use for ALiBI.
- `[llm].attention.clamp_kqv: f32`: Value (`C`) to clamp the values of the `Q`, `K`, and `V` tensors between (`[-C, C]`).

#### RoPE

- `[llm].rope.dimension_count: u32`: The number of rotary dimensions for RoPE.
- `[llm].rope.scale: f32`: A scale factor for RoPE to adjust the context length.

#### Models

The following sections describe the metadata for each model architecture. Each key specified _must_ be present.

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.layer_count`
- `llama.feedforward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`

###### Optional

- `llama.rope.scale`
- `llama.attention.head_count_kv`
- `llama.tensor_data_layout`:
  - `llama.cpp`:
    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```

##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.layer_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.layer_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`

###### Optional

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.layer_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`

###### Optional

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.layer_count`
- `gpt2.attention.head_count`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.layer_count`
- `bloom.feedforward_length`
- `bloom.attention.head_count`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.layer_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`

###### Optional

- `falcon.tensor_data_layout`:

  - `llama.cpp` (this name may be inaccurate depending on where the Falcon implementation ends up):

    ```python
    # The original query_key_value tensor contains n_head_kv "kv groups",
    # each consisting of n_head/n_head_kv query weights followed by one key
    # and one value weight (shared by all query heads in the kv group).
    # This layout makes it a big pain to work with in GGML.
    # So we rearrange them here,, so that we have n_head query weights
    # followed by n_head_kv key weights followed by n_head_kv value weights,
    # in contiguous fashion.

    if "query_key_value" in src:
        qkv = model[src].view(
            n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)

        q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
        k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
        v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)

        model[src] = torch.cat((q,k,v)).reshape_as(model[src])
    ```

##### RWKV

The vocabulary size is the same as the number of rows in the `head` matrix.

- `rwkv.architecture_version: u32`: The only allowed value currently is 4. Version 5 is expected to appear some time in the future.
- `rwkv.context_length: u32`: Length of the context used during training or fine-tuning. RWKV is able to handle larger context than this limit, but the output quality may suffer.
- `rwkv.layer_count: u32`
- `rwkv.embedding_length: u32`
- `rwkv.feedforward_length: u32`

##### Whisper

Keys that do not have types defined should be assumed to share definitions with `llm.` keys.
(For example, `whisper.context_length` is equivalent to `llm.context_length`.)
This is because they are both transformer models.

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.layer_count`
- `whisper.encoder.mels_count: u32`
- `whisper.encoder.attention.head_count`

- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.layer_count`
- `whisper.decoder.attention.head_count`

#### Prompting

**TODO**: Include prompt format, and/or metadata about how it should be used (instruction, conversation, autocomplete, etc).

### LoRA

**TODO**: Figure out what metadata is needed for LoRA. Probably desired features:

- match an existing model exactly, so that it can't be misapplied
- be marked as a LoRA so executors won't try to run it by itself

Should this be an architecture, or should it share the details of the original model with additional fields to mark it as a LoRA?

### Tokenizer

The following keys are used to describe the tokenizer of the model. It is recommended that model authors support as many of these as possible, as it will allow for better tokenization quality with supported executors.

#### GGML

GGML supports an embedded vocabulary that may be lossily compressed from a more complete tokenizer. It is simplistic and specific to GGML. This should enable inferencing of the model, but it may not fully capture the nuances of tokenization. When a more accurate tokenizer is available and supported, it should be used instead.

It is not guaranteed to be standardized across models, and may change in the future. It is recommended that model authors use a more standardized tokenizer if possible.

- `tokenizer.ggml.model: string`: The name of the tokenizer model.
  - `llama`: Llama style SentencePiece (tokens and scores extracted from HF `tokenizer.model`)
  - `replit`: Replit style SentencePiece (tokens and scores extracted from HF `spiece.model`)
  - `gpt2`: GPT-2 / GPT-NeoX style BPE (tokens extracted from HF `tokenizer.json`)
  - `rwkv`: RWKV tokenizer
- `tokenizer.ggml.tokens: array[string]`: A list of tokens indexed by the token ID used by the model.
- `tokenizer.ggml.scores: array[f32]`: If present, the score/probability of each token. If not present, all tokens are assumed to have equal probability. Must be the same length as `tokens`.
- `tokenizer.ggml.merges: array[string]`: If present, the merges of the tokenizer. If not present, the tokens are assumed to be atomic.
- `tokenizer.ggml.bos_token_id: u32`: Beginning of sequence marker
- `tokenizer.ggml.eos_token_id: u32`: End of sequence marker
- `tokenizer.ggml.unknown_token_id: u32`: Unknown token
- `tokenizer.ggml.separator_token_id: u32`: Separator token
- `tokenizer.ggml.padding_token_id: u32`: Padding token

#### Hugging Face

Hugging Face maintains their own `tokenizers` library that supports a wide variety of tokenizers. If your executor uses this library, it may be able to use the model's tokenizer directly.

- `tokenizer.huggingface.json: string`: the entirety of the HF `tokenizer.json` for a given model (e.g. <https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json>). Included for compatibility with executors that support HF tokenizers directly.

#### Other

Other tokenizers may be used, but are not necessarily standardized. They may be executor-specific. They will be documented here as they are discovered/further developed.

- `tokenizer.rwkv.world: string`: a RWKV World tokenizer, like [this](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt). This text file should be included verbatim.

### Computation graph

This is a future extension and still needs to be discussed, and may necessitate a new GGUF version. At the time of writing, the primary blocker is the stabilization of the computation graph format.

A sample computation graph of GGML nodes could be included in the model itself, allowing an executor to run the model without providing its own implementation of the architecture. This would allow for a more consistent experience across executors, and would allow for more complex architectures to be supported without requiring the executor to implement them.

## Migration

All existing Python conversion scripts will be consolidated to use one `gguf` library. They will take models from Hugging Face or elsewhere and produce compliant GGUF files with all of the recommended metadata.

Existing models do not have enough information to be directly converted to GGUF. Instead, a migration tool may be built that takes an existing GGML/GGMF/GGJT file and prompts the user for the missing information. This tool will be executor-agnostic, and will be able to produce a GGUF file that can be used by any executor. This tool may hardcode settings for models with known hashes to ease the migration process, such that a user can run `./migrate nous-hermes-13b.ggmlv3.q5_1.bin` and obtain a `nous-hermes-13b.ggmlv3.q5_1.gguf` file that is ready to use and consistent with uploaded models.

---

## Current State of Affairs

The following information is provided for context, but is not necessary to understand the rest of this document.

### Overview

At present, there are three GGML file formats floating around for LLMs:

- **GGML** (unversioned): baseline format, with no versioning or alignment.
- **GGMF** (versioned): the same as GGML, but with versioning. Only one version exists.
- **GGJT**: Aligns the tensors to allow for use with `mmap`, which requires alignment. v1, v2 and v3 are identical, but the latter versions use a different quantization scheme that is incompatible with previous versions.

GGML is primarily used by the examples in `ggml`, while GGJT is used by `llama.cpp` models. Other executors may use any of the three formats, but this is not 'officially' supported.

These formats share the same fundamental structure:

- a magic number with an optional version number
- model-specific hyperparameters, including
  - metadata about the model, such as the number of layers, the number of heads, etc.
  - a `ftype` that describes the type of the majority of the tensors,
    - for GGML files, the quantization version is encoded in the `ftype` divided by 1000
- an embedded vocabulary, which is a list of strings with length prepended. The GGMF/GGJT formats embed a f32 score next to the strings.
- finally, a list of tensors with their length-prepended name, type, and (aligned, in the case of GGJT) tensor data

Notably, this structure does not identify what model architecture the model belongs to, nor does it offer any flexibility for changing the structure of the hyperparameters. This means that the only way to add new hyperparameters is to add them to the end of the list, which is a breaking change for existing models.

### Drawbacks

Unfortunately, over the last few months, there are a few issues that have become apparent with the existing models:

- There's no way to identify which model architecture a given model is for, because that information isn't present
  - Similarly, existing programs cannot intelligently fail upon encountering new architectures
- Adding or removing any new hyperparameters is a breaking change, which is impossible for a reader to detect without using heuristics
- Each model architecture requires its own conversion script to their architecture's variant of GGML
- Maintaining backwards compatibility without breaking the structure of the format requires clever tricks, like packing the quantization version into the ftype, which are not guaranteed to be picked up by readers/writers, and are not consistent between the two formats

### Why not other formats?

There are a few other formats that could be used, but issues include:

- requiring additional dependencies to load or save the model, which is complicated in a C environment
- limited or no support for 4-bit quantization
- existing cultural expectations (e.g. whether or not the model is a directory or a file)
- lack of support for embedded vocabularies
- lack of control over direction of future development

Ultimately, it is likely that GGUF will remain necessary for the foreseeable future, and it is better to have a single format that is well-documented and supported by all executors than to contort an existing format to fit the needs of GGML.
