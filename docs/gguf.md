# GGUF

GGUF is a file format for storing models for inference with GGML and executors based on GGML. GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML.

It is a successor file format to GGML, GGMF and GGJT, and is designed to be unambiguous by containing all the information needed to load a model. It is also designed to be extensible, so that new information can be added to models without breaking compatibility.

For more information about the motivation behind GGUF, see [Historical State of Affairs](#historical-state-of-affairs).

## Specification

GGUF is a format based on the existing GGJT, but makes a few changes to the format to make it more extensible and easier to use. The following features are desired:

- Single-file deployment: they can be easily distributed and loaded, and do not require any external files for additional information.
- Extensible: new features can be added to GGML-based executors/new information can be added to GGUF models without breaking compatibility with existing models.
- `mmap` compatibility: models can be loaded using `mmap` for fast loading and saving.
- Easy to use: models can be easily loaded and saved using a small amount of code, with no need for external libraries, regardless of the language used.
- Full information: all information needed to load a model is contained in the model file, and no additional information needs to be provided by the user.

The key difference between GGJT and GGUF is the use of a key-value structure for the hyperparameters (now referred to as metadata), rather than a list of untyped values. This allows for new metadata to be added without breaking compatibility with existing models, and to annotate the model with additional information that may be useful for inference or for identifying the model.

### GGUF Naming Convention

GGUF follow a naming convention of `<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf` where each component is delimitated by a `-` if present. Ultimately this is intended to make it easier for humans to at a glance get the most important details of a model. It is not intended to be perfectly parsable in the field due to the diversity of existing gguf filenames.

The components are:
1. **BaseName**: A descriptive name for the model base type or architecture.
    - This can be derived from gguf metadata `general.basename` substituting spaces for dashes.
1. **SizeLabel**: Parameter weight class (useful for leader boards) represented as `<expertCount>x<count><scale-prefix>`
    - This can be derived from gguf metadata `general.size_label` if available or calculated if missing.
    - Rounded decimal point is supported in count with a single letter scale prefix to assist in floating point exponent shown below
      - `Q`: Quadrillion parameters.
      - `T`: Trillion parameters.
      - `B`: Billion parameters.
      - `M`: Million parameters.
      - `K`: Thousand parameters.
    - Additional `-<attributes><count><scale-prefix>` can be appended as needed to indicate other attributes of interest
1. **FineTune**: A descriptive name for the model fine tuning goal (e.g. Chat, Instruct, etc...)
    - This can be derived from gguf metadata `general.finetune` substituting spaces for dashes.
1. **Version**: (Optional) Denotes the model version number, formatted as `v<Major>.<Minor>`
    - If model is missing a version number then assume `v1.0` (First Public Release)
    - This can be derived from gguf metadata `general.version`
1. **Encoding**: Indicates the weights encoding scheme that was applied to the model. Content, type mixture and arrangement however are determined by user code and can vary depending on project needs.
1. **Type**: Indicates the kind of gguf file and the intended purpose for it
  - If missing, then file is by default a typical gguf tensor model file
  - `LoRA` : GGUF file is a LoRA adapter
  - `vocab` : GGUF file with only vocab data and metadata
1. **Shard**: (Optional) Indicates and denotes that the model has been split into multiple shards, formatted as `<ShardNum>-of-<ShardTotal>`.
    - *ShardNum* : Shard position in this model. Must be 5 digits padded by zeros.
      - Shard number always starts from `00001` onwards (e.g. First shard always starts at `00001-of-XXXXX` rather than `00000-of-XXXXX`).
    - *ShardTotal* : Total number of shards in this model. Must be 5 digits padded by zeros.


#### Validating Above Naming Convention

At a minimum all model files should have at least BaseName, SizeLabel, Version, in order to be easily validated as a file that is keeping with the GGUF Naming Convention. An example of this issue is that it is easy for Encoding to be mistaken as a FineTune if Version is omitted.

To validate you can use this regular expression `^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$` which will check that you got the minimum BaseName, SizeLabel and Version present in the correct order.

For example:

  * `Mixtral-8x7B-v0.1-KQ2.gguf`:
    - Model Name: Mixtral
    - Expert Count: 8
    - Parameter Count: 7B
    - Version Number: v0.1
    - Weight Encoding Scheme: KQ2

  * `Hermes-2-Pro-Llama-3-8B-F16.gguf`:
    - Model Name: Hermes 2 Pro Llama 3
    - Expert Count: 0
    - Parameter Count: 8B
    - Version Number: v1.0
    - Weight Encoding Scheme: F16
    - Shard: N/A

  * `Grok-100B-v1.0-Q4_0-00003-of-00009.gguf`
    - Model Name: Grok
    - Expert Count: 0
    - Parameter Count: 100B
    - Version Number: v1.0
    - Weight Encoding Scheme: Q4_0
    - Shard: 3 out of 9 total shards


<details><summary>Example Node.js Regex Function</summary>

```js
#!/usr/bin/env node
const ggufRegex = /^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$/;

function parseGGUFFilename(filename) {
  const match = ggufRegex.exec(filename);
  if (!match)
    return null;
  const {BaseName = null, SizeLabel = null, FineTune = null, Version = "v1.0", Encoding = null, Type = null, Shard = null} = match.groups;
  return {BaseName: BaseName, SizeLabel: SizeLabel, FineTune: FineTune, Version: Version, Encoding: Encoding, Type: Type, Shard: Shard};
}

const testCases = [
  {filename: 'Mixtral-8x7B-v0.1-KQ2.gguf',                         expected: { BaseName: 'Mixtral',              SizeLabel: '8x7B',     FineTune: null, Version: 'v0.1',   Encoding: 'KQ2',  Type: null, Shard: null}},
  {filename: 'Grok-100B-v1.0-Q4_0-00003-of-00009.gguf',            expected: { BaseName: 'Grok',                 SizeLabel: '100B',     FineTune: null, Version: 'v1.0',   Encoding: 'Q4_0', Type: null, Shard: "00003-of-00009"}},
  {filename: 'Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf',              expected: { BaseName: 'Hermes-2-Pro-Llama-3', SizeLabel: '8B', FineTune: null, Version: 'v1.0',   Encoding: 'F16',  Type: null, Shard: null}},
  {filename: 'Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf', expected: { BaseName: 'Phi-3-mini',   SizeLabel: '3.8B-ContextLength4k', FineTune: 'instruct', Version: 'v1.0',   Encoding: null,  Type: null, Shard: null}},
  {filename: 'not-a-known-arrangement.gguf',                       expected: null},
];

testCases.forEach(({ filename, expected }) => {
  const result = parseGGUFFilename(filename);
  const passed = JSON.stringify(result) === JSON.stringify(expected);
  console.log(`${filename}: ${passed ? "PASS" : "FAIL"}`);
  if (!passed) {
      console.log(result);
      console.log(expected);
  }
});
```

</details>


### File Structure

![image](https://github.com/ggerganov/ggml/assets/1991296/c3623641-3a1d-408e-bfaf-1b7c4e16aa63)
*diagram by [@mishig25](https://github.com/mishig25) (GGUF v3)*

GGUF files are structured as follows. They use a global alignment specified in the `general.alignment` metadata field, referred to as `ALIGNMENT` below. Where required, the file is padded with `0x00` bytes to the next multiple of `general.alignment`.

Fields, including arrays, are written sequentially without alignment unless otherwise specified.

Models are little-endian by default. They can also come in big-endian for use with big-endian computers; in this case, all values (including metadata values and tensors) will also be big-endian. At the time of writing, there is no way to determine if a model is big-endian; this may be rectified in future versions. If no additional information is provided, assume the model is little-endian.

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
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
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

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

By convention, most counts/lengths/etc are `uint64` unless otherwise specified. This is to allow for larger models to be supported in the future. Some models may use `uint32` for their values; it is recommended that readers support both.

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
  - `mamba`
  - `rwkv`
- **`general.quantization_version: uint32`**: The version of the quantization format. Not required if the model is not quantized (i.e. no tensors are quantized). If any tensors are quantized, this _must_ be present. This is separate to the quantization scheme of the tensors itself; the quantization version may change without changing the scheme's name (e.g. the quantization scheme is Q5_K, and the quantization version is 4).
- **`general.alignment: uint32`**: the global alignment to use, as described above. This can vary to allow for different alignment schemes, but it must be a multiple of 8. Some writers may not write the alignment. If the alignment is **not** specified, assume it is `32`.

#### General metadata

- `general.name: string`: The name of the model. This should be a human-readable name that can be used to identify the model. It should be unique within the community that the model is defined in.
- `general.author: string`: The author of the model.
- `general.version: string`: The version of the model.
- `general.organization: string`: The organization of the model.
- `general.basename: string`: The base model name / architecture of the model
- `general.finetune: string`: What has the base model been optimized toward.
- `general.description: string`: free-form description of the model including anything that isn't covered by the other fields
- `general.quantized_by: string`: The name of the individual who quantized the model
- `general.size_label: string`: Size class of the model, such as number of weights and experts. (Useful for leader boards)
- `general.license: string`: License of the model, expressed as a [SPDX license expression](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/) (e.g. `"MIT OR Apache-2.0`). Do not include any other information, such as the license text or the URL to the license.
- `general.license.name: string`: Human friendly license name
- `general.license.link: string`: URL to the license.
- `general.url: string`: URL to the model's homepage. This can be a GitHub repo, a paper, etc.
- `general.doi: string`: Digital Object Identifier (DOI) https://www.doi.org/
- `general.uuid: string`: [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.repo_url: string`: URL to the model's repository such as a GitHub repo or HuggingFace repo
- `general.tags: string[]`: List of tags that can be used as search terms for a search engine or social media
- `general.languages: string[]`: What languages can the model speak. Encoded as [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) two letter codes
- `general.datasets: string[]`: Links or references to datasets that the model was trained upon
- `general.file_type: uint32`: An enumerated value describing the type of the majority of the tensors in the file. Optional; can be inferred from the tensor types.
  - `ALL_F32 = 0`
  - `MOSTLY_F16 = 1`
  - `MOSTLY_Q4_0 = 2`
  - `MOSTLY_Q4_1 = 3`
  - `MOSTLY_Q4_1_SOME_F16 = 4`
  - `MOSTLY_Q4_2 = 5` (support removed)
  - `MOSTLY_Q4_3 = 6` (support removed)
  - `MOSTLY_Q8_0 = 7`
  - `MOSTLY_Q5_0 = 8`
  - `MOSTLY_Q5_1 = 9`
  - `MOSTLY_Q2_K = 10`
  - `MOSTLY_Q3_K_S = 11`
  - `MOSTLY_Q3_K_M = 12`
  - `MOSTLY_Q3_K_L = 13`
  - `MOSTLY_Q4_K_S = 14`
  - `MOSTLY_Q4_K_M = 15`
  - `MOSTLY_Q5_K_S = 16`
  - `MOSTLY_Q5_K_M = 17`
  - `MOSTLY_Q6_K = 18`

#### Source metadata

Information about where this model came from. This is useful for tracking the provenance of the model, and for finding the original source if the model is modified. For a model that was converted from GGML, for example, these keys would point to the model that was converted from.

- `general.source.url: string`: URL to the source of the model's homepage. This can be a GitHub repo, a paper, etc.
- `general.source.doi: string`: Source Digital Object Identifier (DOI) https://www.doi.org/
- `general.source.uuid: string`: Source [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.source.repo_url: string`: URL to the source of the model's repository such as a GitHub repo or HuggingFace repo

- `general.base_model.count: uint32`: Number of parent models
- `general.base_model.{id}.name: string`: The name of the parent model.
- `general.base_model.{id}.author: string`: The author of the parent model.
- `general.base_model.{id}.version: string`: The version of the parent model.
- `general.base_model.{id}.organization: string`: The organization of the parent model.
- `general.base_model.{id}.url: string`: URL to the source of the parent model's homepage. This can be a GitHub repo, a paper, etc.
- `general.base_model.{id}.doi: string`: Parent Digital Object Identifier (DOI) https://www.doi.org/
- `general.base_model.{id}.uuid: string`: Parent [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.base_model.{id}.repo_url: string`: URL to the source of the parent model's repository such as a GitHub repo or HuggingFace repo

### LLM

In the following, `[llm]` is used to fill in for the name of a specific LLM architecture. For example, `llama` for LLaMA, `mpt` for MPT, etc. If mentioned in an architecture's section, it is required for that architecture, but not all keys are required for all architectures. Consult the relevant section for more information.

- `[llm].context_length: uint64`: Also known as `n_ctx`. length of the context (in tokens) that the model was trained on. For most architectures, this is the hard limit on the length of the input. Architectures, like RWKV, that are not reliant on transformer-style attention may be able to handle larger inputs, but this is not guaranteed.
- `[llm].embedding_length: uint64`: Also known as `n_embd`. Embedding layer size.
- `[llm].block_count: uint64`: The number of blocks of attention+feed-forward layers (i.e. the bulk of the LLM). Does not include the input or embedding layers.
- `[llm].feed_forward_length: uint64`: Also known as `n_ff`. The length of the feed-forward layer.
- `[llm].use_parallel_residual: bool`: Whether or not the parallel residual logic should be used.
- `[llm].tensor_data_layout: string`: When a model is converted to GGUF, tensors may be rearranged to improve performance. This key describes the layout of the tensor data. This is not required; if not present, it is assumed to be `reference`.
  - `reference`: tensors are laid out in the same order as the original model
  - further options can be found for each architecture in their respective sections
- `[llm].expert_count: uint32`: Number of experts in MoE models (optional for non-MoE arches).
- `[llm].expert_used_count: uint32`: Number of experts used during each token token evaluation (optional for non-MoE arches).

#### Attention

- `[llm].attention.head_count: uint64`: Also known as `n_head`. Number of attention heads.
- `[llm].attention.head_count_kv: uint64`: The number of heads per group used in Grouped-Query-Attention. If not present or if present and equal to `[llm].attention.head_count`, the model does not use GQA.
- `[llm].attention.max_alibi_bias: float32`: The maximum bias to use for ALiBI.
- `[llm].attention.clamp_kqv: float32`: Value (`C`) to clamp the values of the `Q`, `K`, and `V` tensors between (`[-C, C]`).
- `[llm].attention.layer_norm_epsilon: float32`: Layer normalization epsilon.
- `[llm].attention.layer_norm_rms_epsilon: float32`: Layer RMS normalization epsilon.
- `[llm].attention.key_length: uint32`: The optional size of a key head, $d_k$. If not specified, it will be `n_embd / n_head`.
- `[llm].attention.value_length: uint32`: The optional size of a value head, $d_v$. If not specified, it will be `n_embd / n_head`.

#### RoPE

- `[llm].rope.dimension_count: uint64`: The number of rotary dimensions for RoPE.
- `[llm].rope.freq_base: float32`: The base frequency for RoPE.

##### Scaling

The following keys describe RoPE scaling parameters:

- `[llm].rope.scaling.type: string`: Can be `none`, `linear`, or `yarn`.
- `[llm].rope.scaling.factor: float32`: A scale factor for RoPE to adjust the context length.
- `[llm].rope.scaling.original_context_length: uint32_t`: The original context length of the base model.
- `[llm].rope.scaling.finetuned: bool`: True if model has been finetuned with RoPE scaling.

Note that older models may not have these keys, and may instead use the following key:

- `[llm].rope.scale_linear: float32`: A linear scale factor for RoPE to adjust the context length.

It is recommended that models use the newer keys if possible, as they are more flexible and allow for more complex scaling schemes. Executors will need to support both indefinitely.

#### SSM

- `[llm].ssm.conv_kernel: uint32`: The size of the rolling/shift state.
- `[llm].ssm.inner_size: uint32`: The embedding size of the states.
- `[llm].ssm.state_size: uint32`: The size of the recurrent state.
- `[llm].ssm.time_step_rank: uint32`: The rank of time steps.

#### Models

The following sections describe the metadata for each model architecture. Each key specified _must_ be present.

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.block_count`
- `llama.feed_forward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`
- `llama.attention.layer_norm_rms_epsilon`

###### Optional

- `llama.rope.scale`
- `llama.attention.head_count_kv`
- `llama.tensor_data_layout`:
  - `Meta AI original pth`:
    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```
- `llama.expert_count`
- `llama.expert_used_count`

##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.block_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`
- `mpt.attention.layer_norm_epsilon`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.block_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`
- `gptneox.attention.layer_norm_epsilon`

###### Optional

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.block_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`
- `gptj.attention.layer_norm_epsilon`

###### Optional

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.block_count`
- `gpt2.attention.head_count`
- `gpt2.attention.layer_norm_epsilon`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.block_count`
- `bloom.feed_forward_length`
- `bloom.attention.head_count`
- `bloom.attention.layer_norm_epsilon`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.block_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`
- `falcon.attention.layer_norm_epsilon`

###### Optional

- `falcon.tensor_data_layout`:

  - `jploski` (author of the original GGML implementation of Falcon):

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

##### Mamba

- `mamba.context_length`
- `mamba.embedding_length`
- `mamba.block_count`
- `mamba.ssm.conv_kernel`
- `mamba.ssm.inner_size`
- `mamba.ssm.state_size`
- `mamba.ssm.time_step_rank`
- `mamba.attention.layer_norm_rms_epsilon`

##### RWKV

The vocabulary size is the same as the number of rows in the `head` matrix.

- `rwkv.architecture_version: uint32`: The only allowed value currently is 4. Version 5 is expected to appear some time in the future.
- `rwkv.context_length: uint64`: Length of the context used during training or fine-tuning. RWKV is able to handle larger context than this limit, but the output quality may suffer.
- `rwkv.block_count: uint64`
- `rwkv.embedding_length: uint64`
- `rwkv.feed_forward_length: uint64`

##### Whisper

Keys that do not have types defined should be assumed to share definitions with `llm.` keys.
(For example, `whisper.context_length` is equivalent to `llm.context_length`.)
This is because they are both transformer models.

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.block_count`
- `whisper.encoder.mels_count: uint64`
- `whisper.encoder.attention.head_count`

- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.block_count`
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

GGML supports an embedded vocabulary that enables inference of the model, but implementations of tokenization using this vocabulary (i.e. `llama.cpp`'s tokenizer) may have lower accuracy than the original tokenizer used for the model. When a more accurate tokenizer is available and supported, it should be used instead.

It is not guaranteed to be standardized across models, and may change in the future. It is recommended that model authors use a more standardized tokenizer if possible.

- `tokenizer.ggml.model: string`: The name of the tokenizer model.
  - `llama`: Llama style SentencePiece (tokens and scores extracted from HF `tokenizer.model`)
  - `replit`: Replit style SentencePiece (tokens and scores extracted from HF `spiece.model`)
  - `gpt2`: GPT-2 / GPT-NeoX style BPE (tokens extracted from HF `tokenizer.json`)
  - `rwkv`: RWKV tokenizer
- `tokenizer.ggml.tokens: array[string]`: A list of tokens indexed by the token ID used by the model.
- `tokenizer.ggml.scores: array[float32]`: If present, the score/probability of each token. If not present, all tokens are assumed to have equal probability. If present, it must have the same length and index as `tokens`.
- `tokenizer.ggml.token_type: array[int32]`: The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte). If present, it must have the same length and index as `tokens`.
- `tokenizer.ggml.merges: array[string]`: If present, the merges of the tokenizer. If not present, the tokens are assumed to be atomic.
- `tokenizer.ggml.added_tokens: array[string]`: If present, tokens that were added after training.

##### Special tokens

- `tokenizer.ggml.bos_token_id: uint32`: Beginning of sequence marker
- `tokenizer.ggml.eos_token_id: uint32`: End of sequence marker
- `tokenizer.ggml.unknown_token_id: uint32`: Unknown token
- `tokenizer.ggml.separator_token_id: uint32`: Separator token
- `tokenizer.ggml.padding_token_id: uint32`: Padding token

#### Hugging Face

Hugging Face maintains their own `tokenizers` library that supports a wide variety of tokenizers. If your executor uses this library, it may be able to use the model's tokenizer directly.

- `tokenizer.huggingface.json: string`: the entirety of the HF `tokenizer.json` for a given model (e.g. <https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json>). Included for compatibility with executors that support HF tokenizers directly.

#### Other

Other tokenizers may be used, but are not necessarily standardized. They may be executor-specific. They will be documented here as they are discovered/further developed.

- `tokenizer.rwkv.world: string`: a RWKV World tokenizer, like [this](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt). This text file should be included verbatim.
- `tokenizer.chat_template : string`: a Jinja template that specifies the input format expected by the model. For more details see: <https://huggingface.co/docs/transformers/main/en/chat_templating>

### Computation graph

This is a future extension and still needs to be discussed, and may necessitate a new GGUF version. At the time of writing, the primary blocker is the stabilization of the computation graph format.

A sample computation graph of GGML nodes could be included in the model itself, allowing an executor to run the model without providing its own implementation of the architecture. This would allow for a more consistent experience across executors, and would allow for more complex architectures to be supported without requiring the executor to implement them.

## Standardized tensor names

To minimize complexity and maximize compatibility, it is recommended that models using the transformer architecture use the following naming convention for their tensors:

### Base layers

`AA.weight` `AA.bias`

where `AA` can be:

- `token_embd`: Token embedding layer
- `pos_embd`: Position embedding layer
- `output_norm`: Output normalization layer
- `output`: Output layer

### Attention and feed-forward layer blocks

`blk.N.BB.weight` `blk.N.BB.bias`

where N signifies the block number a layer belongs to, and where `BB` could be:

- `attn_norm`: Attention normalization layer
- `attn_norm_2`: Attention normalization layer
- `attn_qkv`: Attention query-key-value layer
- `attn_q`: Attention query layer
- `attn_k`: Attention key layer
- `attn_v`: Attention value layer
- `attn_output`: Attention output layer

- `ffn_norm`: Feed-forward network normalization layer
- `ffn_up`: Feed-forward network "up" layer
- `ffn_gate`: Feed-forward network "gate" layer
- `ffn_down`: Feed-forward network "down" layer
- `ffn_gate_inp`: Expert-routing layer for the Feed-forward network in MoE models
- `ffn_gate_exp`: Feed-forward network "gate" layer per expert in MoE models
- `ffn_down_exp`: Feed-forward network "down" layer per expert in MoE models
- `ffn_up_exp`: Feed-forward network "up" layer per expert in MoE models

- `ssm_in`: State space model input projections layer
- `ssm_conv1d`: State space model rolling/shift layer
- `ssm_x`: State space model selective parametrization layer
- `ssm_a`: State space model state compression layer
- `ssm_d`: State space model skip connection layer
- `ssm_dt`: State space model time step layer
- `ssm_out`: State space model output projection layer

## Version History

This document is actively updated to describe the current state of the metadata, and these changes are not tracked outside of the commits.

However, the format _itself_ has changed. The following sections describe the changes to the format itself.

### v3

Adds big-endian support.

### v2

Most countable values (lengths, etc) were changed from `uint32` to `uint64` to allow for larger models to be supported in the future.

### v1

Initial version.

## Historical State of Affairs

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
- an embedded vocabulary, which is a list of strings with length prepended. The GGMF/GGJT formats embed a float32 score next to the strings.
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
