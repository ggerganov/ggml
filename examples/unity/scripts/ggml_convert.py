# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import struct
from enum import Enum
from io import BufferedWriter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Mapping, Tuple, Union, Sequence, Set, final
import re

import torch
from fairseq2.assets import AssetCard
from fairseq2.models.transformer.frontend import TransformerEmbeddingFrontend
from fairseq2.nn import SinusoidalPositionEncoder
from fairseq2.nn.transformer import RelativePositionalEncoding
from fairseq2.data.text import SentencePieceEncoder, SentencePieceTokenizerBase
from fairseq2.data.typing import PathLike
from fairseq2.typing import Device, finaloverride
from fairseq2.models.utils import TokenizerLoaderBase, ModelLoader
from fairseq2.models.utils.checkpoint import convert_model_state_dict
from fairseq2.assets import asset_store, download_manager

import ggml

Preprocessor = Callable[[Any], Any]
log = logging.getLogger("ggml_convert")


class ModelType(str, Enum):
    AUTO = "auto"  # inferred from the model name
    UNITY = "unity"
    NLLB = "nllb"
    MT = "bitext"
    MTS = "bitext_scripted"


UNITY_SMALLER_MODELS = [
    "unity_nano",
    "unity_micro",
]  # Trained with fairseq2, with custom dict (not original NLLB ones)


NLLB_2_UNITY_KEYMAP = {
    r"^encoder_frontend\.": r"text_encoder_frontend.",
    r"^encoder\."         : r"text_encoder.",
    r"^decoder\."         : r"text_decoder.",
    r"^decoder_frontend\.": r"text_decoder_frontend.",
}


@final
class NllbLikeTokenizer(SentencePieceTokenizerBase):
    """The only difference between this class and NllbTokenizer is it doesn't add a <pad> to control symbol list.
    Since NllbTokenizer is defined as final, we couldn't inherit from it directly. So copying ~everything"""

    langs: Set[str]
    default_lang: str

    def __init__(
        self, pathname: PathLike, langs: Sequence[str], default_lang: str
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param langs:
            The list of supported languages.
        :param default_lang:
            The fall-back language if no language is specified.
        """
        # Each language is represented by a `__lang__` control symbol.
        control_symbols = [f"__{lang}__" for lang in langs]

        # Internal control symbols that are not relevant for eval use.
        control_symbols.extend(["<MINED_DATA>", "<MMT_BT_DATA>", "<SMT_BT_DATA>"])
        super().__init__(pathname, control_symbols)

        self.langs = set(langs)

        self.default_lang = default_lang

    @finaloverride
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> SentencePieceEncoder:
        """Create a token encoder.

        :param task:
            Must be 'translation'. If ``None``, defaults to 'translation'.
        :param lang:
            A language from :attr:`langs`. If ``None``, defaults to
            :attr:`default_lang`.
        :param mode:
            Must be 'source' or 'target'. Set to 'source' if ``lang`` is the
            source language; set to 'target' if ``lang`` is the target language.
            If ``None``, defaults to 'source'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None and task != "translation":
            raise ValueError(f"`task` must be 'translation', but is '{task}' instead.")

        if lang is None:
            lang = self.default_lang

        if lang not in self.langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        if mode is None or mode == "source":
            # NLLB models expect a language token in place of BOS in source
            # sequences.
            prefix_tokens = [f"__{lang}__"]
            suffix_tokens = ["</s>"]
        elif mode == "source_mining":
            prefix_tokens = [f"__{lang}__", "<MINED_DATA>"]
            suffix_tokens = ["</s>"]
        elif mode == "source_mmt_bt":
            prefix_tokens = [f"__{lang}__", "<MMT_BT_DATA>"]
            suffix_tokens = ["</s>"]
        elif mode == "source_smt_bt":
            prefix_tokens = [f"__{lang}__", "<SMT_BT_DATA>"]
            suffix_tokens = ["</s>"]
        elif mode == "target":
            # Target sequences are expected to start with an EOS, followed by
            # the language token.
            prefix_tokens = ["</s>", f"__{lang}__"]
            suffix_tokens = []
        else:
            raise ValueError(
                f"`mode` must be 'source' or 'target', but is '{mode}' instead."
            )

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )


@final
class NllbLikeTokenizerLoader(TokenizerLoaderBase[NllbLikeTokenizer]):
    """Loads tokenizers used by NLLB models."""

    @finaloverride
    def _load(self, pathname: Path, card: AssetCard) -> NllbLikeTokenizer:
        langs = card.field("langs").as_list(str)

        default_lang = card.field("default_lang").as_(str)

        return NllbLikeTokenizer(pathname, langs, default_lang)


def convert_state_dict(
    state_dict: Dict[str, Any], key_map: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:

    if key_map is None:
        return state_dict
    
    state_dict = convert_model_state_dict(state_dict, key_map=key_map)

    # We use the built-in version attribute of `torch.nn.Module`.
    try:
        del state_dict["encoder.version"]
    except KeyError:
        pass
    try:
        del state_dict["decoder.version"]
    except KeyError:
        pass

    try:
        del state_dict["encoder.embed_positions._float_tensor"]
    except KeyError:
        pass
    try:
        del state_dict["decoder.embed_positions._float_tensor"]
    except KeyError:
        pass

    return state_dict


def convert_unity_model(
    model_name: str,
    hparams: Optional[Dict[str, Any]] = None,
):
    from seamless_communication.models import unity
    from seamless_communication.models.unity.builder import UnitYConfig, create_unity_model
    from seamless_communication.models.unity.model import UnitYModel

    load_unity_model_without_conversion = ModelLoader[UnitYModel, UnitYConfig](
        asset_store,
        download_manager,
        unity.load_unity_config,
        create_unity_model,
        None,
        restrict_checkpoints=False,
    )

    model_config = unity.load_unity_config(model_name)
    hparams = flatten_config(
        dataclasses.asdict(model_config), separator="__", overrides=hparams
    )
    hparams["multilingual"] = True
    log.info(hparams)
    # Need the diverge here because current default in SC is to convert from fairseq1 ckpt format
    if model_name in UNITY_SMALLER_MODELS:
        model = load_unity_model_without_conversion(model_name)
        tokenizer = NllbLikeTokenizerLoader(asset_store, download_manager)(model_name)
    else:
        model = unity.load_unity_model(model_name)
        tokenizer = unity.load_unity_text_tokenizer(model_name)

    vocab = read_vocab(tokenizer)

    return model, hparams, vocab


def convert_nllb_model(
    model_name: str,
    hparams: Optional[Dict[str, Any]] = None,
):
    from fairseq2.models.nllb.loader import load_nllb_tokenizer, load_nllb_model, load_nllb_config

    model_config = load_nllb_config(model_name)
    hparams = flatten_config(
        dataclasses.asdict(model_config), separator="__", overrides=hparams,
    )
    hparams["multilingual"] = True

    model = load_nllb_model(model_name)
    tokenizer = load_nllb_tokenizer(model_name)
    vocab = read_vocab(tokenizer)

    return model, hparams, vocab


def convert_bitext_model(
    model_name: str,
    hparams: Optional[Dict[str, Any]] = None,
):
    from mt import load_mt_model, load_vocab  #, test_mt

    hparams = hparams or {}
    hparams["multilingual"] = False
    model = load_mt_model(model_name)
    src_vocab, src_spm = load_vocab(model_name, "src")
    tgt_vocab, tgt_spm = load_vocab(model_name, "tgt")

    # test_mt(model, src_spm, tgt_spm)

    return model, hparams, src_vocab, tgt_vocab


def convert_model(
    model_name: Union[str, torch.nn.Module],
    out: Optional[Path] = None,
    model_type: ModelType = ModelType.AUTO,
    layers: str = "",
    hparams: Optional[Dict[str, Any]] = None,
    fp16: bool = False,
) -> None:
    """
    Entry function for converting different kinds of model into GGML file. Supported model checkpoints:
        - unity models
        - nllb models
        - Bilingual encoder-decoder model (Pytorch) with separate vocabulary for src and tgt languages
        - Bilingual encoder-decoder model (torchscript)
    Args:
        model_name: name of a registered model (discoverable in a fairseq2 asset), path to a checkpoint,\
            or the model object passed directly
        out: path to store the converted .ggml model. If None, the ggml model is stored in the same place\
            as input model
        model_type: type of the model (or inferred from the name, only applied to nllb, unity and seamless)
        layers: wildcard patterns to filter the layers from the model. Does not applied to scripted models
        hparams: override the hparams in the model with the user-defined values
        vocab: Path to  vocabulary files (in case not bundled with the model checkpoint)
        extra_vocab: Path to additional vocabulary files (used in bilingual models with explicit tgt languages)
        fp16: Save to .GGML float16 tensors instead of float32
    """

    key_map: Optional[Dict[str, str]] = None
    tgt_vocab: Optional[List[Tuple[str, float]]] = None
    if isinstance(model_name, str):
        # Load the corresponding fairseq2 model
        if out is None:
            out = Path(model_name).with_suffix(".ggml")

        # Reason the model architecture from the model name or user input
        try:
            if model_type == ModelType.AUTO:
                if "unity" in model_name or "seamlessM4T" in model_name:
                    model_type = ModelType.UNITY
                elif "nllb" in model_name:
                    model_type = ModelType.NLLB

            assert (
                model_type != ModelType.AUTO
            ), "Cannot infer model type from the `model_name`. Please specify `model_type`"

            if model_type == ModelType.UNITY:
                model, hparams, vocab = convert_unity_model(model_name, hparams=hparams)
            elif model_type == ModelType.NLLB:
                model, hparams, vocab = convert_nllb_model(model_name, hparams=hparams)
                key_map = NLLB_2_UNITY_KEYMAP
            elif model_type == ModelType.MTS:
                # TODO: implement the EdgeML model conversion here
                raise NotImplementedError("Scripted model conversion not implemented yet")
            
            # Bilingual non-scripted model
            else:
                model, hparams, vocab, tgt_vocab = convert_bitext_model(model_name, hparams=hparams)
                key_map = NLLB_2_UNITY_KEYMAP
        except Exception as exc:
            raise ValueError(f"Error in loading model: {model_name}") from exc
    else:
        # Use the model passed explicitly
        assert (
            out is not None
        ), "output path is required when explicitly passing a module"
        hparams = hparams or {}
        model = model_name

    state_dict = model.state_dict()
    if layers:
        state_dict = {k: v for k, v in state_dict.items() if re.match(layers, k)}
    fixup_model(model, state_dict, layer_filter=layers)
    state_dict = convert_state_dict(state_dict, key_map=key_map)
    layer_config = read_layer_config(model, layer_filter=layers, key_map=key_map)

    vocab = vocab or []
    tgt_vocab = tgt_vocab or []
    write_ggml_file(out, hparams, layer_config, state_dict=state_dict, vocab=vocab, tgt_vocab=tgt_vocab, fp16=fp16)


def find_children(model: torch.nn.Module, t: type, layer_filter: str = "") -> List[Tuple[str, torch.nn.Module]]:
    queue = list(model._modules.items())
    modules = []
    while queue:
        name, node = queue.pop()
        if node is None:
            continue
        if layer_filter and not re.match(layer_filter, name):
            continue
        if isinstance(node, t):
            modules.append((name, node))
        for child_name, child_node in node._modules.items():
            queue.append((".".join((name, child_name)), child_node))

    return modules


def fixup_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], layer_filter: str) -> None:
    # Bake the embedding scaling into the weights
    frontends = find_children(model, TransformerEmbeddingFrontend, layer_filter)
    if frontends:
        log.info(
            "Upgrading the following TransformerEmbeddingFrontend: {}",
            [x[0] for x in frontends],
        )
    for name, frontend in frontends:
        embed_weights = state_dict[name + ".embed.weight"]
        state_dict[name + ".embed.weight"] = embed_weights * frontend.scale

    # Sinusoidal embeddings are typically not saved since they are easily recomputed,
    # but this allows to avoid porting the sinusoidal logic to GGML
    pos_encoders = find_children(model, SinusoidalPositionEncoder, layer_filter)
    if pos_encoders:
        log.info(
            "Upgrading the following SinusoidalPositionEncoder: {}",
            [x[0] for x in pos_encoders],
        )
    for name, pos_encoder in pos_encoders:
        assert isinstance(pos_encoder.freqs, torch.Tensor)
        assert name not in state_dict
        state_dict[name] = pos_encoder.freqs

    relative_pos_encs = find_children(model, RelativePositionalEncoding, layer_filter)
    # speech_encoder has several copies of the relative_pos_enc module.
    # For efficiency reasons we only make one copy of it to GGML.
    if relative_pos_encs:
        log.info("Merging all speech_encoder RelativePositionalEncoding into one.")
        _, rel_pos_enc = relative_pos_encs[0]
        assert isinstance(rel_pos_enc.freqs, torch.Tensor)
        state_dict["speech_encoder.pos_enc"] = rel_pos_enc.freqs


def read_vocab(tokenizer: Any) -> List[Tuple[str, float]]:
    vocab_info = tokenizer.vocab_info
    vocab = [
        (tokenizer.model.index_to_token(i).replace("▁", " "), -i)
        for i in range(vocab_info.size)
    ]
    return vocab  # type: ignore[return-value]


def write_ggml_file(
    out: Path,
    hparams: Dict[str, Any],
    layer_config: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    vocab: List[Tuple[str, float]],
    tgt_vocab: Optional[List[Tuple[str, float]]] = None,  # tgt_vocab for bilingual models
    fp16: bool = False,
) -> None:
    with out.open("wb") as o:
        write_ggml_header(o)
        write_hparams(o, hparams)
        write_hparams(o, layer_config)
        write_vocab(o, vocab)
        write_state_dict(o, state_dict, fp16)
        write_vocab(o, tgt_vocab)


def write_ggml_header(out: BufferedWriter) -> None:
    """Write GGML header (in reverse cause big-endian)"""
    out.write(b"ggml"[::-1])


def write_hparams(out: BufferedWriter, hparams: Dict[str, Any]) -> None:
    """Write hyper parameters.

    :params hparams:
        flattened dict containing model's hyper parameters.

    """
    simple_vals = {}
    for key, value in hparams.items():
        try:
            simple_vals[key] = to_ctype(value)
        except ValueError:
            logging.warning(f"Skipping config for key {key}={value!r}")
            continue

    out.write(struct.pack("<q", len(simple_vals)))
    for key, (ctype, cvalue) in simple_vals.items():
        write_string(out, key)
        b = struct.pack(ctype, cvalue)
        assert len(b) == 8
        out.write(b)

    logging.info(f"Saved {len(simple_vals)} params.")


def write_vocab(out: BufferedWriter, vocab: List[Tuple[str, float]]) -> None:
    out.write(struct.pack("<q", len(vocab)))

    if len(vocab) == 0:
        return

    # Write all words concatenated in a buffer
    words = [bytes(w, "utf8") for w, score in vocab]
    packed_words = b"\0".join(words)
    # We use i32 to allow reusing the string loading codes
    packed_len = struct.pack("<i", len(packed_words))
    out.write(packed_len)
    out.write(packed_words)

    lengths = torch.tensor([len(w) for w in words], dtype=torch.int8)
    write_tensor(out, lengths)

    scores = torch.tensor([score for w, score in vocab], dtype=torch.float32)
    write_tensor(out, scores)


def write_state_dict(
    out: BufferedWriter, state_dict: Dict[str, torch.Tensor], fp16: bool
) -> None:
    """Write pytorch state dict.

    :params state_dict:
        state dict returned by pytorch model
    :params fp16:
        convert float32 tensors to float16 on disk
    """
    out.write(struct.pack("<q", len(state_dict)))
    # True size of each tensor (before downcasting to float16)
    true_byte_size = sum(x.numel() * x.element_size() for x in state_dict.values())
    out.write(struct.pack("<q", true_byte_size))

    GB = 1024**3
    if not fp16:
        log.warning(
            f"Saving a ggml file with {len(state_dict)} tensors, totalling {true_byte_size / GB:.3f}Gb"
        )
    else:

        def _fp16_byte_size(x: torch.Tensor) -> int:
            full_byte_size = x.numel() * x.element_size()
            if fp16 and x.dtype == torch.float32:
                full_byte_size //= 2
            return full_byte_size

        # Compressed size
        compressed_byte_size = sum(_fp16_byte_size(x) for x in state_dict.values())
        log.warning(
            f"Saving a ggml file with {len(state_dict)} tensors, totalling {true_byte_size / GB:.3f}Gb"
            f". Compressed to {compressed_byte_size / GB:.3f}Gb"
        )

    for key, value in state_dict.items():
        # Rename the layers to make it look like "unity-arch"
        write_string(out, key)
        if key.endswith(".bias") and value.ndim == 1 and "adaptor" not in key:
            # GGML broadcasting isn't as strong as numpy
            value = value.reshape(1, -1)
        if "pointwise_conv" in key:  # pointwise_conv / depthwise_conv
            value = value.squeeze(-1)
        if "depthwise_conv" in key:
            value = value.squeeze(1)
        if fp16 and value.dtype == torch.float32:
            value = value.to(torch.float16)
        write_tensor(out, value.contiguous())


def write_string(out: BufferedWriter, value: str) -> None:
    """Write string in utf-8 format.

    :params value:
        string value to dump.
    """
    str_ = value.encode("utf-8")
    packed_len = struct.pack("<i", len(str_))
    assert len(packed_len) == 4
    out.write(packed_len)
    out.write(str_)


def write_tensor(out: BufferedWriter, value: torch.Tensor) -> None:
    """Write torch tensor in ggml format.

    First we save the number of dimensions and the dtype.
    Then we save the data as numpy array.

    :params value:
        Tensor to dump.
    """
    if value.dtype is torch.int64:
        # GGML doesn't have int64, downcast it
        value = value.to(dtype=torch.int32)

    if value.ndim == 0:
        # GGML doesn't support scalar as tensors.
        value = value.reshape(1)

    data = value.numpy()
    n_dims = data.ndim
    assert n_dims < 5, "ggml doesn't support 5 dims tensors"
    assert n_dims >= 1, "ggml doesn't support 0 dim tensors"

    ftype = torch_to_ggml_type(value.dtype)
    out.write(struct.pack("<i", n_dims))
    out.write(struct.pack("<i", ftype))
    for i in range(n_dims):
        # ggml uses long for shape
        out.write(struct.pack("<q", data.shape[n_dims - 1 - i]))

    data.tofile(out)


def torch_to_ggml_type(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return ggml.GGML_TYPE_F32
    elif dtype is torch.float16:
        return ggml.GGML_TYPE_F16
    elif dtype is torch.int32:
        return ggml.GGML_TYPE_I32
    elif dtype is torch.int8:
        return ggml.GGML_TYPE_I8
    else:
        raise NotImplementedError(f"{dtype} is not mapped to a GGML_TYPE")


def flatten_config(
    config: Dict[str, Any],
    separator: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flatten nested dictionnary

    :param config:
        nested dictionnary containing model config.
    :param separator:
            string separator used when flattening nested hparams
    :param config_preprocessor:
        Preprocessor used for config/hparams values

    :returns:
        flat dictionnary
    """

    def __flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        result = {}
        for key in config:
            new_key = f"{prefix}{key}"
            if isinstance(config[key], dict):
                nested_result = __flatten(config[key], f"{new_key}{separator}")
                result.update(nested_result)
            else:
                new_config = config[key]
                if new_config is not None:
                    result[new_key] = config[key]

        return result

    res_config = __flatten(config)
    if overrides:
        return {**res_config, **overrides}
    else:
        return res_config


def read_layer_config(
    model: torch.nn.Module, layer_filter: str, key_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    layer_config = {}

    def _append_node_config(node: Any, prefix: str) -> None:
        for k, v in node.__dict__.items():
            # Skip special members. In particular all children module and tensors
            # will be hidden in special dicts `_parameters` and `_modules`
            if k.startswith("_"):
                continue
            # All modules have a "training" flag
            if k in ("training", "init_fn"):
                continue
            if v is None:
                continue

            try:
                to_ctype(v)
            except ValueError:
                log.warning(f"Skipping layer config {k}={v!r}")
                continue
            layer_config[prefix + k] = v

    _append_node_config(model, "")
    for name, node in find_children(model, torch.nn.Module, layer_filter):
        _append_node_config(node, name + ".")

    key_map = key_map or {}
    keys_to_replace = []
    for k, v in layer_config.items():
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, k)) != k:
                keys_to_replace.append((k, new_key))
    for old_key, new_key in keys_to_replace:
        layer_config[new_key] = layer_config.pop(old_key)
    return layer_config


def to_ctype(value: Any) -> Tuple[str, Any]:
    """Transform python type to ctype.

    Note: we always use little-endian and 8-byte types.
    This make the format independent of the current platform.

    :params value:
        value to cast into ctype

    :returns:
        A tuple of ctype and cvalue.
    """
    if isinstance(value, int):
        return ("<q", value)
    if isinstance(value, float):
        return ("<d", value)
    if isinstance(value, bool):
        return ("<q", value)
    if isinstance(value, Enum):
        return ("<q", value.value)
    if isinstance(value, tuple) and len(value) == 1:
        return to_ctype(value[0])
    if isinstance(value, str) and len(value) < 8:
        value = bytes(value, "ascii")
        if len(value) < 8:
            value = value + (8 - len(value)) * b"\0"
        return ("8s", value)

    raise ValueError(f"Unsupported type {type(value)}")


def get_cpp_type(value: Any) -> str:
    """Return equivalent cpp type in string format

    :params value:
        value to cast into ctype

    :returns:
        str containing cpp type
    """
    # used to have compatibility between types
    try:
        ctype, _ = to_ctype(value)
    except ValueError as e:
        return f"// Error: {e}"

    if ctype == "i":
        return "std::int32_t"
    if ctype == "l":
        return "std::int64_t"
    if ctype == "f":
        return "float"
    if ctype == "d":
        return "double"
    if ctype == "?":
        return "bool"

    raise RuntimeError(
        f"Should not have reached this part." f"Missing cpp translation for {ctype}"
    )


def generate_hparams_struct(
    hparams: Dict[str, Any],
    struct_name: str,
) -> str:
    """Generate a c++ struct to hold the model hyper-parameters.

    :param hparams:
        Flattened config of the model.
    :param struct_name:
        Name of the generated struct.
    """
    struct = f"struct {struct_name} {{"
    fields = [f"    {get_cpp_type(value)} {key};" for key, value in hparams.items()]
    struct = "\n".join([struct] + fields + ["};\n"])

    valid_fields = [
        key for key, value in hparams.items() if "Error" not in get_cpp_type(value)
    ]
    read_struct = f"void read_{struct_name}({struct_name}& out, std::ifstream &fin) {{"
    read_fields = [
        f"    fin.read((char*) &out.{field}, sizeof(out.{field}));"
        for field in valid_fields
    ]
    read_struct = "\n".join([read_struct] + read_fields + ["};\n"])

    return "\n".join([struct, read_struct])


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(convert_model)
