"""Bounded-memory safetensors loading and Llama 3 rotary metadata.

The production downloader must pin SOURCE_REVISION. This module loads local
files; it does not download files or claim that local weights have been verified
against that revision. Assignment errors leave the supplied part unchanged.
"""

from collections import defaultdict
from collections.abc import Mapping
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch
from torch import nn


SOURCE_MODEL_ID = "dphn/Dolphin3.0-Llama3.2-3B"
SOURCE_REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
TRANSFORMERS_REFERENCE_VERSION = "4.47.1"
MAX_READ_ELEMENTS = 1_048_576


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def load_config(source_dir):
    """Read the local Hugging Face config without importing Transformers."""
    with (Path(source_dir) / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    return config


def _weight_map(source_dir):
    index = source_dir / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as stream:
            weights = json.load(stream).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Safetensors index has no nonempty weight_map")
        return weights
    single = source_dir / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {name: single.name for name in handle.keys()}
    raise FileNotFoundError(f"No safetensors index or single model file in {source_dir}")


def _read_assignment(handle, item, parameter):
    source_name = item["source"]
    source = handle.get_slice(source_name)
    source_shape = tuple(source.get_shape())
    if not source_shape or any(size < 1 for size in source_shape):
        raise ValueError(f"Unsupported empty/scalar source tensor: {source_name}")
    start = 0 if item["row_start"] is None else int(item["row_start"])
    stop = source_shape[0] if item["row_stop"] is None else int(item["row_stop"])
    padding = int(item["pad_rows"])
    if not 0 <= start < stop <= source_shape[0] or padding < 0:
        raise ValueError(f"Invalid source rows/padding for {source_name}")
    columns = math.prod(source_shape[1:])
    rows = stop - start
    shape = tuple(item["reshape"])
    if shape != tuple(parameter.shape) or math.prod(shape) != (rows + padding) * columns:
        raise ValueError(f"Source/destination shape mismatch for {item['destination']}: {source_shape} -> {shape}")
    # Only the final parameter is allocated in full. Temporary source reads are
    # at most MAX_READ_ELEMENTS, except if a single source row is wider than it.
    target = torch.empty(shape, dtype=parameter.dtype, device="cpu")
    target_rows = target.reshape(rows + padding, columns)
    rows_per_read = max(1, MAX_READ_ELEMENTS // columns)
    for offset in range(0, rows, rows_per_read):
        count = min(rows_per_read, rows - offset)
        chunk = source[start + offset:start + offset + count]
        chunk = chunk.to(dtype=parameter.dtype, device="cpu").reshape(count, columns)
        # Check after casting too: finite BF16 values can overflow FP16.
        if not torch.isfinite(chunk).all().item():
            raise ValueError(f"Nonfinite value after conversion in {source_name}, rows {start + offset}:{start + offset + count}")
        target_rows[offset:offset + count].copy_(chunk)
    if padding:
        target_rows[rows:].zero_()
    return target


@torch.no_grad()
def load_part(part, source_dir):
    """Fill every declared parameter, preserving bounded temporary memory.

    Safetensors are opened one shard at a time, with row slices rather than full
    state dictionaries. All final parameter tensors are staged before mutation;
    validation failure therefore leaves the part's parameters and states intact.
    On success parameters are CPU/frozen, states are reset, and eval mode is set.
    """
    source_dir = Path(source_dir).resolve()
    weights = _weight_map(source_dir)
    assignments = list(part.get_source_tensor_assignments())
    parameters = dict(part.named_parameters())
    destinations = [item["destination"] for item in assignments]
    if len(destinations) != len(set(destinations)) or set(destinations) != set(parameters):
        missing = sorted(set(parameters) - set(destinations))
        extra = sorted(set(destinations) - set(parameters))
        raise ValueError(f"Parameter assignments incomplete/duplicated: missing={missing}, extra={extra}")
    by_shard = defaultdict(list)
    for item in assignments:
        name = item["source"]
        if name not in weights:
            raise KeyError(f"Source tensor missing from index: {name}")
        shard = (source_dir / weights[name]).resolve()
        if not shard.is_relative_to(source_dir):
            raise ValueError(f"Shard path leaves source directory: {weights[name]}")
        if not shard.is_file():
            raise FileNotFoundError(f"Required source shard is not complete/available: {shard}")
        by_shard[shard].append(item)
    loaded = {}
    for shard, items in by_shard.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for item in items:
                if item["source"] not in available:
                    raise KeyError(f"Source tensor {item['source']} is absent from indexed shard {shard.name}")
                loaded[item["destination"]] = _read_assignment(handle, item, parameters[item["destination"]])
    for name, tensor in loaded.items():
        module_name, _, leaf = name.rpartition(".")
        module = part.get_submodule(module_name) if module_name else part
        setattr(module, leaf, nn.Parameter(tensor, requires_grad=False))
    if any(parameter.is_meta or parameter.device.type != "cpu" or parameter.requires_grad
           for parameter in part.parameters()):
        raise RuntimeError("Parameter loading left an unmaterialized or unfrozen parameter")
    if hasattr(part, "reset_state"):
        part.reset_state()
    return part.eval()


def make_rope_tables(config, context_length):
    """Return FP16 cos/sin [context_length,head_dim], computed in FP32 on CPU.

    The wavelength interpolation and operation order match Transformers 4.47.1
    _compute_llama3_parameters and LlamaRotaryEmbedding.forward:
    https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/modeling_rope_utils.py
    No theta*factor approximation and no dependence on the export context size.
    """
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    hidden = int(_get(config, "hidden_size"))
    heads = int(_get(config, "num_attention_heads"))
    head_dim = int(_get(config, "head_dim", hidden // heads))
    if head_dim < 2 or head_dim % 2 or _get(config, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("This model rotates the full, even-sized attention head")
    base = float(_get(config, "rope_theta", 10000.0))
    if not math.isfinite(base) or base <= 0:
        raise ValueError("rope_theta must be finite and positive")
    scaling = _get(config, "rope_scaling") or {}
    rope_type = scaling.get("rope_type", scaling.get("type", "default"))
    if rope_type not in {"default", "llama3"}:
        raise ValueError(f"Unsupported RoPE type for this exporter: {rope_type}")
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64, device="cpu").float() / head_dim
        inverse = 1.0 / (base ** exponents)
        if rope_type == "llama3":
            factor = float(scaling["factor"])
            low = float(scaling["low_freq_factor"])
            high = float(scaling["high_freq_factor"])
            original = float(scaling["original_max_position_embeddings"])
            if not all(math.isfinite(x) for x in (factor, low, high, original)) or not (factor > 0 and original > 0 and high > low > 0):
                raise ValueError("Invalid Llama 3 RoPE scaling parameters")
            low_wavelength, high_wavelength = original / low, original / high
            wavelength = 2 * math.pi / inverse
            scaled = torch.where(wavelength > low_wavelength, inverse / factor, inverse)
            smooth = (original / wavelength - low) / (high - low)
            interpolated = (1 - smooth) * scaled / factor + smooth * scaled
            middle = ~(wavelength < high_wavelength) * ~(wavelength > low_wavelength)
            inverse = torch.where(middle, interpolated, scaled)
        positions = torch.arange(context_length, dtype=torch.int64, device="cpu").float()
        frequencies = (inverse[None, :, None] @ positions[None, None, :]).transpose(1, 2)[0]
        doubled = torch.cat((frequencies, frequencies), dim=-1)
        return doubled.cos().to(torch.float16), doubled.sin().to(torch.float16)
