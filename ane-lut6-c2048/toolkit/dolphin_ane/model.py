"""Fixed-token Llama parts for an iOS 18 stateful Core ML export.

All floating inputs/parameters use the selected dtype (FP16 by default). Hidden
states use BC1S, and each invocation consumes exactly one token. The host supplies
the *correct source model's* rotary values and masks; there is no approximate
RoPE implementation here. ANE placement still requires validation on Apple HW.
"""

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _assignment(destination, source, shape, row_start=None, row_stop=None, pad_rows=0):
    """Slice source rows, append zero rows, then reshape to destination shape."""
    return dict(destination=destination, source=source, row_start=row_start,
                row_stop=row_stop, pad_rows=pad_rows, reshape=tuple(shape))


class RMSNorm(nn.Module):
    """RMS normalization through a FP16 layer_norm op, without explicit squares.

    The concat(x, -x) identity is also used by ANEMLL's LlamaRMSNorm:
    https://github.com/Anemll/Anemll/blob/main/anemll/models/llama_model.py
    Its mean is zero and its mean square equals that of x. This preserves epsilon;
    actual finite-precision parity and ANE lowering must be measured separately.
    """

    def __init__(self, hidden_size, eps=1e-5, dtype=torch.float16):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.hidden_size, dtype=dtype))

    def forward(self, hidden_states):
        channels_last = hidden_states.permute(0, 2, 3, 1)
        mirrored = torch.cat((channels_last, -channels_last), dim=-1)
        normalized = F.layer_norm(mirrored, (2 * self.hidden_size,), eps=self.eps)
        normalized = normalized[..., :self.hidden_size] * self.weight
        return normalized.permute(0, 3, 1, 2)


class EmbeddingPart(nn.Module):
    """inputIds int32 [1,1] -> hiddenStates [1,hidden_size,1,1]."""

    def __init__(self, config, dtype=torch.float16, pad_multiple=16):
        super().__init__()
        self.vocab_size = int(_get(config, "vocab_size"))
        self.hidden_size = int(_get(config, "hidden_size"))
        self.padded_vocab_size = _round_up(self.vocab_size, pad_multiple)
        self.embed_tokens = nn.Embedding(self.padded_vocab_size, self.hidden_size, dtype=dtype)

    def forward(self, inputIds):
        return self.embed_tokens(inputIds).transpose(1, 2).unsqueeze(2)

    def get_source_tensor_assignments(self):
        return [_assignment("embed_tokens.weight", "model.embed_tokens.weight",
                            self.embed_tokens.weight.shape, 0, self.vocab_size,
                            self.padded_vocab_size - self.vocab_size)]


class _Attention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        hidden = int(_get(config, "hidden_size"))
        self.num_heads = int(_get(config, "num_attention_heads"))
        self.num_kv_heads = int(_get(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(_get(config, "head_dim", hidden // self.num_heads))
        if self.head_dim % 2 or self.num_heads % self.num_kv_heads:
            raise ValueError("RoPE needs an even head_dim; query heads must group evenly over KV heads")
        if self.num_heads * self.head_dim != hidden:
            raise ValueError("This export requires hidden_size == num_attention_heads * head_dim")
        if _get(config, "attention_bias", False):
            raise ValueError("Biased attention projections are not supported by this Dolphin export")
        self.groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Conv2d(hidden, self.num_heads * self.head_dim, 1, bias=False, dtype=dtype)
        self.k_proj = nn.Conv2d(hidden, self.num_kv_heads * self.head_dim, 1, bias=False, dtype=dtype)
        self.v_proj = nn.Conv2d(hidden, self.num_kv_heads * self.head_dim, 1, bias=False, dtype=dtype)
        self.o_proj = nn.Conv2d(self.num_heads * self.head_dim, hidden, 1, bias=False, dtype=dtype)

    def _rotate(self, head, cosine, sine):
        half = self.head_dim // 2
        rotated_half = torch.cat((-head[:, half:, :, :], head[:, :half, :, :]), dim=1)
        return head * cosine + rotated_half * sine

    def forward(self, hidden_states, old_keys, old_values, ropeCos, ropeSin, writeMask, causalMask):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query_heads = [self._rotate(query[:, h*self.head_dim:(h+1)*self.head_dim, :, :],
                                   ropeCos, ropeSin) for h in range(self.num_heads)]
        key_heads = [self._rotate(key[:, h*self.head_dim:(h+1)*self.head_dim, :, :],
                                 ropeCos, ropeSin) for h in range(self.num_kv_heads)]
        new_key = torch.cat(key_heads, dim=1)
        # Masks are a host-validated one-hot write position and a causal prefix.
        keys = old_keys * (1.0 - writeMask) + new_key * writeMask
        values = old_values * (1.0 - writeMask) + value * writeMask
        heads = []
        for h in range(self.num_heads):
            kv = h // self.groups
            begin, end = kv * self.head_dim, (kv + 1) * self.head_dim
            q = query_heads[h].permute(0, 2, 3, 1)
            k = keys[:, begin:end, :, :].permute(0, 2, 1, 3)
            v = values[:, begin:end, :, :].permute(0, 2, 3, 1)
            # Pre-scaling avoids an unnecessarily large intermediate dot product.
            scores = torch.matmul(q * self.scale, k) + causalMask
            probabilities = F.softmax(scores, dim=-1)
            heads.append(torch.matmul(probabilities, v).permute(0, 3, 1, 2))
        return self.o_proj(torch.cat(heads, dim=1)), keys, values


class _MLP(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        hidden = int(_get(config, "hidden_size"))
        intermediate = int(_get(config, "intermediate_size"))
        if _get(config, "mlp_bias", False) or _get(config, "hidden_act", "silu") != "silu":
            raise ValueError("This export requires a bias-free SiLU gated MLP")
        self.gate_proj = nn.Conv2d(hidden, intermediate, 1, bias=False, dtype=dtype)
        self.up_proj = nn.Conv2d(hidden, intermediate, 1, bias=False, dtype=dtype)
        self.down_proj = nn.Conv2d(intermediate, hidden, 1, bias=False, dtype=dtype)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _DecoderLayer(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        hidden = int(_get(config, "hidden_size"))
        eps = float(_get(config, "rms_norm_eps", 1e-5))
        self.input_layernorm = RMSNorm(hidden, eps, dtype)
        self.self_attn = _Attention(config, dtype)
        self.post_attention_layernorm = RMSNorm(hidden, eps, dtype)
        self.mlp = _MLP(config, dtype)

    def forward(self, hidden_states, keys, values, ropeCos, ropeSin, writeMask, causalMask):
        attention, keys, values = self.self_attn(self.input_layernorm(hidden_states),
                                                keys, values, ropeCos, ropeSin, writeMask, causalMask)
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, keys, values


class BlockPart(nn.Module):
    """A contiguous layer range [start_layer,end_layer), with two local states.

    keyCache/valueCache: [local_layers, KV_heads*head_dim, 1, context_length].
    States are allocated on CPU even under a meta-device parameter constructor.
    This inference-only module must run under torch.no_grad(), with no overlapping
    invocations. Its two buffers are exported as ct.StateType by the caller.
    """

    def __init__(self, config, start_layer, end_layer, context_length=2048, dtype=torch.float16):
        super().__init__()
        self.start_layer, self.end_layer = int(start_layer), int(end_layer)
        total_layers = int(_get(config, "num_hidden_layers"))
        if not 0 <= self.start_layer < self.end_layer <= total_layers or context_length < 1:
            raise ValueError("Invalid layer range or context length")
        self.context_length = int(context_length)
        self.layers = nn.ModuleList([_DecoderLayer(config, dtype)
                                     for _ in range(self.end_layer - self.start_layer)])
        attention = self.layers[0].self_attn
        state_shape = (len(self.layers), attention.num_kv_heads * attention.head_dim, 1, self.context_length)
        self.register_buffer("keyCache", torch.zeros(state_shape, dtype=dtype, device="cpu"))
        self.register_buffer("valueCache", torch.zeros(state_shape, dtype=dtype, device="cpu"))

    def forward(self, hiddenStates, ropeCos, ropeSin, writeMask, causalMask):
        for local_index, layer in enumerate(self.layers):
            hiddenStates, keys, values = layer(hiddenStates,
                self.keyCache[local_index:local_index+1], self.valueCache[local_index:local_index+1],
                ropeCos, ropeSin, writeMask, causalMask)
            self.keyCache[local_index:local_index+1].copy_(keys)
            self.valueCache[local_index:local_index+1].copy_(values)
        return hiddenStates

    @torch.no_grad()
    def reset_state(self):
        self.keyCache.zero_()
        self.valueCache.zero_()

    def get_source_tensor_assignments(self):
        assignments = []
        for local_index, layer in enumerate(self.layers):
            for name, parameter in layer.named_parameters():
                assignments.append(_assignment(f"layers.{local_index}.{name}",
                    f"model.layers.{self.start_layer + local_index}.{name}", parameter.shape))
        return assignments


class HeadPart(nn.Module):
    """Final RMSNorm plus split 1x1 vocabulary projections, with no lost tokens.

    Dolphin's eight true output widths are (16032,)*7 + (16034,). The final
    projection has 16048 weight rows for LUT group-16 padding; unused logits are
    removed inside the graph. The host samples over all returned true widths.
    """

    def __init__(self, config, dtype=torch.float16, num_splits=8, pad_multiple=16):
        super().__init__()
        hidden = int(_get(config, "hidden_size"))
        self.vocab_size = int(_get(config, "vocab_size"))
        if self.vocab_size < 1 or num_splits < 1 or pad_multiple < 1:
            raise ValueError("Vocabulary, split count, and padding multiple must be positive")
        self.num_splits = min(int(num_splits), self.vocab_size)
        base = (self.vocab_size // self.num_splits // pad_multiple) * pad_multiple
        if base == 0:  # Permit small synthetic vocabularies in numerical tests.
            base = self.vocab_size // self.num_splits
        self.head_widths = (base,) * (self.num_splits - 1) + (self.vocab_size - base * (self.num_splits - 1),)
        self.padded_head_widths = tuple(_round_up(width, pad_multiple) for width in self.head_widths)
        self.head_offsets = tuple(sum(self.head_widths[:i]) for i in range(self.num_splits))
        self.norm = RMSNorm(hidden, float(_get(config, "rms_norm_eps", 1e-5)), dtype)
        self.lm_heads = nn.ModuleList([nn.Conv2d(hidden, width, 1, bias=False, dtype=dtype)
                                       for width in self.padded_head_widths])
        self.source_head_name = "model.embed_tokens.weight" if _get(config, "tie_word_embeddings", False) else "lm_head.weight"

    def forward(self, hiddenStates):
        normalized = self.norm(hiddenStates)
        return tuple(head(normalized)[:, :width, :, :]
                     for head, width in zip(self.lm_heads, self.head_widths))

    def get_source_tensor_assignments(self):
        result = [_assignment("norm.weight", "model.norm.weight", self.norm.weight.shape)]
        for index, (head, offset, width, padded) in enumerate(zip(
                self.lm_heads, self.head_offsets, self.head_widths, self.padded_head_widths)):
            result.append(_assignment(f"lm_heads.{index}.weight", self.source_head_name,
                                      head.weight.shape, offset, offset + width, padded - width))
        return result
