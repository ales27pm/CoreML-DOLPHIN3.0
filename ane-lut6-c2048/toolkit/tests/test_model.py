"""Independent numerical checks against Transformers 4.47.1 eager Llama.

These are CPU PyTorch tests of the exported architecture, not Core ML runtime
or ANE placement tests. No downloaded model weights are required.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dolphin_ane.model import BlockPart, EmbeddingPart, HeadPart


CONTEXT = 40


@pytest.fixture
def reference():
    torch.manual_seed(2718)
    torch.set_num_threads(2)
    config = LlamaConfig(
        vocab_size=259, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=131072, rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 32.0,
                      "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                      "original_max_position_embeddings": 8192},
        rms_norm_eps=1e-5, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=[257], pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float().eval()
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.6, 1.7, 64))
    return model


def load_reference(part, reference):
    """Load the declared tensor mapping, while expected results come from HF."""
    source = reference.state_dict()
    mapped = {}
    for item in part.get_source_tensor_assignments():
        value = source[item["source"]].detach()
        if item["row_start"] is not None:
            value = value[item["row_start"]:item["row_stop"]]
        if item["pad_rows"]:
            padding = value.new_zeros((item["pad_rows"], *value.shape[1:]))
            value = torch.cat((value, padding), dim=0)
        mapped[item["destination"]] = value.reshape(item["reshape"]).clone()
    missing, unexpected = part.load_state_dict(mapped, strict=False)
    assert not unexpected
    assert set(missing) <= {"keyCache", "valueCache"}
    return part.eval()


def make_parts(reference, chunks):
    config = reference.config
    embedding = load_reference(EmbeddingPart(config, dtype=torch.float32), reference)
    boundaries = [(0, 2)] if chunks == 1 else [(0, 1), (1, 2)]
    blocks = [load_reference(BlockPart(config, start, end, CONTEXT, torch.float32), reference)
              for start, end in boundaries]
    head = load_reference(HeadPart(config, dtype=torch.float32), reference)
    return embedding, blocks, head


def position_inputs(reference, position):
    # Obtain RoPE from the independent HF implementation, not exporter helpers.
    cos, sin = reference.model.rotary_emb(torch.zeros(1, 1, 64), torch.tensor([[position]]))
    write = torch.zeros(1, 1, 1, CONTEXT)
    write[..., position] = 1
    causal = torch.full((1, 1, 1, CONTEXT), -65504.0)
    causal[..., :position + 1] = 0
    return cos.reshape(1, 16, 1, 1), sin.reshape(1, 16, 1, 1), write, causal


@torch.no_grad()
def consume(reference, parts, tokens):
    embedding, blocks, head = parts
    outputs = []
    for position, token in enumerate(tokens):
        hidden = embedding(torch.tensor([[token]], dtype=torch.int32))
        for block in blocks:
            hidden = block(hidden, *position_inputs(reference, position))
        outputs.append(torch.cat(head(hidden), dim=1).flatten())
    return torch.stack(outputs)


@pytest.mark.parametrize("chunks", [1, 2])
@pytest.mark.parametrize("traced", [False, True])
@torch.no_grad()
def test_streaming_matches_hf_full_prefix_at_multiple_positions(reference, chunks, traced):
    parts = make_parts(reference, chunks)
    if traced:
        hidden = torch.zeros(1, 64, 1, 1)
        blocks = [torch.jit.trace(block, (hidden, *position_inputs(reference, 0)),
                                  check_trace=False) for block in parts[1]]
        for block in blocks:
            block.keyCache.zero_()
            block.valueCache.zero_()
        parts = parts[0], blocks, parts[2]
    # Exercise position 32 and both vocabulary rows lost by stock floor splitting.
    tokens = [1, 257, 258, 7] + [(i * 37 + 11) % 259 for i in range(32)]
    actual = consume(reference, parts, tokens)
    expected = torch.stack([
        reference(torch.tensor([tokens[:i + 1]]), use_cache=False).logits[0, -1]
        for i in range(len(tokens))
    ])
    assert actual.shape == (len(tokens), 259)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


@torch.no_grad()
def test_final_vocabulary_rows_and_padded_logits(reference):
    # Force each of the final two IDs to be the winner on a different input.
    weight = reference.model.embed_tokens.weight
    weight.zero_()
    weight[257, 0], weight[258, 1] = 4.0, 5.0
    embedding, _, head = make_parts(reference, 2)
    for token in (257, 258):
        actual_embedding = embedding(torch.tensor([[token]], dtype=torch.int32))
        torch.testing.assert_close(actual_embedding.flatten(), weight[token])
    assert sum(head.head_widths) == 259
    assert all(width % 16 == 0 for width in head.padded_head_widths)
    # Give padded rows huge weights: graph slicing must still exclude them.
    for projection, true_width in zip(head.lm_heads, head.head_widths):
        projection.weight[true_width:].fill_(10000.0)
    for axis, expected_id in ((0, 257), (1, 258)):
        hidden = torch.zeros(1, 64, 1, 1)
        hidden[0, axis, 0, 0] = 1.0
        actual = torch.cat(head(hidden), dim=1).flatten()
        expected = reference.lm_head(reference.model.norm(hidden.flatten()))
        assert actual.numel() == 259
        assert actual.argmax().item() == expected_id
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


@torch.no_grad()
def test_state_reset_and_independent_conversations(reference):
    parts = make_parts(reference, 2)
    consume(reference, parts, [1, 257, 22, 258])
    snapshots = [(b.keyCache.clone(), b.valueCache.clone()) for b in parts[1]]
    other = make_parts(reference, 2)
    for block in other[1]:
        assert torch.count_nonzero(block.keyCache) == 0
        assert torch.count_nonzero(block.valueCache) == 0
    consume(reference, other, [1, 11, 13])
    for block, (keys, values) in zip(parts[1], snapshots):
        torch.testing.assert_close(block.keyCache, keys, rtol=0, atol=0)
        torch.testing.assert_close(block.valueCache, values, rtol=0, atol=0)
        block.reset_state()
        assert torch.count_nonzero(block.keyCache) == 0
        assert torch.count_nonzero(block.valueCache) == 0
    tokens = [1, 8, 258]
    actual = consume(reference, parts, tokens)[-1]
    expected = reference(torch.tensor([tokens]), use_cache=False).logits[0, -1]
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


@torch.no_grad()
def test_mask_excludes_unwritten_future_cache(reference):
    parts = make_parts(reference, 2)
    for block in parts[1]:
        block.keyCache.normal_(mean=20.0, std=2.0)
        block.valueCache.normal_(mean=30.0, std=2.0)
    tokens = [1, 257, 12, 258, 13]
    actual = consume(reference, parts, tokens)
    expected = reference(torch.tensor([tokens]), use_cache=False).logits[0]
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
