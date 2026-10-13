"""Custom palette correctness and Core ML API integration; no Apple runtime."""

import numpy as np
import pytest
import torch

from dolphin_ane.palette import palette, palette4, palette6


def uniform_error(weight):
    """Independent nearest-palette baseline, computed using all samples."""
    values = weight.astype(np.float64).reshape(-1)
    lut = np.linspace(values.min(), values.max(), 16).astype(np.float16).astype(np.float64)
    distance = np.abs(values[:, None] - lut[None, :])
    return np.mean(np.square(distance.min(axis=1)))


@pytest.mark.parametrize("constant", [0.0, -0.0, 0.125, -65504.0, 65504.0])
@pytest.mark.parametrize("bits", [4, 6])
def test_constants_are_exact_and_finite(constant, bits):
    value = np.full((16, 128, 1, 1), constant, dtype=np.float16)
    lut, indices = palette(value, bits=bits)
    assert lut.shape == (1 << bits,) and lut.dtype == np.float16
    assert indices.shape == (value.size,) and indices.dtype == np.uint8
    assert np.isfinite(lut).all()
    np.testing.assert_array_equal(lut[indices], value.reshape(-1))


def test_small_discrete_alphabet_and_padding_zeros_are_lossless():
    values = np.array([-65504, -2, -0.0, 0.0, 1e-7, 0.0625, 1, 65504], np.float16)
    source = np.tile(values, 4096).reshape(16, -1)
    source[8:] = 0
    lut, indices = palette4(source)
    np.testing.assert_array_equal(lut[indices].reshape(source.shape), source)


@pytest.mark.parametrize("distribution", ["normal", "bf16", "outliers", "zeros"])
def test_reconstruction_error_determinism_and_nonmutation(distribution):
    rng = np.random.default_rng(9871)
    source = rng.normal(0, 0.04, (16, 3072)).astype(np.float32)
    if distribution == "bf16":
        source = torch.from_numpy(source).bfloat16().float().numpy()
    elif distribution == "outliers":
        source[0, :4] = [-100, -2, 3, 60]
    elif distribution == "zeros":
        source[:12] = 0
    source = source.astype(np.float16)
    original = source.copy()
    lut, indices = palette4(source)
    again = palette4(source)
    assert indices.max() < 16
    assert np.isfinite(lut).all()
    np.testing.assert_array_equal(lut.view(np.uint16), again[0].view(np.uint16))
    np.testing.assert_array_equal(indices, again[1])
    np.testing.assert_array_equal(source.view(np.uint16), original.view(np.uint16))
    recovered = lut[indices].reshape(source.shape).astype(np.float64)
    mse = np.mean(np.square(recovered - source.astype(np.float64)))
    assert mse <= uniform_error(source) + 1e-14
    if distribution in ("normal", "bf16"):
        assert mse / np.mean(np.square(source.astype(np.float64))) < 0.02


def test_noncontiguous_input_preserves_flattening_order():
    source = np.arange(512, dtype=np.float16).reshape(16, 32)[:, ::2]
    lut, indices = palette4(source)
    contiguous = palette4(np.ascontiguousarray(source))
    np.testing.assert_array_equal(lut, contiguous[0])
    np.testing.assert_array_equal(indices, contiguous[1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("callback", [palette4, palette6])
def test_nonfinite_weights_fail(bad, callback):
    with pytest.raises(ValueError, match="finite"):
        callback(np.array([0, bad], dtype=np.float16))


@pytest.mark.parametrize("callback", [palette4, palette6])
def test_empty_and_wrong_dtype_fail(callback):
    with pytest.raises(ValueError, match="empty"):
        callback(np.empty(0, dtype=np.float16))
    with pytest.raises(TypeError, match="float16"):
        callback(np.ones(16, dtype=np.float32))


def test_default_six_bits_is_deterministic_and_leaves_weights_unchanged():
    source = np.random.default_rng(811).normal(0, .03, (16, 3072)).astype(np.float16)
    before = source.copy()
    lut, indices = palette(source)
    explicit = palette6(source)
    assert lut.shape == (64,) and indices.dtype == np.uint8 and indices.max() < 64
    np.testing.assert_array_equal(lut.view(np.uint16), explicit[0].view(np.uint16))
    np.testing.assert_array_equal(indices, explicit[1])
    np.testing.assert_array_equal(source.view(np.uint16), before.view(np.uint16))
    mse = np.square(lut[indices].astype(np.float64) - source.reshape(-1)).mean()
    assert mse / np.square(source.astype(np.float64)).mean() < .001
    with pytest.raises(ValueError, match="4 and 6"):
        palette(source, bits=5)


def test_six_bit_solver_preserves_rare_activation_important_tails():
    # The failed real model had rare down_proj weights on highly active input
    # channels. Average scalar MSE alone missed the resulting output collapse.
    source = np.random.default_rng(444).normal(0, .018, (16, 8192)).astype(np.float16)
    source[0, 289], source[1, 6057], source[2, 289] = .5, -.375, -.625
    activation = np.zeros(8192, dtype=np.float64)
    activation[289], activation[6057] = -15.1875, -10.0234375
    expected = source.astype(np.float64) @ activation
    error = {}
    for bits in (4, 6):
        lut, indices = palette(source, bits=bits)
        reconstructed = lut[indices].reshape(source.shape).astype(np.float64)
        delta = reconstructed @ activation - expected
        error[bits] = float(delta @ delta / (expected @ expected))
    assert error[6] < .005
    assert error[6] < error[4] / 10


def test_coreml_custom_callback_infers_four_bits_for_grouped_conv():
    import coremltools as ct

    torch.manual_seed(111)
    layer = torch.nn.Conv2d(128, 32, 1, bias=False, dtype=torch.float16).eval()
    sample = torch.zeros(1, 128, 1, 1, dtype=torch.float16)
    traced = torch.jit.trace(layer, sample)
    model = ct.convert(traced, inputs=[ct.TensorType(shape=sample.shape, dtype=np.float16)],
                       convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    config = ct.optimize.coreml.OptimizationConfig(global_config=
        ct.optimize.coreml.OpPalettizerConfig(mode="custom", lut_function=palette4,
            granularity="per_grouped_channel", group_size=16))
    compressed = ct.optimize.coreml.palettize_weights(model, config)
    ops = [op for function in compressed._mil_program.functions.values()
           for op in function.operations if op.op_type == "constexpr_lut_to_dense"]
    assert len(ops) == 1
    assert ops[0].indices.dtype.get_bitwidth() == 4
    assert ops[0].lut.shape == (2, 1, 1, 1, 16, 1)
