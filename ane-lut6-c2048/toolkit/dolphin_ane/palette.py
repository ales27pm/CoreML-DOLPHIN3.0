"""Deterministic FP16 palettes for Core ML's public custom-LUT API.

The default ``palette(weight, bits=6)`` uses Core ML Tools 9's bundled
``kmeans1d`` solver: globally optimal weighted one-dimensional k-means on the
sorted unique FP16 values, with occurrence counts as weights. Each invocation
handles one output-channel group of 16; 64 FP16 centers produce uint6 indices.
There is no fallback to a local Lloyd fit or to a lower-MSE candidate.

The legacy ``palette4`` below is retained solely to reproduce the rejected
four-bit export and its evidence. It has poor downstream accuracy on Dolphin:
rare, activation-important weight tails are lost even when average weight MSE
looks small. ``palette(weight, bits=4)`` intentionally reproduces that legacy
algorithm. Neither bit width establishes Apple runtime or ANE performance.

Legacy algorithm details:

``palette4`` fits 16 centroids to the exact histogram of FP16 weight values.
Weighted one-dimensional Lloyd iterations use cumulative interval statistics;
their cost depends on occupied FP16 values, not repeated weight samples. This
is a local Lloyd solution, not a globally optimal k-means solver. Weighted
quantile initialization and a maximum of 30 updates make work deterministic.

After FP16 rounding, the candidate with the smaller weighted squared error is
selected between the fitted palette and a uniform min/max palette. This bounds
per-group weight MSE by that uniform baseline; it does not establish downstream
model quality. No calibration data or task-aware optimization is performed.

Usage with coremltools 9.0 (do not supply ``nbits`` in custom mode)::

    OpPalettizerConfig(mode="custom", lut_function=palette4,
                      granularity="per_grouped_channel", group_size=16)

Core ML Tools infers four index bits from the 16-entry LUT. The callback is
called once per group; channel dimensions must already be divisible by 16.
"""

import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time


PALETTE_ALGORITHMS = {
    6: "Core ML Tools 9 bundled kmeans1d: globally optimal weighted scalar 1D k-means over unique FP16 values; 64 centers; no fallback",
    4: "Legacy deterministic histogram-weighted 1D Lloyd, <=30 iterations, with lower-MSE uniform fallback; rejected model quality",
}


def palette(weight, bits=6):
    """Return a scalar FP16 LUT and flat uint8 indices for one group.

    Six bits is the release default. The backend's sample-count weights preserve
    the original objective without expanding duplicate FP16 values. Its maximum
    temporary input is one group, not the full tensor. FP16 center rounding and
    original solver labels match Core ML Tools' own k-means implementation.
    Four bits selects the preserved legacy callback for evidence reproduction.
    """
    if bits == 4:
        return palette4(weight)
    if bits != 6:
        raise ValueError("Supported palette bit widths are 4 and 6")
    if not isinstance(weight, np.ndarray) or weight.dtype != np.dtype(np.float16):
        raise TypeError("palette requires a NumPy array with dtype float16")
    if weight.size == 0:
        raise ValueError("palette cannot quantize an empty weight array")
    if not np.isfinite(weight).all():
        raise ValueError("palette requires finite FP16 weights; found NaN or infinity")
    # Force the bundled solver for tiny tests as well as real groups. The stock
    # helper otherwise selects sklearn below 10,000 samples. No silent fallback.
    from coremltools.models.neural_network.quantization_utils import (
        _get_kmeans_lookup_table_and_weight,
    )
    lut, indices = _get_kmeans_lookup_table_and_weight(
        bits, np.ascontiguousarray(weight), force_kmeans1d=True)
    lut = np.asarray(lut, dtype=np.float16).reshape(1 << bits)
    indices = np.asarray(indices, dtype=np.uint8).reshape(-1)
    if indices.size != weight.size or np.any(indices >= (1 << bits)) or not np.isfinite(lut).all():
        raise RuntimeError("Invalid result from bundled weighted kmeans1d solver")
    return lut, indices


def palette6(weight):
    """Default 64-center globally optimal weighted 1D k-means callback."""
    return palette(weight, bits=6)


def fp16_group_digest(weight):
    """Shape-independent identity of one finite FP16 group in C order."""
    if not isinstance(weight, np.ndarray) or weight.dtype != np.dtype(np.float16):
        raise TypeError("Palette cache requires FP16 NumPy arrays")
    contiguous = np.ascontiguousarray(weight)
    return hashlib.sha256(memoryview(contiguous).cast("B")).digest()


class CachedPalettes:
    """Fit groups concurrently; serve the exact results to a serial callback.

    Input arrays are never retained. At most eight groups are being fitted at
    once. Cache identity depends only on the FP16 bytes, so equivalent rank-two
    and rank-four views have the same result. Reference counts preserve repeated
    groups, and results are discarded after their final expected use. Cache
    misses fail explicitly; they never silently change the fitting policy.
    """

    def __init__(self, bits=6, workers=4, max_bytes=512 * 1024 * 1024):
        if bits not in (4, 6) or workers < 1:
            raise ValueError("Cached palettes need bits 4 or 6 and positive workers")
        self.bits, self.workers = bits, workers
        self.batch_size = min(8, 2 * workers)
        self.max_bytes = max_bytes
        self._entries = {}
        self._remaining = Counter()
        self.group_count = self.unique_groups = self.hits = self.misses = 0
        self.current_bytes = self.peak_bytes = 0
        self.fit_seconds = 0.0

    def prepare(self, named_weights, progress=None):
        started = time.monotonic()
        pending, pending_keys = [], set()

        def fit(item):
            key, group = item
            return key, palette(group, bits=self.bits)

        def flush(executor):
            for key, result in executor.map(fit, pending):
                size = sum(array.nbytes for array in result)
                if self.current_bytes + size > self.max_bytes:
                    raise MemoryError("Palette cache budget exceeded; use fewer layers or --palette-workers 1")
                self._entries[key] = result
                self.unique_groups += 1
                self.current_bytes += size
                self.peak_bytes = max(self.peak_bytes, self.current_bytes)
            pending.clear()
            pending_keys.clear()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for name, weight in named_weights:
                if weight.ndim < 2 or weight.shape[0] % 16:
                    raise ValueError(f"Expected full groups of 16 output channels: {name}")
                for row in range(0, weight.shape[0], 16):
                    group = weight[row:row + 16]
                    key = fp16_group_digest(group)
                    self.group_count += 1
                    self._remaining[key] += 1
                    if key not in self._entries and key not in pending_keys:
                        pending.append((key, group))
                        pending_keys.add(key)
                    if len(pending) >= self.batch_size:
                        flush(executor)
                flush(executor)
                if progress is not None:
                    progress(name, self.group_count, self.peak_bytes)
        self.fit_seconds += time.monotonic() - started
        return self

    def __call__(self, weight):
        key = fp16_group_digest(weight)
        result = self._entries.get(key)
        if result is None or self._remaining[key] <= 0:
            self.misses += 1
            raise RuntimeError("Unexpected FP16 group in precomputed palette callback")
        if result[1].size != weight.size:
            raise RuntimeError("Palette cache index count differs from callback input")
        self.hits += 1
        self._remaining[key] -= 1
        if self._remaining[key] == 0:
            self.current_bytes -= sum(array.nbytes for array in result)
            del self._entries[key]
            del self._remaining[key]
        return result

    def assert_consumed(self):
        if self._entries or self._remaining or self.hits != self.group_count:
            raise RuntimeError("Not every precomputed palette group was consumed by Core ML")

    def report(self):
        return {"workers": self.workers, "batch_size": self.batch_size,
                "group_count": self.group_count, "unique_groups": self.unique_groups,
                "cache_hits": self.hits, "cache_misses": self.misses,
                "remaining_groups": sum(self._remaining.values()),
                "peak_cache_bytes": self.peak_bytes, "fit_seconds": self.fit_seconds,
                "mechanism": "bounded thread fitting; byte-addressed CUSTOM callback with reference counts"}


def _evaluate(values, counts, centers):
    """Nearest representable centers; exact midpoint ties choose the lower."""
    lut = np.sort(np.asarray(centers, dtype=np.float16))
    widened = lut.astype(np.float64)
    boundaries = (widened[:-1] + widened[1:]) * 0.5
    labels = np.searchsorted(boundaries, values, side="left")
    error = np.sum(counts * np.square(values - widened[labels]), dtype=np.float64)
    return float(error), lut, labels.astype(np.uint8)


def _lloyd(values, counts):
    cumulative = np.cumsum(counts, dtype=np.float64)
    quantiles = (np.arange(16, dtype=np.float64) + 0.5) * cumulative[-1] / 16
    centers = values[np.searchsorted(cumulative, quantiles, side="left")].copy()
    prefix_mass = np.concatenate(([0.0], cumulative))
    prefix_sum = np.concatenate(([0.0], np.cumsum(values * counts, dtype=np.float64)))
    previous_edges = None
    for _ in range(30):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        edges = np.concatenate(([0], np.searchsorted(values, boundaries, side="right"),
                                [values.size]))
        if previous_edges is not None and np.array_equal(edges, previous_edges):
            break
        mass = np.diff(prefix_mass[edges])
        weighted_sum = np.diff(prefix_sum[edges])
        centers = np.sort(np.divide(weighted_sum, mass, out=centers.copy(), where=mass > 0))
        previous_edges = edges
    # Weighted means of finite FP16 values are bounded by the input range.
    return np.clip(centers, values[0], values[-1])


def palette4(weight):
    """Return ``(FP16 lut[16], flat uint8 indices)`` without mutating input.

    Requires a nonempty finite FP16 NumPy array. Constants and groups containing
    at most 16 distinct FP16 bit patterns are represented exactly. Repeated
    centers and signed zeros are valid; nonfinite values fail before fitting.
    A contiguous copy is made only when needed for the bit-pattern histogram.
    """
    if not isinstance(weight, np.ndarray) or weight.dtype != np.dtype(np.float16):
        raise TypeError("palette4 requires a NumPy array with dtype float16")
    if weight.size == 0:
        raise ValueError("palette4 cannot quantize an empty weight array")
    flat = np.ascontiguousarray(weight).reshape(-1)
    bits = flat.view(np.uint16)
    if np.any((bits & np.uint16(0x7C00)) == np.uint16(0x7C00)):
        raise ValueError("palette4 requires finite FP16 weights; found NaN or infinity")

    histogram = np.bincount(bits, minlength=65536)
    present = np.flatnonzero(histogram).astype(np.uint16)
    values = present.view(np.float16).astype(np.float64)
    order = np.argsort(values, kind="stable")
    present, values = present[order], values[order]
    counts = histogram[present].astype(np.float64)

    if values.size <= 16:
        lut = np.full(16, values[-1], dtype=np.float16)
        lut[:values.size] = values.astype(np.float16)
        labels = np.arange(values.size, dtype=np.uint8)
    else:
        fitted = _evaluate(values, counts, _lloyd(values, counts))
        uniform = _evaluate(values, counts, np.linspace(values[0], values[-1], 16))
        _, lut, labels = fitted if fitted[0] <= uniform[0] else uniform

    lookup = np.zeros(65536, dtype=np.uint8)
    lookup[present] = labels
    return lut, lookup[bits]
