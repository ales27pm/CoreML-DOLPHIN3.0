"""End-to-end pipeline for converting Dolphin3.0-Llama3.1-8B to a compressed Core ML package.

This script performs the following steps:
    1. Download (or locate) the base model checkpoint from Hugging Face.
    2. Load the tokenizer, configuration, and base PyTorch model.
    3. Merge mandatory LoRA adapters into the base model weights.
    4. Attach an LLM2Vec encoder head for embedding generation.
    5. Build PyTorch wrapper modules for init, decode, and encode passes.
    6. Convert the wrappers into a multifunction Core ML mlprogram.
    7. Apply palettization and linear quantization for W{wbits} compression.
    8. Optionally validate the exported model and clean temporary artifacts.

The resulting `.mlpackage` contains three entry points for chat initialization,
decode with KV-cache, and embedding generation.

The script intentionally avoids placeholders and is designed for production usage
with comprehensive error handling and logging.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import shlex
import statistics
import textwrap
import time


from quantization import (
    NEURAL_ENGINE_GROUP_SIZES,
    SUPPORTED_MIXED_PRECISION_KEYS,
    SUPPORTED_WBITS,
    _cosine_similarity,
    _mixed_precision_arg,
    _resolve_mixed_precision_plan,
    _validate_group_size_for_backend,
    sweep_group_size_arg,
    sweep_wbits_arg,
)


# ---------------------------------------------------------------------------
# Rich console initialisation (installed on-demand)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
except ImportError:  # pragma: no cover - executed only when dependency missing
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

console = Console()


def _module_available(name: str) -> bool:
    """Return True when the module can be imported."""

    return importlib.util.find_spec(name) is not None


def _derive_module_name(package_spec: str) -> str:
    """Return the importable module name for a given pip package spec."""

    base = package_spec.split("[")[0]
    base = base.split("==")[0]
    base = base.split(">=")[0]
    base = base.split("<=")[0]
    return base.replace("-", "_")


def ensure_packages(packages: Iterable[str]) -> None:
    """Ensure that the required Python packages are available with retries."""

    pending: List[Tuple[str, str]] = []
    for pkg in packages:
        module_name = _derive_module_name(pkg)
        if not _module_available(module_name):
            pending.append((pkg, module_name))

    if not pending:
        return

    pkgs_to_install: List[str] = [pkg for pkg, _ in pending]
    modules_to_verify: List[str] = [module for _, module in pending]

    console.print(
        Panel.fit(
            ("[bold yellow]Installing dependencies:[/] " + " ".join(pkgs_to_install)),
            border_style="yellow",
        )
    )

    max_attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        wait_seconds = min(2**attempt, 10)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-input",
            "--upgrade",
            "--progress-bar",
            "off",
            *pkgs_to_install,
        ]
        console.log(
            f"[pip] Attempt {attempt}/{max_attempts}: {' '.join(shlex.quote(part) for part in command)}"
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == max_attempts:
                console.print(
                    Panel.fit(
                        (
                            "[bold red]❌ Dependency installation failed after retries.[/]\n"
                            "Check your network connection or install manually with: "
                            f"{sys.executable} -m pip install {' '.join(pkgs_to_install)}"
                        ),
                        border_style="red",
                    )
                )
                raise RuntimeError("pip could not install required packages") from exc
            console.print(
                Panel.fit(
                    (
                        "[bold yellow]⚠️ pip install failed:[/] "
                        f"{exc}. Retrying in {wait_seconds} seconds…"
                    ),
                    border_style="yellow",
                )
            )
            time.sleep(wait_seconds)
            continue

        missing_after = [
            module for module in modules_to_verify if not _module_available(module)
        ]
        if not missing_after:
            console.print(Panel.fit("[green]Dependencies installed successfully."))
            return

        pkgs_to_install = [pkg for pkg, module in pending if module in missing_after]
        modules_to_verify = missing_after
        console.print(
            Panel.fit(
                (
                    "[bold yellow]⚠️ Some modules remain unavailable after installation:[/] "
                    + ", ".join(modules_to_verify)
                    + ". Retrying…"
                ),
                border_style="yellow",
            )
        )
        time.sleep(wait_seconds)

    if last_error is None:
        raise RuntimeError(
            "Dependencies were installed but modules remain unavailable. "
            "Check that pip is targeting the correct Python environment."
        )
    raise RuntimeError("Dependency installation failed") from last_error


def prepare_tmp_dir(base_path: Path) -> Path:
    """Create an isolated temporary directory under ``base_path`` with validation."""

    resolved_base = base_path.expanduser().resolve()
    if resolved_base.exists() and not resolved_base.is_dir():
        raise RuntimeError(
            f"Temporary path {resolved_base} exists but is not a directory. "
            "Delete it or provide a different --tmp path."
        )

    try:
        resolved_base.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise RuntimeError(
            f"Unable to create temporary root directory {resolved_base}: {exc}"
        ) from exc

    if not (os.access(resolved_base, os.W_OK) and os.access(resolved_base, os.X_OK)):
        raise RuntimeError(
            f"Temporary root {resolved_base} is not writable. Adjust permissions or choose a different path."
        )

    run_dir = (
        resolved_base / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError:  # pragma: no cover - highly unlikely
        run_dir = resolved_base / f"run-{uuid.uuid4().hex}"
        run_dir.mkdir(exist_ok=False)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise RuntimeError(
            f"Unable to create isolated temporary directory at {run_dir}: {exc}"
        ) from exc

    console.print(
        Panel.fit(
            f"[bold green]Working directory prepared:[/] {run_dir}",
            border_style="green",
        )
    )
    return run_dir


def cleanup_tmp_dir(path: Path) -> None:
    """Attempt to remove the temporary directory and report failures."""

    if not path.exists():
        return

    try:
        shutil.rmtree(path)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        console.print(
            Panel.fit(
                (
                    "[bold yellow]⚠️ Unable to fully clean temporary directory:[/] "
                    f"{path} ({exc})"
                ),
                border_style="yellow",
            )
        )
    else:
        console.print("[green]Temporary build directory cleaned up.")


try:  # pragma: no cover - fallback when optional deps are missing
    import torch
except ImportError:  # pragma: no cover
    ensure_packages(["torch>=2.0.0"])
    import torch

try:  # pragma: no cover
    import coremltools as ct
except ImportError:  # pragma: no cover
    ensure_packages(["coremltools>=8.0.0"])
    import coremltools as ct

coreml_version = getattr(ct, "__version__", "0")
try:
    _COREMLTOOLS_MAJOR = int(coreml_version.split(".")[0])
except ValueError:  # pragma: no cover - unexpected version string
    _COREMLTOOLS_MAJOR = 0
if _COREMLTOOLS_MAJOR < 8:  # pragma: no cover - ensures runtime alignment
    raise RuntimeError("coremltools>=8.0 is required for mixed-precision overrides.")

try:  # pragma: no cover
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    ensure_packages(["huggingface_hub"])
    from huggingface_hub import snapshot_download

try:  # pragma: no cover
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    ensure_packages(
        [
            "transformers>=4.44.0",
            "accelerate",
            "sentencepiece",
            "tokenizers",
        ]
    )
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Optional packages are installed on demand later in the workflow.
HAS_UNSLOTH = _module_available("unsloth")
HAS_LLM2VEC = _module_available("llm2vec")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@dataclass
class ShellResult:
    """Container for shell execution results."""

    returncode: int
    command: str


def sh(command: Sequence[str], *, check: bool = True) -> ShellResult:
    """Execute a subprocess without invoking the shell."""

    pretty = " ".join(shlex.quote(part) for part in command)
    console.log(f"[shell] {pretty}")
    result = subprocess.run(list(command), check=check)
    return ShellResult(returncode=result.returncode, command=pretty)


@dataclass(frozen=True)
class GoldenPrompt:
    """Representative prompt exercised during deterministic validation."""

    prompt: str
    max_new_tokens: int = 32


@dataclass(frozen=True)
class QuantizationVariant:
    """Combination of bit-width and group size evaluated during a sweep."""

    wbits: int
    group_size: int
    output_path: Path


@dataclass
class QuantizationResult:
    """Outcome of quantizing and benchmarking a single variant."""

    variant: QuantizationVariant
    size_bytes: Optional[int]
    compression_ratio: Optional[float]
    performance: Optional[Dict[str, Any]]
    performance_error: Optional[str]
    error: Optional[str]
    validation_passed: Optional[bool] = None


GOLDEN_PROMPTS: Sequence[GoldenPrompt] = (
    GoldenPrompt(
        "You are Dolphin, a helpful AI assistant. Explain why deterministic validation matters when exporting Core ML models.",
        max_new_tokens=32,
    ),
    GoldenPrompt(
        "List three legitimate uses for packet capture during ethical security research and mention one required safeguard.",
        max_new_tokens=32,
    ),
    GoldenPrompt(
        "Summarize the Dolphin 3.0 Core ML export workflow in under sixty words.",
        max_new_tokens=32,
    ),
)


EMBEDDING_BENCHMARKS: Sequence[str] = (
    "Legitimate packet capture helps incident responders observe beaconing without disrupting production traffic.",
    "Deterministic validation builds trust in Core ML exports by catching parity gaps before shipping to devices.",
    "Batched LLM2Vec embeddings enable semantic clustering of security advisories for faster triage.",
)


EMBEDDING_COSINE_THRESHOLD = 0.99


def _prepare_prompt_arrays(
    tokenizer: "AutoTokenizer", prompt: str, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return padded input IDs, mask, trimmed prompt IDs, and prompt length."""

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
    )
    ids = encoded["input_ids"].astype(np.int32, copy=False)
    mask = encoded["attention_mask"].astype(np.int32, copy=False)

    prompt_len = int(mask.sum())
    if prompt_len == 0:
        raise ValueError("Prompt tokenization produced an empty sequence.")

    padded_ids = np.zeros((1, seq_len), dtype=np.int32)
    padded_mask = np.zeros((1, seq_len), dtype=np.int32)
    usable = min(seq_len, ids.shape[1])
    padded_ids[0, :usable] = ids[0, :usable]
    padded_mask[0, :usable] = mask[0, :usable]

    effective_len = min(prompt_len, seq_len)
    return (
        padded_ids,
        padded_mask,
        ids[:, :effective_len],
        effective_len,
    )


def _prepare_embedding_arrays(
    tokenizer: "AutoTokenizer", sentence: str, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return padded input ID and mask arrays suitable for Core ML embedding calls."""

    encoded = tokenizer(
        sentence,
        return_tensors="np",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        add_special_tokens=True,
    )
    ids = encoded["input_ids"].astype(np.int32, copy=False)
    mask = encoded["attention_mask"].astype(np.int32, copy=False)
    return ids, mask


def _torch_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:  # pragma: no cover - defensive branch
        return torch.device("cpu")


def _validate_embedding_parity(
    mlmodel: "ct.models.MLModel",
    embedding_module: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    seq_len: int,
    sentences: Sequence[str],
    *,
    minimum_cosine: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """Return cosine similarity metrics for LLM2Vec embeddings."""

    device = _torch_device(embedding_module)
    results: List[Dict[str, Any]] = []
    min_cosine = float("inf")

    for sentence in sentences:
        ids_np, mask_np = _prepare_embedding_arrays(tokenizer, sentence, seq_len)
        ids_tensor = torch.from_numpy(ids_np).to(device=device, dtype=torch.long)
        mask_tensor = torch.from_numpy(mask_np).to(device=device, dtype=torch.long)

        with torch.no_grad():
            reference_embedding = embedding_module.encode(
                input_ids=ids_tensor,
                attention_mask=mask_tensor,
            )

        reference_np = reference_embedding.detach().to("cpu").float().numpy()
        coreml_out = mlmodel.predict(
            {"input_ids": ids_np, "attention_mask": mask_np},
            function_name="encode",
        )
        coreml_np = np.asarray(coreml_out["embedding"], dtype=np.float32)
        cosine = _cosine_similarity(reference_np, coreml_np)
        min_cosine = min(min_cosine, cosine)
        results.append(
            {"sentence": sentence, "cosine": cosine, "pass": cosine >= minimum_cosine}
        )

    if not results:
        raise ValueError(
            "Embedding validation requires at least one benchmark sentence."
        )

    return results, min_cosine


def _run_reference_decode(
    model: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    trimmed_prompt: np.ndarray,
    prompt_len: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Generate deterministic tokens using PyTorch as the golden reference."""

    device = _torch_device(model)
    torch_input = torch.from_numpy(trimmed_prompt[:, :prompt_len]).to(
        device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(torch_input, dtype=torch.long)

    pad_token = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    eos_id = tokenizer.eos_token_id
    if pad_token is None:
        raise ValueError(
            "Tokenizer is missing both pad and EOS token IDs for generation."
        )

    torch.manual_seed(0)
    with torch.no_grad():
        generated = model.generate(
            input_ids=torch_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token,
            eos_token_id=eos_id,
        )

    new_tokens = generated[0, prompt_len : prompt_len + max_new_tokens].tolist()
    golden_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"tokens": new_tokens, "text": golden_text}


def _trim_coreml_cache(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(
            f"Expected rank-4 KV cache tensor, received rank {array.ndim}."
        )
    if array.shape[2] <= 1:
        raise ValueError("KV cache tensor does not have enough timesteps to trim.")
    return np.ascontiguousarray(array[:, :, 1:, :])


def _run_coreml_decode(
    mlmodel: "ct.models.MLModel",
    tokenizer: "AutoTokenizer",
    padded_ids: np.ndarray,
    padded_mask: np.ndarray,
    prompt_len: int,
    seq_len: int,
    num_layers: int,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> Dict[str, Any]:
    """Run deterministic decode against the Core ML model and collect metrics."""

    if seq_len <= 0:
        raise ValueError("Sequence length must be positive for validation.")

    init_start = time.perf_counter()
    init_out = mlmodel.predict(
        {"input_ids": padded_ids, "attention_mask": padded_mask},
        function_name="init",
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0

    past_k = [
        np.ascontiguousarray(init_out[f"past_k_{layer}"]) for layer in range(num_layers)
    ]
    past_v = [
        np.ascontiguousarray(init_out[f"past_v_{layer}"]) for layer in range(num_layers)
    ]

    logits = init_out["logits"]
    generated: List[int] = []
    decode_latencies: List[float] = []
    residency_samples: List[float] = []
    evicted_tokens = 0
    total_context = prompt_len

    for step in range(max_new_tokens):
        if logits.ndim != 3:
            raise ValueError("Logits tensor must be rank-3 during validation.")
        if logits.shape[1] == seq_len:
            token_logits = logits[0, min(max(prompt_len, 1), seq_len) - 1, :]
        else:
            token_logits = logits[0, -1, :]
        next_token = int(np.argmax(token_logits))
        generated.append(next_token)

        overflow_before = max(0, total_context - seq_len)
        total_context += 1
        overflow_after = max(0, total_context - seq_len)
        residency_samples.append(min(1.0, total_context / seq_len))
        if overflow_after > overflow_before:
            evicted_tokens += overflow_after - overflow_before

        if eos_token_id is not None and next_token == eos_token_id:
            break

        if step == max_new_tokens - 1:
            break

        decode_inputs: Dict[str, Any] = {
            "input_ids": np.array([[next_token]], dtype=np.int32),
            "attention_mask": np.ones((1, 1), dtype=np.int32),
        }
        for layer in range(num_layers):
            decode_inputs[f"in_k_{layer}"] = past_k[layer]
            decode_inputs[f"in_v_{layer}"] = past_v[layer]

        decode_start = time.perf_counter()
        decode_out = mlmodel.predict(decode_inputs, function_name="decode")
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        decode_latencies.append(decode_ms)

        past_k = [
            _trim_coreml_cache(np.ascontiguousarray(decode_out[f"out_k_{layer}"]))
            for layer in range(num_layers)
        ]
        past_v = [
            _trim_coreml_cache(np.ascontiguousarray(decode_out[f"out_v_{layer}"]))
            for layer in range(num_layers)
        ]
        logits = decode_out["logits"]

    decoded_text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "tokens": generated,
        "text": decoded_text,
        "init_ms": init_ms,
        "decode_ms": decode_latencies,
        "residency": residency_samples,
        "evicted": evicted_tokens,
    }


def _compute_package_size(path: Path) -> int:
    """Return the size of a saved Core ML package in bytes."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot determine size for missing path: {path}")

    if path.is_file():
        return path.stat().st_size

    if not path.is_dir():
        raise RuntimeError(f"Unsupported path type for size computation: {path}")

    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _remove_existing_path(path: Path) -> None:
    """Remove an existing file or directory path before overwriting."""

    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _build_quantization_plan(
    *,
    root: Path,
    default_wbits: int,
    default_group_size: int,
    compute_units: str,
    requested_wbits: Optional[Sequence[int]],
    requested_group_sizes: Optional[Sequence[int]],
) -> List[QuantizationVariant]:
    """Return the ordered list of quantization variants to evaluate."""

    root.mkdir(parents=True, exist_ok=True)

    wbits_sequence: List[int] = [default_wbits]
    if requested_wbits:
        wbits_sequence.extend(
            [value for value in requested_wbits if value not in wbits_sequence]
        )
    else:
        wbits_sequence.extend(
            [value for value in SUPPORTED_WBITS if value not in wbits_sequence]
        )

    base_group_sequence: List[int] = [default_group_size]
    if requested_group_sizes:
        base_group_sequence.extend(
            [value for value in requested_group_sizes if value not in base_group_sequence]
        )
    else:
        ne_sizes = list(NEURAL_ENGINE_GROUP_SIZES)
        if compute_units == "CPU_ONLY" and default_group_size not in ne_sizes:
            ne_sizes.append(default_group_size)
        base_group_sequence.extend(
            [value for value in ne_sizes if value not in base_group_sequence]
        )

    variants: List[QuantizationVariant] = []
    seen: set[Tuple[int, int]] = set()

    for wbits in wbits_sequence:
        for group_size in base_group_sequence:
            key = (wbits, group_size)
            if key in seen:
                continue
            try:
                _validate_group_size_for_backend(group_size, compute_units)
            except ValueError as exc:
                raise ValueError(
                    f"Sweep configuration wbits={wbits}, group_size={group_size} is invalid: {exc}"
                ) from exc
            output_path = root / f"dolphin_w{wbits}_g{group_size}.mlpackage"
            variants.append(
                QuantizationVariant(
                    wbits=wbits, group_size=group_size, output_path=output_path
                )
            )
            seen.add(key)

    return variants


def _collect_coreml_performance(
    *,
    mlmodel: "ct.models.MLModel",
    tokenizer: "AutoTokenizer",
    seq_len: int,
    num_layers: int,
    prompts: Sequence[GoldenPrompt],
) -> Dict[str, Any]:
    """Run deterministic decode to capture latency and residency statistics."""

    prompt_rows: List[Dict[str, Any]] = []
    all_decode: List[float] = []
    all_residency: List[float] = []
    init_samples: List[float] = []
    total_evicted = 0

    def _percentile(samples: Sequence[float], percentile: float) -> float:
        if not samples:
            return 0.0
        return float(np.percentile(np.array(samples, dtype=np.float64), percentile))

    for golden in prompts:
        padded_ids, padded_mask, _, effective_len = _prepare_prompt_arrays(
            tokenizer, golden.prompt, seq_len
        )
        metrics = _run_coreml_decode(
            mlmodel,
            tokenizer,
            padded_ids,
            padded_mask,
            effective_len,
            seq_len,
            num_layers,
            golden.max_new_tokens,
            tokenizer.eos_token_id,
        )

        decode_ms = metrics["decode_ms"]
        init_samples.append(float(metrics["init_ms"]))
        all_decode.extend(float(sample) for sample in decode_ms)
        all_residency.extend(float(sample) for sample in metrics["residency"])
        total_evicted += int(metrics["evicted"])

        avg_decode = float(sum(decode_ms) / len(decode_ms)) if decode_ms else 0.0
        final_residency = (
            float(metrics["residency"][-1] * 100.0)
            if metrics["residency"]
            else 0.0
        )

        prompt_rows.append(
            {
                "prompt": golden.prompt,
                "init_ms": float(metrics["init_ms"]),
                "avg_decode_ms": avg_decode,
                "decode_p50_ms": _percentile(decode_ms, 50),
                "decode_p90_ms": _percentile(decode_ms, 90),
                "decode_p99_ms": _percentile(decode_ms, 99),
                "final_residency_pct": final_residency,
                "evicted_tokens": int(metrics["evicted"]),
            }
        )

    aggregate = {
        "init_mean_ms": statistics.fmean(init_samples) if init_samples else 0.0,
        "decode_p50_ms": _percentile(all_decode, 50),
        "decode_p90_ms": _percentile(all_decode, 90),
        "decode_p99_ms": _percentile(all_decode, 99),
        "avg_residency_pct": (statistics.fmean(all_residency) * 100.0)
        if all_residency
        else 0.0,
        "min_residency_pct": (min(all_residency) * 100.0) if all_residency else 0.0,
        "max_residency_pct": (max(all_residency) * 100.0) if all_residency else 0.0,
        "total_evicted_tokens": total_evicted,
    }

    return {"prompts": prompt_rows, "aggregate": aggregate}


def _render_sweep_summary(
    results: Sequence[QuantizationResult], *, baseline_size: Optional[int]
) -> None:
    """Print a Rich table summarising sweep outcomes."""

    table = Table(title="Quantization Sweep Summary")
    table.add_column("Variant", justify="left")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Vs FP16", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p90 ms", justify="right")
    table.add_column("p99 ms", justify="right")
    table.add_column("Avg residency %", justify="right")
    table.add_column("Notes", justify="left")

    for index, result in enumerate(results):
        variant = result.variant
        label = f"W{variant.wbits}/G{variant.group_size}"
        if index == 0:
            label += " (primary)"

        size_display = (
            f"{result.size_bytes / 1e9:.3f}"
            if result.size_bytes is not None
            else "-"
        )
        if result.compression_ratio is not None:
            ratio_display = f"{result.compression_ratio:.3f}×"
        elif baseline_size and result.size_bytes is not None:
            ratio_display = f"{(result.size_bytes / baseline_size):.3f}×"
        else:
            ratio_display = "-"

        aggregate = result.performance.get("aggregate") if result.performance else {}
        p50_display = (
            f"{aggregate.get('decode_p50_ms', 0.0):.2f}" if aggregate else "-"
        )
        p90_display = (
            f"{aggregate.get('decode_p90_ms', 0.0):.2f}" if aggregate else "-"
        )
        p99_display = (
            f"{aggregate.get('decode_p99_ms', 0.0):.2f}" if aggregate else "-"
        )
        residency_display = (
            f"{aggregate.get('avg_residency_pct', 0.0):.2f}"
            if aggregate
            else "-"
        )

        notes: List[str] = []
        if result.error:
            notes.append(f"❌ {result.error}")
        if result.performance_error:
            notes.append(f"⚠️ {result.performance_error}")
        if result.validation_passed is True:
            notes.append("Validation ✅")
        elif result.validation_passed is False:
            notes.append("Validation ❌")

        table.add_row(
            label,
            size_display,
            ratio_display,
            p50_display,
            p90_display,
            p99_display,
            residency_display,
            "\n".join(notes) if notes else "",
        )

    console.print(table)


def _write_sweep_report(
    path: Path,
    *,
    model: str,
    revision: Optional[str],
    compute_units: str,
    deployment_target: str,
    seq_len: int,
    baseline_size: Optional[int],
    results: Sequence[QuantizationResult],
    evaluated_wbits: Sequence[int],
    evaluated_group_sizes: Sequence[int],
) -> None:
    """Persist a machine-readable summary of sweep results."""

    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": model,
        "revision": revision,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "compute_units": compute_units,
        "minimum_deployment_target": deployment_target,
        "seq_len": seq_len,
        "evaluated_wbits": list(evaluated_wbits),
        "evaluated_group_sizes": list(evaluated_group_sizes),
        "variants": [],
    }

    if baseline_size is not None:
        payload["baseline_size_bytes"] = baseline_size

    for result in results:
        entry: Dict[str, Any] = {
            "wbits": result.variant.wbits,
            "group_size": result.variant.group_size,
            "output_path": str(result.variant.output_path.resolve()),
        }
        if result.size_bytes is not None:
            entry["size_bytes"] = result.size_bytes
        if result.compression_ratio is not None:
            entry["compression_ratio"] = result.compression_ratio
        if result.performance is not None:
            entry["performance"] = result.performance
        if result.performance_error:
            entry["performance_error"] = result.performance_error
        if result.error:
            entry["error"] = result.error
        if result.validation_passed is not None:
            entry["validation_passed"] = result.validation_passed
        payload["variants"].append(entry)

    try:
        resolved_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem specific
        raise RuntimeError(f"Failed to write sweep report to {resolved_path}: {exc}") from exc

    console.print(
        Panel.fit(
            f"[bold green]Quantization sweep report written to {resolved_path}",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Argument parsing utilities
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full Dolphin→CoreML pipeline (chat + embeddings) with compression "
            "and optional validation"
        )
    )
    parser.add_argument(
        "--model", required=True, help="HF repo or local path for the base model"
    )
    parser.add_argument(
        "--revision", default=None, help="Specific revision or commit to fetch"
    )
    parser.add_argument(
        "--hf-token", dest="hf_token", default=None, help="Optional Hugging Face token"
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        default=str(Path("hf_cache").absolute()),
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--lora-checkpoint",
        required=True,
        help="Directory containing the LoRA adapters to merge",
    )
    parser.add_argument(
        "--llm2vec-checkpoint",
        required=True,
        help="Checkpoint directory for the LLM2Vec encoder head",
    )
    parser.add_argument(
        "--seq-len",
        dest="seq_len",
        type=int,
        required=True,
        help="Maximum sequence length for export",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the final .mlpackage (or directory)",
    )
    parser.add_argument(
        "--tmp",
        default="build_tmp",
        help="Temporary working directory for intermediate artifacts",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=4,
        choices=[2, 4, 6, 8],
        help="Weight bit-width for palettization",
    )
    parser.add_argument(
        "--palett-granularity",
        dest="palett_granularity",
        choices=["per_tensor", "per_grouped_channel", "per_channel"],
        default="per_grouped_channel",
        help="Granularity for the LUT palettization",
    )
    parser.add_argument(
        "--palett-group-size",
        dest="palett_group_size",
        type=int,
        default=16,
        help="Group size when using grouped-channel palettization",
    )
    parser.add_argument(
        "--quant-sweep",
        dest="quant_sweep",
        action="store_true",
        help="Sweep quantization configs across supported bit-widths/group sizes and report trade-offs.",
    )
    parser.add_argument(
        "--sweep-wbits",
        dest="sweep_wbits",
        type=sweep_wbits_arg,
        default=None,
        help="Comma-separated bit-widths (e.g., '2,4,6') to include in the quantization sweep.",
    )
    parser.add_argument(
        "--sweep-group-sizes",
        dest="sweep_group_sizes",
        type=sweep_group_size_arg,
        default=None,
        help="Comma-separated palettization group sizes to include in the sweep report.",
    )
    parser.add_argument(
        "--sweep-report",
        dest="sweep_report",
        default=None,
        help="Optional JSON file path for CI consumption of sweep metrics.",
    )
    parser.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        default=None,
        type=_mixed_precision_arg,
        help=(
            "Comma separated overrides for palettization bit-width by component. "
            "Example: 'attention=6,mlp=4'. Supported keys: attention, mlp."
        ),
    )
    parser.add_argument(
        "--compute-units",
        dest="compute_units",
        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY"],
        default="ALL",
        help="Core ML compute units",
    )
    parser.add_argument(
        "--minimum-deployment-target",
        dest="minimum_deployment_target",
        default="iOS18",
        help="Deployment target (e.g., iOS18 or macOS15)",
    )
    parser.add_argument(
        "--profile-validate",
        dest="profile_validate",
        action="store_true",
        help="Run a lightweight validation against the exported model",
    )
    parser.add_argument(
        "--clean-tmp",
        dest="clean_tmp",
        action="store_true",
        help="Remove the temporary build directory on success",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def resolve_compute_units(name: str) -> ct.ComputeUnit:
    if not hasattr(ct.ComputeUnit, name):
        raise ValueError(f"Unsupported compute unit '{name}'.")
    return getattr(ct.ComputeUnit, name)


def resolve_target(name: str) -> ct.target.Target:
    if not hasattr(ct.target, name):
        raise ValueError(
            f"Unsupported deployment target '{name}'. Update coremltools if the target is new."
        )
    return getattr(ct.target, name)


def ensure_unsloth() -> None:
    global HAS_UNSLOTH
    if HAS_UNSLOTH:
        return
    ensure_packages(["unsloth"])
    HAS_UNSLOTH = _module_available("unsloth")
    if not HAS_UNSLOTH:
        raise RuntimeError("Unsloth installation did not make the module importable.")


def ensure_llm2vec() -> None:
    global HAS_LLM2VEC
    if HAS_LLM2VEC:
        return
    ensure_packages(["llm2vec"])
    HAS_LLM2VEC = _module_available("llm2vec")
    if not HAS_LLM2VEC:
        raise RuntimeError("LLM2Vec installation did not make the module importable.")


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------
def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if (args.sweep_wbits or args.sweep_group_sizes or args.sweep_report) and not args.quant_sweep:
        console.print(
            Panel.fit(
                "[bold red]❌ Sweep options require --quant-sweep to be enabled.",
                border_style="red",
            )
        )
        return 2

    try:
        _validate_group_size_for_backend(args.palett_group_size, args.compute_units)
    except ValueError as exc:
        console.print(Panel.fit(f"[bold red]❌ {exc}", border_style="red"))
        return 2

    mixed_precision_overrides = args.mixed_precision or {}

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = prepare_tmp_dir(Path(args.tmp))

    compression_line = (
        "• Sweep quantization variants"
        if args.quant_sweep
        else f"• Apply W{args.wbits} compression"
    )

    console.print(
        Panel.fit(
            (
                "[bold cyan]🐬 Dolphin 3.0-Llama3.1-8B → Core ML (chat + embeddings) Pipeline[/]\n"
                "• Download base model\n"
                "• Merge LoRA adapters\n"
                "• Attach LLM2Vec encoder\n"
                "• Export init/decode/encode wrappers\n"
                "• Convert to Core ML multifunction mlprogram\n"
                f"{compression_line} • Validate"
            ),
            title="Pipeline Start",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------
    # Step 1: Fetch base model (local path or snapshot download)
    # ------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            "[green]Fetching base model from Hugging Face…", total=None
        )
        try:
            if Path(args.model).exists():
                local_base = str(Path(args.model).resolve())
            else:
                local_base = snapshot_download(
                    repo_id=args.model,
                    revision=args.revision,
                    token=args.hf_token,
                    cache_dir=str(cache_dir),
                    local_files_only=False,
                )
            progress.update(
                task_id, description=f"[green]Base model ready at {local_base}"
            )
        except Exception as exc:  # pragma: no cover - network interaction
            console.print(
                Panel.fit(
                    f"[bold red]❌ Failed to download base model:[/] {exc}",
                    border_style="red",
                )
            )
            return 1

    # ------------------------------------------------------------------
    # Step 2: Load tokenizer, config, and base model
    # ------------------------------------------------------------------
    console.print(Panel.fit("[bold green]Loading tokenizer, config, and base model…"))
    tokenizer = AutoTokenizer.from_pretrained(local_base, use_fast=True)
    config = AutoConfig.from_pretrained(local_base)
    config.use_cache = True
    config.output_hidden_states = True

    model = AutoModelForCausalLM.from_pretrained(
        local_base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    console.print("[green]Base model loaded.")

    # ------------------------------------------------------------------
    # Step 3: Merge LoRA adapters
    # ------------------------------------------------------------------
    ensure_unsloth()
    console.print(
        Panel.fit(f"[bold green]Merging LoRA adapters from: {args.lora_checkpoint}")
    )
    try:
        from peft import (
            PeftModel,
        )  # imported lazily to avoid unnecessary dependency during help
    except ImportError:  # pragma: no cover
        ensure_packages(["peft"])
        from peft import PeftModel

    lora_model = PeftModel.from_pretrained(model, args.lora_checkpoint)
    merged_model = lora_model.merge_and_unload()
    model = merged_model
    model.eval()
    console.print("[green]LoRA merged successfully.")

    # ------------------------------------------------------------------
    # Step 4: Attach LLM2Vec encoder head
    # ------------------------------------------------------------------
    ensure_llm2vec()
    console.print(
        Panel.fit(
            f"[bold green]Loading LLM2Vec encoder from: {args.llm2vec_checkpoint}"
        )
    )
    try:
        from llm2vec import LLM2Vec

        embed_encoder = LLM2Vec.from_pretrained(
            args.llm2vec_checkpoint,
            base_model=model,
        )
        embedding_module = embed_encoder
        console.print("[green]LLM2Vec encoder head loaded.")
    except Exception as exc:  # pragma: no cover - external dependency
        console.print(
            Panel.fit(
                f"[bold red]❌ Failed loading LLM2Vec encoder head:[/] {exc}",
                border_style="red",
            )
        )
        return 1

    # ------------------------------------------------------------------
    # Step 5: Construct wrapper modules for export
    # ------------------------------------------------------------------
    console.print(
        Panel.fit("[bold green]Constructing PyTorch wrapper modules for export…")
    )

    num_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads

    class InitWrapper(torch.nn.Module):
        def __init__(self, base: torch.nn.Module):
            super().__init__()
            self.base = base

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, ...]:
            input_ids = input_ids.long()
            attention_mask = attention_mask.long()
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = out.logits.to(dtype=torch.float16)
            last_hidden = out.hidden_states[-1].to(dtype=torch.float16)
            flat: List[torch.Tensor] = []
            for key_tensor, value_tensor in out.past_key_values:
                flat.append(key_tensor.to(dtype=torch.float16))
                flat.append(value_tensor.to(dtype=torch.float16))
            return logits, last_hidden, *flat

    class DecodeWrapper(torch.nn.Module):
        def __init__(self, base: torch.nn.Module, layers: int):
            super().__init__()
            self.base = base
            self.layers = layers

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            *flat_past: torch.Tensor,
        ) -> Tuple[torch.Tensor, ...]:
            input_ids = input_ids.long()
            attention_mask = attention_mask.long()
            past: List[Tuple[torch.Tensor, torch.Tensor]] = []
            iterator = iter(flat_past)
            for _ in range(self.layers):
                key_tensor = next(iterator)
                value_tensor = next(iterator)
                past.append((key_tensor, value_tensor))

            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=tuple(past),
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = out.logits.to(dtype=torch.float16)
            last_hidden = out.hidden_states[-1].to(dtype=torch.float16)
            flat: List[torch.Tensor] = []
            for key_tensor, value_tensor in out.past_key_values:
                flat.append(key_tensor.to(dtype=torch.float16))
                flat.append(value_tensor.to(dtype=torch.float16))
            return logits, last_hidden, *flat

    class EncodeWrapper(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module):
            super().__init__()
            self.encoder = encoder

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            input_ids = input_ids.long()
            attention_mask = attention_mask.long()
            embedding = self.encoder.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return embedding.to(dtype=torch.float16)

    init_module = InitWrapper(model)
    decode_module = DecodeWrapper(model, layers=num_layers)
    encode_module = EncodeWrapper(embedding_module)

    console.print("[green]Wrappers built.")

    try:
        with torch.no_grad():
            try:
                probe_device = next(encode_module.parameters()).device
            except (StopIteration, AttributeError):
                probe_device = torch.device("cpu")
            _ids = torch.zeros((1, 1), dtype=torch.long, device=probe_device)
            _mask = torch.ones((1, 1), dtype=torch.long, device=probe_device)
            _probe = encode_module(_ids, _mask)
            embed_dim = int(_probe.shape[-1])
    except Exception:
        embed_dim = int(config.hidden_size)

    # ------------------------------------------------------------------
    # Step 6: Prepare export metadata for conversion
    # ------------------------------------------------------------------
    console.print(Panel.fit("[bold green]Preparing export shapes for conversion…"))
    batch = 1
    seq_len = args.seq_len
    past_shape = (batch, n_heads, seq_len, head_dim)
    console.print("[green]Export shape metadata prepared.")

    # ------------------------------------------------------------------
    # Step 7: Convert to Core ML multifunction model
    # ------------------------------------------------------------------
    console.print(
        Panel.fit(
            "[bold green]Converting to Core ML (mlprogram) with multiple functions…"
        )
    )

    def tensor_type(name: str, shape: Tuple[int, ...], dtype: Any) -> ct.TensorType:
        return ct.TensorType(name=name, shape=shape, dtype=dtype)

    init_inputs = [
        tensor_type("input_ids", (batch, seq_len), np.int32),
        tensor_type("attention_mask", (batch, seq_len), np.int32),
    ]
    init_outputs: List[ct.TensorType] = [
        tensor_type("logits", (batch, seq_len, config.vocab_size), np.float16),
        tensor_type("last_hidden", (batch, seq_len, config.hidden_size), np.float16),
    ]
    for layer_idx in range(num_layers):
        init_outputs.append(tensor_type(f"past_k_{layer_idx}", past_shape, np.float16))
        init_outputs.append(tensor_type(f"past_v_{layer_idx}", past_shape, np.float16))

    decode_inputs = [
        tensor_type("input_ids", (batch, 1), np.int32),
        tensor_type("attention_mask", (batch, 1), np.int32),
    ]
    for layer_idx in range(num_layers):
        decode_inputs.append(tensor_type(f"in_k_{layer_idx}", past_shape, np.float16))
        decode_inputs.append(tensor_type(f"in_v_{layer_idx}", past_shape, np.float16))

    decode_outputs: List[ct.TensorType] = [
        tensor_type("logits", (batch, 1, config.vocab_size), np.float16),
        tensor_type("last_hidden", (batch, 1, config.hidden_size), np.float16),
    ]
    decode_out_shape = (batch, n_heads, seq_len + 1, head_dim)
    for layer_idx in range(num_layers):
        decode_outputs.append(
            tensor_type(f"out_k_{layer_idx}", decode_out_shape, np.float16)
        )
        decode_outputs.append(
            tensor_type(f"out_v_{layer_idx}", decode_out_shape, np.float16)
        )

    encode_inputs = [
        tensor_type("input_ids", (batch, seq_len), np.int32),
        tensor_type("attention_mask", (batch, seq_len), np.int32),
    ]
    encode_outputs = [
        tensor_type("embedding", (batch, embed_dim), np.float16),
    ]

    model_converted = ct.convert(
        {"init": init_module, "decode": decode_module, "encode": encode_module},
        source="pytorch",
        convert_to="mlprogram",
        inputs={
            "init": init_inputs,
            "decode": decode_inputs,
            "encode": encode_inputs,
        },
        outputs={
            "init": init_outputs,
            "decode": decode_outputs,
            "encode": encode_outputs,
        },
        minimum_deployment_target=resolve_target(args.minimum_deployment_target),
        compute_units=resolve_compute_units(args.compute_units),
    )

    console.print("[green]Core ML conversion successful.")

    # ------------------------------------------------------------------
    # Step 8: Apply compression
    # ------------------------------------------------------------------
    console.print(
        Panel.fit(
            "[bold green]Preparing Core ML compression (palettization + linear quantization)…"
        )
    )

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        get_weights_metadata,
        linear_quantize_weights,
        palettize_weights,
    )

    mixed_plan: Dict[str, int] = {}
    mixed_summary: Counter[str] = Counter()

    if mixed_precision_overrides:
        try:
            weight_metadata = get_weights_metadata(model_converted)
        except Exception as exc:  # pragma: no cover - depends on Core ML runtime availability
            raise RuntimeError(
                "Mixed precision overrides require Core ML weight metadata support. "
                "Ensure coremltools is installed with libcoremlpython bindings."
            ) from exc

        mixed_plan, mixed_summary = _resolve_mixed_precision_plan(
            weight_metadata.keys(), mixed_precision_overrides
        )

        unused_overrides = [
            key for key in sorted(mixed_precision_overrides) if key not in mixed_summary
        ]
        if unused_overrides:
            console.print(
                Panel.fit(
                    "[yellow]Warning: The following mixed precision overrides were not used "
                    "(no matching weights found): " + ", ".join(unused_overrides),
                    title="Unused Mixed Precision Overrides",
                    border_style="yellow",
                )
            )
        if not mixed_plan:
            console.print(
                Panel.fit(
                    "[bold yellow]⚠️ No weights matched the requested mixed-precision overrides. "
                    "Falling back to global bit-width.",
                    border_style="yellow",
                )
            )

    if mixed_summary:
        lines = ["[bold green]Mixed-precision overrides applied:"]
        for key, description in SUPPORTED_MIXED_PRECISION_KEYS.items():
            if key in mixed_summary:
                bits = mixed_precision_overrides[key]
                count = mixed_summary[key]
                lines.append(
                    f"• {key} ({description}): {bits}-bit across {count} weights"
                )
        console.print(Panel.fit("\n".join(lines), border_style="green"))

    lin_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            granularity="per_tensor",
        )
    )

    quant_results: List[QuantizationResult] = []
    baseline_size_bytes: Optional[int] = None
    output_root = Path(args.output)
    sweep_wbits: Tuple[int, ...] = (
        tuple(args.sweep_wbits) if args.sweep_wbits else ()
    )
    sweep_group_sizes: Tuple[int, ...] = (
        tuple(args.sweep_group_sizes) if args.sweep_group_sizes else ()
    )

    if args.quant_sweep:
        if output_root.suffix == ".mlpackage":
            console.print(
                Panel.fit(
                    "[bold red]❌ --output must be a directory when --quant-sweep is enabled.",
                    border_style="red",
                )
            )
            return 2
        if output_root.exists() and not output_root.is_dir():
            console.print(
                Panel.fit(
                    f"[bold red]❌ Sweep output path {output_root} exists and is not a directory.",
                    border_style="red",
                )
            )
            return 2
        try:
            quant_plan = _build_quantization_plan(
                root=output_root,
                default_wbits=args.wbits,
                default_group_size=args.palett_group_size,
                compute_units=args.compute_units,
                requested_wbits=list(sweep_wbits) if sweep_wbits else None,
                requested_group_sizes=list(sweep_group_sizes) if sweep_group_sizes else None,
            )
        except ValueError as exc:
            console.print(Panel.fit(f"[bold red]❌ {exc}", border_style="red"))
            return 2
        baseline_tmp = tmp_dir / "baseline_fp16.mlpackage"
        _remove_existing_path(baseline_tmp)
        model_converted.save(str(baseline_tmp))
        baseline_size_bytes = _compute_package_size(baseline_tmp)
        _remove_existing_path(baseline_tmp)
        evaluated_wbits = tuple(
            dict.fromkeys(variant.wbits for variant in quant_plan)
        )
        evaluated_group_sizes = tuple(
            dict.fromkeys(variant.group_size for variant in quant_plan)
        )
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        quant_plan = [
            QuantizationVariant(
                wbits=args.wbits,
                group_size=args.palett_group_size,
                output_path=output_root,
            )
        ]
        evaluated_wbits = (args.wbits,)
        evaluated_group_sizes = (args.palett_group_size,)

    if not quant_plan:
        raise RuntimeError("Quantization plan did not produce any variants.")

    for index, variant in enumerate(quant_plan):
        descriptor = f"W{variant.wbits}/G{variant.group_size}"
        border = "green" if index == 0 else "cyan"
        console.print(
            Panel.fit(
                f"[bold {border}]Applying compression for {descriptor}",
                border_style=border,
            )
        )
        try:
            variant.output_path.parent.mkdir(parents=True, exist_ok=True)
            _remove_existing_path(variant.output_path)
            pal_config = OptimizationConfig(
                global_config=OpPalettizerConfig(
                    nbits=variant.wbits,
                    granularity=args.palett_granularity,
                    group_size=variant.group_size,
                ),
                op_name_configs={
                    name: OpPalettizerConfig(
                        nbits=nbits,
                        granularity=args.palett_granularity,
                        group_size=variant.group_size,
                    )
                    for name, nbits in mixed_plan.items()
                }
                if mixed_plan
                else None,
            )
            model_pal = palettize_weights(model_converted, pal_config)
            model_quant = linear_quantize_weights(
                model_pal,
                lin_config,
                joint_compression=True,
            )
            model_quant.save(str(variant.output_path))
            size_bytes = _compute_package_size(variant.output_path)
        except Exception as exc:
            message = f"Failed to quantize variant {descriptor}: {exc}"
            if index == 0:
                console.print(
                    Panel.fit(f"[bold red]❌ {message}", border_style="red")
                )
                raise
            console.print(
                Panel.fit(f"[bold yellow]⚠️ {message}", border_style="yellow")
            )
            quant_results.append(
                QuantizationResult(
                    variant=variant,
                    size_bytes=None,
                    compression_ratio=None,
                    performance=None,
                    performance_error=None,
                    error=str(exc),
                )
            )
            continue

        compression_ratio = (
            (size_bytes / baseline_size_bytes) if baseline_size_bytes else None
        )

        performance: Optional[Dict[str, Any]] = None
        performance_error: Optional[str] = None
        if args.quant_sweep:
            try:
                mlmodel_variant = ct.models.MLModel(
                    str(variant.output_path),
                    compute_units=resolve_compute_units(args.compute_units),
                )
                performance = _collect_coreml_performance(
                    mlmodel=mlmodel_variant,
                    tokenizer=tokenizer,
                    seq_len=seq_len,
                    num_layers=num_layers,
                    prompts=GOLDEN_PROMPTS,
                )
            except Exception as exc:
                performance_error = str(exc)
                console.print(
                    Panel.fit(
                        f"[bold yellow]⚠️ Unable to collect performance metrics for {descriptor}:[/] {exc}",
                        border_style="yellow",
                    )
                )

        quant_results.append(
            QuantizationResult(
                variant=variant,
                size_bytes=size_bytes,
                compression_ratio=compression_ratio,
                performance=performance,
                performance_error=performance_error,
                error=None,
            )
        )
        console.print(
            f"[green]Saved {descriptor} to {variant.output_path} ({size_bytes / 1e9:.3f} GB)."
        )

    primary_result = quant_results[0]
    primary_output_path = primary_result.variant.output_path
    console.print(
        Panel.fit(
            f"[bold green]Primary quantized model available at {primary_output_path}",
            border_style="green",
        )
    )
    if primary_result.size_bytes is not None:
        console.print(
            f"[blue]Final package size: {primary_result.size_bytes / 1e9:.3f} GB"
        )

    # ------------------------------------------------------------------
    # Step 10: Optional validation
    # ------------------------------------------------------------------
    validation_failed = False

    if args.profile_validate:
        console.print(Panel.fit("[bold green]Running deterministic validation suite…"))
        try:
            mlmodel = ct.models.MLModel(
                str(primary_output_path),
                compute_units=resolve_compute_units(args.compute_units),
            )

            transcript_rows: List[Dict[str, Any]] = []
            decode_lat_all: List[float] = []
            residency_all: List[float] = []
            total_evicted = 0
            all_match = True

            for golden in GOLDEN_PROMPTS:
                padded_ids, padded_mask, trimmed_prompt, effective_len = (
                    _prepare_prompt_arrays(tokenizer, golden.prompt, seq_len)
                )
                reference = _run_reference_decode(
                    model,
                    tokenizer,
                    trimmed_prompt,
                    effective_len,
                    golden.max_new_tokens,
                )
                coreml_metrics = _run_coreml_decode(
                    mlmodel,
                    tokenizer,
                    padded_ids,
                    padded_mask,
                    effective_len,
                    seq_len,
                    num_layers,
                    golden.max_new_tokens,
                    tokenizer.eos_token_id,
                )
                match = reference["tokens"] == coreml_metrics["tokens"]
                if not match:
                    all_match = False

                transcript_rows.append(
                    {
                        "prompt": golden.prompt,
                        "reference": reference,
                        "coreml": coreml_metrics,
                        "match": match,
                    }
                )
                decode_lat_all.extend(coreml_metrics["decode_ms"])
                residency_all.extend(coreml_metrics["residency"])
                total_evicted += coreml_metrics["evicted"]

            transcript_table = Table(title="Golden Transcript Comparison")
            transcript_table.add_column("Prompt", justify="left")
            transcript_table.add_column("PyTorch", justify="left")
            transcript_table.add_column("Core ML", justify="left")
            transcript_table.add_column("Match", justify="center")

            metrics_table = Table(title="Core ML Decode Latency & KV Residency")
            metrics_table.add_column("Prompt", justify="left")
            metrics_table.add_column("Init ms", justify="right")
            metrics_table.add_column("Avg ms/token", justify="right")
            metrics_table.add_column("p50 ms", justify="right")
            metrics_table.add_column("p90 ms", justify="right")
            metrics_table.add_column("p99 ms", justify="right")
            metrics_table.add_column("Final residency %", justify="right")
            metrics_table.add_column("Evicted tokens", justify="right")

            def _percentile(values: Sequence[float], percentile: float) -> float:
                if not values:
                    return 0.0
                return float(
                    np.percentile(np.array(values, dtype=np.float64), percentile)
                )

            for row in transcript_rows:
                prompt_snippet = textwrap.shorten(
                    row["prompt"], width=58, placeholder="…"
                )
                torch_text = row["reference"]["text"] or ""
                coreml_text = row["coreml"]["text"] or ""
                transcript_table.add_row(
                    prompt_snippet,
                    textwrap.shorten(torch_text, width=64, placeholder="…"),
                    textwrap.shorten(coreml_text, width=64, placeholder="…"),
                    "✅" if row["match"] else "❌",
                )

                decode_ms = row["coreml"]["decode_ms"]
                avg_ms = float(sum(decode_ms) / len(decode_ms)) if decode_ms else 0.0
                metrics_table.add_row(
                    prompt_snippet,
                    f"{row['coreml']['init_ms']:.2f}",
                    f"{avg_ms:.2f}",
                    f"{_percentile(decode_ms, 50):.2f}",
                    f"{_percentile(decode_ms, 90):.2f}",
                    f"{_percentile(decode_ms, 99):.2f}",
                    f"{(row['coreml']['residency'][-1] * 100.0) if row['coreml']['residency'] else 0.0:.2f}",
                    str(row["coreml"]["evicted"]),
                )

            console.print(transcript_table)
            console.print(metrics_table)

            try:
                embedding_rows, min_cosine = _validate_embedding_parity(
                    mlmodel,
                    embedding_module,
                    tokenizer,
                    seq_len,
                    EMBEDDING_BENCHMARKS,
                    minimum_cosine=EMBEDDING_COSINE_THRESHOLD,
                )
            except Exception as exc:
                validation_failed = True
                console.print(
                    Panel.fit(
                        f"[bold red]❌ LLM2Vec embedding validation failed:[/] {exc}",
                        border_style="red",
                    )
                )
            else:
                embed_table = Table(title="LLM2Vec Embedding Cosine Similarity")
                embed_table.add_column("Sentence", justify="left")
                embed_table.add_column("Cosine", justify="right")
                embed_table.add_column("Pass", justify="center")
                for row in embedding_rows:
                    embed_table.add_row(
                        textwrap.shorten(row["sentence"], width=58, placeholder="…"),
                        f"{row['cosine']:.5f}",
                        "✅" if row["pass"] else "❌",
                    )
                console.print(embed_table)
                if min_cosine < EMBEDDING_COSINE_THRESHOLD:
                    validation_failed = True
                    console.print(
                        Panel.fit(
                            "[bold red]❌ LLM2Vec embedding parity FAILED.[/]",
                            border_style="red",
                        )
                    )
                else:
                    console.print(
                        Panel.fit(
                            f"[green]LLM2Vec embedding parity confirmed (min cosine {min_cosine:.4f})."
                        )
                    )

            if decode_lat_all:
                overall = np.percentile(
                    np.array(decode_lat_all, dtype=np.float64), [50, 90, 99]
                )
                console.print(
                    Panel.fit(
                        "[green]Aggregate decode latency — "
                        f"p50: {overall[0]:.2f} ms, p90: {overall[1]:.2f} ms, p99: {overall[2]:.2f} ms"
                    )
                )

            if residency_all:
                avg_residency = statistics.fmean(residency_all) * 100.0
                min_residency = min(residency_all) * 100.0
                max_residency = max(residency_all) * 100.0
                console.print(
                    Panel.fit(
                        "[green]KV-cache residency — "
                        f"avg: {avg_residency:.2f}% • min: {min_residency:.2f}% • "
                        f"max: {max_residency:.2f}% • evicted tokens: {total_evicted}"
                    )
                )

            if not all_match:
                validation_failed = True
                console.print(
                    Panel.fit(
                        "[bold red]❌ Golden transcript parity FAILED.[/]",
                        border_style="red",
                    )
                )
            else:
                console.print(
                    "[green]Validation run complete. Numerical parity confirmed."
                )
        except Exception as exc:  # pragma: no cover - device specific behaviour
            console.print(
                Panel.fit(
                    f"[bold yellow]⚠️ Validation encountered an issue:[/] {exc}",
                    border_style="yellow",
                )
            )
            raise

    if quant_results and args.profile_validate:
        quant_results[0].validation_passed = not validation_failed

    if args.quant_sweep:
        _render_sweep_summary(quant_results, baseline_size_bytes)
        if args.sweep_report:
            _write_sweep_report(
                Path(args.sweep_report),
                model=args.model,
                revision=args.revision,
                compute_units=args.compute_units,
                deployment_target=args.minimum_deployment_target,
                seq_len=args.seq_len,
                baseline_size=baseline_size_bytes,
                results=quant_results,
                evaluated_wbits=evaluated_wbits,
                evaluated_group_sizes=evaluated_group_sizes,
            )

    # ------------------------------------------------------------------
    # Step 11: Optional cleanup
    # ------------------------------------------------------------------
    if args.clean_tmp:
        cleanup_tmp_dir(tmp_dir)

    if validation_failed:
        return 1

    console.print(
        Panel.fit(
            "[bold green]✅ Pipeline complete. Model ready for production integration.",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
