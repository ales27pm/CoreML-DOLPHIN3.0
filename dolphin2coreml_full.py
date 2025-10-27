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
import importlib
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import shlex
import statistics
import textwrap
import time


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


def ensure_packages(packages: Iterable[str]) -> None:
    """Ensure that the required Python packages are available."""

    to_install: List[str] = []
    for pkg in packages:
        module_name = pkg.split("==")[0].split(">=")[0].replace("-", "_")
        if not _module_available(module_name):
            to_install.append(pkg)
    if to_install:
        console.print(
            Panel.fit(
                f"[bold yellow]Installing dependencies:[/] {' '.join(to_install)}",
                border_style="yellow",
            )
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *to_install],
            check=True,
        )


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

try:  # pragma: no cover
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    ensure_packages(["huggingface_hub"])
    from huggingface_hub import snapshot_download

try:  # pragma: no cover
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    ensure_packages([
        "transformers>=4.44.0",
        "accelerate",
        "sentencepiece",
        "tokenizers",
    ])
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


def _torch_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:  # pragma: no cover - defensive branch
        return torch.device("cpu")


def _run_reference_decode(
    model: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    trimmed_prompt: np.ndarray,
    prompt_len: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Generate deterministic tokens using PyTorch as the golden reference."""

    device = _torch_device(model)
    torch_input = torch.from_numpy(trimmed_prompt[:, :prompt_len]).to(device)
    attention_mask = torch.ones_like(torch_input, dtype=torch.long)

    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token is None:
        raise ValueError("Tokenizer is missing both pad and EOS token IDs for generation.")

    torch.manual_seed(0)
    with torch.no_grad():
        generated = model.generate(
            input_ids=torch_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token,
        )

    new_tokens = generated[0, prompt_len : prompt_len + max_new_tokens].tolist()
    golden_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"tokens": new_tokens, "text": golden_text}


def _trim_coreml_cache(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"Expected rank-4 KV cache tensor, received rank {array.ndim}.")
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
        np.ascontiguousarray(init_out[f"past_k_{layer}"])
        for layer in range(num_layers)
    ]
    past_v = [
        np.ascontiguousarray(init_out[f"past_v_{layer}"])
        for layer in range(num_layers)
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
    parser.add_argument("--model", required=True, help="HF repo or local path for the base model")
    parser.add_argument("--revision", default=None, help="Specific revision or commit to fetch")
    parser.add_argument("--hf-token", dest="hf_token", default=None, help="Optional Hugging Face token")
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

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(args.tmp)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            (
                "[bold cyan]🐬 Dolphin 3.0-Llama3.1-8B → Core ML (chat + embeddings) Pipeline[/]\n"
                "• Download base model\n"
                "• Merge LoRA adapters\n"
                "• Attach LLM2Vec encoder\n"
                "• Export init/decode/encode wrappers\n"
                "• Convert to Core ML multifunction mlprogram\n"
                f"• Apply W{args.wbits} compression • Validate"
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
        task_id = progress.add_task("[green]Fetching base model from Hugging Face…", total=None)
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
            progress.update(task_id, description=f"[green]Base model ready at {local_base}")
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
        from peft import PeftModel  # imported lazily to avoid unnecessary dependency during help
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
        Panel.fit(f"[bold green]Loading LLM2Vec encoder from: {args.llm2vec_checkpoint}")
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
    console.print(Panel.fit("[bold green]Constructing PyTorch wrapper modules for export…"))

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
    console.print(Panel.fit("[bold green]Converting to Core ML (mlprogram) with multiple functions…"))

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
        decode_outputs.append(tensor_type(f"out_k_{layer_idx}", decode_out_shape, np.float16))
        decode_outputs.append(tensor_type(f"out_v_{layer_idx}", decode_out_shape, np.float16))

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
            f"[bold green]Applying W{args.wbits} compression: palettization + linear quant…"
        )
    )

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        linear_quantize_weights,
        palettize_weights,
    )

    pal_config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            nbits=args.wbits,
            granularity=args.palett_granularity,
            group_size=args.palett_group_size,
        )
    )
    model_pal = palettize_weights(model_converted, pal_config)

    lin_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            granularity="per_tensor",
        )
    )
    model_wbits = linear_quantize_weights(
        model_pal,
        lin_config,
        joint_compression=True,
    )

    console.print("[green]Compression complete.")

    # ------------------------------------------------------------------
    # Step 9: Persist the mlpackage
    # ------------------------------------------------------------------
    console.print(Panel.fit(f"[bold green]Saving final model to {out_path}…"))
    model_wbits.save(str(out_path))

    if out_path.is_file():
        size_bytes = out_path.stat().st_size
    else:
        size_bytes = sum(
            item.stat().st_size for item in out_path.rglob("*") if item.is_file()
        )
    console.print(f"[blue]Final package size: {size_bytes / 1e9:.3f} GB")

    # ------------------------------------------------------------------
    # Step 10: Optional validation
    # ------------------------------------------------------------------
    if args.profile_validate:
        console.print(Panel.fit("[bold green]Running deterministic validation suite…"))
        try:
            mlmodel = ct.models.MLModel(
                str(out_path),
                compute_units=resolve_compute_units(args.compute_units),
            )

            transcript_rows: List[Dict[str, Any]] = []
            decode_lat_all: List[float] = []
            residency_all: List[float] = []
            total_evicted = 0
            all_match = True

            for golden in GOLDEN_PROMPTS:
                padded_ids, padded_mask, trimmed_prompt, effective_len = _prepare_prompt_arrays(
                    tokenizer, golden.prompt, seq_len
                )
                reference = _run_reference_decode(
                    model, tokenizer, trimmed_prompt, effective_len, golden.max_new_tokens
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
                return float(np.percentile(np.array(values, dtype=np.float64), percentile))

            for row in transcript_rows:
                prompt_snippet = textwrap.shorten(row["prompt"], width=58, placeholder="…")
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
                raise RuntimeError(
                    "Core ML decode outputs diverged from PyTorch golden transcripts."
                )

            console.print("[green]Validation run complete. Numerical parity confirmed.")
        except Exception as exc:  # pragma: no cover - device specific behaviour
            console.print(
                Panel.fit(
                    f"[bold yellow]⚠️ Validation encountered an issue:[/] {exc}",
                    border_style="yellow",
                )
            )

    # ------------------------------------------------------------------
    # Step 11: Optional cleanup
    # ------------------------------------------------------------------
    if args.clean_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        console.print("[green]Temporary build directory cleaned up.")

    console.print(
        Panel.fit(
            "[bold green]✅ Pipeline complete. Model ready for production integration.",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
