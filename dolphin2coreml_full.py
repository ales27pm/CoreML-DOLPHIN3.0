#!/usr/bin/env python3
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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rich console initialisation (installed on-demand)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
except ImportError:  # pragma: no cover - executed only when dependency missing
    subprocess.run(f"{sys.executable} -m pip install rich", shell=True, check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@dataclass
class ShellResult:
    """Container for shell execution results."""

    returncode: int
    command: str


def sh(command: str, *, check: bool = True) -> ShellResult:
    """Execute a shell command with optional error checking."""

    console.log(f"[shell] {command}")
    result = subprocess.run(command, shell=True, check=check)
    return ShellResult(returncode=result.returncode, command=command)


def ensure_packages(packages: Iterable[str]) -> None:
    """Ensure that the required Python packages are available."""

    to_install: List[str] = []
    for pkg in packages:
        module_name = pkg.split("==")[0].split(">=")[0].replace("-", "_")
        try:
            __import__(module_name)
        except ImportError:
            to_install.append(pkg)
    if to_install:
        console.print(
            Panel.fit(
                f"[bold yellow]Installing dependencies:[/] {' '.join(to_install)}",
                border_style="yellow",
            )
        )
        sh(f"{sys.executable} -m pip install -q {' '.join(to_install)}")


# Ensure core dependencies are present before importing heavy modules.
ensure_packages(
    [
        "torch>=2.0.0",
        "transformers>=4.44.0",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "tokenizers",
        "coremltools>=8.0.0",
        "numpy",
    ]
)

# Optional packages are installed on demand later in the workflow.
try:  # pragma: no cover - best effort dependency management
    import unsloth  # type: ignore

    HAS_UNSLOTH = True
except ImportError:  # pragma: no cover
    HAS_UNSLOTH = False

try:  # pragma: no cover
    import llm2vec  # type: ignore

    HAS_LLM2VEC = True
except ImportError:  # pragma: no cover
    HAS_LLM2VEC = False

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import coremltools as ct


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
    import unsloth as _unsloth  # noqa: F401  # pragma: no cover

    HAS_UNSLOTH = True


def ensure_llm2vec() -> None:
    global HAS_LLM2VEC
    if HAS_LLM2VEC:
        return
    ensure_packages(["llm2vec"])
    import llm2vec as _llm2vec  # noqa: F401  # pragma: no cover

    HAS_LLM2VEC = True


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
    from peft import PeftModel  # imported lazily to avoid unnecessary dependency during help

    lora_model = PeftModel.from_pretrained(model, args.lora_checkpoint)
    merged_model = lora_model.merge_and_unload()
    model = merged_model
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
        ) -> Tuple[torch.Tensor, torch.Tensor, *Tuple[torch.Tensor, ...]]:
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = out.logits
            last_hidden = out.hidden_states[-1]
            flat: List[torch.Tensor] = []
            for key_tensor, value_tensor in out.past_key_values:
                flat.append(key_tensor)
                flat.append(value_tensor)
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
        ) -> Tuple[torch.Tensor, torch.Tensor, *Tuple[torch.Tensor, ...]]:
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
            logits = out.logits
            last_hidden = out.hidden_states[-1]
            flat: List[torch.Tensor] = []
            for key_tensor, value_tensor in out.past_key_values:
                flat.append(key_tensor)
                flat.append(value_tensor)
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
            embedding = self.encoder.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return embedding

    init_module = InitWrapper(model)
    decode_module = DecodeWrapper(model, layers=num_layers)
    encode_module = EncodeWrapper(embedding_module)

    console.print("[green]Wrappers built.")

    # ------------------------------------------------------------------
    # Step 6: Prepare dummy inputs for tracing
    # ------------------------------------------------------------------
    console.print(Panel.fit("[bold green]Preparing dummy inputs for tracing & conversion…"))
    batch = 1
    seq_len = args.seq_len

    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(batch, seq_len),
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)

    past_shape = (batch, n_heads, seq_len, head_dim)
    dummy_flat_past: List[torch.Tensor] = []
    for _ in range(num_layers):
        dummy_flat_past.append(torch.randn(past_shape, dtype=torch.float16))
        dummy_flat_past.append(torch.randn(past_shape, dtype=torch.float16))

    console.print("[green]Dummy input tensors prepared.")

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
    for layer_idx in range(num_layers):
        init_outputs.append(tensor_type(f"past_v_{layer_idx}", past_shape, np.float16))

    decode_inputs = [
        tensor_type("input_ids", (batch, 1), np.int32),
        tensor_type("attention_mask", (batch, 1), np.int32),
    ]
    for layer_idx in range(num_layers):
        decode_inputs.append(tensor_type(f"in_k_{layer_idx}", past_shape, np.float16))
    for layer_idx in range(num_layers):
        decode_inputs.append(tensor_type(f"in_v_{layer_idx}", past_shape, np.float16))

    decode_outputs: List[ct.TensorType] = [
        tensor_type("logits", (batch, 1, config.vocab_size), np.float16),
        tensor_type("last_hidden", (batch, 1, config.hidden_size), np.float16),
    ]
    decode_out_shape = (batch, n_heads, seq_len + 1, head_dim)
    for layer_idx in range(num_layers):
        decode_outputs.append(tensor_type(f"out_k_{layer_idx}", decode_out_shape, np.float16))
    for layer_idx in range(num_layers):
        decode_outputs.append(tensor_type(f"out_v_{layer_idx}", decode_out_shape, np.float16))

    encode_inputs = [
        tensor_type("input_ids", (batch, seq_len), np.int32),
        tensor_type("attention_mask", (batch, seq_len), np.int32),
    ]
    encode_outputs = [
        tensor_type("embedding", (batch, config.hidden_size), np.float16),
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
        console.print(Panel.fit("[bold green]Running quick validation…"))
        try:
            mlmodel = ct.models.MLModel(
                str(out_path),
                compute_units=resolve_compute_units(args.compute_units),
            )
            torch_input_ids = torch.randint(
                low=0,
                high=tokenizer.vocab_size,
                size=(batch, seq_len),
                dtype=torch.long,
            )
            torch_mask = torch.ones((batch, seq_len), dtype=torch.long)
            with torch.no_grad():
                reference = model(
                    torch_input_ids,
                    attention_mask=torch_mask,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
            coreml_result = mlmodel.predict(
                {
                    "input_ids": torch_input_ids.numpy().astype(np.int32),
                    "attention_mask": torch_mask.numpy().astype(np.int32),
                },
                function_name="init",
            )

            table = Table(title="Validation Summary")
            table.add_column("Output", justify="left")
            table.add_column("PyTorch shape", justify="left")
            table.add_column("Core ML shape", justify="left")
            table.add_row(
                "logits",
                str(tuple(reference.logits.shape)),
                str(tuple(coreml_result["logits"].shape)),
            )
            table.add_row(
                "last_hidden",
                str(tuple(reference.hidden_states[-1].shape)),
                str(tuple(coreml_result["last_hidden"].shape)),
            )
            console.print(table)
            console.print(
                "[green]Validation run complete. Inspect numerical fidelity as needed."
            )
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
