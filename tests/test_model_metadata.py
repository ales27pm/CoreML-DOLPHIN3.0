import json
import sys
from pathlib import Path
from typing import ClassVar, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dolphin2coreml_full import (  # imported after sys.path mutation
    METADATA_FILENAME,
    _build_model_metadata,
    _load_llm2vec_descriptor,
    _load_lora_descriptor,
    _write_model_metadata,
    resolve_model_variant_label,
)


class DummyConfig:
    model_type = "llama3"
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    vocab_size = 128256
    rope_theta = 10000.0
    rope_scaling: ClassVar[Dict[str, float]] = {"type": "linear", "factor": 1.0}


def test_resolve_model_variant_label_auto() -> None:
    assert resolve_model_variant_label("auto", hidden_size=4096, num_layers=32) == "8B"
    assert resolve_model_variant_label("auto", hidden_size=8192, num_layers=80) == "70B"
    assert resolve_model_variant_label("auto", hidden_size=3072, num_layers=28) == "3B"
    assert (
        resolve_model_variant_label("CUSTOM", hidden_size=4096, num_layers=32)
        == "CUSTOM"
    )
    assert resolve_model_variant_label("70B", hidden_size=1024, num_layers=8) == "70B"


def test_load_lora_descriptor(tmp_path: Path) -> None:
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "base",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj"],
                "peft_type": "LORA",
                "scaling": 0.5,
            }
        ),
        encoding="utf-8",
    )

    descriptor = _load_lora_descriptor(lora_dir)
    assert descriptor["config_available"] is True
    assert descriptor["rank"] == 16
    assert descriptor["target_modules"] == ["q_proj", "k_proj"]


def test_load_llm2vec_descriptor(tmp_path: Path) -> None:
    head_dir = tmp_path / "llm2vec"
    head_dir.mkdir()
    (head_dir / "config.json").write_text(
        json.dumps({"projection_dim": 4096, "pooling": "mean", "normalize": True}),
        encoding="utf-8",
    )

    descriptor = _load_llm2vec_descriptor(head_dir, embedding_dim=4096)
    assert descriptor["embedding_dimension"] == 4096
    assert descriptor["projection_dim"] == 4096
    assert descriptor["pooling"] == "mean"
    assert descriptor.get("config_available", True)


def test_build_and_write_model_metadata(tmp_path: Path) -> None:
    package_dir = tmp_path / "model.mlpackage"
    package_dir.mkdir()
    payload = _build_model_metadata(
        model_identifier="repo/model",
        revision="main",
        variant_label="8B",
        config=DummyConfig(),
        seq_len=8192,
        embedding_dim=4096,
        quantization={
            "wbits": 4,
            "group_size": 16,
            "palett_granularity": "per_grouped_channel",
            "mixed_precision_overrides": {"attention": 6},
            "variant_index": 0,
            "variant_count": 1,
        },
        compute_units="ALL",
        lora={"path": "lora"},
        llm2vec={"path": "llm2vec", "embedding_dimension": 4096},
        generated_at="2024-07-01T00:00:00Z",
        size_bytes=123456,
    )

    metadata_path = _write_model_metadata(package_dir, payload)
    assert metadata_path.name == METADATA_FILENAME

    written = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert written["model"]["variant"] == "8B"
    assert written["quantization"]["size_bytes"] == 123456
    assert written["pipeline"]["compute_units"] == "ALL"
    assert written["model"]["embedding_dimension"] == 4096


def test_write_model_metadata_rejects_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing.mlpackage"
    with pytest.raises(FileNotFoundError):
        _write_model_metadata(missing_dir, {"model": {}})
