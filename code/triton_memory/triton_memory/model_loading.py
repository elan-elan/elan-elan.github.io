from __future__ import annotations

from pathlib import Path
from typing import Any


ADAPTER_CONFIG = "adapter_config.json"


def create_timm_backbone(
    model_name: str,
    *,
    pretrained: bool = True,
    num_classes: int = 0,
) -> Any:
    try:
        import timm  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CUDA host setup
        raise RuntimeError("Install timm to create the real ConvNeXt-DINOv3 backbone") from exc

    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def load_first_peft_adapter(base_model: Any, adapter_path: str | Path, *, adapter_name: str) -> Any:
    validate_peft_adapter_path(adapter_path)
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CUDA host setup
        raise RuntimeError("Install peft to load LoRA adapters") from exc

    return PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        adapter_name=adapter_name,
        is_trainable=False,
    )


def load_additional_peft_adapter(model: Any, adapter_path: str | Path, *, adapter_name: str) -> Any:
    validate_peft_adapter_path(adapter_path)
    load_adapter = getattr(model, "load_adapter", None)
    if not callable(load_adapter):
        raise TypeError("model does not provide load_adapter(); expected a PEFT model")
    load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
    return model


def validate_peft_adapter_path(adapter_path: str | Path) -> Path:
    path = Path(adapter_path)
    path_text = str(path)
    if path_text.startswith("/path/to/"):
        raise RuntimeError(
            f"{path_text!r} is a placeholder. Pass a real PEFT adapter directory "
            f"containing {ADAPTER_CONFIG!r}, or use --synthetic-cuda to test the CUDA pipeline first."
        )
    if not path.exists():
        raise RuntimeError(
            f"PEFT adapter path does not exist: {path}. Pass a local adapter directory "
            f"containing {ADAPTER_CONFIG!r}, or use --synthetic-cuda to test the CUDA pipeline first."
        )
    if not path.is_dir():
        raise RuntimeError(f"PEFT adapter path must be a directory, got: {path}")
    config_path = path / ADAPTER_CONFIG
    if not config_path.is_file():
        raise RuntimeError(
            f"PEFT adapter directory is missing {ADAPTER_CONFIG!r}: {path}. "
            "This file is created by PEFT save_pretrained()."
        )
    return path


def create_linear_head(*, feature_dim: int, num_classes: int, device: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CUDA host setup
        raise RuntimeError("Install torch to create fallback linear heads") from exc

    return torch.nn.Linear(feature_dim, num_classes).to(device).eval()


def load_torch_module(path: str | Path, *, device: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CUDA host setup
        raise RuntimeError("Install torch to load serialized heads") from exc

    module = torch.load(str(path), map_location=device)
    if not callable(module):
        raise TypeError(f"Expected {path} to contain a callable torch module")
    to = getattr(module, "to", None)
    if callable(to):
        module = module.to(device)
    eval_fn = getattr(module, "eval", None)
    if callable(eval_fn):
        module.eval()
    return module
