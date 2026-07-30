from __future__ import annotations

from pathlib import Path
from typing import Any


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
    load_adapter = getattr(model, "load_adapter", None)
    if not callable(load_adapter):
        raise TypeError("model does not provide load_adapter(); expected a PEFT model")
    load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
    return model


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
