from __future__ import annotations

from typing import Any


class ToyCudaBackbone:
    """CUDA-friendly adapter model for pipeline tests without PEFT assets."""

    def __init__(
        self,
        *,
        torch_module: Any,
        device: str,
        feature_dim: int,
        hidden_dim: int,
        adapter_rank: int,
    ) -> None:
        torch = torch_module
        self.device = device
        self.feature_dim = feature_dim
        self.active_adapter = "task_a"
        self.base = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        ).to(device).eval()
        self.adapters = {
            "task_a": _ToyAdapter(torch, feature_dim, adapter_rank).to(device).eval(),
            "task_b": _ToyAdapter(torch, feature_dim, adapter_rank).to(device).eval(),
        }

    def eval(self) -> "ToyCudaBackbone":
        self.base.eval()
        for adapter in self.adapters.values():
            adapter.eval()
        return self

    def set_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.adapters:
            raise KeyError(f"Unknown toy adapter {adapter_name!r}")
        self.active_adapter = adapter_name

    def __call__(self, image: Any) -> Any:
        features = self.base(image)
        return features + self.adapters[self.active_adapter](features)

    def to(self, device: str) -> "ToyCudaBackbone":
        self.device = device
        self.base.to(device)
        for adapter in self.adapters.values():
            adapter.to(device)
        return self


class _ToyAdapter:
    def __init__(self, torch_module: Any, feature_dim: int, adapter_rank: int) -> None:
        torch = torch_module
        self.module = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, adapter_rank, bias=False),
            torch.nn.Linear(adapter_rank, feature_dim, bias=False),
        )

    def __call__(self, features: Any) -> Any:
        return self.module(features)

    def train(self, mode: bool = True) -> "_ToyAdapter":
        self.module.train(mode)
        return self

    def eval(self) -> "_ToyAdapter":
        self.module.eval()
        return self

    def to(self, device: str) -> "_ToyAdapter":
        self.module.to(device)
        return self


def make_toy_backbone(
    torch_module: Any,
    *,
    device: str,
    feature_dim: int,
    hidden_dim: int,
    adapter_rank: int,
) -> ToyCudaBackbone:
    return ToyCudaBackbone(
        torch_module=torch_module,
        device=device,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        adapter_rank=adapter_rank,
    )
