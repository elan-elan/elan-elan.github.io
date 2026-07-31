from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


DEFAULT_ADAPTER_MARKERS = ("adapter", "lora")


def find_trainable_non_adapter_parameters(
    module: Any,
    *,
    adapter_markers: Iterable[str] = DEFAULT_ADAPTER_MARKERS,
) -> list[str]:
    """Return trainable parameter names that do not look adapter-specific."""

    markers = tuple(marker.lower() for marker in adapter_markers)
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("module must provide named_parameters()")

    violations: list[str] = []
    for name, parameter in named_parameters():
        requires_grad = bool(getattr(parameter, "requires_grad", False))
        if requires_grad and not any(marker in name.lower() for marker in markers):
            violations.append(name)
    return violations


def assert_no_trainable_non_adapter_parameters(module: Any) -> None:
    violations = find_trainable_non_adapter_parameters(module)
    if violations:
        joined = ", ".join(violations)
        raise AssertionError(f"Non-adapter parameters are trainable: {joined}")


def find_state_dict_mismatches(
    left: Any,
    right: Any,
    *,
    ignored_markers: Iterable[str] = DEFAULT_ADAPTER_MARKERS,
) -> list[str]:
    """Compare two state dicts while ignoring adapter-specific keys."""

    left_state = _state_dict(left)
    right_state = _state_dict(right)
    markers = tuple(marker.lower() for marker in ignored_markers)
    keys = sorted(set(left_state) | set(right_state))

    mismatches: list[str] = []
    for key in keys:
        lowered = key.lower()
        if any(marker in lowered for marker in markers):
            continue
        if key not in left_state or key not in right_state:
            mismatches.append(key)
            continue
        if not _values_equal(left_state[key], right_state[key]):
            mismatches.append(key)
    return mismatches


def _state_dict(module: Any) -> dict[str, Any]:
    if isinstance(module, dict):
        return module
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("module must be a state dict or provide state_dict()")
    return dict(state_dict())


def _values_equal(left: Any, right: Any) -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    if torch is not None and hasattr(torch, "is_tensor"):
        if torch.is_tensor(left) or torch.is_tensor(right):
            return bool(torch.equal(left, right))
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))
