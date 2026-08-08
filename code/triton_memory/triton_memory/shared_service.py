import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    adapter_name: str
    head_name: str


class SharedMultiAdapterService:
    """One backbone object shared by multiple adapter/head routes.

    The service is deliberately small so it can be tested with mock objects on a
    MacBook and reused with real PyTorch/PEFT modules on a CUDA host.
    """

    def __init__(
        self,
        backbone: Any,
        heads: Mapping[str, Any],
        *,
        tasks: Mapping[str, TaskSpec] | None = None,
        output_name: str = "logits",
    ) -> None:
        if not heads:
            raise ValueError("At least one task head is required")

        self.backbone = backbone
        self.heads = dict(heads)
        self.tasks = dict(
            tasks
            or {
                "task_a": TaskSpec("task_a", "task_a"),
                "task_b": TaskSpec("task_b", "task_b"),
            }
        )
        self.output_name = output_name
        self._adapter_lock = threading.Lock()

        self._maybe_eval(self.backbone)
        for head in self.heads.values():
            self._maybe_eval(head)

    def infer_task(self, task_name: str, image: Any) -> dict[str, np.ndarray]:
        try:
            spec = self.tasks[task_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.tasks))
            raise KeyError(f"Unknown task {task_name!r}; expected one of: {known}") from exc

        return self._infer(image, adapter_name=spec.adapter_name, head_name=spec.head_name)

    def infer_a(self, image: Any) -> dict[str, np.ndarray]:
        return self.infer_task("task_a", image)

    def infer_b(self, image: Any) -> dict[str, np.ndarray]:
        return self.infer_task("task_b", image)

    def _infer(self, image: Any, *, adapter_name: str, head_name: str) -> dict[str, np.ndarray]:
        if head_name not in self.heads:
            raise KeyError(f"No head named {head_name!r}")

        with self._adapter_lock:
            if hasattr(self.backbone, "set_adapter"):
                self.backbone.set_adapter(adapter_name)

            with _inference_mode_if_available():
                features = self._call(self.backbone, image)
                logits = self._call(self.heads[head_name], features)
                output = self._to_numpy(logits)

        return {self.output_name: output}

    @staticmethod
    def _call(module: Any, value: Any) -> Any:
        if not callable(module):
            raise TypeError(f"Expected callable module, got {type(module).__name__}")
        return module(value)

    @staticmethod
    def _maybe_eval(module: Any) -> None:
        eval_fn = getattr(module, "eval", None)
        if callable(eval_fn):
            eval_fn()

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        float_fn = getattr(value, "float", None)
        if callable(float_fn):
            value = float_fn()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_fn = getattr(value, "numpy", None)
        if callable(numpy_fn):
            return np.asarray(numpy_fn())
        return np.asarray(value)


class _inference_mode_if_available:
    def __enter__(self) -> None:
        try:
            import torch  # type: ignore
        except Exception:
            self._context = None
            return None

        self._context = torch.inference_mode()
        self._context.__enter__()
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self._context is not None:
            return bool(self._context.__exit__(exc_type, exc, tb))
        return False
