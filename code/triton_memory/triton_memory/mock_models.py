import threading
import time
from typing import Iterable

import numpy as np


class MockAdapterBackbone:
    """Small adapter-aware model used for local tests without CUDA or PEFT."""

    def __init__(self, *, feature_dim: int = 4, delay_seconds: float = 0.0) -> None:
        self.feature_dim = feature_dim
        self.delay_seconds = delay_seconds
        self.active_adapter = "task_a"
        self.history: list[str] = []
        self._lock = threading.Lock()
        self._active_forwards = 0
        self.max_concurrent_forwards = 0

    def eval(self) -> "MockAdapterBackbone":
        return self

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        self.history.append(adapter_name)

    def __call__(self, image: object) -> np.ndarray:
        with self._lock:
            self._active_forwards += 1
            self.max_concurrent_forwards = max(self.max_concurrent_forwards, self._active_forwards)

        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)

            array = np.asarray(image, dtype=np.float32)
            if array.ndim == 0:
                array = array.reshape(1, 1)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            batch = array.shape[0]
            base = array.reshape(batch, -1).mean(axis=1, keepdims=True)
            offset = 1.0 if self.active_adapter == "task_a" else 10.0
            columns = np.arange(self.feature_dim, dtype=np.float32).reshape(1, -1)
            return base + columns + offset
        finally:
            with self._lock:
                self._active_forwards -= 1


class MockHead:
    def __init__(self, *, num_classes: int, offset: float) -> None:
        self.num_classes = num_classes
        self.offset = offset

    def eval(self) -> "MockHead":
        return self

    def __call__(self, features: object) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        base = array.mean(axis=1, keepdims=True) + self.offset
        columns = np.arange(self.num_classes, dtype=np.float32).reshape(1, -1)
        return base + columns


def make_mock_inputs(*, batch_size: int = 2, shape: Iterable[int] = (3, 8, 8)) -> np.ndarray:
    full_shape = (batch_size, *tuple(shape))
    return np.linspace(0.0, 1.0, num=int(np.prod(full_shape)), dtype=np.float32).reshape(full_shape)
