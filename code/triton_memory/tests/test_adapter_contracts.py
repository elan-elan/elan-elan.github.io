from __future__ import annotations

import unittest

import numpy as np

from triton_memory.adapter_contracts import (
    find_state_dict_mismatches,
    find_trainable_non_adapter_parameters,
)


class FakeParameter:
    def __init__(self, *, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class FakeModule:
    def __init__(self) -> None:
        self.parameters = {
            "stem.weight": FakeParameter(requires_grad=False),
            "norm.weight": FakeParameter(requires_grad=True),
            "blocks.0.lora_A.weight": FakeParameter(requires_grad=True),
        }

    def named_parameters(self):
        return self.parameters.items()


class AdapterContractsTest(unittest.TestCase):
    def test_finds_trainable_non_adapter_parameters(self) -> None:
        violations = find_trainable_non_adapter_parameters(FakeModule())
        self.assertEqual(violations, ["norm.weight"])

    def test_ignores_adapter_state_dict_differences(self) -> None:
        left = {
            "stem.weight": np.array([1.0, 2.0]),
            "blocks.0.lora_A.weight": np.array([1.0]),
        }
        right = {
            "stem.weight": np.array([1.0, 3.0]),
            "blocks.0.lora_A.weight": np.array([9.0]),
        }

        mismatches = find_state_dict_mismatches(left, right)

        self.assertEqual(mismatches, ["stem.weight"])


if __name__ == "__main__":
    unittest.main()
