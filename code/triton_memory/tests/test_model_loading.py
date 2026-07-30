from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from triton_memory.model_loading import ADAPTER_CONFIG, validate_peft_adapter_path


class ModelLoadingTest(unittest.TestCase):
    def test_rejects_placeholder_adapter_path(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "placeholder"):
            validate_peft_adapter_path("/path/to/adapter_a")

    def test_rejects_directory_without_adapter_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, ADAPTER_CONFIG):
                validate_peft_adapter_path(directory)

    def test_accepts_directory_with_adapter_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ADAPTER_CONFIG).write_text("{}\n", encoding="utf-8")

            self.assertEqual(validate_peft_adapter_path(path), path)


if __name__ == "__main__":
    unittest.main()
