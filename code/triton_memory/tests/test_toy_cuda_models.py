import unittest


class ToyCudaModelsTest(unittest.TestCase):
    def test_toy_backbone_runs_on_cpu_for_shape_check(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch is not installed")

        from triton_memory.toy_cuda_models import make_toy_backbone

        backbone = make_toy_backbone(
            torch,
            device="cpu",
            feature_dim=8,
            hidden_dim=16,
            adapter_rank=2,
        )
        inputs = torch.randn(2, 3, 8, 8)

        backbone.set_adapter("task_a")
        output_a = backbone(inputs)
        backbone.set_adapter("task_b")
        output_b = backbone(inputs)

        self.assertEqual(tuple(output_a.shape), (2, 8))
        self.assertEqual(tuple(output_b.shape), (2, 8))


if __name__ == "__main__":
    unittest.main()
