import threading
import unittest

from triton_memory.mock_models import MockAdapterBackbone, MockHead, make_mock_inputs
from triton_memory.shared_service import SharedMultiAdapterService


class SharedMultiAdapterServiceTest(unittest.TestCase):
    def test_routes_two_tasks_through_one_backbone_object(self) -> None:
        backbone = MockAdapterBackbone()
        service = SharedMultiAdapterService(
            backbone,
            {
                "task_a": MockHead(num_classes=5, offset=100.0),
                "task_b": MockHead(num_classes=12, offset=200.0),
            },
        )
        inputs = make_mock_inputs(batch_size=3)

        output_a = service.infer_a(inputs)["logits"]
        output_b = service.infer_b(inputs)["logits"]

        self.assertIs(service.backbone, backbone)
        self.assertEqual(output_a.shape, (3, 5))
        self.assertEqual(output_b.shape, (3, 12))
        self.assertEqual(backbone.history, ["task_a", "task_b"])
        self.assertGreater(output_b.mean(), output_a.mean())

    def test_adapter_lock_serializes_backbone_forward(self) -> None:
        backbone = MockAdapterBackbone(delay_seconds=0.01)
        service = SharedMultiAdapterService(
            backbone,
            {
                "task_a": MockHead(num_classes=5, offset=100.0),
                "task_b": MockHead(num_classes=12, offset=200.0),
            },
        )
        inputs = make_mock_inputs(batch_size=1)
        threads: list[threading.Thread] = []

        for index in range(12):
            target = service.infer_a if index % 2 == 0 else service.infer_b
            thread = threading.Thread(target=target, args=(inputs,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        self.assertEqual(backbone.max_concurrent_forwards, 1)


if __name__ == "__main__":
    unittest.main()
