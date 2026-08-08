#!/usr/bin/env python

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from triton_memory.mock_models import MockAdapterBackbone, MockHead, make_mock_inputs
from triton_memory.shared_service import SharedMultiAdapterService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress the adapter lock with mock local calls.")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backbone = MockAdapterBackbone(delay_seconds=args.delay_seconds)
    service = SharedMultiAdapterService(
        backbone,
        {
            "task_a": MockHead(num_classes=5, offset=100.0),
            "task_b": MockHead(num_classes=12, offset=200.0),
        },
    )
    inputs = make_mock_inputs(batch_size=2)

    def call(index: int) -> tuple[int, tuple[int, ...]]:
        output = service.infer_a(inputs) if index % 2 == 0 else service.infer_b(inputs)
        return index, tuple(output["logits"].shape)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(call, range(args.requests)))

    print(f"requests={len(results)}")
    print(f"max_concurrent_backbone_forwards={backbone.max_concurrent_forwards}")
    if backbone.max_concurrent_forwards != 1:
        raise RuntimeError("Adapter lock failed: concurrent backbone forwards observed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
