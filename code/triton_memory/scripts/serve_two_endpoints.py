#!/usr/bin/env python
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve two PyTriton endpoints from one shared service object.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classes-a", type=int, default=5)
    parser.add_argument("--classes-b", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    parse_args()
    raise SystemExit(
        "PyTriton serving is intentionally left as a CUDA-host step. "
        "Run cuda_verify_memory.py first; then wire real adapter/head paths into this server."
    )


if __name__ == "__main__":
    raise SystemExit(main())
