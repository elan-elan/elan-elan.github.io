#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from triton_memory.memory_report import CudaMemorySnapshot, capture_cuda_memory, format_markdown_table
from triton_memory.mock_models import MockAdapterBackbone, MockHead, make_mock_inputs
from triton_memory.model_loading import (
    attach_first_random_lora_adapter,
    attach_random_lora_adapter,
    create_lora_config,
    create_linear_head,
    create_timm_backbone,
    find_lora_target_module_names,
    load_additional_peft_adapter,
    load_first_peft_adapter,
    load_torch_module,
)
from triton_memory.shared_service import SharedMultiAdapterService
from triton_memory.toy_cuda_models import make_toy_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify shared-backbone CUDA memory behavior for multi-LoRA PyTriton deployment.",
    )
    parser.add_argument("--mock", action="store_true", help="Run the local mock pipeline without requiring CUDA.")
    parser.add_argument(
        "--synthetic-cuda",
        action="store_true",
        help="Run a CUDA allocator test with synthetic torch adapters instead of real PEFT adapter files.",
    )
    parser.add_argument(
        "--random-lora",
        action="store_true",
        help="Create random PEFT LoRA adapters in memory instead of loading adapter directories.",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device for the real verification path.")
    parser.add_argument("--base-model", default="convnext_tiny.dinov3_lvd1689m")
    parser.add_argument("--no-pretrained", action="store_true", help="Create the timm backbone without downloading pretrained weights.")
    parser.add_argument("--adapter-a", type=Path)
    parser.add_argument("--adapter-b", type=Path)
    parser.add_argument("--head-a", type=Path)
    parser.add_argument("--head-b", type=Path)
    parser.add_argument("--classes-a", type=int, default=5)
    parser.add_argument("--classes-b", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--toy-hidden-dim", type=int, default=4096)
    parser.add_argument("--toy-adapter-rank", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-limit", type=int, default=16)
    parser.add_argument("--lora-target-kinds", default="linear,conv2d")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--sample-nvidia-smi", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mock:
            result = run_mock_verification(args)
        elif args.synthetic_cuda:
            result = run_synthetic_cuda_verification(args)
        elif args.random_lora:
            result = run_random_lora_cuda_verification(args)
        else:
            result = run_cuda_verification(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_path, markdown_path = write_outputs(result, args.output_dir)
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {markdown_path}")
    return 0


def run_mock_verification(args: argparse.Namespace) -> dict[str, Any]:
    backbone = MockAdapterBackbone(feature_dim=4)
    service = SharedMultiAdapterService(
        backbone,
        {
            "task_a": MockHead(num_classes=args.classes_a, offset=100.0),
            "task_b": MockHead(num_classes=args.classes_b, offset=200.0),
        },
    )
    inputs = make_mock_inputs(batch_size=args.batch_size)
    output_a = service.infer_a(inputs)["logits"]
    output_b = service.infer_b(inputs)["logits"]

    snapshots = [
        CudaMemorySnapshot("mock start", 0.0, 0.0, 0.0, "local mock; no CUDA measurement"),
        CudaMemorySnapshot("mock after task A", 0.0, 0.0, 0.0, f"shape={tuple(output_a.shape)}"),
        CudaMemorySnapshot("mock after task B", 0.0, 0.0, 0.0, f"shape={tuple(output_b.shape)}"),
    ]
    return {
        "mode": "mock",
        "created_at": now_iso(),
        "environment": environment_metadata(cuda_required=False),
        "snapshots": [snapshot.as_dict() for snapshot in snapshots],
        "deltas": {},
        "notes": [
            "Mock mode validates the local pipeline only.",
            "Run without --mock on an NVIDIA CUDA host for the blog memory claim.",
        ],
    }


def run_cuda_verification(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_cuda(args.device)
    require_real_inputs(args)

    snapshots: list[CudaMemorySnapshot] = []
    smi_samples: list[dict[str, str]] = []

    reset_cuda(torch, args.device)
    snapshots.append(capture_cuda_memory("start after CUDA init", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "start")

    base = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    snapshots.append(capture_cuda_memory("one base backbone", device=args.device, require_cuda=True))

    shared = load_first_peft_adapter(base, args.adapter_a, adapter_name="task_a").to(args.device).eval()
    snapshots.append(capture_cuda_memory("shared + adapter A", device=args.device, require_cuda=True))

    load_additional_peft_adapter(shared, args.adapter_b, adapter_name="task_b")
    snapshots.append(capture_cuda_memory("shared + adapters A/B", device=args.device, require_cuda=True))

    heads = {
        "task_a": load_torch_module(args.head_a, device=args.device)
        if args.head_a
        else create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_a, device=args.device),
        "task_b": load_torch_module(args.head_b, device=args.device)
        if args.head_b
        else create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_b, device=args.device),
    }
    service = SharedMultiAdapterService(shared, heads)
    snapshots.append(capture_cuda_memory("shared + adapters + heads", device=args.device, require_cuda=True))

    inputs = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=args.device,
        dtype=torch.float32,
    )
    service.infer_a(inputs)
    snapshots.append(capture_cuda_memory("after warm-up task A", device=args.device, require_cuda=True))
    service.infer_b(inputs)
    snapshots.append(capture_cuda_memory("after warm-up task B", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "after shared warm-up")

    shared_allocated = snapshots[-1].allocated_mib
    del service, heads, shared, base, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base_a = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    model_a = load_first_peft_adapter(base_a, args.adapter_a, adapter_name="task_a").to(args.device).eval()
    _head_a = load_torch_module(args.head_a, device=args.device) if args.head_a else create_linear_head(
        feature_dim=args.feature_dim,
        num_classes=args.classes_a,
        device=args.device,
    )
    snapshots.append(capture_cuda_memory("duplicated baseline model A", device=args.device, require_cuda=True))

    base_b = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    model_b = load_first_peft_adapter(base_b, args.adapter_b, adapter_name="task_b").to(args.device).eval()
    _head_b = load_torch_module(args.head_b, device=args.device) if args.head_b else create_linear_head(
        feature_dim=args.feature_dim,
        num_classes=args.classes_b,
        device=args.device,
    )
    snapshots.append(capture_cuda_memory("duplicated baseline models A/B", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "after duplicated baseline")

    allocated_by_label = {snapshot.label: snapshot.allocated_mib for snapshot in snapshots}
    deltas = {
        "adapter_b_incremental_mib": allocated_by_label["shared + adapters A/B"]
        - allocated_by_label["shared + adapter A"],
        "duplicated_minus_shared_mib": allocated_by_label["duplicated baseline models A/B"] - shared_allocated,
        "warmup_peak_mib": snapshots[-3].peak_mib,
    }

    return {
        "mode": "cuda",
        "created_at": now_iso(),
        "environment": environment_metadata(cuda_required=True, torch_module=torch),
        "arguments": serializable_args(args),
        "snapshots": [snapshot.as_dict() for snapshot in snapshots],
        "deltas": deltas,
        "nvidia_smi_samples": smi_samples,
        "notes": [
            "Use memory_allocated for parameter-copy comparisons; reserved memory can remain high because of the PyTorch allocator.",
            "nvidia-smi samples are process-level corroboration, not the primary allocator metric.",
        ],
    }


def run_random_lora_cuda_verification(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_cuda(args.device)
    snapshots: list[CudaMemorySnapshot] = []
    smi_samples: list[dict[str, str]] = []
    target_kinds = parse_lora_target_kinds(args.lora_target_kinds)

    reset_cuda(torch, args.device)
    snapshots.append(capture_cuda_memory("random LoRA start after CUDA init", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "random LoRA start")

    base = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    snapshots.append(capture_cuda_memory("random LoRA one base backbone", device=args.device, require_cuda=True))

    target_modules = find_lora_target_module_names(
        base,
        torch_module=torch,
        target_kinds=target_kinds,
        target_limit=args.lora_target_limit,
    )
    lora_config = create_lora_config(
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    shared = attach_first_random_lora_adapter(base, lora_config, adapter_name="task_a").to(args.device).eval()
    snapshots.append(capture_cuda_memory("random LoRA shared + adapter A", device=args.device, require_cuda=True))

    attach_random_lora_adapter(shared, lora_config, adapter_name="task_b")
    snapshots.append(capture_cuda_memory("random LoRA shared + adapters A/B", device=args.device, require_cuda=True))

    heads = {
        "task_a": create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_a, device=args.device),
        "task_b": create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_b, device=args.device),
    }
    service = SharedMultiAdapterService(shared, heads)
    snapshots.append(capture_cuda_memory("random LoRA shared + adapters + heads", device=args.device, require_cuda=True))

    inputs = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=args.device,
        dtype=torch.float32,
    )
    service.infer_a(inputs)
    snapshots.append(capture_cuda_memory("random LoRA warm-up task A", device=args.device, require_cuda=True))
    service.infer_b(inputs)
    snapshots.append(capture_cuda_memory("random LoRA warm-up task B", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "random LoRA shared warm-up")

    shared_allocated = snapshots[-1].allocated_mib
    del service, heads, shared, base, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base_a = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    target_modules_a = find_lora_target_module_names(
        base_a,
        torch_module=torch,
        target_kinds=target_kinds,
        target_limit=args.lora_target_limit,
    )
    lora_config_a = create_lora_config(target_modules=target_modules_a, rank=args.lora_rank, alpha=args.lora_alpha)
    model_a = attach_first_random_lora_adapter(base_a, lora_config_a, adapter_name="task_a").to(args.device).eval()
    _head_a = create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_a, device=args.device)
    snapshots.append(capture_cuda_memory("random LoRA duplicated model A", device=args.device, require_cuda=True))

    base_b = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    target_modules_b = find_lora_target_module_names(
        base_b,
        torch_module=torch,
        target_kinds=target_kinds,
        target_limit=args.lora_target_limit,
    )
    lora_config_b = create_lora_config(target_modules=target_modules_b, rank=args.lora_rank, alpha=args.lora_alpha)
    model_b = attach_first_random_lora_adapter(base_b, lora_config_b, adapter_name="task_b").to(args.device).eval()
    _head_b = create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_b, device=args.device)
    snapshots.append(capture_cuda_memory("random LoRA duplicated models A/B", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "random LoRA duplicated baseline")

    allocated_by_label = {snapshot.label: snapshot.allocated_mib for snapshot in snapshots}
    deltas = {
        "adapter_b_incremental_mib": allocated_by_label["random LoRA shared + adapters A/B"]
        - allocated_by_label["random LoRA shared + adapter A"],
        "duplicated_minus_shared_mib": allocated_by_label["random LoRA duplicated models A/B"] - shared_allocated,
        "warmup_peak_mib": snapshots[-3].peak_mib,
    }
    keep_alive = (model_a, model_b, _head_a, _head_b)
    if keep_alive is None:  # pragma: no cover - keeps variables live for measurement clarity
        raise AssertionError("unreachable")

    return {
        "mode": "random-lora-cuda",
        "created_at": now_iso(),
        "environment": environment_metadata(cuda_required=True, torch_module=torch),
        "arguments": serializable_args(args),
        "lora_target_modules": target_modules,
        "snapshots": [snapshot.as_dict() for snapshot in snapshots],
        "deltas": deltas,
        "nvidia_smi_samples": smi_samples,
        "notes": [
            "Random LoRA mode creates in-memory PEFT adapters and measures memory without trained adapter files.",
            "The weights are random; use this for memory behavior, not model quality.",
        ],
    }


def run_synthetic_cuda_verification(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_cuda(args.device)
    snapshots: list[CudaMemorySnapshot] = []
    smi_samples: list[dict[str, str]] = []

    reset_cuda(torch, args.device)
    snapshots.append(capture_cuda_memory("synthetic start after CUDA init", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "synthetic start")

    shared = make_toy_backbone(
        torch,
        device=args.device,
        feature_dim=args.feature_dim,
        hidden_dim=args.toy_hidden_dim,
        adapter_rank=args.toy_adapter_rank,
    )
    snapshots.append(capture_cuda_memory("synthetic shared backbone + adapters", device=args.device, require_cuda=True))

    heads = {
        "task_a": create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_a, device=args.device),
        "task_b": create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_b, device=args.device),
    }
    service = SharedMultiAdapterService(shared, heads)
    snapshots.append(capture_cuda_memory("synthetic shared + heads", device=args.device, require_cuda=True))

    inputs = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=args.device,
        dtype=torch.float32,
    )
    service.infer_a(inputs)
    snapshots.append(capture_cuda_memory("synthetic warm-up task A", device=args.device, require_cuda=True))
    service.infer_b(inputs)
    snapshots.append(capture_cuda_memory("synthetic warm-up task B", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "synthetic shared warm-up")

    shared_allocated = snapshots[-1].allocated_mib
    del service, heads, shared, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    duplicated_a = make_toy_backbone(
        torch,
        device=args.device,
        feature_dim=args.feature_dim,
        hidden_dim=args.toy_hidden_dim,
        adapter_rank=args.toy_adapter_rank,
    )
    _head_a = create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_a, device=args.device)
    snapshots.append(capture_cuda_memory("synthetic duplicated model A", device=args.device, require_cuda=True))

    duplicated_b = make_toy_backbone(
        torch,
        device=args.device,
        feature_dim=args.feature_dim,
        hidden_dim=args.toy_hidden_dim,
        adapter_rank=args.toy_adapter_rank,
    )
    _head_b = create_linear_head(feature_dim=args.feature_dim, num_classes=args.classes_b, device=args.device)
    snapshots.append(capture_cuda_memory("synthetic duplicated models A/B", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "synthetic duplicated baseline")

    allocated_by_label = {snapshot.label: snapshot.allocated_mib for snapshot in snapshots}
    deltas = {
        "synthetic_duplicated_minus_shared_mib": allocated_by_label["synthetic duplicated models A/B"]
        - shared_allocated,
        "synthetic_warmup_peak_mib": snapshots[-3].peak_mib,
    }
    keep_alive = (duplicated_a, duplicated_b, _head_a, _head_b)
    if keep_alive is None:  # pragma: no cover - keeps variables live for measurement clarity
        raise AssertionError("unreachable")

    return {
        "mode": "synthetic-cuda",
        "created_at": now_iso(),
        "environment": environment_metadata(cuda_required=True, torch_module=torch),
        "arguments": serializable_args(args),
        "snapshots": [snapshot.as_dict() for snapshot in snapshots],
        "deltas": deltas,
        "nvidia_smi_samples": smi_samples,
        "notes": [
            "Synthetic CUDA mode validates the allocator/reporting pipeline without PEFT adapter files.",
            "Use real adapter directories for the final blog claim.",
        ],
    }


def require_cuda(device: str):
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyTorch is required for real CUDA verification") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Re-run with --mock locally or use an NVIDIA GPU host.")
    torch.cuda.set_device(device)
    return torch


def require_real_inputs(args: argparse.Namespace) -> None:
    missing = [name for name in ("adapter_a", "adapter_b") if getattr(args, name) is None]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise RuntimeError(
            f"Real CUDA verification requires adapter paths: {joined}. "
            "Use --random-lora to create random PEFT adapters in memory, or --synthetic-cuda for a non-PEFT smoke test."
        )


def reset_cuda(torch: Any, device: str) -> None:
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def environment_metadata(*, cuda_required: bool, torch_module: Any | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import numpy as np

        metadata["numpy"] = np.__version__
    except Exception:
        metadata["numpy"] = None

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch  # type: ignore

            torch = imported_torch
        except Exception:
            torch = None
    if torch is not None:
        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["torch_cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            metadata["gpu_name"] = props.name
            metadata["gpu_total_memory_mib"] = props.total_memory / (1024 ** 2)
    elif cuda_required:
        metadata["torch"] = None

    for package_name in ("timm", "peft", "pytriton"):
        metadata[package_name] = package_version(package_name)
    return metadata


def package_version(package_name: str) -> str | None:
    try:
        module = __import__(package_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def sample_smi_if_requested(args: argparse.Namespace, samples: list[dict[str, str]], label: str) -> None:
    if not args.sample_nvidia_smi:
        return
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        samples.append({"label": label, "error": str(exc)})
        return
    samples.append({"label": label, "output": completed.stdout.strip()})


def parse_lora_target_kinds(value: str) -> tuple[str, ...]:
    kinds = tuple(kind.strip().lower() for kind in value.split(",") if kind.strip())
    allowed = {"linear", "conv2d"}
    unknown = sorted(set(kinds) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise RuntimeError(f"Unknown LoRA target kind(s): {joined}. Expected linear, conv2d, or both.")
    if not kinds:
        raise RuntimeError("At least one LoRA target kind is required")
    return kinds


def write_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}-{result['mode']}-memory"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshots = [CudaMemorySnapshot(**snapshot) for snapshot in result["snapshots"]]
    markdown = [
        f"# {result['mode'].upper()} Memory Verification",
        "",
        f"Created: `{result['created_at']}`",
        "",
        format_markdown_table(snapshots),
        "## Deltas",
        "",
    ]
    if result["deltas"]:
        for name, value in result["deltas"].items():
            markdown.append(f"- `{name}`: {value:.1f} MiB")
    else:
        markdown.append("- No CUDA deltas in mock mode.")
    markdown.append("")
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    return json_path, markdown_path


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
