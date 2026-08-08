#!/usr/bin/env python

import argparse
import gc
import importlib
import json
import platform
import subprocess
import sys
import time
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
)
from triton_memory.shared_service import SharedMultiAdapterService
from triton_memory.shared_service import TaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify shared-backbone CUDA memory behavior for multi-LoRA PyTriton deployment.",
    )
    parser.add_argument("--mock", action="store_true", help="Run the local mock pipeline without requiring CUDA.")
    parser.add_argument(
        "--random-lora",
        action="store_true",
        help="Start PyTriton endpoints with random in-memory PEFT LoRA adapters.",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device for the real verification path.")
    parser.add_argument("--base-model", default="convnext_tiny.dinov3_lvd1689m")
    parser.add_argument("--no-pretrained", action="store_true", help="Create the timm backbone without downloading pretrained weights.")
    parser.add_argument("--classes-a", type=int, default=5)
    parser.add_argument("--classes-b", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-limit", type=int, default=16)
    parser.add_argument("--lora-target-kinds", default="linear,conv2d")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--sample-nvidia-smi", action="store_true")
    parser.add_argument("--pytriton-http-port", type=int, default=8100)
    parser.add_argument("--pytriton-grpc-port", type=int, default=8101)
    parser.add_argument("--pytriton-metrics-port", type=int, default=8102)
    parser.add_argument("--pytriton-model-a", default="TaskA")
    parser.add_argument("--pytriton-model-b", default="TaskB")
    parser.add_argument("--pytriton-settle-seconds", type=float, default=2.0)
    parser.add_argument("--pytriton-client-protocol", choices=("grpc", "http"), default="grpc")
    parser.add_argument("--pytriton-client-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mock:
            result = run_mock_verification(args)
        elif args.random_lora:
            result = run_pytriton_random_lora_cuda_verification(args)
        else:
            raise RuntimeError("CUDA verification requires --random-lora and always starts active PyTriton endpoints")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
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


def run_pytriton_random_lora_cuda_verification(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_cuda(args.device)
    pytriton_modules = require_pytriton_modules()
    snapshots: list[CudaMemorySnapshot] = []
    smi_samples: list[dict[str, str]] = []

    reset_cuda(torch, args.device)
    snapshots.append(capture_cuda_memory("pytriton start after CUDA init", device=args.device, require_cuda=True))
    sample_smi_if_requested(args, smi_samples, "pytriton start")

    shared_service, shared_targets = create_random_lora_shared_service(args, torch, tasks=("task_a", "task_b"))
    shared_endpoint = PyTritonMultiAdapterEndpoint(shared_service, torch_module=torch, device=args.device)
    snapshots.append(capture_cuda_memory("pytriton shared service ready", device=args.device, require_cuda=True))

    input_batch = None
    shared_triton = create_pytriton_server(args, pytriton_modules)
    try:
        bind_pytriton_endpoints(args, shared_triton, shared_endpoint, pytriton_modules)
        snapshots.append(capture_cuda_memory("pytriton shared endpoints bound", device=args.device, require_cuda=True))
        start_pytriton(shared_triton)
        time.sleep(args.pytriton_settle_seconds)
        wait_for_pytriton_model(args, pytriton_modules, args.pytriton_model_a)
        wait_for_pytriton_model(args, pytriton_modules, args.pytriton_model_b)
        snapshots.append(capture_cuda_memory("pytriton shared endpoints active", device=args.device, require_cuda=True))
        sample_smi_if_requested(args, smi_samples, "pytriton shared endpoints active")

        input_batch = torch.randn(
            args.batch_size,
            3,
            args.image_size,
            args.image_size,
            device=args.device,
            dtype=torch.float32,
        ).detach().cpu().numpy()
        infer_pytriton_batch(args, pytriton_modules, args.pytriton_model_a, input_batch)
        snapshots.append(capture_cuda_memory("pytriton shared warm-up task A", device=args.device, require_cuda=True))
        infer_pytriton_batch(args, pytriton_modules, args.pytriton_model_b, input_batch)
        snapshots.append(capture_cuda_memory("pytriton shared warm-up task B", device=args.device, require_cuda=True))
        sample_smi_if_requested(args, smi_samples, "pytriton shared warm-up")
    finally:
        stop_pytriton(shared_triton)

    shared_allocated = snapshots[-1].allocated_mib
    del shared_triton, shared_endpoint, shared_service, input_batch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1.0)

    service_a, targets_a = create_random_lora_shared_service(args, torch, tasks=("task_a",))
    service_b, targets_b = create_random_lora_shared_service(args, torch, tasks=("task_b",))
    endpoint_a = PyTritonMultiAdapterEndpoint(service_a, torch_module=torch, device=args.device)
    endpoint_b = PyTritonMultiAdapterEndpoint(service_b, torch_module=torch, device=args.device)
    snapshots.append(capture_cuda_memory("pytriton duplicated services ready", device=args.device, require_cuda=True))

    duplicated_input_batch = None
    duplicated_triton = create_pytriton_server(args, pytriton_modules)
    try:
        bind_pytriton_model(
            args,
            duplicated_triton,
            args.pytriton_model_a,
            endpoint_a.infer_a,
            args.classes_a,
            pytriton_modules,
        )
        bind_pytriton_model(
            args,
            duplicated_triton,
            args.pytriton_model_b,
            endpoint_b.infer_b,
            args.classes_b,
            pytriton_modules,
        )
        snapshots.append(capture_cuda_memory("pytriton duplicated endpoints bound", device=args.device, require_cuda=True))
        start_pytriton(duplicated_triton)
        time.sleep(args.pytriton_settle_seconds)
        wait_for_pytriton_model(args, pytriton_modules, args.pytriton_model_a)
        wait_for_pytriton_model(args, pytriton_modules, args.pytriton_model_b)
        snapshots.append(capture_cuda_memory("pytriton duplicated endpoints active", device=args.device, require_cuda=True))
        sample_smi_if_requested(args, smi_samples, "pytriton duplicated endpoints active")

        duplicated_input_batch = torch.randn(
            args.batch_size,
            3,
            args.image_size,
            args.image_size,
            device=args.device,
            dtype=torch.float32,
        ).detach().cpu().numpy()
        infer_pytriton_batch(args, pytriton_modules, args.pytriton_model_a, duplicated_input_batch)
        snapshots.append(capture_cuda_memory("pytriton duplicated warm-up task A", device=args.device, require_cuda=True))
        infer_pytriton_batch(args, pytriton_modules, args.pytriton_model_b, duplicated_input_batch)
        snapshots.append(capture_cuda_memory("pytriton duplicated warm-up task B", device=args.device, require_cuda=True))
        sample_smi_if_requested(args, smi_samples, "pytriton duplicated warm-up")
    finally:
        stop_pytriton(duplicated_triton)

    snapshot_by_label = {snapshot.label: snapshot for snapshot in snapshots}
    allocated_by_label = {label: snapshot.allocated_mib for label, snapshot in snapshot_by_label.items()}
    deltas = {
        "pytriton_shared_endpoint_overhead_mib": allocated_by_label["pytriton shared endpoints active"]
        - allocated_by_label["pytriton shared service ready"],
        "pytriton_duplicated_minus_shared_mib": allocated_by_label["pytriton duplicated warm-up task B"]
        - shared_allocated,
        "pytriton_shared_warmup_allocated_mib": allocated_by_label["pytriton shared warm-up task B"],
        "pytriton_shared_warmup_peak_mib": snapshot_by_label["pytriton shared warm-up task B"].peak_mib,
    }
    keep_alive = (duplicated_triton, endpoint_a, endpoint_b, service_a, service_b, duplicated_input_batch)
    if keep_alive is None:  # pragma: no cover - keeps variables live for measurement clarity
        raise AssertionError("unreachable")

    return {
        "mode": "pytriton-random-lora-cuda",
        "created_at": now_iso(),
        "environment": environment_metadata(cuda_required=True, torch_module=torch),
        "arguments": serializable_args(args),
        "lora_target_modules": shared_targets,
        "duplicated_lora_target_modules": {"task_a": targets_a, "task_b": targets_b},
        "snapshots": [snapshot.as_dict() for snapshot in snapshots],
        "deltas": deltas,
        "nvidia_smi_samples": smi_samples,
        "notes": [
            "This mode starts real PyTriton endpoints and warms both endpoints through PyTriton ModelClient.",
            "The shared path binds two endpoint names to one service object and one CUDA-resident backbone.",
            "The duplicated baseline binds two endpoint names backed by two independently constructed CUDA-resident backbones.",
            "Random LoRA weights validate memory behavior, not prediction quality.",
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


class PyTritonMultiAdapterEndpoint:
    def __init__(self, service: SharedMultiAdapterService, *, torch_module: Any, device: str) -> None:
        self.service = service
        self.torch = torch_module
        self.device = device

    def infer_a(self, image: Any) -> dict[str, Any]:
        return self._infer("task_a", image)

    def infer_b(self, image: Any) -> dict[str, Any]:
        return self._infer("task_b", image)

    def _infer(self, task_name: str, image: Any) -> dict[str, Any]:
        tensor = self.torch.as_tensor(image, device=self.device, dtype=self.torch.float32)
        return self.service.infer_task(task_name, tensor)


def create_random_lora_shared_service(
    args: argparse.Namespace,
    torch: Any,
    *,
    tasks: tuple[str, ...],
) -> tuple[SharedMultiAdapterService, list[str]]:
    base = create_timm_backbone(args.base_model, pretrained=not args.no_pretrained, num_classes=0).to(args.device).eval()
    target_modules = find_lora_target_module_names(
        base,
        torch_module=torch,
        target_kinds=parse_lora_target_kinds(args.lora_target_kinds),
        target_limit=args.lora_target_limit,
    )
    lora_config = create_lora_config(target_modules=target_modules, rank=args.lora_rank, alpha=args.lora_alpha)

    first_task = tasks[0]
    shared = attach_first_random_lora_adapter(base, lora_config, adapter_name=first_task).to(args.device).eval()
    for task_name in tasks[1:]:
        attach_random_lora_adapter(shared, lora_config, adapter_name=task_name)

    heads: dict[str, Any] = {}
    task_specs: dict[str, TaskSpec] = {}
    for task_name in tasks:
        num_classes = args.classes_a if task_name == "task_a" else args.classes_b
        heads[task_name] = create_linear_head(feature_dim=args.feature_dim, num_classes=num_classes, device=args.device)
        task_specs[task_name] = TaskSpec(adapter_name=task_name, head_name=task_name)

    return SharedMultiAdapterService(shared, heads, tasks=task_specs), target_modules


def require_pytriton_modules() -> dict[str, Any]:
    try:
        import numpy as np

        client_module = importlib.import_module("pytriton.client")
        decorators_module = importlib.import_module("pytriton.decorators")
        model_config_module = importlib.import_module("pytriton.model_config")
        triton_module = importlib.import_module("pytriton.triton")
    except Exception as exc:  # pragma: no cover - depends on CUDA host setup
        raise RuntimeError("Install pytriton to run active-endpoint CUDA verification") from exc

    return {
        "ModelClient": client_module.ModelClient,
        "Triton": triton_module.Triton,
        "TritonConfig": triton_module.TritonConfig,
        "Tensor": model_config_module.Tensor,
        "batch": decorators_module.batch,
        "np": np,
    }


def create_pytriton_server(args: argparse.Namespace, pytriton_modules: dict[str, Any]) -> Any:
    Triton = pytriton_modules["Triton"]
    TritonConfig = pytriton_modules["TritonConfig"]
    return Triton(
        config=TritonConfig(
            http_port=args.pytriton_http_port,
            grpc_port=args.pytriton_grpc_port,
            metrics_port=args.pytriton_metrics_port,
        )
    )


def bind_pytriton_endpoints(
    args: argparse.Namespace,
    triton: Any,
    endpoint: PyTritonMultiAdapterEndpoint,
    pytriton_modules: dict[str, Any],
) -> None:
    bind_pytriton_model(args, triton, args.pytriton_model_a, endpoint.infer_a, args.classes_a, pytriton_modules)
    bind_pytriton_model(args, triton, args.pytriton_model_b, endpoint.infer_b, args.classes_b, pytriton_modules)


def bind_pytriton_model(
    args: argparse.Namespace,
    triton: Any,
    model_name: str,
    infer_func: Any,
    num_classes: int,
    pytriton_modules: dict[str, Any],
) -> None:
    Tensor = pytriton_modules["Tensor"]
    batch = pytriton_modules["batch"]
    np = pytriton_modules["np"]
    triton.bind(
        model_name=model_name,
        infer_func=batch(infer_func),
        inputs=[Tensor(name="image", dtype=np.float32, shape=(3, args.image_size, args.image_size))],
        outputs=[Tensor(name="logits", dtype=np.float32, shape=(num_classes,))],
    )


def start_pytriton(triton: Any) -> None:
    run = getattr(triton, "run", None)
    if not callable(run):
        raise TypeError("PyTriton object does not provide run()")
    run()


def stop_pytriton(triton: Any) -> None:
    stop = getattr(triton, "stop", None)
    if callable(stop):
        stop()
        time.sleep(1.0)


def infer_pytriton_batch(
    args: argparse.Namespace,
    pytriton_modules: dict[str, Any],
    model_name: str,
    input_batch: Any,
) -> dict[str, Any]:
    ModelClient = pytriton_modules["ModelClient"]
    with ModelClient(
        pytriton_client_url(args),
        model_name,
        init_timeout_s=args.pytriton_client_timeout_seconds,
        inference_timeout_s=args.pytriton_client_timeout_seconds,
    ) as client:
        return client.infer_batch(image=input_batch)


def wait_for_pytriton_model(args: argparse.Namespace, pytriton_modules: dict[str, Any], model_name: str) -> None:
    ModelClient = pytriton_modules["ModelClient"]
    with ModelClient(
        pytriton_client_url(args),
        model_name,
        init_timeout_s=args.pytriton_client_timeout_seconds,
        inference_timeout_s=args.pytriton_client_timeout_seconds,
    ) as client:
        client.wait_for_model(args.pytriton_client_timeout_seconds)


def pytriton_client_url(args: argparse.Namespace) -> str:
    if args.pytriton_client_protocol == "grpc":
        return f"grpc://localhost:{args.pytriton_grpc_port}"
    return f"http://localhost:{args.pytriton_http_port}"


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
