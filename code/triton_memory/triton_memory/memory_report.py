from dataclasses import asdict, dataclass
from typing import Iterable


MIB = 1024 ** 2


@dataclass(frozen=True)
class CudaMemorySnapshot:
    label: str
    allocated_mib: float
    reserved_mib: float
    peak_mib: float
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _import_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError("PyTorch is required for CUDA memory reporting") from exc
    return torch


def cuda_available() -> bool:
    try:
        torch = _import_torch()
    except RuntimeError:
        return False
    return bool(torch.cuda.is_available())


def capture_cuda_memory(
    label: str,
    *,
    device: str | None = None,
    require_cuda: bool = False,
) -> CudaMemorySnapshot:
    """Capture PyTorch CUDA allocator state in MiB.

    When `require_cuda` is false, this returns a zero-valued snapshot with a note
    on non-CUDA hosts so local Mac tests can exercise the reporting pipeline.
    """

    try:
        torch = _import_torch()
    except RuntimeError:
        if require_cuda:
            raise
        return CudaMemorySnapshot(label, 0.0, 0.0, 0.0, "PyTorch unavailable")

    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA is not available on this host")
        return CudaMemorySnapshot(label, 0.0, 0.0, 0.0, "CUDA unavailable")

    if device is not None:
        torch.cuda.set_device(device)
    torch.cuda.synchronize()
    return CudaMemorySnapshot(
        label=label,
        allocated_mib=torch.cuda.memory_allocated(device) / MIB,
        reserved_mib=torch.cuda.memory_reserved(device) / MIB,
        peak_mib=torch.cuda.max_memory_allocated(device) / MIB,
    )


def format_markdown_table(snapshots: Iterable[CudaMemorySnapshot]) -> str:
    lines = [
        "| Checkpoint | Allocated MiB | Reserved MiB | Peak MiB | Note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for snapshot in snapshots:
        lines.append(
            "| "
            f"{snapshot.label} | "
            f"{snapshot.allocated_mib:.1f} | "
            f"{snapshot.reserved_mib:.1f} | "
            f"{snapshot.peak_mib:.1f} | "
            f"{snapshot.note} |"
        )
    return "\n".join(lines) + "\n"
