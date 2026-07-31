# Saving GPU Memory With PyTriton And Multiple LoRA Adapters

<p class="blog-post-date">2026-07-30</p>

The easiest way to waste GPU memory in a multi-task LoRA deployment is to load the same backbone more than once.

Suppose two endpoints use the same ConvNeXt backbone, but different LoRA adapters and different prediction heads. A direct implementation often creates two complete model objects:

```text
endpoint A -> backbone copy A + LoRA A + head A
endpoint B -> backbone copy B + LoRA B + head B
```

That works, but it pays for the backbone twice. The better shape is to keep one CUDA-resident backbone object, attach both adapters to it, and let the endpoints route through the same Python service object.

```text
endpoint A -> shared backbone + LoRA A + head A
endpoint B -> shared backbone + LoRA B + head B
```

<p class="cvpr-callout">The key idea is simple: PyTriton can expose multiple endpoints, but GPU memory is saved only if those endpoints share the same Python model object.</p>

This post walks through the pattern with small code examples, then shows the CUDA memory measurement from an active PyTriton random-LoRA test run on a Tesla T4. The reproducible command is included below, and the measurement script is linked so the exact serving setup is easy to inspect.

## The Deployment Shape

The service owns one backbone and a small task table. Each task names the adapter and head it should use.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    adapter_name: str
    head_name: str


tasks = {
    "task_a": TaskSpec(adapter_name="task_a", head_name="task_a"),
    "task_b": TaskSpec(adapter_name="task_b", head_name="task_b"),
}
```

The important part is that the backbone is passed in once:

```python
service = SharedMultiAdapterService(
    backbone=shared_peft_model,
    heads={
        "task_a": task_a_head,
        "task_b": task_b_head,
    },
    tasks=tasks,
)
```

Both endpoints should call methods on this same `service` object. They should not each build their own backbone.

## Switching Adapters Safely

PEFT's `set_adapter()` changes active adapter state on the model. That means adapter selection and forward inference should be protected by a lock when the same model object serves multiple endpoints.

The core service logic is small:

```python
import threading
import torch


class SharedMultiAdapterService:
    def __init__(self, backbone, heads, tasks):
        self.backbone = backbone
        self.heads = heads
        self.tasks = tasks
        self._adapter_lock = threading.Lock()

    def infer_task(self, task_name, image):
        spec = self.tasks[task_name]

        with self._adapter_lock:
            self.backbone.set_adapter(spec.adapter_name)

            with torch.inference_mode():
                features = self.backbone(image)
                logits = self.heads[spec.head_name](features)

        return {"logits": logits.detach().float().cpu().numpy()}
```

This is not clever. That is the point. The service makes object sharing explicit, and the lock makes adapter switching explicit.

## What PyTriton Adds

PyTriton gives the service two model endpoints. The endpoints can still point back to the same service instance:

```python
from pytriton.decorators import batch


class TritonEndpoints:
    def __init__(self, service):
        self.service = service

    @batch
    def infer_task_a(self, image):
        return self.service.infer_task("task_a", image)

    @batch
    def infer_task_b(self, image):
        return self.service.infer_task("task_b", image)
```

The exact `Triton.bind()` code depends on the input and output tensor contracts, but the memory rule does not change: bind methods on the same object if the model should be shared.

## Building Random LoRA Adapters

For memory testing, the adapters do not need to be trained. Random LoRA adapters are enough because the question is parameter memory, not prediction quality.

The test harness creates a timm backbone, finds supported LoRA targets, and attaches two in-memory PEFT adapters:

```python
base = timm.create_model(
    "convnext_tiny.dinov3_lvd1689m",
    pretrained=False,
    num_classes=0,
).to("cuda:0").eval()

target_modules = find_lora_target_module_names(
    base,
    torch_module=torch,
    target_kinds=("linear", "conv2d"),
    target_limit=16,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.0,
    bias="none",
)

shared = get_peft_model(base, lora_config, adapter_name="task_a")
shared.add_adapter("task_b", lora_config)
```

One practical detail matters for ConvNeXt: depthwise/grouped `Conv2d` layers are skipped. PEFT supports LoRA on grouped convolutions only when the rank is compatible with the group count, and ConvNeXt has layers such as `groups=96`. Skipping grouped convolutions keeps the memory test focused and avoids a target-selection error.

## The CUDA Test

The [measurement script](https://github.com/elan-elan/elan-elan.github.io/blob/triton-test/code/triton_memory/scripts/cuda_verify_memory.py) starts a real PyTriton server and compares two paths:

- **Shared path:** `TaskA` and `TaskB` are two PyTriton model names bound to one service object, one CUDA-resident backbone, two random LoRA adapters, and two heads.
- **Duplicated path:** `TaskA` and `TaskB` are two PyTriton model names backed by two independently constructed CUDA-resident backbones.

Both endpoints are warmed through `pytriton.client.ModelClient`, so the result is a serving-path measurement rather than a direct eager-mode call.

The most reproducible way to run it is inside NVIDIA's Triton Server container. The [Dockerfile](https://github.com/elan-elan/elan-elan.github.io/blob/triton-test/code/triton_memory/docker/Dockerfile.cuda) starts from `nvcr.io/nvidia/tritonserver:24.10-py3`, then installs PyTorch, timm, PEFT, and PyTriton.

If this is a fresh CUDA host, first check that Docker can see the GPU:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

From the repository root on a CUDA host, build the image:

```bash
docker build \
    -f code/triton_memory/docker/Dockerfile.cuda \
    -t triton-memory:24.10 \
    code/triton_memory
```

Then run the active PyTriton memory test:

```bash
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=8g \
    -v "$PWD":/workspace \
    -w /workspace \
    -p 8200:8200 \
    -p 8201:8201 \
    -p 8202:8202 \
    triton-memory:24.10 \
    python3 code/triton_memory/scripts/cuda_verify_memory.py \
        --device cuda:0 \
        --random-lora \
        --no-pretrained \
        --base-model convnext_tiny.dinov3_lvd1689m \
        --classes-a 5 \
        --classes-b 12 \
        --output-dir code/triton_memory/results \
        --sample-nvidia-smi \
        --pytriton-http-port 8200 \
        --pytriton-grpc-port 8201 \
        --pytriton-metrics-port 8202 \
        --pytriton-client-protocol grpc \
        --pytriton-client-timeout-seconds 300
```

The script writes JSON and Markdown summaries under `code/triton_memory/results/`. The raw JSON for this run is checked in as [`20260731T011619Z-pytriton-random-lora-cuda-memory.json`](https://github.com/elan-elan/elan-elan.github.io/blob/triton-test/code/triton_memory/results/20260731T011619Z-pytriton-random-lora-cuda-memory.json).

The run used a Tesla T4 with PyTriton `0.7.0`, PyTorch `2.6.0+cu124`, PEFT `0.20.0`, timm `1.0.28`, and CUDA `12.4`.

The recorded configuration used batch size `2`, image size `224`, feature dimension `768`, LoRA rank `8`, LoRA alpha `16`, and `16` selected LoRA target modules from ConvNeXt. The selected targets were the stem convolution, early MLP `fc1`/`fc2` layers, and early downsample convolutions; the full module list is in the JSON.

## Result

| Checkpoint | Allocated MiB | Reserved MiB | Peak MiB |
| --- | ---: | ---: | ---: |
| Shared service ready | 107.8 | 128.0 | 107.8 |
| Shared endpoints active | 107.8 | 128.0 | 107.8 |
| Shared after TaskA warm-up | 115.9 | 172.0 | 152.8 |
| Shared after TaskB warm-up | 124.1 | 192.0 | 160.9 |
| Duplicated services ready | 230.6 | 254.0 | 230.6 |
| Duplicated endpoints active | 230.6 | 254.0 | 230.6 |
| Duplicated after both warm-ups | 230.6 | 298.0 | 267.0 |

Binding two model names to the shared service added `0.0 MiB` of PyTorch allocated memory before warm-up. The duplicated two-model PyTriton baseline used `106.5 MiB` more allocated memory than the shared warmed-up service.

Put another way:

| Comparison | Value |
| --- | ---: |
| Shared after both PyTriton warm-ups | 124.1 MiB |
| Duplicated after both PyTriton warm-ups | 230.6 MiB |
| Allocated memory saved | 106.5 MiB |
| Savings vs duplicated baseline | 46.2% |
| Duplicated/shared ratio | 1.86x |

The `nvidia-smi` process samples pointed in the same direction:

| `nvidia-smi` checkpoint | Python process MiB | Triton backend MiB | Total MiB |
| --- | ---: | ---: | ---: |
| PyTriton start | 102 | - | 102 |
| Shared endpoints active | 230 | 166 | 396 |
| Shared after warm-up | 332 | 166 | 498 |
| Duplicated endpoints active | 394 | 166 | 560 |
| Duplicated after warm-up | 438 | 166 | 604 |

After the shared warm-up, the Python process plus Triton backend process summed to about `498 MiB` (`332 + 166`). After the duplicated warm-up, they summed to about `604 MiB` (`438 + 166`). That process-level difference is also `106 MiB`, but the absolute number is different from `torch.cuda.memory_allocated()` because `nvidia-smi` sees the whole process, CUDA context, and Triton backend, not only live tensors tracked by the PyTorch allocator.

## How To Read The Numbers

The important comparison is not reserved memory. PyTorch's caching allocator may keep reserved blocks after tensors are freed. For parameter-copy comparisons, `memory_allocated()` is the cleaner signal.

The result says three useful things:

- The shared service sits at about `108 MiB` before requests.
- Adding two served PyTriton model names to that same service adds no extra PyTorch parameter allocation by itself.
- Duplicating the model uses `106.5 MiB` more allocated memory for this two-task test.

That is exactly the deployment lesson: memory sharing is an object-graph property. PyTriton will serve what you bind, but it will not deduplicate two independently constructed Python model objects for you.

## Caveats

This test used random LoRA adapters and `pretrained=False`. That is fine for memory behavior because the tensors have the same shapes, but it says nothing about prediction quality.

The run measures an active PyTriton serving path, but it is still a small controlled memory test. It is not a throughput or latency benchmark.

The exact MiB values will change with the backbone, adapter rank, target modules, batch size, dtype, GPU, PyTorch version, and allocator state. The durable lesson is the shape of the comparison: one shared backbone plus small adapters is much cheaper than one full backbone per endpoint.

## Takeaway

If two LoRA endpoints use the same backbone, make that sharing explicit in Python:

```text
one process
one service object
one CUDA-resident backbone
many adapters
many small heads
many PyTriton endpoints
```

That pattern keeps deployment simple and makes GPU memory scale with adapters and heads instead of with repeated backbone copies.

The supporting code lives in the [`code/triton_memory`](https://github.com/elan-elan/elan-elan.github.io/tree/triton-test/code/triton_memory) directory. The two most relevant files are the [measurement script](https://github.com/elan-elan/elan-elan.github.io/blob/triton-test/code/triton_memory/scripts/cuda_verify_memory.py) and the [CUDA Dockerfile](https://github.com/elan-elan/elan-elan.github.io/blob/triton-test/code/triton_memory/docker/Dockerfile.cuda).