# PyTriton Multi-LoRA Memory Analysis

This directory contains the analysis code for the blog post on sharing one CUDA-resident backbone across multiple PyTriton endpoints with different LoRA adapters and heads.

The local development path is intentionally lightweight:

```bash
cd code/triton_memory
python -m unittest discover -s tests
python scripts/cuda_verify_memory.py --help
python scripts/cuda_verify_memory.py --mock --output-dir results
```

Those commands validate the service shape, adapter locking, and result-writing pipeline on an Apple Silicon Mac. They do not prove the CUDA memory claim.

Run the active PyTriton memory verification on an NVIDIA CUDA machine without trained adapters by creating random adapters in memory:

```bash
python code/triton_memory/scripts/cuda_verify_memory.py \
  --device cuda:0 \
  --random-lora \
  --no-pretrained \
  --base-model convnext_tiny.dinov3_lvd1689m \
  --classes-a 5 \
  --classes-b 12 \
  --output-dir code/triton_memory/results \
  --sample-nvidia-smi
```

This command starts real PyTriton endpoints twice:

1. `TaskA` and `TaskB` bound to one shared CUDA-resident backbone with two random LoRA adapters.
2. `TaskA` and `TaskB` bound to two independently constructed CUDA-resident backbones.

Both endpoints are warmed through `pytriton.client.ModelClient`, so the resulting table is about served endpoints rather than eager PyTorch calls. Random LoRA weights validate memory behavior, not prediction quality. Use `--no-pretrained` if the CUDA host should avoid downloading pretrained weights; memory behavior is still useful because the architecture and randomly initialized parameter tensors are the same size.

Send back the generated JSON and Markdown files from `results/`; those are the source material for the final blog memory table.

## Docker on a CUDA Host

The most reliable setup is to run inside NVIDIA's Triton Server container so the Triton Python backend, CUDA runtime, and shared `libpython` library line up.

Build the image from the repository root on the CUDA host:

```bash
docker build \
  -f code/triton_memory/docker/Dockerfile.cuda \
  -t triton-memory:24.10 \
  code/triton_memory
```

Run the active PyTriton verification in the container:

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

If Docker cannot see the GPU, install or fix the NVIDIA Container Toolkit on the host and verify this command first:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the default PyTriton ports are busy, override them:

```bash
python code/triton_memory/scripts/cuda_verify_memory.py \
  --device cuda:0 \
  --random-lora \
  --no-pretrained \
  --pytriton-http-port 8200 \
  --pytriton-grpc-port 8201 \
  --pytriton-metrics-port 8202 \
  --output-dir code/triton_memory/results \
  --sample-nvidia-smi
```
