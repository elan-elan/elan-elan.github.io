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
