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

Run the real memory verification on an NVIDIA CUDA machine:

```bash
python code/triton_memory/scripts/cuda_verify_memory.py \
  --device cuda:0 \
  --base-model convnext_tiny.dinov3_lvd1689m \
  --adapter-a /path/to/adapter_a \
  --adapter-b /path/to/adapter_b \
  --classes-a 5 \
  --classes-b 12 \
  --output-dir code/triton_memory/results
```

Send back the generated JSON and Markdown files from `results/`; those are the source material for the final blog memory table.
