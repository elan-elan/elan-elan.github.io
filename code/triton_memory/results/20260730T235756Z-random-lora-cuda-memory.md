# RANDOM-LORA-CUDA Memory Verification

Created: `2026-07-30T23:57:56.990692+00:00`

| Checkpoint | Allocated MiB | Reserved MiB | Peak MiB | Note |
| --- | ---: | ---: | ---: | --- |
| random LoRA start after CUDA init | 0.0 | 0.0 | 0.0 |  |
| random LoRA one base backbone | 107.0 | 128.0 | 107.0 |  |
| random LoRA shared + adapter A | 107.4 | 128.0 | 107.4 |  |
| random LoRA shared + adapters A/B | 107.8 | 128.0 | 107.8 |  |
| random LoRA shared + adapters + heads | 107.8 | 128.0 | 107.8 |  |
| random LoRA warm-up task A | 118.9 | 172.0 | 153.8 |  |
| random LoRA warm-up task B | 118.9 | 172.0 | 153.8 |  |
| random LoRA duplicated model A | 115.8 | 128.0 | 115.8 |  |
| random LoRA duplicated models A/B | 222.8 | 254.0 | 222.8 |  |

## Deltas

- `adapter_b_incremental_mib`: 0.4 MiB
- `duplicated_minus_shared_mib`: 103.9 MiB
- `warmup_peak_mib`: 153.8 MiB
