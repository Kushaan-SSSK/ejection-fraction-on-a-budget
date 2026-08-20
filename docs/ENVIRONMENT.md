# Environment

## Training / measurement machine (authoritative for all cost rows)

- PC with NVIDIA RTX 2070 Super (8 GB VRAM, Turing), 32 GB RAM
- Fill in when first run starts: exact CPU, driver version, CUDA/cuDNN
  versions, `pip freeze` snapshot (commit it as `docs/pip-freeze.txt`)
- Python 3.11 (required if X3D via pytorchvideo is used)

## Development machine (code authoring only — no GPU, no cost rows)

- Windows 11 Home, Python 3.13.5, torch 2.6.0+cu124 (CUDA unavailable)

## Setup on the training PC

```
git clone <this repo>
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install pytorchvideo==0.1.5        # X3D backbones; skip if it fails
pip freeze > docs/pip-freeze.txt
```

## Memory plan for 8 GB (AMP on, effective batch 20 everywhere)

| frames | physical batch | grad accumulation |
|---|---|---|
| 8 | 20 | 1× |
| 16 | 10 | 2× |
| 32 | 10 | 2× |
| 64 | 4 | 5× |

If a config OOMs, halve its `physical_batch` in the YAML (keep it a divisor
of 20) and note it — the registry logs the accumulation factor automatically.

## Decode cache

`cache_dir` in a config enables a one-time AVI→npy uint8 cache per resolution.
Full 112² cache ≈ 60–65 GB on disk — check free space first, or cache only
TRAIN. Cache lives outside the repo and never leaves the machine (agreement).
