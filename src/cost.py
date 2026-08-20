"""The cost harness. docs/PROTOCOL.md is the human-readable contract; this file is
the executable one. Frozen at Gate 2 under git tag cost-harness-v1 — any
post-freeze change means a new tag and re-profiling every configuration.

profile(model, input_shape, ...) -> dict (one row of results/costs.csv)

Protocol (do not edit without a new tag):
  FLOPs        fvcore FlopCountAnalysis counts MACs; we report GFLOPs = 2*GMACs.
               Unsupported ops are logged into the row, never silently dropped.
  GPU latency  batch 1, fp32, eval, cudnn.benchmark=True, torch.cuda.synchronize
               around each timed forward, 20 warmup discarded, 100 timed,
               median + IQR. Skipped (NaN) when CUDA is unavailable.
  CPU latency  torch.set_num_threads(4) — a handheld-device proxy — same
               warmup/repeat/median protocol.
  Peak memory  torch.cuda.max_memory_allocated over one inference forward.
  Disk         fp32 state_dict bytes.
  Per-video    per-clip cost * mean non-overlapping clips per test video for
               that clip geometry (caller passes clips_per_video).
"""

from __future__ import annotations

import argparse
import io
import time
import warnings
from pathlib import Path

import numpy as np
import torch

PROTOCOL_VERSION = "cost-harness-v1"
WARMUP = 20
RUNS = 100
CPU_THREADS = 4


def _time_forward(model, x, sync=lambda: None, warmup=WARMUP, runs=RUNS):
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return (float(np.median(t)),
            float(np.percentile(t, 75) - np.percentile(t, 25)))


def profile(model, input_shape, clips_per_video=1.0, device_gpu="cuda",
            warmup=WARMUP, runs=RUNS):
    """input_shape: (3, T, H, W) for one clip. Returns a flat dict."""
    model = model.eval()
    row = {"protocol": PROTOCOL_VERSION,
           "input_shape": "x".join(map(str, input_shape)),
           "clips_per_video": round(float(clips_per_video), 3)}

    # --- Parameters and disk size ------------------------------------------
    row["params_total"] = sum(p.numel() for p in model.parameters())
    row["params_trainable"] = sum(p.numel() for p in model.parameters()
                                  if p.requires_grad)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    row["disk_mb"] = round(buf.getbuffer().nbytes / 1e6, 2)

    # --- FLOPs -------------------------------------------------------------
    from fvcore.nn import FlopCountAnalysis
    model_cpu = model.to("cpu")
    x = torch.zeros(1, *input_shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops = FlopCountAnalysis(model_cpu, x)
        flops.unsupported_ops_warnings(False)
        gmacs = flops.total() / 1e9
        row["gflops_per_clip"] = round(2 * gmacs, 6)
        row["flops_unsupported_ops"] = ";".join(sorted(flops.unsupported_ops()))

    # --- CPU latency -------------------------------------------------------
    old_threads = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        med, iqr = _time_forward(model_cpu, x, warmup=warmup, runs=runs)
    finally:
        torch.set_num_threads(old_threads)
    row["cpu_ms_per_clip_median"] = round(med, 2)
    row["cpu_ms_per_clip_iqr"] = round(iqr, 2)
    row["cpu_threads"] = CPU_THREADS

    # --- GPU latency + peak memory ----------------------------------------
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model_gpu = model.to(device_gpu)
        xg = x.to(device_gpu)
        med, iqr = _time_forward(model_gpu, xg, sync=torch.cuda.synchronize,
                                 warmup=warmup, runs=runs)
        row["gpu_ms_per_clip_median"] = round(med, 2)
        row["gpu_ms_per_clip_iqr"] = round(iqr, 2)
        row["gpu_name"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model_gpu(xg)
        torch.cuda.synchronize()
        row["gpu_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        model.to("cpu")
    else:
        row.update(gpu_ms_per_clip_median=float("nan"),
                   gpu_ms_per_clip_iqr=float("nan"),
                   gpu_name="", gpu_peak_mem_mb=float("nan"))

    # --- Per-video columns -------------------------------------------------
    for k in ("gflops", "cpu_ms", "gpu_ms"):
        per_clip = row[f"{k}_per_clip" if k == "gflops" else f"{k}_per_clip_median"]
        row[f"{k}_per_video"] = round(per_clip * row["clips_per_video"], 6)
    return row


def profile_config(config_path, dataset_root=None, output_csv=None):
    """CLI entry: profile the model a config describes and append the row."""
    import pandas as pd
    import yaml
    from .models import build_model

    cfg = yaml.safe_load(Path(config_path).read_text())
    model = build_model(cfg["backbone"], pretrained=False)  # weights don't affect cost
    shape = (3, cfg["frames"], cfg["resolution"], cfg["resolution"])

    clips_per_video = 1.0
    if dataset_root:
        from .data import EchoNetDataset
        ds = EchoNetDataset(dataset_root, "TEST", frames=cfg["frames"],
                            period=cfg["period"], resolution=cfg["resolution"],
                            mode="all_clips")
        clips_per_video = ds.clips_per_video()

    row = {"config": Path(config_path).stem, **cfg,
           **profile(model, shape, clips_per_video)}
    if output_csv:
        out = Path(output_csv)
        df = pd.DataFrame([row])
        if out.exists():
            df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df.to_csv(out, index=False)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", default=None, help="dataset root (for clips/video)")
    ap.add_argument("--out", default="results/costs.csv")
    a = ap.parse_args()
    print(profile_config(a.config, a.data, a.out))
