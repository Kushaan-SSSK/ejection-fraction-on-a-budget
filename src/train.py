"""Config-driven trainer. Takes a YAML path and nothing else (Invariant 1:
the recipe lives in the config, identical across the sweep; only swept axes
differ between files).

The fast recipe (decided 2026-08-14, applied identically to every run):
  SGD lr 1e-4, momentum 0.9, weight decay 1e-4
  step LR x0.1 every 10 epochs, 30 epochs total   (scaled from EchoNet's 15/45)
  effective batch 20 via gradient accumulation    (physical batch from config)
  AMP (fp16 autocast + GradScaler) on CUDA
  MSE loss on raw EF percent
  one random clip per video per epoch; val = deterministic first clip, MAE

Checkpoints every epoch to checkpoints/<config>/last.pt (+ best.pt by val MAE)
and auto-resumes, so a multi-day sweep survives reboots. On completion appends
one row to results/registry.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import EchoNetDataset
from .models import build_model

DEFAULTS = dict(
    backbone="r2plus1d_18", pretrained=True,
    frames=32, period=2, resolution=112,
    epochs=30, lr=1e-4, momentum=0.9, weight_decay=1e-4, lr_step=10,
    effective_batch=20, physical_batch=10,
    seed=20260814, num_workers=4, amp=True, cache_dir=None,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path) -> dict:
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text())}
    if cfg["effective_batch"] % cfg["physical_batch"]:
        raise ValueError("effective_batch must be a multiple of physical_batch")
    cfg["name"] = Path(path).stem
    return cfg


def train(config_path, dataset_root, ckpt_root="checkpoints",
          registry="results/registry.csv", device=None, max_epochs=None):
    cfg = load_config(config_path)
    if max_epochs:  # smoke tests only; a real run never passes this
        cfg["epochs"] = max_epochs
    seed_everything(cfg["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["amp"]) and device == "cuda"
    accum = cfg["effective_batch"] // cfg["physical_batch"]

    train_ds = EchoNetDataset(dataset_root, "TRAIN", frames=cfg["frames"],
                              period=cfg["period"], resolution=cfg["resolution"],
                              mode="train", cache_dir=cfg["cache_dir"])
    val_ds = EchoNetDataset(dataset_root, "VAL", frames=cfg["frames"],
                            period=cfg["period"], resolution=cfg["resolution"],
                            mode="train", pad_pixels=0,
                            cache_dir=cfg["cache_dir"])
    train_dl = DataLoader(train_ds, batch_size=cfg["physical_batch"],
                          shuffle=True, num_workers=cfg["num_workers"],
                          pin_memory=(device == "cuda"), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["physical_batch"], shuffle=False,
                        num_workers=cfg["num_workers"])

    model = build_model(cfg["backbone"], pretrained=cfg["pretrained"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg["lr_step"], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(ckpt_root) / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last, best = ckpt_dir / "last.pt", ckpt_dir / "best.pt"

    start_epoch, best_val = 0, float("inf")
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        print(f"resumed {cfg['name']} at epoch {start_epoch}")

    t_start = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        for step, (clips, efs) in enumerate(train_dl):
            clips = clips.to(device, non_blocking=True)
            efs = efs.float().to(device)
            with torch.autocast("cuda", enabled=use_amp):
                loss = torch.nn.functional.mse_loss(model(clips), efs)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * len(efs)
            seen += len(efs)
        sched.step()

        model.eval()
        abs_err, n = 0.0, 0
        with torch.inference_mode():
            for clips, efs in val_dl:
                clips = clips.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=use_amp):
                    preds = model(clips)
                abs_err += (preds.float().cpu() - efs.float()).abs().sum().item()
                n += len(efs)
        val_mae = abs_err / max(n, 1)

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_val": min(best_val, val_mae),
                 "config": cfg}
        torch.save(state, last)
        if val_mae < best_val:
            best_val = val_mae
            torch.save(state, best)
        print(f"[{cfg['name']}] epoch {epoch + 1}/{cfg['epochs']} "
              f"train_mse {running / max(seen, 1):.3f} val_mae {val_mae:.3f} "
              f"(best {best_val:.3f})")

    _append_registry(registry, cfg, config_path, best_val,
                     hours=(time.time() - t_start) / 3600, device=device,
                     accum=accum, amp=use_amp, ckpt=str(best))
    return best_val


def _append_registry(registry, cfg, config_path, best_val, hours, device,
                     accum, amp, ckpt):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    deviations = [f"grad_accum_x{accum}"] if accum > 1 else []
    if cfg.get("native_resolution") and cfg["native_resolution"] != cfg["resolution"]:
        # X3D only: decoded at cfg["resolution"], upsampled internally (see
        # src/models.py and docs/PROTOCOL.md)
        deviations.append(f"upsampled_{cfg['resolution']}_to_{cfg['native_resolution']}"
                          f"_pretrained_weights_expect_native_res_not_upsampled")
    row = {"run_id": cfg["name"],
           "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": commit, "config_path": str(config_path),
           "seed": cfg["seed"], "epochs": cfg["epochs"],
           "best_val_mae": round(best_val, 4), "train_hours": round(hours, 2),
           "device": device, "amp": amp,
           "deviations": ";".join(deviations),
           "checkpoint": ckpt}
    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", required=True, help="EchoNet-Dynamic root")
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--registry", default="results/registry.csv")
    a = ap.parse_args()
    train(a.config, a.data, a.ckpt_root, a.registry)
