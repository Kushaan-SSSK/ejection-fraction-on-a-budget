"""Accuracy + clinical metrics with patient-level bootstrap CIs.

Two entry points:
  predict()  — run a trained checkpoint over the test split with the frozen
               all-clips-averaged protocol, write per-video predictions CSV.
  report()   — turn a predictions CSV into the full metric families with
               10,000-resample bootstrap CIs (Invariant 5: a config without
               all families is incomplete, not partially reportable).

paired_bootstrap_diff() implements the pre-stated comparison rule: two configs
only differ if the CI of their MAE difference excludes zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EF_BANDS = ((0, 30), (30, 40), (40, 55), (55, 101))
CUTPOINTS = (40.0, 50.0)
N_BOOT = 10_000
BOOT_SEED = 20260814  # fixed so every config's CIs use identical resamples


# --------------------------------------------------------------------------
# Inference (frozen test protocol: all non-overlapping clips, mean prediction)
# --------------------------------------------------------------------------

def predict(config_path, checkpoint, dataset_root, out_csv, device=None):
    import yaml
    from .data import EchoNetDataset
    from .models import build_model

    cfg = yaml.safe_load(Path(config_path).read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["backbone"], pretrained=False).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)

    ds = EchoNetDataset(dataset_root, "TEST", frames=cfg["frames"],
                        period=cfg["period"], resolution=cfg["resolution"],
                        mode="all_clips", cache_dir=cfg.get("cache_dir"))
    rows = []
    with torch.inference_mode():
        for i in range(len(ds)):
            clips, ef = ds[i]  # (n_clips, 3, T, H, W)
            preds = model(clips.to(device)).float().cpu().numpy()
            rows.append({"file": ds.names[i], "ef_true": ef,
                         "ef_pred": float(preds.mean()),
                         "n_clips": len(preds)})
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": _r2(y_true, y_pred),
        "bland_altman_bias": float(np.mean(err)),
        "bland_altman_loa_lo": float(np.mean(err) - 1.96 * np.std(err, ddof=1)),
        "bland_altman_loa_hi": float(np.mean(err) + 1.96 * np.std(err, ddof=1)),
    }
    for cut in CUTPOINTS:
        pos = y_true < cut          # "reduced EF" is the positive class
        pred_pos = y_pred < cut
        tp = int(np.sum(pos & pred_pos))
        fn = int(np.sum(pos & ~pred_pos))
        tn = int(np.sum(~pos & ~pred_pos))
        fp = int(np.sum(~pos & pred_pos))
        c = int(cut)
        out[f"sens_at_{c}"] = tp / (tp + fn) if (tp + fn) else float("nan")
        out[f"spec_at_{c}"] = tn / (tn + fp) if (tn + fp) else float("nan")
        out[f"auc_at_{c}"] = _auc(pos, -y_pred)  # lower predicted EF => positive
        out[f"n_pos_at_{c}"] = tp + fn
    for lo, hi in EF_BANDS:
        m = (y_true >= lo) & (y_true < hi)
        out[f"mae_band_{lo}_{hi}"] = (float(np.mean(np.abs(err[m])))
                                      if m.any() else float("nan"))
        out[f"n_band_{lo}_{hi}"] = int(m.sum())
    return out


def _r2(y_true, y_pred) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _auc(pos: np.ndarray, score: np.ndarray) -> float:
    if pos.all() or not pos.any():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos.astype(int), score))


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def _resample_indices(n, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_metrics(y_true, y_pred, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Point estimate + 95% percentile CI for every metric. One video = one
    patient in EchoNet-Dynamic, so video-level resampling IS patient-level."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    point = point_metrics(y_true, y_pred)
    samples = {k: [] for k, v in point.items() if isinstance(v, float)}
    for idx in _resample_indices(len(y_true), n_boot, seed):
        m = point_metrics(y_true[idx], y_pred[idx])
        for k in samples:
            samples[k].append(m[k])
    out = dict(point)
    for k, vals in samples.items():
        vals = np.asarray(vals)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[f"{k}_ci_lo"] = float(np.percentile(vals, 2.5))
            out[f"{k}_ci_hi"] = float(np.percentile(vals, 97.5))
    return out


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=N_BOOT,
                          seed=BOOT_SEED) -> dict:
    """CI of MAE(a) - MAE(b) on shared resamples. 'a dominates b' requires
    ci_hi < 0; anything straddling zero is reported as indistinguishable."""
    y = np.asarray(y_true, float)
    ea = np.abs(np.asarray(pred_a, float) - y)
    eb = np.abs(np.asarray(pred_b, float) - y)
    diffs = [float(np.mean(ea[idx]) - np.mean(eb[idx]))
             for idx in _resample_indices(len(y), n_boot, seed)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mae_diff": float(np.mean(ea) - np.mean(eb)),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "distinguishable": bool(hi < 0 or lo > 0)}


def report(pred_csv, out_json=None, n_boot=N_BOOT) -> dict:
    df = pd.read_csv(pred_csv)
    metrics = bootstrap_metrics(df["ef_true"].values, df["ef_pred"].values,
                                n_boot=n_boot)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("config"); p.add_argument("checkpoint")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("pred_csv"); r.add_argument("--out", default=None)
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a.config, a.checkpoint, a.data, a.out)
    else:
        m = report(a.pred_csv, a.out, a.n_boot)
        print(json.dumps({k: v for k, v in m.items()
                          if not k.endswith(("_ci_lo", "_ci_hi"))}, indent=2))
