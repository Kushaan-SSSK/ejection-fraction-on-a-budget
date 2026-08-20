"""Cost-accuracy frontier figures (F1/F2/F3), reading results/.

Encoding (fixed in Phase 0, before any training):
  x            cost, log scale (primary axis: CPU ms per video, 4 threads)
  y            F1: test MAE  |  F2: sensitivity at EF < 40   (95% CI whiskers)
  color        backbone family — 3 validated categorical slots
  marker shape backbone within family
  marker size  clip frames (8/16/32/64)
  ink line     Pareto frontier (non-dominated points)
  gray         prior work (not protocol-matched unless re-profiled)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_COLORS = {"resnet3d": "#2a78d6", "x3d": "#eb6834", "2d": "#1baf7a"}
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PRIOR = "#898781"

BACKBONE_STYLE = {  # backbone -> (family, marker, pretty label)
    "r2plus1d_18": ("resnet3d", "o", "R(2+1)D-18"),
    "r3d_18": ("resnet3d", "s", "R3D-18"),
    "mc3_18": ("resnet3d", "^", "MC3-18"),
    "x3d_s": ("x3d", "o", "X3D-S"),
    "x3d_m": ("x3d", "s", "X3D-M"),
    "r2d_18_pool": ("2d", "D", "2D ResNet-18 + pool"),
}
FRAME_SIZE = {8: 45, 16: 80, 32: 120, 64: 170}


def pareto_front(df: pd.DataFrame, xcol: str, ycol: str, maximize: bool = False) -> pd.DataFrame:
    """Non-dominated subset: no other point is both cheaper and better on ycol.
    maximize=False (F1): better = lower ycol (error). maximize=True (F2):
    better = higher ycol (sensitivity) -- higher-is-better metrics need the
    running-best comparison flipped, or every point looks like a "frontier"."""
    d = df.sort_values(xcol).reset_index(drop=True)
    keep, best = [], (-np.inf if maximize else np.inf)
    for _, row in d.iterrows():
        is_better = row[ycol] > best if maximize else row[ycol] < best
        if is_better:
            keep.append(row)
            best = row[ycol]
    return pd.DataFrame(keep)


def _style_axes(ax, xlabel, ylabel):
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.set_xlabel(xlabel, color=SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=SECONDARY, fontsize=10)
    ax.set_xscale("log")


def plot_panel(ax, df, ycol, ylo, yhi, ylabel, xcol="cpu_ms_per_video",
               xlabel="CPU latency per video, ms (log, 4 threads)",
               reference="3a_r2p1d_f32_p2", prior=None, label_frontier=True,
               maximize=False):
    """df: one row per config with columns [config, backbone, frames, xcol,
    ycol, ylo, yhi]. prior: optional df with [label, x, y]. maximize: True
    for higher-is-better ycols (e.g. sensitivity), False for error metrics."""
    _style_axes(ax, xlabel, ylabel)

    front = pareto_front(df, xcol, ycol, maximize=maximize)
    ax.plot(front[xcol], front[ycol], color=INK, linewidth=1.2, zorder=2,
            drawstyle="steps-post", alpha=0.75)

    for _, r in df.iterrows():
        fam, marker, _ = BACKBONE_STYLE[r["backbone"]]
        is_ref = r["config"] == reference
        ax.errorbar(r[xcol], r[ycol], yerr=[[r[ycol] - r[ylo]], [r[yhi] - r[ycol]]],
                    fmt="none", ecolor=SECONDARY, elinewidth=0.8, capsize=2,
                    zorder=3, alpha=0.7)
        ax.scatter(r[xcol], r[ycol], s=FRAME_SIZE[int(r["frames"])],
                   marker="*" if is_ref else marker, zorder=4,
                   facecolor=INK if is_ref else FAMILY_COLORS[fam],
                   edgecolor="#ffffff", linewidth=1.2)
    # model identity is carried by the legend only; no in-plot name labels
    # explicit limits: errorbar on a log axis can leak log10(x) into dataLim
    ax.set_xlim(df[xcol].min() * 0.75, df[xcol].max() * 1.3)
    if prior is not None:
        for _, r in prior.iterrows():
            ax.scatter(r["x"], r["y"], s=70, marker="d", facecolor="none",
                       edgecolor=PRIOR, linewidth=1.4, zorder=3)
            ax.annotate(r["label"], (r["x"], r["y"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=MUTED, style="italic")


def add_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, linestyle="", markersize=7,
                      markerfacecolor=FAMILY_COLORS[fam], markeredgecolor="#fff",
                      label=lbl)
               for fam, m, lbl in BACKBONE_STYLE.values()]
    handles += [Line2D([], [], marker="*", linestyle="", markersize=10,
                       markerfacecolor=INK, markeredgecolor="#fff",
                       label="reference config"),
                Line2D([], [], color=INK, linewidth=1.2, alpha=0.75,
                       label="Pareto frontier"),
                Line2D([], [], marker="d", linestyle="", markersize=8,
                       markerfacecolor="none", markeredgecolor=PRIOR,
                       label="prior work")]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(s),
                       markerfacecolor="#c3c2b7", markeredgecolor="#fff",
                       label=f"{f} frames") for f, s in FRAME_SIZE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.0), labelcolor=SECONDARY)


def load_results(costs_csv="results/costs.csv", metrics_dir="results/metrics"):
    """Join costs.csv with per-config bootstrapped metrics JSONs."""
    costs = pd.read_csv(costs_csv)
    rows = []
    for _, c in costs.iterrows():
        mpath = Path(metrics_dir) / f"{c['config']}.metrics.json"
        if not mpath.exists():
            continue  # Invariant 5: incomplete configs are not plotted
        m = json.loads(mpath.read_text())
        rows.append({"config": c["config"], "backbone": c["backbone"],
                     "frames": c["frames"], "period": c["period"],
                     "cpu_ms_per_video": c["cpu_ms_per_video"],
                     "gflops_per_video": c["gflops_per_video"],
                     "mae": m["mae"], "mae_lo": m["mae_ci_lo"],
                     "mae_hi": m["mae_ci_hi"],
                     "sens40": m["sens_at_40"],
                     "sens40_lo": m.get("sens_at_40_ci_lo", np.nan),
                     "sens40_hi": m.get("sens_at_40_ci_hi", np.nan)})
    return pd.DataFrame(rows)


def make(df, out_dir="figures/out", prior=None, suffix=""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ycol, ylo, yhi, ylabel, fname, extra, maximize, xcol, xlabel in (
            ("mae", "mae_lo", "mae_hi", "Test MAE, EF points",
             f"F1_frontier{suffix}.png", ("hline", 4.1, "published ref 4.1"), False,
             "cpu_ms_per_video", "CPU latency per video, ms (log, 4 threads)"),
            ("sens40", "sens40_lo", "sens40_hi",
             "Sensitivity at EF < 40 (95% CI)",
             f"F2_sensitivity{suffix}.png", None, True,
             "cpu_ms_per_video", "CPU latency per video, ms (log, 4 threads)"),
            ("mae", "mae_lo", "mae_hi", "Test MAE, EF points",
             f"F3_frontier_flops{suffix}.png", ("hline", 4.1, "published ref 4.1"), False,
             "gflops_per_video", "GFLOPs per video (log)")):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
        fig.patch.set_facecolor("#ffffff")
        plot_panel(ax, df, ycol, ylo, yhi, ylabel, prior=prior, maximize=maximize,
                   xcol=xcol, xlabel=xlabel)
        if extra:
            ax.axhline(extra[1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.annotate(extra[2], (ax.get_xlim()[0], extra[1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=MUTED)
        add_legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out / fname, facecolor="#ffffff")
        fig.savefig((out / fname).with_suffix(".pdf"), facecolor="#ffffff")
        fig.savefig((out / fname).with_suffix(".tif"), facecolor="#ffffff",
                    dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
        print(f"wrote {out / fname} (+.pdf, +.tif)")


if __name__ == "__main__":
    make(load_results())
