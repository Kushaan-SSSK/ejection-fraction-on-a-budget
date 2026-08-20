"""F3: clinical error structure. Panel (a) MAE by EF band (bootstrap CIs);
panels (b)/(c) Bland-Altman for the reference and the recommended R3D-18
config. Reads only results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures" / "out"

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BANDS = ((0, 30), (30, 40), (40, 55), (55, 101))
BAND_LABELS = ["EF < 30", "30–40", "40–55", "≥ 55"]

CONFIGS = [  # (name, label, color)
    ("3a_r2p1d_f32_p2", "R(2+1)D 32f×p2 (ref.)", "#0b0b0b"),
    ("3b_r3d_f32_p2", "R3D 32f×p2", "#2a78d6"),
    ("3a_r2p1d_f16_p4", "R(2+1)D 16f×p4", "#7aade8"),
    ("3b_x3ds_f16_p4", "X3D-S 16f×p4", "#eb6834"),
    ("3b_r2dpool_f16_p4", "2D pool 16f×p4", "#1baf7a"),
]

N_BOOT, SEED = 10_000, 20260814


def load_pred(name):
    df = pd.read_csv(RES / "metrics" / f"{name}.pred.csv")
    return df["ef_true"].values.astype(float), df["ef_pred"].values.astype(float)


def band_mae_ci(y, p):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    out = []
    err = np.abs(p - y)
    for lo, hi in BANDS:
        m = (y >= lo) & (y < hi)
        point = err[m].mean()
        boots = []
        for i in idx:
            mi = (y[i] >= lo) & (y[i] < hi)
            if mi.any():
                boots.append(err[i][mi].mean())
        out.append((point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
    return out


def style(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECONDARY, labelsize=8)


def bland_altman(ax, y, p, title):
    # Standard Bland-Altman: difference against the MEAN of the two methods
    # (difference vs one method alone can induce a spurious trend).
    style(ax)
    diff = p - y
    mean = (p + y) / 2.0
    bias = diff.mean()
    sd = diff.std(ddof=1)
    ax.scatter(mean, diff, s=8, facecolor="#2a78d6", edgecolor="none",
               alpha=0.35, zorder=3)
    for v, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        ax.axhline(v, color=INK, linewidth=0.9, linestyle=ls, alpha=0.8, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=2)
    ax.annotate(f"bias {bias:+.2f}", (2, bias), fontsize=7, color=INK,
                textcoords="offset points", xytext=(0, 3))
    ax.annotate(f"+1.96 SD {bias + 1.96 * sd:+.2f}", (2, bias + 1.96 * sd),
                fontsize=7, color=SECONDARY, textcoords="offset points", xytext=(0, 3))
    ax.annotate(f"−1.96 SD {bias - 1.96 * sd:+.2f}", (2, bias - 1.96 * sd),
                fontsize=7, color=SECONDARY, textcoords="offset points", xytext=(0, -10))
    ax.set_xlabel("Mean of predicted and reference EF (%)",
                  color=SECONDARY, fontsize=9)
    ax.set_ylabel("Predicted − reference (EF points)", color=SECONDARY, fontsize=9)
    ax.set_title(title, color=INK, fontsize=9)
    ax.set_ylim(-25, 25)
    ax.set_xlim(0, 85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.6, 3.6), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.32,
                          left=0.055, right=0.975, top=0.86, bottom=0.16)

    # (a) MAE by band
    ax = fig.add_subplot(gs[0])
    style(ax)
    n_cfg = len(CONFIGS)
    width = 0.8 / n_cfg
    for j, (name, label, color) in enumerate(CONFIGS):
        y, p = load_pred(name)
        stats = band_mae_ci(y, p)
        xs = np.arange(len(BANDS)) + (j - (n_cfg - 1) / 2) * width
        vals = [s[0] for s in stats]
        los = [s[0] - s[1] for s in stats]
        his = [s[2] - s[0] for s in stats]
        ax.bar(xs, vals, width=width * 0.92, color=color, zorder=3, label=label)
        ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=SECONDARY,
                    elinewidth=0.8, capsize=1.5, zorder=4)
    ax.set_xticks(np.arange(len(BANDS)))
    ax.set_xticklabels(BAND_LABELS)
    ax.set_ylim(0, 14.5)
    ax.set_ylabel("Test MAE, EF points (95% CI)", color=SECONDARY, fontsize=9)
    ax.set_xlabel("EF band (n = 83 / 77 / 241 / 876)", color=SECONDARY, fontsize=9)
    ax.set_title("(a) Error by EF band", color=INK, fontsize=9)
    # short labels keep the legend narrow, over the low bars on the right
    ax.legend(frameon=False, fontsize=6.5, labelcolor=SECONDARY,
              loc="upper right", borderaxespad=0.3, handlelength=1.4)

    # (b)/(c) Bland-Altman
    for slot, (name, title) in zip(
            (gs[1], gs[2]),
            (("3a_r2p1d_f32_p2", "(b) Reference: R(2+1)D-18 32f×p2"),
             ("3b_r3d_f32_p2", "(c) Recommended: R3D-18 32f×p2"))):
        y, p = load_pred(name)
        bland_altman(fig.add_subplot(slot), y, p, title)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F4_clinical.{ext}", facecolor="#ffffff")
        print(f"wrote {OUT / f'F4_clinical.{ext}'}")
    fig.savefig(OUT / "F4_clinical.tif", facecolor="#ffffff", dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {OUT / 'F4_clinical.tif'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
