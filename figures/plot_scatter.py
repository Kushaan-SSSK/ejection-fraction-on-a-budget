"""F5: predicted vs reference EF scatter for the reference config and the
recommended R3D-18 config. Reads only results/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures" / "out"

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

PANELS = [
    ("3a_r2p1d_f32_p2", "(a) Reference: R(2+1)D-18 32f×p2"),
    ("3b_r3d_f32_p2", "(b) Recommended: R3D-18 32f×p2"),
]


def style(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECONDARY, labelsize=8)


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    for ax, (name, title) in zip(axes, PANELS):
        df = pd.read_csv(RES / "metrics" / f"{name}.pred.csv")
        y = df["ef_true"].values.astype(float)
        p = df["ef_pred"].values.astype(float)
        style(ax)
        lim = (5, 90)
        ax.plot(lim, lim, color=MUTED, linewidth=0.9, zorder=2)
        ax.scatter(y, p, s=8, facecolor="#2a78d6", edgecolor="none",
                   alpha=0.35, zorder=3)
        ax.axvline(40, color=MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
        ax.axhline(40, color=MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"$R^2$ = {r2(y, p):.2f}\nMAE = {np.abs(p - y).mean():.2f}",
                    (0.05, 0.86), xycoords="axes fraction", fontsize=8, color=INK)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Reference EF (%)", color=SECONDARY, fontsize=9)
        ax.set_ylabel("Predicted EF (%)", color=SECONDARY, fontsize=9)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_aspect("equal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F5_scatter.{ext}", facecolor="#ffffff")
        print(f"wrote {OUT / f'F5_scatter.{ext}'}")
    fig.savefig(OUT / "F5_scatter.tif", facecolor="#ffffff", dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {OUT / 'F5_scatter.tif'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
