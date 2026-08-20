"""Stage-3a analysis: joins registry/costs/metrics into one table, runs
paired bootstrap comparisons (shared resamples, per docs/PROTOCOL.md), determines
the cost-vs-MAE and cost-vs-sensitivity Pareto frontiers, and applies the
pre-registered Phase-3 rule (docs/PLAN.md line 65) to pick 3b's two clip settings.

Writes:
  results/stage3a_summary.csv        one row per config, all 3 metric families
  results/stage3a_comparisons.csv    paired bootstrap MAE diff vs reference
                                      and vs the single best-MAE config

    .venv\\Scripts\\python scripts/analyze_3a.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import paired_bootstrap_diff  # noqa: E402
RESULTS = ROOT / "results"
REFERENCE = "3a_r2p1d_f32_p2"
NONINFERIORITY_MARGIN = 0.5  # EF points, pre-stated in docs/PLAN.md Phase 4


def load_table() -> pd.DataFrame:
    costs = pd.read_csv(RESULTS / "costs.csv").set_index("config")
    rows = []
    for name in costs.index:
        m = json.loads((RESULTS / "metrics" / f"{name}.metrics.json").read_text())
        c = costs.loc[name]
        rows.append({
            "config": name, "frames": int(c["frames"]), "period": int(c["period"]),
            "mae": m["mae"], "mae_ci_lo": m["mae_ci_lo"], "mae_ci_hi": m["mae_ci_hi"],
            "rmse": m["rmse"], "r2": m["r2"],
            "bland_altman_bias": m["bland_altman_bias"],
            "bland_altman_loa_lo": m["bland_altman_loa_lo"],
            "bland_altman_loa_hi": m["bland_altman_loa_hi"],
            "sens_at_40": m["sens_at_40"], "sens_at_40_ci_lo": m["sens_at_40_ci_lo"],
            "sens_at_40_ci_hi": m["sens_at_40_ci_hi"],
            "spec_at_40": m["spec_at_40"],
            "sens_at_50": m["sens_at_50"], "spec_at_50": m["spec_at_50"],
            "mae_band_0_30": m["mae_band_0_30"], "n_band_0_30": m["n_band_0_30"],
            "mae_band_30_40": m["mae_band_30_40"], "n_band_30_40": m["n_band_30_40"],
            "mae_band_40_55": m["mae_band_40_55"], "n_band_40_55": m["n_band_40_55"],
            "mae_band_55_101": m["mae_band_55_101"], "n_band_55_101": m["n_band_55_101"],
            "gflops_per_clip": c["gflops_per_clip"], "params_total": c["params_total"],
            "disk_mb": c["disk_mb"],
            "gpu_ms_per_clip_median": c["gpu_ms_per_clip_median"],
            "cpu_ms_per_clip_median": c["cpu_ms_per_clip_median"],
            "gpu_peak_mem_mb": c["gpu_peak_mem_mb"],
            "cpu_ms_per_video": c["cpu_ms_per_video"],
            "gpu_ms_per_video": c["gpu_ms_per_video"],
        })
    df = pd.DataFrame(rows).sort_values("cpu_ms_per_video").reset_index(drop=True)
    return df


def pareto_front(df: pd.DataFrame, xcol: str, ycol: str, maximize: bool) -> list[str]:
    d = df.sort_values(xcol).reset_index(drop=True)
    keep, best = [], (-np.inf if maximize else np.inf)
    for _, row in d.iterrows():
        better = row[ycol] > best if maximize else row[ycol] < best
        if better:
            keep.append(row["config"])
            best = row[ycol]
    return keep


def load_pred(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "metrics" / f"{name}.pred.csv").sort_values("file").reset_index(drop=True)


def pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    best_config = df.loc[df["mae"].idxmin(), "config"]
    ref_pred = load_pred(REFERENCE)
    best_pred = load_pred(best_config)
    assert list(ref_pred["file"]) == list(best_pred["file"]), "test-set order mismatch"

    rows = []
    for name in df["config"]:
        p = load_pred(name)
        assert list(p["file"]) == list(ref_pred["file"]), f"{name}: test-set order mismatch"
        y = p["ef_true"].values
        vs_ref = paired_bootstrap_diff(y, p["ef_pred"].values, ref_pred["ef_pred"].values)
        vs_best = paired_bootstrap_diff(y, p["ef_pred"].values, best_pred["ef_pred"].values)
        rows.append({
            "config": name,
            "mae_diff_vs_reference": vs_ref["mae_diff"],
            "ci_lo_vs_reference": vs_ref["ci_lo"], "ci_hi_vs_reference": vs_ref["ci_hi"],
            "distinguishable_vs_reference": vs_ref["distinguishable"],
            "mae_diff_vs_best": vs_best["mae_diff"],
            "ci_lo_vs_best": vs_best["ci_lo"], "ci_hi_vs_best": vs_best["ci_hi"],
            "distinguishable_vs_best": vs_best["distinguishable"],
        })
    return pd.DataFrame(rows), best_config


def pick_3b_clips(df: pd.DataFrame) -> tuple[tuple[int, int], tuple[int, int], str]:
    """docs/PLAN.md line 65: reference (32x2) + cheapest config within 0.5 MAE of
    the best 3a run, pre-registered on the primary cost axis (cpu_ms_per_video)."""
    best_mae = df["mae"].min()
    candidates = df[df["mae"] <= best_mae + NONINFERIORITY_MARGIN]
    candidates = candidates[candidates["config"] != REFERENCE]
    cheapest = candidates.sort_values("cpu_ms_per_video").iloc[0]
    ref_row = df[df["config"] == REFERENCE].iloc[0]
    explanation = (
        f"Best 3a test MAE = {best_mae:.3f} (config {df.loc[df['mae'].idxmin(), 'config']}). "
        f"Configs within {NONINFERIORITY_MARGIN} MAE points of that: "
        f"{', '.join(candidates.sort_values('cpu_ms_per_video')['config'])}. "
        f"Cheapest of those by CPU ms/video: {cheapest['config']} "
        f"({cheapest['cpu_ms_per_video']:.1f} ms/video, MAE {cheapest['mae']:.3f})."
    )
    return (int(ref_row["frames"]), int(ref_row["period"])), \
           (int(cheapest["frames"]), int(cheapest["period"])), explanation


def main():
    df = load_table()
    df.to_csv(RESULTS / "stage3a_summary.csv", index=False)

    comparisons, best_config = pairwise_comparisons(df)
    comparisons.to_csv(RESULTS / "stage3a_comparisons.csv", index=False)

    mae_frontier = pareto_front(df, "cpu_ms_per_video", "mae", maximize=False)
    sens_frontier = pareto_front(df, "cpu_ms_per_video", "sens_at_40", maximize=True)

    clip_a, clip_b, explanation = pick_3b_clips(df)

    print("=== SUMMARY TABLE ===")
    print(df.to_string(index=False))
    print("\n=== best-MAE config ===", best_config)
    print("\n=== PAIRED BOOTSTRAP COMPARISONS ===")
    print(comparisons.to_string(index=False))
    print("\n=== cost-vs-MAE Pareto frontier (cpu_ms_per_video, mae) ===")
    print(mae_frontier)
    print("\n=== cost-vs-sensitivity@40 Pareto frontier (cpu_ms_per_video, sens_at_40) ===")
    print(sens_frontier)
    print("\n=== 3b clip-setting selection ===")
    print(explanation)
    print(f"chosen: {clip_a} (reference) + {clip_b} (cheapest non-inferior)")


if __name__ == "__main__":
    main()
