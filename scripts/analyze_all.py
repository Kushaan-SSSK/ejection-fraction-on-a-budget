"""Combined 3a+3b(+X3D)+3d analysis: the full study, not just stage 3a.

Writes:
  results/stage3ab_summary.csv     22 primary configs (12 stage-3a clip-geometry
                                    sweep + 6 stage-3b backbones + 4 X3D), all
                                    3 metric families, with a `caveat` column
                                    (X3D rows only -- see PROTOCOL.md)
  results/stage3ab_comparisons.csv paired bootstrap MAE diff vs reference and
                                    vs best-MAE config, across all 22
  results/stage3d_seed_variance.csv per-seed MAE/sens@40 for the 4 replicated
                                    configs, plus the 3x3 cross-seed diff grid
                                    for both headline claims

    .venv\\Scripts\\python scripts/analyze_all.py
"""

from __future__ import annotations

import itertools
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
NONINFERIORITY_MARGIN = 0.5

X3D_CAVEAT = ("Input upsampled 112->native (160/X3D-S, 224/X3D-M) inside the "
             "model; pretrained Kinetics-400 weights expect a sharp native-"
             "resolution decode, not upsampled 112px. Cannot cleanly separate "
             "\"less accuracy-efficient\" from \"handicapped by input "
             "degradation.\" See PROTOCOL.md.")

SEED_STUDY = {
    "3a_r2p1d_f32_p2": ["3a_r2p1d_f32_p2", "3d_3a_r2p1d_f32_p2_s20260815", "3d_3a_r2p1d_f32_p2_s20260816"],
    "3a_r2p1d_f16_p4": ["3a_r2p1d_f16_p4", "3d_3a_r2p1d_f16_p4_s20260815", "3d_3a_r2p1d_f16_p4_s20260816"],
    "3a_r2p1d_f8_p1": ["3a_r2p1d_f8_p1", "3d_3a_r2p1d_f8_p1_s20260815", "3d_3a_r2p1d_f8_p1_s20260816"],
    "3a_r2p1d_f8_p4": ["3a_r2p1d_f8_p4", "3d_3a_r2p1d_f8_p4_s20260815", "3d_3a_r2p1d_f8_p4_s20260816"],
}
SEEDS = [20260814, 20260815, 20260816]


def load_metrics(name: str) -> dict:
    return json.loads((RESULTS / "metrics" / f"{name}.metrics.json").read_text())


def load_pred(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "metrics" / f"{name}.pred.csv").sort_values("file").reset_index(drop=True)


def load_table() -> pd.DataFrame:
    costs = pd.read_csv(RESULTS / "costs.csv").set_index("config")
    rows = []
    for name in costs.index:
        m = load_metrics(name)
        c = costs.loc[name]
        rows.append({
            "config": name, "backbone": c["backbone"],
            "frames": int(c["frames"]), "period": int(c["period"]),
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
            "caveat": X3D_CAVEAT if c["backbone"] in ("x3d_s", "x3d_m") else "",
        })
    return pd.DataFrame(rows).sort_values("cpu_ms_per_video").reset_index(drop=True)


def pareto_front(df: pd.DataFrame, xcol: str, ycol: str, maximize: bool) -> list[str]:
    d = df.sort_values(xcol).reset_index(drop=True)
    keep, best = [], (-np.inf if maximize else np.inf)
    for _, row in d.iterrows():
        better = row[ycol] > best if maximize else row[ycol] < best
        if better:
            keep.append(row["config"])
            best = row[ycol]
    return keep


def pairwise_comparisons(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    best_config = df.loc[df["mae"].idxmin(), "config"]
    ref_pred = load_pred(REFERENCE)
    best_pred = load_pred(best_config)

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


def seed_variance_table() -> pd.DataFrame:
    rows = []
    for base, names in SEED_STUDY.items():
        maes = [load_metrics(n)["mae"] for n in names]
        sens = [load_metrics(n)["sens_at_40"] for n in names]
        for seed, name, mae, s in zip(SEEDS, names, maes, sens):
            rows.append({"base_config": base, "run": name, "seed": seed,
                         "mae": mae, "sens_at_40": s})
    df = pd.DataFrame(rows)
    return df


def cross_seed_grid(a: str, b: str) -> dict:
    a_names, b_names = SEED_STUDY[a], SEED_STUDY[b]
    a_mae = [load_metrics(n)["mae"] for n in a_names]
    b_mae = [load_metrics(n)["mae"] for n in b_names]
    a_sens = [load_metrics(n)["sens_at_40"] for n in a_names]
    b_sens = [load_metrics(n)["sens_at_40"] for n in b_names]
    mae_diffs = [x - y for x, y in itertools.product(a_mae, b_mae)]
    sens_diffs = [x - y for x, y in itertools.product(a_sens, b_sens)]
    return {
        "mae_diffs_9": mae_diffs, "mae_diff_min": min(mae_diffs), "mae_diff_max": max(mae_diffs),
        "mae_diff_mean": float(np.mean(mae_diffs)),
        "n_of_9_a_worse_than_b_on_mae": sum(1 for d in mae_diffs if d > 0),
        "sens_diffs_9": sens_diffs, "sens_diff_min": min(sens_diffs), "sens_diff_max": max(sens_diffs),
        "sens_diff_mean": float(np.mean(sens_diffs)),
    }


def main():
    df = load_table()
    df.to_csv(RESULTS / "stage3ab_summary.csv", index=False)

    comparisons, best_config = pairwise_comparisons(df)
    comparisons.to_csv(RESULTS / "stage3ab_comparisons.csv", index=False)

    mae_frontier = pareto_front(df, "cpu_ms_per_video", "mae", maximize=False)
    sens_frontier = pareto_front(df, "cpu_ms_per_video", "sens_at_40", maximize=True)

    seed_df = seed_variance_table()
    seed_df.to_csv(RESULTS / "stage3d_seed_variance.csv", index=False)

    grid_ref = cross_seed_grid("3a_r2p1d_f16_p4", "3a_r2p1d_f32_p2")
    grid_sparse = cross_seed_grid("3a_r2p1d_f8_p1", "3a_r2p1d_f8_p4")

    print("=== FULL 22-CONFIG SUMMARY (sorted by cost) ===")
    print(df[["config", "backbone", "frames", "period", "mae", "mae_ci_lo", "mae_ci_hi",
              "sens_at_40", "cpu_ms_per_video", "caveat"]].to_string(index=False))
    print("\n=== best-MAE config ===", best_config)
    print("\n=== cost-vs-MAE Pareto frontier ===", mae_frontier)
    print("=== cost-vs-sensitivity@40 Pareto frontier ===", sens_frontier)
    print("\n=== paired bootstrap: distinguishable from reference? ===")
    print(comparisons[["config", "mae_diff_vs_reference", "ci_lo_vs_reference",
                       "ci_hi_vs_reference", "distinguishable_vs_reference"]].to_string(index=False))

    print("\n=== SEED VARIANCE ===")
    print(seed_df.to_string(index=False))

    print("\n=== f16_p4 vs reference: 3x3 cross-seed grid ===")
    for k, v in grid_ref.items():
        print(f"  {k}: {v}")
    print("\n=== sparse-beats-dense (f8_p1 vs f8_p4): 3x3 cross-seed grid ===")
    for k, v in grid_sparse.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
