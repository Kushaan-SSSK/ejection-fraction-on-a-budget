"""Revision reanalysis for PDIG-D-26-01680.

Uses only the per-video test-set predictions released in S1 Dataset, so no
model is retrained or re-run. Usage:

    python reanalysis.py <path/to/S1_Dataset> <output_dir>

Outputs (CSV):
  threshold_metrics_ci.csv   sensitivity/specificity at EF 40 and 50 with 95% CIs
  comparisons_adjusted.csv   paired MAE differences vs reference, Holm-adjusted
                             p values, Bonferroni-adjusted intervals, and
                             non-inferiority at margins 0.25, 0.30, 0.50
  seed_summary.csv           seed mean and SD per replicated configuration
  seed_comparisons.csv       seed-averaged paired differences with test-set CI
  coverage_matched.csv       paired MAE differences at matched temporal span
  bootstrap_check.csv        reproduction check of the published MAE intervals
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = 10_000
SEED = 20260814
REF = "3a_r2p1d_f32_p2"
CUTS = (40, 50)
MARGINS = (0.25, 0.30, 0.50)


def load_preds(s1: Path) -> dict:
    preds = {}
    for f in sorted((s1 / "predictions").glob("*.pred.csv")):
        run = f.name.replace(".pred.csv", "")
        df = pd.read_csv(f).sort_values("file").reset_index(drop=True)
        preds[run] = df
    files = preds[REF]["file"].values
    for run, df in preds.items():
        assert (df["file"].values == files).all(), run
    return preds


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main(s1: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    preds = load_preds(s1)
    summary = pd.read_csv(s1 / "summary_all_configs.csv")
    primary = list(summary["config"])
    y = preds[REF]["ef_true"].values
    n = len(y)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))

    abs_err = {r: np.abs(df["ef_pred"].values - y) for r, df in preds.items()}
    boot_mae = {r: e[idx].mean(axis=1) for r, e in abs_err.items()}

    # 1. reproduction check of published MAE intervals
    rows = []
    for _, s in summary.iterrows():
        c = s["config"]
        lo, hi = np.percentile(boot_mae[c], [2.5, 97.5])
        rows.append(dict(config=c, mae=abs_err[c].mean(), mae_pub=s["mae"],
                         ci_lo=lo, ci_hi=hi, ci_lo_pub=s["mae_ci_lo"],
                         ci_hi_pub=s["mae_ci_hi"]))
    pd.DataFrame(rows).to_csv(out / "bootstrap_check.csv", index=False)

    # 2. sensitivity / specificity with 95% bootstrap CIs
    rows = []
    for c in primary:
        yhat = preds[c]["ef_pred"].values
        row = dict(config=c)
        for cut in CUTS:
            pos = y < cut
            tp_v = (yhat < cut) & pos
            tn_v = (yhat >= cut) & ~pos
            sens = tp_v.sum() / pos.sum()
            spec = tn_v.sum() / (~pos).sum()
            sens_b = tp_v[idx].sum(1) / pos[idx].sum(1)
            spec_b = tn_v[idx].sum(1) / (~pos)[idx].sum(1)
            row[f"sens_{cut}"] = sens
            row[f"sens_{cut}_lo"], row[f"sens_{cut}_hi"] = np.percentile(sens_b, [2.5, 97.5])
            row[f"spec_{cut}"] = spec
            row[f"spec_{cut}_lo"], row[f"spec_{cut}_hi"] = np.percentile(spec_b, [2.5, 97.5])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "threshold_metrics_ci.csv", index=False)

    # 3. paired comparisons vs reference with multiplicity adjustment
    others = [c for c in primary if c != REF]
    m = len(others)
    q_lo, q_hi = 0.025 / m, 1 - 0.025 / m
    rows = []
    for c in others:
        d = boot_mae[c] - boot_mae[REF]
        point = abs_err[c].mean() - abs_err[REF].mean()
        p = min(1.0, 2 * min(((d <= 0).sum() + 1) / (B + 1), ((d >= 0).sum() + 1) / (B + 1)))
        lo, hi = np.quantile(d, [0.025, 0.975])
        blo, bhi = np.quantile(d, [q_lo, q_hi])
        row = dict(config=c, diff=point, ci_lo=lo, ci_hi=hi, p_raw=p,
                   bonf_lo=blo, bonf_hi=bhi)
        for mg in MARGINS:
            row[f"ni_{mg:.2f}"] = hi < mg
            row[f"ni_{mg:.2f}_bonf"] = bhi < mg
        rows.append(row)
    cmp = pd.DataFrame(rows)
    cmp["p_holm"] = holm(cmp["p_raw"].values)
    cmp["dist_raw"] = (cmp["ci_lo"] > 0) | (cmp["ci_hi"] < 0)
    cmp["dist_holm"] = cmp["p_holm"] < 0.05
    cmp.to_csv(out / "comparisons_adjusted.csv", index=False)

    # 4. seed replication: training variability vs test-set uncertainty
    sv = pd.read_csv(s1 / "seed_variance.csv")
    rows = []
    groups = {}
    for base, g in sv.groupby("base_config", sort=False):
        runs = list(g["run"])
        groups[base] = runs
        maes = np.array([abs_err[r].mean() for r in runs])
        ci_w = np.mean([np.diff(np.percentile(boot_mae[r], [2.5, 97.5]))[0] for r in runs])
        rows.append(dict(config=base, mae_mean=maes.mean(), mae_sd=maes.std(ddof=1),
                         mae_min=maes.min(), mae_max=maes.max(),
                         sens40_mean=g["sens_at_40"].mean(), sens40_sd=g["sens_at_40"].std(ddof=1),
                         mean_testset_ci_width=ci_w))
    pd.DataFrame(rows).to_csv(out / "seed_summary.csv", index=False)

    rows = []
    for a, b in [("3a_r2p1d_f16_p4", REF), ("3a_r2p1d_f8_p4", "3a_r2p1d_f8_p1")]:
        ma = np.mean([abs_err[r].mean() for r in groups[a]])
        mb = np.mean([abs_err[r].mean() for r in groups[b]])
        da = np.mean([boot_mae[r] for r in groups[a]], axis=0)
        db = np.mean([boot_mae[r] for r in groups[b]], axis=0)
        lo, hi = np.quantile(da - db, [0.025, 0.975])
        pair = np.array([abs_err[i].mean() - abs_err[j].mean() for i in groups[a] for j in groups[b]])
        rows.append(dict(a=a, b=b, seedmean_diff=ma - mb, testset_ci_lo=lo, testset_ci_hi=hi,
                         pairings_sd=pair.std(ddof=1), pairings_min=pair.min(), pairings_max=pair.max()))
    pd.DataFrame(rows).to_csv(out / "seed_comparisons.csv", index=False)

    # 5. temporal coverage: matched span F*P (also matched clips per video)
    grid = summary[summary["config"].str.startswith("3a_")].copy()
    grid["span"] = grid["frames"] * grid["period"]
    rows = []
    for span, g in grid.groupby("span"):
        g = g.sort_values("frames")
        cfgs = list(g["config"])
        for i in range(len(cfgs)):
            for j in range(i + 1, len(cfgs)):
                a, b = cfgs[i], cfgs[j]  # a sparser (fewer frames), b denser
                d = boot_mae[a] - boot_mae[b]
                lo, hi = np.quantile(d, [0.025, 0.975])
                rows.append(dict(span=span, sparser=a, denser=b,
                                 mae_sparser=abs_err[a].mean(), mae_denser=abs_err[b].mean(),
                                 diff=abs_err[a].mean() - abs_err[b].mean(), ci_lo=lo, ci_hi=hi,
                                 clips_per_video=preds[a]["n_clips"].mean()))
    pd.DataFrame(rows).to_csv(out / "coverage_matched.csv", index=False)
    print("done:", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
