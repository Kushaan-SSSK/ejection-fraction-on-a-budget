"""Generate LaTeX table bodies for the revised manuscript.

    python make_tables.py <path/to/S1_Dataset> <outputs_dir>

Reads S1 Dataset summary/cost files and the reanalysis outputs, and writes
one .tex fragment per table into <outputs_dir>/tables/. Fragments are pasted
into final_manuscript.tex (PLOS requires a single source file).
"""

import sys
from pathlib import Path

import pandas as pd

REF = "3a_r2p1d_f32_p2"
NAMES = {"r2plus1d_18": "R(2+1)D-18", "r3d_18": "R3D-18", "mc3_18": "MC3-18",
         "x3d_s": "X3D-S", "x3d_m": "X3D-M", "r2d_18_pool": "2D ResNet-18 + pool"}


def label(row) -> str:
    s = f"{NAMES[row['backbone']]} {int(row['frames'])}f$\\times$p{int(row['period'])}"
    if row["backbone"].startswith("x3d"):
        s += "\\,$^{\\dagger}$"
    if row["config"] == REF:
        s += " (ref.)"
    return s


def ci(x, lo, hi, fmt="{:.3f}"):
    return f"{fmt.format(x)} ({fmt.format(lo)}--{fmt.format(hi)})"


def signed(x):
    return f"{x:+.3f}".replace("-", "$-$") if x < 0 else f"{x:+.3f}"


def yn(b):
    return "yes" if b else "no"


def main(s1: Path, out: Path) -> None:
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(s1 / "summary_all_configs.csv")
    costs = pd.read_csv(s1 / "costs.csv").set_index("config")
    summ = summ.sort_values("cpu_ms_per_video", kind="stable").reset_index(drop=True)
    summ["label"] = summ.apply(label, axis=1)
    lab = dict(zip(summ["config"], summ["label"]))
    order = list(summ["config"])

    # Table 1: cost, with measured per-clip CPU latency added
    lines = []
    for _, r in summ.iterrows():
        c = costs.loc[r["config"]]
        lines.append(" & ".join([
            r["label"], f"{c['gflops_per_clip']:.1f}", f"{c['params_total'] / 1e6:.1f}",
            f"{c['disk_mb']:.0f}", f"{c['clips_per_video']:.2f}",
            f"{c['cpu_ms_per_clip_median']:.0f}", f"{c['cpu_ms_per_video']:.0f}",
            f"{c['gpu_ms_per_video']:.1f}", f"{c['gpu_peak_mem_mb']:.0f}"]) + " \\\\")
    (tdir / "t1_cost.tex").write_text("\n".join(lines) + "\n")

    # Table 2: accuracy (threshold metrics moved to Table 3)
    lines = []
    for _, r in summ.iterrows():
        lines.append(" & ".join([
            r["label"], ci(r["mae"], r["mae_ci_lo"], r["mae_ci_hi"], "{:.2f}"),
            f"{r['rmse']:.2f}", f"{r['r2']:.3f}"]) + " \\\\")
    (tdir / "t2_accuracy.tex").write_text("\n".join(lines) + "\n")

    # Table 3: threshold metrics with CIs
    th = pd.read_csv(out / "threshold_metrics_ci.csv").set_index("config")
    lines = []
    for c in order:
        t = th.loc[c]
        cells = [lab[c]]
        for k in ("sens_40", "spec_40", "sens_50", "spec_50"):
            cells.append(ci(t[k], t[k + "_lo"], t[k + "_hi"], "{:.2f}"))
        lines.append(" & ".join(cells) + " \\\\")
    (tdir / "t3_threshold.tex").write_text("\n".join(lines) + "\n")

    # Table 4: matched temporal span
    cov = pd.read_csv(out / "coverage_matched.csv")
    lines = []
    for _, r in cov.iterrows():
        lines.append(" & ".join([
            f"{int(r['span'])}", f"{r['clips_per_video']:.2f}",
            lab[r["sparser"]].replace("R(2+1)D-18 ", ""), f"{r['mae_sparser']:.2f}",
            lab[r["denser"]].replace("R(2+1)D-18 ", "").replace(" (ref.)", " (ref.)"),
            f"{r['mae_denser']:.2f}",
            f"{signed(r['diff'])} ({signed(r['ci_lo'])}, {signed(r['ci_hi'])})"]) + " \\\\")
    (tdir / "t4_span.tex").write_text("\n".join(lines) + "\n")

    # Table 5: comparisons with multiplicity adjustment
    cmp = pd.read_csv(out / "comparisons_adjusted.csv").set_index("config")
    lines = []
    for c in order:
        if c == REF:
            lines.append(f"{lab[c]} & -- & -- & -- & -- & -- \\\\")
            continue
        r = cmp.loc[c]
        p = r["p_holm"]
        ptxt = "$<$0.005" if p < 0.005 else f"{p:.3f}" if p < 1 else "1.000"
        lines.append(" & ".join([
            lab[c], f"{signed(r['diff'])} ({signed(r['ci_lo'])}, {signed(r['ci_hi'])})", ptxt,
            f"{yn(r['dist_raw'])} / {yn(r['dist_holm'])}",
            f"{yn(r['ni_0.50'])} / {yn(r['ni_0.50_bonf'])}",
            f"{yn(r['ni_0.25'])} / {yn(r['ni_0.25_bonf'])}"]) + " \\\\")
    (tdir / "t5_comparisons.tex").write_text("\n".join(lines) + "\n")

    # Table 6: seeds
    sv = pd.read_csv(s1 / "seed_variance.csv")
    ss = pd.read_csv(out / "seed_summary.csv").set_index("config")
    lines = []
    for base, g in sv.groupby("base_config", sort=False):
        row = summ[summ["config"] == base].iloc[0]
        s = ss.loc[base]
        lines.append(" & ".join([
            lab[base], " / ".join(f"{v:.3f}" for v in g["mae"]),
            f"{s['mae_mean']:.3f} ({s['mae_sd']:.3f})",
            " / ".join(f"{v:.3f}" for v in g["sens_at_40"]),
            f"{s['sens40_mean']:.3f} ({s['sens40_sd']:.3f})"]) + " \\\\")
    (tdir / "t6_seeds.tex").write_text("\n".join(lines) + "\n")

    # S2 Table: full adjusted comparisons as CSV
    s2 = cmp.reset_index()
    s2.insert(1, "label", s2["config"].map(lambda c: lab[c].replace("$\\times$", "x")
                                         .replace("\\,$^{\\dagger}$", " (X3D caveat)")))
    s2.to_csv(out / "S2_Table.csv", index=False)
    print("tables written to", tdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
