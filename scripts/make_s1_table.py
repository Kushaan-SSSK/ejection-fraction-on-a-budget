"""Build S1 Table (threshold metrics for every primary configuration).

    python scripts/make_s1_table.py <path/to/S1_Dataset> <out.csv>

For each of the 22 primary configurations in summary_all_configs.csv:
  auc_<cut>, auc_<cut>_ci_lo, auc_<cut>_ci_hi   from metrics/<config>.metrics.json
                                                 (rounded to 4 decimals)
  tp/fn/fp/tn_<cut>                              confusion-matrix counts computed
                                                 from predictions/<config>.pred.csv
at the EF cut points 40 and 50. Reduced EF (below the cut point) is the
positive class, for both the reference and the predicted EF. Reads only
S1 Dataset files; no model is run.
"""

import json
import sys
from pathlib import Path

import pandas as pd

CUTS = (40, 50)


def main(s1: Path, out: Path) -> None:
    summary = pd.read_csv(s1 / "summary_all_configs.csv")
    rows = []
    for _, s in summary.sort_values("config").iterrows():
        c = s["config"]
        m = json.loads((s1 / "metrics" / f"{c}.metrics.json").read_text())
        p = pd.read_csv(s1 / "predictions" / f"{c}.pred.csv")
        row = dict(config=c, backbone=s["backbone"], frames=int(s["frames"]),
                   period=int(s["period"]))
        for cut in CUTS:
            for suffix in ("", "_ci_lo", "_ci_hi"):
                row[f"auc_{cut}{suffix}"] = round(m[f"auc_at_{cut}{suffix}"], 4)
            pos = p["ef_true"] < cut
            pred_pos = p["ef_pred"] < cut
            row[f"tp_{cut}"] = int((pos & pred_pos).sum())
            row[f"fn_{cut}"] = int((pos & ~pred_pos).sum())
            row[f"fp_{cut}"] = int((~pos & pred_pos).sum())
            row[f"tn_{cut}"] = int((~pos & ~pred_pos).sum())
        rows.append(row)
    # "\n" line endings so the output is byte-identical on every OS
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
