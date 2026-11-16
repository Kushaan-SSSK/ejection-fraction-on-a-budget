"""Replace / insert table floats in final_manuscript.tex from generated fragments.

    python apply_tables.py <manuscript.tex> <tables_dir>
"""

import re
import sys
from pathlib import Path

tex_path, tdir = Path(sys.argv[1]), Path(sys.argv[2])
tex = tex_path.read_text(encoding="utf-8")
frag = {p.stem: p.read_text(encoding="utf-8").rstrip("\n") for p in tdir.glob("*.tex")}
DAG = "$^{\\dagger}$X3D caveat as in Table~\\ref{tab:cost}."

T1 = r"""\begin{table}[!ht]
\singlespacing
\centering
\caption{{\bf Cost profile of every configuration, sorted by estimated CPU
latency per video.} Measured under the frozen protocol (fp32, batch 1; CPU
limited to 4 threads; median of 100 runs after 20 warmup runs). The CPU
ms/clip column is measured directly; the per-video columns are estimates
equal to the measured per-clip value multiplied by the mean number of
clips per test video, assuming sequential processing and excluding video
decoding and preprocessing. $^{\dagger}$X3D rows: input upsampled from
112 pixels to the architecture's native resolution inside the model; see
Materials and Methods.}
\centering
\footnotesize
% table body generated from the results CSVs
\begin{tabular}{lrrrrrrrr}
\toprule
Configuration & \multicolumn{1}{c}{GFLOPs} & \multicolumn{1}{c}{Params} &
\multicolumn{1}{c}{Disk} & \multicolumn{1}{c}{Clips} &
\multicolumn{1}{c}{CPU} & \multicolumn{1}{c}{CPU} &
\multicolumn{1}{c}{GPU} & \multicolumn{1}{c}{Mem} \\
 & \multicolumn{1}{c}{/clip} & \multicolumn{1}{c}{(M)} &
\multicolumn{1}{c}{(MB)} & \multicolumn{1}{c}{/video} &
\multicolumn{1}{c}{(ms/clip)} & \multicolumn{1}{c}{(ms/video)} &
\multicolumn{1}{c}{(ms/video)} & \multicolumn{1}{c}{(MB)} \\
\midrule
""" + frag["t1_cost"] + r"""
\bottomrule
\end{tabular}
\label{tab:cost}
\end{table}"""

T2 = r"""\begin{table}[!ht]
\singlespacing
\centering
\caption{{\bf Accuracy for every configuration, ordered as in
Table~\ref{tab:cost}.} MAE with 95\% patient-level bootstrap CI.
""" + DAG + r"""}
\centering
\footnotesize
% table body generated from the results CSVs
\begin{tabular}{lccc}
\toprule
Configuration & MAE (95\% CI) & RMSE & $R^2$ \\
\midrule
""" + frag["t2_accuracy"] + r"""
\bottomrule
\end{tabular}
\label{tab:accuracy}
\end{table}

\begin{table}[!ht]
\singlespacing
\centering
\caption{{\bf Sensitivity and specificity at the 40\% and 50\% cut points,
ordered as in Table~\ref{tab:cost}.} Reduced \ef{} is the positive class
(predicted and reference \ef{} below the cut point). Values with 95\%
patient-level bootstrap CIs; 160 condition-positive videos at 40\% and 285
at 50\% among 1{,}277 test videos. """ + DAG + r"""}
\centering
\scriptsize
% table body generated from the results CSVs
\begin{tabular}{lcccc}
\toprule
Configuration & Sens$_{40}$ (95\% CI) & Spec$_{40}$ (95\% CI) &
Sens$_{50}$ (95\% CI) & Spec$_{50}$ (95\% CI) \\
\midrule
""" + frag["t3_threshold"] + r"""
\bottomrule
\end{tabular}
\label{tab:threshold}
\end{table}"""

T4 = r"""\begin{table}[!ht]
\singlespacing
\centering
\caption{{\bf R(2+1)D-18 configurations compared at matched temporal
span.} Within each span ($F \cdot P$, raw frames), configurations cover
the same stretch of video and average the same number of clips per video,
differing only in frame density. $\Delta$MAE is sparser minus denser, with
the 95\% CI of the paired bootstrap difference (unadjusted; exploratory).}
\centering
\footnotesize
% table body generated from the results CSVs
\begin{tabular}{rrlclcc}
\toprule
Span & Clips/video & Sparser & MAE & Denser & MAE & $\Delta$MAE (95\% CI) \\
\midrule
""" + frag["t4_span"] + r"""
\bottomrule
\end{tabular}
\label{tab:span}
\end{table}"""

T5 = r"""\begin{table}[!ht]
\singlespacing
\centering
\caption{{\bf Paired bootstrap comparison of every configuration against
the reference.} Identical resamples, 10{,}000 draws. $\Delta$MAE with the
unadjusted 95\% CI; Holm-adjusted p value across the 21 comparisons. A
configuration is distinguishable when the unadjusted CI excludes zero
(unadj.) or when the Holm-adjusted p value is below 0.05 (Holm), and
non-inferior at a margin when the upper bound of the paired difference is
below that margin, using the unadjusted 95\% CI (unadj.) or the
Bonferroni-adjusted one-sided bound (adj.). The 0.50 margin was
pre-specified; 0.25 is a sensitivity analysis. Results at a 0.30 margin
are in \nameref{S2_Table}. """ + DAG + r"""}
\scriptsize
% table body generated from the results CSVs
\begin{tabular}{lccccc}
\toprule
 & & & Distinguishable & Non-inferior, 0.50 & Non-inferior, 0.25 \\
Configuration & $\Delta$MAE vs.\ ref.\ (95\% CI) & Holm p &
unadj. / Holm & unadj. / adj. & unadj. / adj. \\
\midrule
""" + frag["t5_comparisons"] + r"""
\bottomrule
\end{tabular}
\label{tab:comparisons}
\end{table}"""

T6 = r"""\begin{table}[!ht]
\singlespacing
\centering
\caption{{\bf Seed replication of the four pre-registered configurations.}
Seeds 20260814 / 20260815 / 20260816; mean and standard deviation (SD)
across the three seeds.}
\footnotesize
% table body generated from the results CSVs
\begin{tabular}{lcccc}
\toprule
Configuration & MAE by seed & MAE mean (SD) & Sens$_{40}$ by seed &
Sens$_{40}$ mean (SD) \\
\midrule
""" + frag["t6_seeds"] + r"""
\bottomrule
\end{tabular}
\label{tab:seeds}
\end{table}"""


def block_span(text: str, label: str):
    i = text.index(f"\\label{{{label}}}")
    s = text.rindex("\\begin{table}", 0, i)
    e = text.index("\\end{table}", i) + len("\\end{table}")
    return s, e


for label, new in [("tab:cost", T1), ("tab:accuracy", T2),
                   ("tab:comparisons", T5), ("tab:seeds", T6)]:
    s, e = block_span(tex, label)
    tex = tex[:s] + new + tex[e:]

anchor = "% SPAN TABLE HERE"
assert anchor in tex, "span anchor missing"
tex = tex.replace(anchor, T4)
assert "—" not in tex
tex_path.write_text(tex, encoding="utf-8")
print("tables applied")
