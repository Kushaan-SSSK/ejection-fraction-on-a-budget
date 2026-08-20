# Ejection Fraction on a Budget — Execution Plan (v1.1, 2026-08-14)

**v1.1 decisions (locked):** full EchoNet-Dynamic kept (smaller datasets like
CAMUS gut the test-set statistics the paper depends on). Time cut instead via
the **fast recipe**: AMP (fp16 autocast) + 30 epochs (LR step ×0.1 @ 10) +
optional decode cache — applied identically to every run, so Invariant 1
holds. Training machine: **RTX 2070 Super (8 GB) / 32 GB RAM**; effective
batch 20 preserved everywhere via gradient accumulation (see ENVIRONMENT.md
table). Fallback if time runs short: subsample the *train* set to 50%, keep
the full test set. Repo is built and smoke-tested; sweep driver is
`scripts/run_sweep.ps1`.

Accuracy-versus-compute study of video models for EF estimation on EchoNet-Dynamic.

## Invariants (never violated)

1. **One training recipe** for every sweep run — same optimizer, schedule, epochs, augmentation, seed. Memory issues handled by gradient accumulation (preserving effective batch), deviation logged.
2. **One cost protocol**, frozen at Gate 2 (`cost-harness-v1` tag). A forced fix = new tag + re-profile *everything*. Never mix protocols in one table.
3. **One test protocol**: all non-overlapping clips per video, predictions averaged (a custom deployment tiling rule, applied uniformly; differs from the EchoNet repo's all-starts, zero-padded evaluation). Per-clip and per-video costs are separate columns, never conflated.
4. **Agreement compliance**: individual Redivis registration, no Kaggle mirror, no redistribution of videos/frames. Local transcodes stay local.
5. **No config reported without all three metric families** (accuracy, clinical, cost).
6. **Verify before citing**: R² = 0.81 is unverified — check the Nature results table first. (MAE 4.1 / AUC 0.97 verified from abstract.)

## Phase 0 — Setup (Week 1)

- [ ] Day 1: every member registers at echonet.github.io/dynamic, accepts agreement, starts ~7 GB Redivis download
- [ ] `git init`, MIT license, lock environment (python 3.11, torch/torchvision, pytorchvideo, fvcore, opencv-headless); record versions + GPU/driver in `ENVIRONMENT.md`
- [ ] Validate data: 10,030 videos decode at 112×112; `FileList.csv` patient-level splits match ~7,465/1,288/1,277 (record exact)
- [ ] Tabulate test-set n per EF band (<30, 30–40, 40–55, >55) — sets expectations for stratified-metric CIs
- [ ] Compute train-set pixel mean/std once, store, use everywhere
- [ ] Mock figures F1 (cost log-x vs MAE, frontier) and F2 (sensitivity@EF40 vs cost) with fake data; fix axes and encodings now

## Phase 1 — Baseline reproduction (Weeks 1–2)

Reference config (EchoNet repo defaults): r2plus1d_18 Kinetics-400 pretrained, 32 frames × period 2, 112², SGD lr 1e-4 / momentum 0.9 / wd 1e-4, step ×0.1 @ 15 epochs, 45 epochs, batch 20, random clip per video per epoch (train), all-clips-averaged (test).

- [ ] Train reference; evaluate with **our** metrics code; cross-check our metrics vs repo's on identical predictions
- [ ] **GATE 1: test MAE ≤ 4.6%** (within 0.5 of 4.1). If missed: debug list (weights loaded? normalization? clip sampling? label parsing? leakage? LR off-by-one), time-boxed to 1 week; fallback = use our reproduction as internal reference, disclosed in paper
- [ ] Week 2: literature re-check (Echo-E3Net, Mobile U-Net, EchoCoTr, on-device EF). Code available → re-profile under our harness later; else plot published numbers, flagged not-protocol-matched

## Phase 2 — Cost harness (Weeks 1–2, parallel)

`profile(model, input_shape) -> CostProfile`, one CSV row per config in `results/costs.csv`.

- FLOPs: fvcore (report GFLOPs = 2 × GMACs, convention stated; cross-check ptflops; log unsupported ops)
- Params (trainable + total); size on disk (fp32 state_dict)
- GPU latency: batch 1, fp32, eval, cuda.synchronize, 20 warmup discarded, ≥100 runs, median + IQR, cudnn.benchmark fixed on
- CPU latency: `torch.set_num_threads(4)`, same protocol, no background load
- Peak GPU memory: `max_memory_allocated`, inference only
- Both **per-clip** and **per-video** (per-clip × mean clips/video for that clip-length/period)

- [ ] **GATE 2: harness reviewed by second person, unit-tested, `PROTOCOL.md` written, tagged `cost-harness-v1`**

## Phase 3 — Sweep (Weeks 3–7)

| Stage | Varies | Fixed | Runs |
|---|---|---|---|
| 3a | frames {8,16,32,64} × period {1,2,4} | r2plus1d_18 pretrained | 12 |
| 3b | r3d_18, mc3_18, X3D-S, X3D-M, 2D ResNet-18 + temporal mean-pool | best two clip settings from 3a | 10 |
| 3c (opt.) | resolution 112 vs 64 | two cheapest 3b survivors | 4 |
| 3d (robustness) | +2 seeds on 3–4 frontier-knee configs | everything | 6–8 |

Decisions made now:
- "Best two" from 3a = reference (32×2) + cheapest config within 0.5 MAE of best 3a run (pre-registered, not post-hoc)
- X3D fed 112² despite 160/224 nominal — **superseded 2026-08-18: does not work.** Verified failure (hard `RuntimeError`, not a distribution-shift-but-functional caveat as originally assumed) at both 3b clip settings for both X3D backbones. Fix: internal upsample to native resolution inside the model only (`src/models.py`), decode pipeline stays 112² like every other backbone. Full detail and the resulting pretrained-weight-mismatch limitation: `PROTOCOL.md`.
- MoViNet = stretch goal only; skip if no maintained pretrained PyTorch port found in Week 2
- 2D floor: ImageNet-pretrained ResNet-18, per-frame features mean-pooled
- One fixed seed for the grid; 3d supplies error bars where ranking matters

Mechanics: one YAML per run in `configs/` (`3a_r2p1d_f32_p2.yaml`); every run appends to `results/registry.csv` (run id, config hash, git commit, seed, times, best val MAE, checkpoint, deviations); checkpoint + auto-resume; run 3a cheapest-first. Budget ≈ 150–250 GPU-h total on one 12–24 GB card.

- [ ] **GATE 3 (end of 3a): 12 configs × all three metric families, F1 drawn with real data — minimal publishable unit**

### Stage 3d — pre-registered 2026-08-18, before any seed run started

Written and committed before `configs/3d_*.yaml` exist or any seed-replicate
training begins, specifically so config choice cannot be second-guessed
after seeing which seeds would help. This instantiates the stage-3d row
above (2 seeds × 3–4 configs, 6–8 runs) with concrete choices:

**Configs (4), and why each:** `3a_r2p1d_f32_p2` (the reference — every
equivalence claim in the paper is measured against this point, so its own
run-to-run stability has to be known), `3a_r2p1d_f16_p4` (the "half the
compute, no measurable loss" headline claim — this is the single most
reviewer-exposed number in stage 3a), `3a_r2p1d_f8_p1` and `3a_r2p1d_f8_p4`
(the matched pair, same frame count, differing only in period, anchoring
the "sparse sampling beats dense sampling" finding). This is a
claim-driven selection rather than a pure geometric-frontier-knee
selection — it directly stress-tests the two specific results a reviewer
is most likely to question, which the plain "frontier-knee" language above
under-specifies anyway. `3a_r2p1d_f64_p4` (best point-estimate MAE in
stage 3a, the other true Pareto-frontier point) is a reasonable candidate
for a *future* extension of this same exercise but is not included now —
kept to 4 configs to match the original 6–8 run budget rather than
expanding scope unprompted.

**Seed values, by a stated rule:** the original recipe seed is `20260814`
(the date the fast recipe was frozen). The two replicate seeds are
`20260815` and `20260816` — the following two calendar dates, chosen only
for a simple, auditable, monotonic rule with nothing tuned to any run's
outcome (no run has used these seeds yet at the time this section was
written).

**Runs:** 4 configs × 2 new seeds = 8 training runs, naming pattern
`3d_<base-config-name>_s<seed>.yaml`, otherwise byte-identical to the base
config (same backbone/frames/period/resolution/physical_batch) with only
`seed` changed.

**Cost harness:** skipped for all 8 seed-replicate runs. Confirmed, not
assumed — `src/cost.py::profile_config` always builds the model with
`pretrained=False` and never loads a checkpoint or touches
`torch.manual_seed`; every quantity it measures (FLOPs, params, disk size,
GPU/CPU latency, peak memory) is a property of the fixed architecture, not
the trained weights. A seed-replicate's cost is by construction identical
to its base config's already-measured row in `results/costs.csv` — look
it up by base config name (`3d_3a_r2p1d_f32_p2_s20260815` → cost row
`3a_r2p1d_f32_p2`), do not expect or add a duplicate row.

**Expected cost:** ~17 GPU-hours (2 × the sum of the 4 base configs' actual
stage-3a training hours: f32_p2 3.92h + f16_p4 2.02h + f8_p1 1.20h + f8_p4
1.13h ≈ 8.27h × 2 seeds).

## Phase 4 — Metrics & analysis (Weeks 5–8, rolling as runs finish)

Per config:
- **Accuracy**: MAE, RMSE, R², Bland–Altman bias + 95% LoA
- **Clinical**: sens/spec at EF<40 and EF<50, AUC reduced-EF, MAE by band with per-band n
- **Cost**: GFLOPs, params, GPU/CPU latency, peak memory, disk — per-clip and per-video

Statistics (pre-stated):
- 10,000 patient-level bootstrap resamples; 95% CIs on everything
- Config comparisons: paired bootstrap on per-video errors; differences of a few tenths MAE expected to straddle zero → say "indistinguishable", don't rank; frontier dominance requires paired CI excluding zero
- **Headline comparison**: cheapest config non-inferior to reference at 0.5-point margin (one-sided 95%) = deployment recommendation
- Frontier per cost axis (primary: CPU latency per video; secondary: FLOPs, memory); prior work overlaid

Figures: F1 cost-vs-MAE frontier · F2 sensitivity@EF40 vs cost · F3 error-by-band + Bland–Altman (reference vs recommended) · F4 resolution (if 3c ran). All generated by script from `results/`, never by hand.

## Phase 5 — Write-up & release (Weeks 8–10)

- Venue: Bioengineering / Diagnostics / Sensors; draft to strictest format, decide Week 9
- Release: code (MIT), config YAMLs, registry + costs + metrics CSVs, `PROTOCOL.md`. No videos/frames. Check agreement before releasing weights; if unclear, code + CSVs only
- Checklist: R² claim verified or dropped; every number traceable to a registry row; clean-machine repro of one small config from the public repo; limitations (single dataset/view/site, 112², one GPU model, no quantization, single-seed grid + 3d bars, cost harness excludes decode/preprocessing — see PROTOCOL.md, X3D-S/X3D-M run on 112px input upsampled to their native 160/224px rather than a genuine higher-res decode — pretrained-weight mismatch confound, cannot cleanly separate "less efficient" from "handicapped by upsampling," full detail in PROTOCOL.md, must be flagged wherever X3D numbers are reported)

## Timeline

| Wk | Work | Gate |
|---|---|---|
| 1 | data access, env, validation, figure mocks, harness draft, baseline starts | data verified |
| 2 | baseline evaluated; harness frozen; lit re-check | G1 + G2 |
| 3–4 | stage 3a, rolling eval | — |
| 5 | 3a analysis, pick top-2 settings; 3b starts | G3 |
| 6 | 3b completes | — |
| 7 | 3c + 3d; profile prior-work code | sweep frozen |
| 8 | full stats, figures final, results drafted | frontier fixed |
| 9 | full draft, internal review, venue, repo cleanup | draft done |
| 10 | revisions, repro check, submit | submission |

Behind at Week 5 → cut 3c first, then 3d, then trim 3b to three backbones. 3a is never cut.

## Repo layout

```
ef-on-a-budget/
├── docs/               PLAN.md  PROTOCOL.md  ENVIRONMENT.md
├── configs/            one YAML per run
├── src/                data.py  models.py  train.py  evaluate.py  cost.py
├── scripts/            sweep drivers and analysis
├── results/            registry.csv  costs.csv  metrics/
├── figures/            script-generated only
└── tests/              harness determinism, metric cross-checks
```

**First three actions:** ① register on Redivis + start download · ② `git init` + lock environment · ③ mock F1/F2 and fix their axes.
