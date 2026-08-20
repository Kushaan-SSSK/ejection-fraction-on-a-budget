# Cost Measurement Protocol — DRAFT (freezes at Gate 2 as `cost-harness-v1`)

The executable contract is `src/cost.py`; this file is the human-readable one.
After the Gate 2 review, tag the repo `cost-harness-v1`. Any change after the
tag requires a new tag and re-profiling **every** configuration. Cost numbers
from different tags never appear in the same table.

## X3D-S / X3D-M: input is upsampled, and every X3D number is affected

**Applies to every row where `backbone` is `x3d_s` or `x3d_m`.** X3D cannot
run on this study's native 112×112 decode at all — verified 2026-08-18: its
downsampling stride pattern is built for 160px (X3D-S) / 224px (X3D-M) input,
and at 112px a deep-layer feature map shrinks below that layer's own
convolution kernel, raising a hard `RuntimeError` (confirmed at every clip
length X3D is run at in this study). The fix (`src/models.py::UpsampleThenModel`)
upsamples the already-decoded 112px clip to X3D's native size immediately
before X3D's own first layer — no other backbone in this codebase carries
this wrapper, and the data pipeline (decode, `data.py`, `train.py`, `cost.py`)
is otherwise identical across every config in the study.

**This is a real, unresolved limitation, not a footnote:** X3D's Kinetics-400
weights were pretrained on sharp native-resolution video. An upsampled 112px
source has the right *shape* to satisfy the architecture but not the same
*detail level* the network was trained on — X3D may not use its pretrained
weights as effectively here as it would on a genuine 160/224px decode.
**Consequently, X3D's accuracy numbers in this study cannot be cleanly
attributed to "X3D is less accuracy-efficient for this task" versus "X3D was
handicapped by input degradation that no other backbone in the sweep
experiences."** Both cost and accuracy tables must carry this caveat directly
next to X3D's rows — see `results/stage3ab_summary.csv`'s `caveat` column —
never only in prose a reader has to go find.

The upsample's own compute cost is *not* excluded from X3D's cost numbers —
`cost.py` times the model's whole `forward()` call, which includes the
upsample, so `gflops_per_clip` / `cpu_ms_per_clip_median` / `gpu_ms_per_clip_median`
for X3D rows honestly reflect the full cost of making X3D work at this
study's decode resolution, upsample tax included. (fvcore does not have a
FLOP counter registered for the interpolate op itself; that shows up as an
entry in `flops_unsupported_ops`, per the existing never-drop-silently policy
below — the omitted cost is small relative to the convolution stack.)

Also see `registry.csv`'s `deviations` column, which flags this per training
run (`upsampled_112_to_160` / `upsampled_112_to_224`), and PLAN.md's
limitations checklist.

## Measured quantities

| Quantity | Method | Conditions |
|---|---|---|
| GFLOPs | fvcore `FlopCountAnalysis` | fvcore counts MACs; reported GFLOPs = 2 × GMACs. Unsupported ops recorded in the row, never dropped silently. |
| Parameters | direct count | trainable and total |
| GPU latency | wall clock around `forward()` | batch 1, fp32, eval mode, `cudnn.benchmark=True`, `torch.cuda.synchronize` around each timed run, 20 warmup discarded, 100 timed, median + IQR |
| CPU latency | same | `torch.set_num_threads(4)` (handheld proxy), no background load, same warmup/repeat/median |
| Peak GPU memory | `torch.cuda.max_memory_allocated` | reset before one inference forward |
| Size on disk | fp32 `state_dict` bytes | no quantization claims |

## Views

Every quantity that scales with input is reported **per-clip** (one forward)
and **per-video** (per-clip × mean non-overlapping clips per official TEST
video at that config's frames × period). The two never share a column.

### What `cpu_ms_per_video` (the primary cost axis) actually contains

`cpu_ms_per_video = cpu_ms_per_clip_median × clips_per_video`. Both factors
are separate columns in `results/costs.csv` — the composite is always
decomposable, never reported alone.

- **`cpu_ms_per_clip_median`** is pure model-forward time: one `forward()`
  call on a single already-materialized clip tensor (`torch.zeros(1, 3,
  frames, 112, 112)` — synthetic, not a real decoded clip), batch 1, fp32,
  eval mode, 4 threads, median of 100 timed runs after 20 discarded warmup
  runs (`src/cost.py::_time_forward`). **It does not include video
  decoding, frame extraction, resizing, normalization, or any dataloader
  I/O.** Those costs are excluded on purpose, because they depend on
  deployment-specific decode/storage choices (codec, disk vs. cached
  tensors, etc.) that are out of scope for a model-cost comparison — see
  Limitation below.
- **`clips_per_video`** is a data-dependent tiling count, not a timed
  quantity: for each *official TEST-split* video, `max(video_frame_count
  // (frames × period), 1)` non-overlapping clips (matching the frozen
  test protocol below), averaged over all TEST videos
  (`src/data.py::EchoNetDataset.clips_per_video`). It depends only on
  `frames × period` and the real distribution of video lengths in the
  test set — not on anything measured with a stopwatch.

This is why, at a fixed frame count, `cpu_ms_per_clip_median` is
near-constant across period (the model always processes exactly `frames`
frames per forward call, regardless of stride) while `clips_per_video` —
and therefore `cpu_ms_per_video` — falls roughly in proportion to `1/period`:
a longer stride between sampled frames means each clip covers more real
video time, so fewer non-overlapping clips are needed to tile a video.
Example at 8 frames: `clips_per_video` is 21.7 / 10.6 / 5.0 at period
1/2/4, driving `cpu_ms_per_video` from ~3267ms down to ~761ms even though
`cpu_ms_per_clip_median` stays at ~150ms and GFLOPs/clip stays at 40.6
across all three.

**Limitation (carry into the paper's limitations checklist):** `cpu_ms_per_video`
is model-compute cost only. It is not an end-to-end deployment latency —
real-world serving also pays for decode/preprocessing, which this protocol
deliberately excludes and does not currently measure at all, at any
config.

## Fixed environment

All rows in `results/costs.csv` are measured on one machine (the 2070 Super
PC), hardware recorded in the row (`gpu_name`) and in ENVIRONMENT.md, with no
other GPU/CPU load. Rows measured elsewhere are invalid.

## Test protocol (accuracy side, same freeze discipline)

Video-level EF prediction = mean over all non-overlapping clips starting at
frame 0 (≥1 clip; short videos loop-padded). This is a custom
deployment-oriented tiling rule, applied uniformly to every configuration;
it differs from the original EchoNet repository's test-time evaluation,
which uses all possible clip start positions and zero-pads short videos.
Used for every accuracy number in the paper.
