"""Generate sweep config YAMLs. Stage 3a is generated up front; 3b/3c are
generated AFTER 3a's analysis picks the best two clip settings:

    python -m src.make_configs 3a
    python -m src.make_configs 3b --clip 32x2 --clip 16x2
    python -m src.make_configs 3c --clip 16x2 --backbone x3d_s --backbone r2d_18_pool

Physical batch sizes are per-config so effective batch 20 always holds on an
8 GB card (2070 Super, AMP on): scaled down as frames grow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Physical batch that fits 8 GB with AMP, by frame count. Effective stays 20.
PHYSICAL_BATCH = {8: 20, 16: 10, 32: 10, 64: 4}

BACKBONES_3B = ["r3d_18", "mc3_18", "x3d_s", "x3d_m", "r2d_18_pool"]

# X3D cannot run at this project's native 112px decode (see src/models.py) --
# it upsamples internally to its own native resolution instead. Physical
# batch for these two is memory-probed, not looked up in PHYSICAL_BATCH; the
# placeholder here should always be treated as provisional.
X3D_NATIVE_RESOLUTION = {"x3d_s": 160, "x3d_m": 224}


def _write(name: str, **overrides):
    cfg = dict(backbone="r2plus1d_18", pretrained=True, frames=32, period=2,
               resolution=112, effective_batch=20)
    cfg.update(overrides)
    if cfg["backbone"] in X3D_NATIVE_RESOLUTION:
        cfg["native_resolution"] = X3D_NATIVE_RESOLUTION[cfg["backbone"]]
    cfg["physical_batch"] = PHYSICAL_BATCH[cfg["frames"]]
    if cfg["effective_batch"] % cfg["physical_batch"]:
        cfg["physical_batch"] = max(
            b for b in range(1, cfg["physical_batch"] + 1)
            if cfg["effective_batch"] % b == 0)
    CONFIG_DIR.mkdir(exist_ok=True)
    (CONFIG_DIR / f"{name}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True))
    return name


def stage_3a():
    names = []
    for frames in (8, 16, 32, 64):
        for period in (1, 2, 4):
            names.append(_write(f"3a_r2p1d_f{frames}_p{period}",
                                frames=frames, period=period))
    return names


def stage_3b(clips):
    names = []
    for backbone in BACKBONES_3B:
        for frames, period in clips:
            short = backbone.replace("_18", "").replace("_", "")
            names.append(_write(f"3b_{short}_f{frames}_p{period}",
                                backbone=backbone, frames=frames,
                                period=period))
    return names


def stage_3d(base_configs, seeds):
    """Seed replicates: copies a base config's fully-resolved dict, changing
    only `seed`. Writes 3d_<base>_s<seed>.yaml. See docs/PLAN.md's pre-registered
    stage-3d section for which base configs and seeds, and why."""
    names = []
    for base in base_configs:
        base_cfg = yaml.safe_load((CONFIG_DIR / f"{base}.yaml").read_text())
        for seed in seeds:
            cfg = dict(base_cfg)
            cfg["seed"] = seed
            name = f"3d_{base}_s{seed}"
            (CONFIG_DIR / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
            names.append(name)
    return names


def stage_3c(clips, backbones):
    names = []
    for backbone in backbones:
        for frames, period in clips:
            for res in (112, 64):
                short = backbone.replace("_18", "").replace("_", "")
                names.append(_write(f"3c_{short}_f{frames}_p{period}_r{res}",
                                    backbone=backbone, frames=frames,
                                    period=period, resolution=res))
    return names


def _parse_clip(s: str):
    f, p = s.lower().split("x")
    return int(f), int(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["3a", "3b", "3c", "3d"])
    ap.add_argument("--clip", action="append", default=[],
                    help="frames x period, e.g. 32x2 (repeatable)")
    ap.add_argument("--backbone", action="append", default=[])
    ap.add_argument("--base", action="append", default=[],
                    help="3d only: base config stem to replicate (repeatable)")
    ap.add_argument("--seed", action="append", default=[], type=int,
                    help="3d only: new seed value (repeatable)")
    a = ap.parse_args()
    clips = [_parse_clip(c) for c in a.clip]
    if a.stage == "3a":
        made = stage_3a()
    elif a.stage == "3b":
        assert len(clips) == 2, "3b needs exactly two --clip settings (from 3a)"
        made = stage_3b(clips)
    elif a.stage == "3d":
        assert a.base and a.seed, "3d needs --base and --seed"
        made = stage_3d(a.base, a.seed)
    else:
        assert clips and a.backbone, "3c needs --clip and --backbone"
        made = stage_3c(clips, a.backbone)
    print(f"wrote {len(made)} configs: {', '.join(made)}")
