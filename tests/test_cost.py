import torch.nn as nn

from src.cost import PROTOCOL_VERSION, profile


def _tiny():
    return nn.Sequential(nn.Conv3d(3, 4, 3, padding=1), nn.AdaptiveAvgPool3d(1),
                         nn.Flatten(), nn.Linear(4, 1))


def test_profile_keys_and_determinism():
    shape = (3, 4, 16, 16)
    a = profile(_tiny(), shape, clips_per_video=2.5, warmup=2, runs=10)
    b = profile(_tiny(), shape, clips_per_video=2.5, warmup=2, runs=10)
    assert a["protocol"] == PROTOCOL_VERSION
    # Analytic quantities are exactly reproducible across calls
    for k in ("gflops_per_clip", "params_total", "disk_mb"):
        assert a[k] == b[k]
    # Per-video = per-clip x clips_per_video
    assert abs(a["gflops_per_video"] - a["gflops_per_clip"] * 2.5) < 1e-6
    assert a["cpu_ms_per_clip_median"] > 0
    assert a["cpu_threads"] == 4


def test_flops_scale_with_frames():
    a = profile(_tiny(), (3, 4, 16, 16), warmup=1, runs=3)
    b = profile(_tiny(), (3, 8, 16, 16), warmup=1, runs=3)
    assert 1.8 < b["gflops_per_clip"] / a["gflops_per_clip"] < 2.2
