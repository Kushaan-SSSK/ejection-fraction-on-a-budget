import importlib.util

import pytest
import torch

from src.models import build_model

CPU_TESTABLE = ["r2plus1d_18", "r3d_18", "mc3_18", "r2d_18_pool"]


@pytest.mark.parametrize("name", CPU_TESTABLE)
def test_forward_shape(name):
    model = build_model(name, pretrained=False).eval()
    x = torch.randn(2, 3, 8, 64, 64)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (2,)


@pytest.mark.skipif(importlib.util.find_spec("pytorchvideo") is None,
                    reason="pytorchvideo not installed (training PC only)")
@pytest.mark.parametrize("name", ["x3d_s", "x3d_m"])
@pytest.mark.parametrize("frames", [16, 32])  # the two clip lengths 3b actually uses for X3D
def test_x3d_forward_shape(name, frames):
    # 112px input like every other backbone; the wrapper upsamples internally.
    # 8 frames is not tested: x3d_m's temporal kernel needs kT=16.
    model = build_model(name, pretrained=False).eval()
    x = torch.randn(1, 3, frames, 112, 112)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (1,)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_model("resnet50_3d")
