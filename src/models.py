"""Backbone factory: every architecture in the sweep behind one interface.

build_model(name, pretrained) -> nn.Module mapping (B, 3, T, H, W) -> (B,)
EF regression output in percent (raw, no sigmoid; targets are 0-100).

Backbones:
  r2plus1d_18, r3d_18, mc3_18  torchvision, Kinetics-400 weights
  x3d_s, x3d_m                 pytorchvideo (torch.hub), Kinetics-400 weights;
                               cannot run at 112px (stride pattern built for
                               160/224px input), so UpsampleThenModel resizes
                               the decoded clip to native size inside the
                               model. See docs/PROTOCOL.md for the resulting
                               pretrained-weight caveat.
  r2d_18_pool                  ImageNet ResNet-18 per frame, features
                               mean-pooled over time (the 2D floor)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TORCHVISION_3D = {"r2plus1d_18", "r3d_18", "mc3_18"}
X3D = {"x3d_s", "x3d_m"}
ALL_BACKBONES = sorted(TORCHVISION_3D | X3D | {"r2d_18_pool"})

X3D_NATIVE_RESOLUTION = {"x3d_s": 160, "x3d_m": 224}


class UpsampleThenModel(nn.Module):
    """Bilinearly upsamples each frame to `size` (spatial only) before
    calling the wrapped model."""

    def __init__(self, model: nn.Module, size: int):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if (h, w) != (self.size, self.size):
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(x, size=(self.size, self.size),
                              mode="bilinear", align_corners=False)
            x = x.reshape(b, t, c, self.size, self.size).permute(0, 2, 1, 3, 4)
        return self.model(x)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name in TORCHVISION_3D:
        return _torchvision_3d(name, pretrained)
    if name in X3D:
        return _x3d(name, pretrained)
    if name == "r2d_18_pool":
        return Frame2DPool(pretrained)
    raise ValueError(f"unknown backbone {name!r}; choose from {ALL_BACKBONES}")


def _torchvision_3d(name: str, pretrained: bool) -> nn.Module:
    import torchvision.models.video as tvv

    weights = {"r2plus1d_18": tvv.R2Plus1D_18_Weights.KINETICS400_V1,
               "r3d_18": tvv.R3D_18_Weights.KINETICS400_V1,
               "mc3_18": tvv.MC3_18_Weights.KINETICS400_V1}[name] if pretrained else None
    model = getattr(tvv, name)(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return Squeeze(model)


def _x3d(name: str, pretrained: bool) -> nn.Module:
    try:
        model = torch.hub.load("facebookresearch/pytorchvideo", name,
                               pretrained=pretrained, trust_repo=True)
    except Exception as e:  # pytorchvideo missing or hub unreachable
        raise RuntimeError(
            f"{name} needs the pytorchvideo package (python<=3.11) and torch.hub "
            f"access. Install pytorchvideo==0.1.5 on the training machine, or "
            f"drop X3D per the plan's fallback. Original error: {e}") from e
    # X3D head ends in a Linear projection to 400 Kinetics classes; find and
    # replace the last Linear with a 1-output regressor.
    last_linear_parent, last_linear_name = None, None
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                last_linear_parent, last_linear_name = module, child_name
    if last_linear_parent is None:
        raise RuntimeError(f"no Linear head found in {name}")
    old = getattr(last_linear_parent, last_linear_name)
    setattr(last_linear_parent, last_linear_name, nn.Linear(old.in_features, 1))
    return UpsampleThenModel(Squeeze(model), X3D_NATIVE_RESOLUTION[name])


class Frame2DPool(nn.Module):
    """The does-temporal-modeling-matter-at-all floor: each frame through an
    ImageNet ResNet-18, mean-pool features over time, linear head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # -> (N,512,1,1)
        self.head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feats = self.features(x).flatten(1)          # (B*T, 512)
        feats = feats.reshape(b, t, -1).mean(dim=1)  # (B, 512)
        return self.head(feats).squeeze(-1)


class Squeeze(nn.Module):
    """Wraps a classifier-turned-regressor so output is (B,) not (B, 1)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.reshape(out.shape[0])
