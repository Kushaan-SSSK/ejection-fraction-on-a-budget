"""EchoNet-Dynamic dataset with the project's two frozen clip protocols.

Train protocol : one random clip of `frames` frames sampled every `period`
                 raw frames, per video per epoch.
Test protocol  : ALL non-overlapping clips of the same shape, predictions
                 averaged by the caller (evaluate.py). This is a custom
                 deployment tiling rule (Invariant 3 of the plan) — it
                 differs from the EchoNet repo's test-time evaluation,
                 which uses all possible clip starts and zero padding.

Videos shorter than frames*period are padded by looping from the start.
Normalization uses train-set statistics computed once by compute_mean_std()
and stored in data/stats.json — never ImageNet's.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("TRAIN", "VAL", "TEST")


def load_video(path: str, resolution: int = 112) -> np.ndarray:
    """Decode an AVI to uint8 array of shape (T, H, W, 3), RGB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            frame = cv2.resize(frame, (resolution, resolution),
                               interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded: {path}")
    return np.stack(frames)


class EchoNetDataset(Dataset):
    """One item = (clip(s), ef).

    mode="train": clip is a float tensor (3, frames, H, W), randomly placed.
    mode="all_clips": clip is (n_clips, 3, frames, H, W), all non-overlapping
    clips starting at frame 0 (at least one clip even for short videos).
    """

    def __init__(self, root, split, frames=32, period=2, resolution=112,
                 mode="train", pad_pixels=12, cache_dir=None, stats=None,
                 file_list=None):
        assert split in SPLITS, split
        assert mode in ("train", "all_clips"), mode
        self.root = Path(root)
        self.frames = frames
        self.period = period
        self.resolution = resolution
        self.mode = mode
        self.pad_pixels = pad_pixels if mode == "train" else 0
        self.cache_dir = Path(cache_dir) if cache_dir else None

        df = pd.read_csv(file_list or self.root / "FileList.csv")
        df["FileName"] = df["FileName"].astype(str)
        # FileList.csv sometimes omits the .avi extension.
        df.loc[~df["FileName"].str.endswith(".avi"), "FileName"] += ".avi"
        df = df[df["Split"].str.upper() == split].reset_index(drop=True)
        self.names = df["FileName"].tolist()
        self.efs = df["EF"].astype(np.float32).tolist()

        if stats is None:
            stats_path = self.root / "stats.json"
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())
            else:
                # Safe fallback for smoke tests only; real runs must use
                # train-set stats (see compute_mean_std / Phase 0 checklist).
                stats = {"mean": [0.0] * 3, "std": [255.0] * 3}
        self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(3, 1, 1, 1)
        self.std = np.asarray(stats["std"], dtype=np.float32).reshape(3, 1, 1, 1)

    def __len__(self):
        return len(self.names)

    def _video(self, name: str) -> np.ndarray:
        if self.cache_dir is not None:
            npy = self.cache_dir / (name + f".{self.resolution}.npy")
            if npy.exists():
                return np.load(npy, mmap_mode="r")
            video = load_video(str(self.root / "Videos" / name), self.resolution)
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, video)
            return video
        return load_video(str(self.root / "Videos" / name), self.resolution)

    def _sample(self, video: np.ndarray, start: int) -> np.ndarray:
        span = self.frames * self.period
        idx = start + np.arange(self.frames) * self.period
        idx = idx % max(len(video), 1)  # loop-pad short videos
        clip = np.asarray(video[idx], dtype=np.float32)  # (T, H, W, 3)
        clip = clip.transpose(3, 0, 1, 2)  # (3, T, H, W)
        return (clip - self.mean) / self.std

    def __getitem__(self, i):
        video = self._video(self.names[i])
        ef = self.efs[i]
        span = self.frames * self.period

        if self.mode == "train":
            max_start = max(len(video) - span, 0)
            start = np.random.randint(0, max_start + 1)
            clip = self._sample(video, start)
            if self.pad_pixels:
                clip = _random_shift(clip, self.pad_pixels)
            return torch.from_numpy(np.ascontiguousarray(clip)), ef

        n_clips = max(len(video) // span, 1)
        clips = np.stack([self._sample(video, k * span) for k in range(n_clips)])
        return torch.from_numpy(np.ascontiguousarray(clips)), ef

    def clips_per_video(self) -> float:
        """Mean number of non-overlapping clips per video in this split —
        feeds the per-video cost columns in the harness."""
        span = self.frames * self.period
        counts = []
        for name in self.names:
            counts.append(max(self._n_frames(name) // span, 1))
        return float(np.mean(counts))

    def _n_frames(self, name: str) -> int:
        if self.cache_dir is not None:
            npy = self.cache_dir / (name + f".{self.resolution}.npy")
            if npy.exists():
                return np.load(npy, mmap_mode="r").shape[0]
        cap = cv2.VideoCapture(str(self.root / "Videos" / name))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 1)


def _random_shift(clip: np.ndarray, pad: int) -> np.ndarray:
    """EchoNet's spatial augmentation: zero-pad `pad` px on each side, take a
    random crop of the original size (a random shift of up to ±pad)."""
    c, t, h, w = clip.shape
    padded = np.zeros((c, t, h + 2 * pad, w + 2 * pad), dtype=clip.dtype)
    padded[:, :, pad:pad + h, pad:pad + w] = clip
    y = np.random.randint(0, 2 * pad + 1)
    x = np.random.randint(0, 2 * pad + 1)
    return padded[:, :, y:y + h, x:x + w]


def compute_mean_std(root, resolution=112, max_videos=None, seed=0):
    """Train-set per-channel mean/std on the 0–255 scale. Phase 0, run once:
        python -m src.data <dataset_root>
    Writes <root>/stats.json used by every subsequent run."""
    root = Path(root)
    df = pd.read_csv(root / "FileList.csv")
    df["FileName"] = df["FileName"].astype(str)
    df.loc[~df["FileName"].str.endswith(".avi"), "FileName"] += ".avi"
    names = df[df["Split"].str.upper() == "TRAIN"]["FileName"].tolist()
    if max_videos:
        rng = np.random.default_rng(seed)
        names = list(rng.choice(names, size=min(max_videos, len(names)),
                                replace=False))
    n, s1, s2 = 0, np.zeros(3), np.zeros(3)
    for name in names:
        v = load_video(str(root / "Videos" / name), resolution).astype(np.float64)
        pixels = v.reshape(-1, 3)
        n += pixels.shape[0]
        s1 += pixels.sum(0)
        s2 += (pixels ** 2).sum(0)
    mean = s1 / n
    std = np.sqrt(s2 / n - mean ** 2)
    stats = {"mean": mean.tolist(), "std": std.tolist(),
             "n_videos": len(names), "resolution": resolution}
    (root / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    import sys
    print(compute_mean_std(sys.argv[1],
                           max_videos=int(sys.argv[2]) if len(sys.argv) > 2 else None))
