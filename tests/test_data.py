import numpy as np
import torch

from src.data import EchoNetDataset


def test_train_mode_shapes_and_padding(fake_root):
    ds = EchoNetDataset(fake_root, "TRAIN", frames=16, period=2, mode="train")
    assert len(ds) == 3
    for i in range(len(ds)):  # includes the 25-frame video (needs loop-pad)
        clip, ef = ds[i]
        assert clip.shape == (3, 16, 112, 112)
        assert clip.dtype == torch.float32
        assert 0 < ef < 100


def test_all_clips_mode(fake_root):
    ds = EchoNetDataset(fake_root, "TEST", frames=8, period=2, mode="all_clips")
    clips, _ = ds[ds.names.index("f.avi")]  # 48 frames / span 16 -> 3 clips
    assert clips.shape == (3, 3, 8, 112, 112)
    clips, _ = ds[ds.names.index("e.avi")]  # 6 frames, shorter than span -> 1
    assert clips.shape[0] == 1


def test_clips_per_video_matches_manual(fake_root):
    ds = EchoNetDataset(fake_root, "TEST", frames=8, period=2, mode="all_clips")
    assert ds.clips_per_video() == np.mean([1, 3])


def test_cache_roundtrip(fake_root, tmp_path):
    ds = EchoNetDataset(fake_root, "TRAIN", frames=8, period=1, mode="train",
                        cache_dir=tmp_path)
    a, _ = ds[0]
    assert (tmp_path / "a.avi.112.npy").exists()
    b, _ = EchoNetDataset(fake_root, "TRAIN", frames=8, period=1,
                          mode="train", cache_dir=tmp_path)[0]
    assert a.shape == b.shape
