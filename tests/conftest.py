"""Synthetic mini-EchoNet: a few tiny AVIs + FileList.csv, so the whole
pipeline is testable without the gated dataset."""

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake_echonet")
    (root / "Videos").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    specs = [  # (name, n_frames, split) — includes one too-short video
        ("a.avi", 40, "TRAIN"), ("b.avi", 60, "TRAIN"), ("c.avi", 25, "TRAIN"),
        ("d.avi", 50, "VAL"), ("e.avi", 6, "TEST"), ("f.avi", 48, "TEST"),
    ]
    for name, n, split in specs:
        path = str(root / "Videos" / name)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 50,
                             (112, 112))
        for _ in range(n):
            vw.write(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8))
        vw.release()
        rows.append({"FileName": name.removesuffix(".avi"),
                     "EF": float(rng.uniform(20, 75)), "Split": split})
    pd.DataFrame(rows).to_csv(root / "FileList.csv", index=False)
    return root
