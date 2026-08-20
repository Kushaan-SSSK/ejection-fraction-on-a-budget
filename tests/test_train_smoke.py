"""End-to-end: 1-epoch train on the synthetic dataset (2D floor backbone for
CPU speed), then predict + report through the real pipeline."""

import json

import pandas as pd
import yaml

from src.evaluate import predict, report
from src.train import train


def test_train_predict_report_roundtrip(fake_root, tmp_path):
    cfg = dict(backbone="r2d_18_pool", pretrained=False, frames=4, period=1,
               resolution=112, effective_batch=2, physical_batch=1,
               epochs=1, num_workers=0, amp=False)
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    registry = tmp_path / "registry.csv"
    best_val = train(cfg_path, fake_root, ckpt_root=tmp_path / "ckpt",
                     registry=registry)
    assert best_val > 0
    reg = pd.read_csv(registry)
    assert reg.loc[0, "run_id"] == "smoke"
    assert reg.loc[0, "deviations"] == "grad_accum_x2"

    pred_csv = tmp_path / "pred.csv"
    df = predict(cfg_path, tmp_path / "ckpt" / "smoke" / "best.pt",
                 fake_root, pred_csv)
    assert len(df) == 2  # two TEST videos
    assert {"file", "ef_true", "ef_pred", "n_clips"} <= set(df.columns)

    out_json = tmp_path / "metrics.json"
    m = report(pred_csv, out_json, n_boot=50)
    assert "mae" in m and "sens_at_40" in m
    assert json.loads(out_json.read_text())["n"] == 2


def test_resume_from_checkpoint(fake_root, tmp_path):
    cfg = dict(backbone="r2d_18_pool", pretrained=False, frames=4, period=1,
               resolution=112, effective_batch=1, physical_batch=1,
               epochs=2, num_workers=0, amp=False)
    cfg_path = tmp_path / "resume.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    kw = dict(ckpt_root=tmp_path / "ckpt", registry=tmp_path / "reg.csv")
    train(cfg_path, fake_root, max_epochs=1, **kw)   # epoch 0 only
    train(cfg_path, fake_root, **kw)                 # resumes at epoch 1
    reg = pd.read_csv(tmp_path / "reg.csv")
    assert len(reg) == 2  # both invocations logged
