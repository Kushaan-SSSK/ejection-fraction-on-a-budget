"""Stage-3b X3D sweep: run after sweep_3b.py finishes and the GPU
is idle. Two phases: (1) probe the largest physical_batch that fits the GPU
(X3D at native resolution needs more activation memory than the other
backbones) and patch the 4 configs in place; (2) train each fresh (30
epochs, no resume) and run predict -> report -> cost -> commit/push, same
structure as the other stage drivers. See src/models.py and docs/PROTOCOL.md for
the X3D upsample wrapper and its caveat.

    .venv\\Scripts\\python scripts/sweep_3b_x3d.py
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = sys.executable
DATA = "data/EchoNet-Dynamic"
CONFIGS_DIR = ROOT / "configs"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
REGISTRY = RESULTS / "registry.csv"

# Cheapest-first, by latency probe scaled against known r2plus1d_18 epoch times.
CONFIGS = [
    "3b_x3ds_f16_p4",
    "3b_x3dm_f16_p4",
    "3b_x3ds_f32_p2",
    "3b_x3dm_f32_p2",
]

PHYSICAL_BATCH_CANDIDATES = [20, 10, 5, 4, 2, 1]  # divisors of effective_batch=20
MEM_SAFETY_MB = 7200  # ~88% of this card's 8192MB, matching existing configs' peaks

EPOCH_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\] epoch (?P<epoch>\d+)/(?P<total>\d+) "
    r"train_mse (?P<mse>[\d.]+) val_mae (?P<mae>[\d.]+) \(best (?P<best>[\d.]+)\)$"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Phase 1: memory probe
# ---------------------------------------------------------------------------

def probe_physical_batch(backbone: str, frames: int, probe_log: Path) -> int:
    from src.models import build_model

    device = "cuda"
    for pb in PHYSICAL_BATCH_CANDIDATES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(backbone, pretrained=True).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            clips = torch.zeros(pb, 3, frames, 112, 112, device=device)
            efs = torch.zeros(pb, device=device)
            with torch.autocast("cuda", enabled=True):
                loss = F.mse_loss(model(clips), efs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6
            del model, opt, scaler, clips, efs, loss
            torch.cuda.empty_cache()
            ok = peak <= MEM_SAFETY_MB
            append(probe_log, f"{now_iso()} {backbone} frames={frames} physical_batch={pb}: "
                              f"peak_mem={peak:.0f}MB {'OK' if ok else f'over {MEM_SAFETY_MB}MB safety limit'}\n")
            if ok:
                return pb
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            append(probe_log, f"{now_iso()} {backbone} frames={frames} physical_batch={pb}: OOM\n")
            torch.cuda.empty_cache()
    raise RuntimeError(f"no physical_batch in {PHYSICAL_BATCH_CANDIDATES} fits {backbone} "
                       f"frames={frames} under {MEM_SAFETY_MB}MB -- needs a real failure, not a guess")


def run_memory_probe():
    probe_log = LOGS / "sweep3b_x3d_probe.log"
    LOGS.mkdir(exist_ok=True)
    append(probe_log, f"{now_iso()} === X3D memory probe started (GPU must be idle) ===\n")

    used_mb, _ = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip(), None
    append(probe_log, f"{now_iso()} GPU memory in use before probe: {used_mb}MB "
                      f"(should be near-zero -- confirms no other job is running)\n")

    for name in CONFIGS:
        cfg_path = CONFIGS_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        pb = probe_physical_batch(cfg["backbone"], cfg["frames"], probe_log)
        cfg["physical_batch"] = pb
        if cfg["effective_batch"] % pb:
            raise RuntimeError(f"probed physical_batch={pb} for {name} does not divide "
                               f"effective_batch={cfg['effective_batch']}")
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        accum = cfg["effective_batch"] // pb
        append(probe_log, f"{now_iso()} {name}: physical_batch={pb} "
                          f"(grad_accum_x{accum}), written to {cfg_path}\n")

    append(probe_log, f"{now_iso()} === probe complete, all 4 configs patched ===\n")


# ---------------------------------------------------------------------------
# Phase 2: sweep (identical structure to sweep_3a.py / sweep_3b.py)
# ---------------------------------------------------------------------------

def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        mem, util = out.stdout.strip().split(",")
        return float(mem), float(util)
    except Exception:
        return None, None


class GpuSampler(threading.Thread):
    def __init__(self, csv_path: Path, interval: float = 10.0):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.interval = interval
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.buffer: list[tuple[float, float]] = []
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp_utc", "mem_used_mib", "util_pct"])

    def run(self):
        while not self._stop.is_set():
            mem, util = gpu_sample()
            if mem is not None:
                with self.lock:
                    self.buffer.append((mem, util))
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([now_iso(), mem, util])
            self._stop.wait(self.interval)

    def pop_epoch_stats(self):
        with self.lock:
            buf, self.buffer = self.buffer, []
        if not buf:
            return None, None
        mems = [b[0] for b in buf]
        utils = [b[1] for b in buf]
        return max(mems), sum(utils) / len(utils)

    def stop(self):
        self._stop.set()


def run_training(name: str, config_path: Path):
    progress_log = LOGS / f"{name}_progress.log"
    raw_log = LOGS / f"{name}_raw.log"
    alert_log = LOGS / f"{name}_ALERT.txt"
    gpu_csv = LOGS / f"{name}_gpu_samples.csv"
    LOGS.mkdir(exist_ok=True)

    progress_log.write_text(
        f"=== run '{name}' started {now_iso()} "
        f"(fresh start, full 30 epochs, no early stop, no resume; "
        f"X3D upsample caveat applies -- see docs/PROTOCOL.md) ===\n"
        f"raw training output: {raw_log}\n"
        f"gpu samples: {gpu_csv}\n\n"
    )

    sampler = GpuSampler(gpu_csv)
    sampler.start()

    cmd = [PYTHON, "-u", "-m", "src.train", str(config_path), "--data", DATA]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)

    t_epoch_start = time.time()
    t_run_start = t_epoch_start
    raw_lines: list[str] = []
    with open(raw_log, "w", encoding="utf-8") as rf:
        for line in proc.stdout:
            line = line.rstrip("\n")
            raw_lines.append(line)
            rf.write(f"[{now_iso()}] {line}\n")
            rf.flush()

            m = EPOCH_RE.match(line)
            if not m:
                continue
            wall = time.time() - t_epoch_start
            t_epoch_start = time.time()
            peak_mem, avg_util = sampler.pop_epoch_stats()
            epoch_i = int(m.group("epoch"))
            mem_s = f"{peak_mem:.0f}" if peak_mem is not None else "NA"
            util_s = f"{avg_util:.1f}" if avg_util is not None else "NA"
            append(progress_log,
                   f"{now_iso()} EPOCH {epoch_i}/{m.group('total')} "
                   f"wall_time_s={wall:.1f} train_mse={m.group('mse')} "
                   f"val_mae={m.group('mae')} best_val_mae={m.group('best')} "
                   f"peak_gpu_mem_MiB={mem_s} avg_gpu_util_pct={util_s}\n")
            if epoch_i == 1:
                total = int(m.group("total"))
                eta_h = wall * total / 3600
                append(progress_log,
                       f"{now_iso()} ESTIMATE after epoch 1: ~{wall/60:.1f} "
                       f"min/epoch -> ~{eta_h:.2f} GPU-h for this run.\n")

    ret = proc.wait()
    sampler.stop()
    elapsed_h = (time.time() - t_run_start) / 3600

    if ret != 0:
        alert_log.write_text(
            f"Training process for run '{name}' failed (exit code {ret}).\n\n"
            f"Last output lines:\n" + "\n".join(raw_lines[-40:]) + "\n\n"
            f"This was a FRESH run (no resume). checkpoints/{name}/last.pt may "
            f"hold a partial in-progress state. Per policy for this project, "
            f"do not silently resume it -- a human should look at the failure "
            f"first and decide whether to resume or restart from scratch.\n"
            f"Full output: {raw_log}\n"
        )
        append(progress_log, f"{now_iso()} FAILURE exit={ret} -- see {alert_log}\n")
        return False, elapsed_h

    append(progress_log, f"{now_iso()} training complete, {elapsed_h:.2f} GPU-h\n")
    return True, elapsed_h


def run_eval_and_cost(name: str, config_path: Path):
    ckpt_best = CHECKPOINTS / name / "best.pt"
    pred_csv = RESULTS / "metrics" / f"{name}.pred.csv"
    metrics_json = RESULTS / "metrics" / f"{name}.metrics.json"
    costs_csv = RESULTS / "costs.csv"

    subprocess.run([PYTHON, "-m", "src.evaluate", "predict", str(config_path),
                     str(ckpt_best), "--data", DATA, "--out", str(pred_csv)],
                    cwd=ROOT, check=True)
    subprocess.run([PYTHON, "-m", "src.evaluate", "report", str(pred_csv),
                     "--out", str(metrics_json)], cwd=ROOT, check=True)
    subprocess.run([PYTHON, "-m", "src.cost", str(config_path), "--data", DATA,
                     "--out", str(costs_csv)], cwd=ROOT, check=True)


def git_commit_and_push(message: str) -> str:
    subprocess.run(["git", "add", "results", "logs", "configs"], cwd=ROOT, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if diff.returncode == 0:
        return "nothing to commit"
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    try:
        push = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True,
                               text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "WARNING: git push timed out, committed locally only"
    if push.returncode != 0:
        return f"WARNING: git push failed, committed locally only: {push.stderr.strip()[:300]}"
    return "committed and pushed"


def main():
    LOGS.mkdir(exist_ok=True)
    sweep_log = LOGS / "sweep3b_x3d_progress.log"
    sweep_alert = LOGS / "sweep3b_x3d_ALERT.txt"

    append(sweep_log, f"{now_iso()} === X3D follow-up sweep starting: memory probe first ===\n")
    run_memory_probe()
    git_commit_and_push("Stage 3b X3D: memory-probed physical_batch for all 4 configs")

    append(sweep_log, f"{now_iso()} === X3D training sweep started, {len(CONFIGS)} configs, "
                       f"cheapest-first, no resumes ===\n")

    total_start = time.time()
    for name in CONFIGS:
        cfg_path = CONFIGS_DIR / f"{name}.yaml"

        if REGISTRY.exists() and any(
                row.split(",", 1)[0] == name
                for row in REGISTRY.read_text().splitlines()[1:]):
            append(sweep_log, f"{now_iso()} {name} already complete this sweep, skipping\n")
            continue

        ckpt_dir = CHECKPOINTS / name
        if ckpt_dir.exists():
            msg = (f"checkpoints/{name} already exists but this is supposed to be "
                   f"a from-scratch sweep with no resumes -- refusing to guess, halting.")
            append(sweep_log, f"{now_iso()} HALTED: {msg}\n")
            sweep_alert.write_text(msg + "\n")
            return

        append(sweep_log, f"{now_iso()} --- starting {name} ---\n")

        ok, hours = run_training(name, cfg_path)
        if not ok:
            append(sweep_log, f"{now_iso()} SWEEP HALTED at {name} (training failure)\n")
            sweep_alert.write_text(
                f"Sweep halted at '{name}': training failed. "
                f"See logs/{name}_ALERT.txt and logs/{name}_raw.log\n")
            status = git_commit_and_push(f"Stage 3b X3D sweep HALTED at {name}: training failure (see logs)")
            append(sweep_log, f"{now_iso()} git: {status}\n")
            return

        try:
            run_eval_and_cost(name, cfg_path)
        except subprocess.CalledProcessError as e:
            append(sweep_log, f"{now_iso()} SWEEP HALTED at {name} (eval/cost failure: {e})\n")
            sweep_alert.write_text(
                f"Sweep halted at '{name}': prediction, metrics, or cost-harness step "
                f"failed after training completed successfully.\nError: {e}\n")
            status = git_commit_and_push(f"Stage 3b X3D sweep HALTED at {name}: eval/cost failure (see logs)")
            append(sweep_log, f"{now_iso()} git: {status}\n")
            return

        status = git_commit_and_push(f"Stage 3b X3D: complete fresh 30-epoch run for {name}")
        append(sweep_log, f"{now_iso()} {name} complete ({hours:.2f} GPU-h). git: {status}\n")

    total_h = (time.time() - total_start) / 3600
    append(sweep_log, f"{now_iso()} === X3D SWEEP COMPLETE, {total_h:.2f} GPU-h total ===\n")
    status = git_commit_and_push("Stage 3b X3D sweep complete: all 4 configs, fresh 30-epoch runs")
    append(sweep_log, f"{now_iso()} final git: {status}\n")


if __name__ == "__main__":
    main()
