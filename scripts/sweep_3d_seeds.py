"""Stage-3d seed-replicate sweep: trains the 8 3d_*.yaml configs (4 base
configs x 2 extra seeds, pre-registered in docs/PLAN.md) fresh with the same
driver structure as the other stage sweeps. The cost harness is skipped:
cost is a property of the architecture, so each replicate shares its base
config's row in results/costs.csv (see BASE_CONFIG_OF).

    .venv\\Scripts\\python scripts/sweep_3d_seeds.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = sys.executable
DATA = "data/EchoNet-Dynamic"
CONFIGS_DIR = ROOT / "configs"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
REGISTRY = RESULTS / "registry.csv"

# Cheapest-first, by the base config's actual measured stage-3a training
# hours (f8_p4 1.13h < f8_p1 1.19h < f16_p4 2.02h < f32_p2 3.92h), grouped
# by config then seed.
CONFIGS = [
    "3d_3a_r2p1d_f8_p4_s20260815", "3d_3a_r2p1d_f8_p4_s20260816",
    "3d_3a_r2p1d_f8_p1_s20260815", "3d_3a_r2p1d_f8_p1_s20260816",
    "3d_3a_r2p1d_f16_p4_s20260815", "3d_3a_r2p1d_f16_p4_s20260816",
    "3d_3a_r2p1d_f32_p2_s20260815", "3d_3a_r2p1d_f32_p2_s20260816",
]

# name -> its stage-3a base config, for cost-lookup and for the combined
# analysis to join a seed replicate's accuracy against the right cost row.
_BASE_RE = re.compile(r"^3d_(?P<base>.+)_s\d+$")
BASE_CONFIG_OF = {name: _BASE_RE.match(name).group("base") for name in CONFIGS}

EPOCH_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\] epoch (?P<epoch>\d+)/(?P<total>\d+) "
    r"train_mse (?P<mse>[\d.]+) val_mae (?P<mae>[\d.]+) \(best (?P<best>[\d.]+)\)$"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


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
        f"stage-3d seed replicate of {BASE_CONFIG_OF[name]}) ===\n"
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


def run_eval_no_cost(name: str, config_path: Path):
    """predict -> report only. Cost harness deliberately skipped -- see
    module docstring for why this is confirmed safe, not assumed."""
    ckpt_best = CHECKPOINTS / name / "best.pt"
    pred_csv = RESULTS / "metrics" / f"{name}.pred.csv"
    metrics_json = RESULTS / "metrics" / f"{name}.metrics.json"

    subprocess.run([PYTHON, "-m", "src.evaluate", "predict", str(config_path),
                     str(ckpt_best), "--data", DATA, "--out", str(pred_csv)],
                    cwd=ROOT, check=True)
    subprocess.run([PYTHON, "-m", "src.evaluate", "report", str(pred_csv),
                     "--out", str(metrics_json)], cwd=ROOT, check=True)


def git_commit_and_push(message: str) -> str:
    subprocess.run(["git", "add", "results", "logs"], cwd=ROOT, check=True)
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
    sweep_log = LOGS / "sweep3d_progress.log"
    sweep_alert = LOGS / "sweep3d_ALERT.txt"

    append(sweep_log, f"{now_iso()} === stage-3d seed-replicate sweep started, "
                       f"{len(CONFIGS)} configs, cost harness skipped (seed-independent, "
                       f"see module docstring), cheapest-first, no resumes ===\n")

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

        append(sweep_log, f"{now_iso()} --- starting {name} (base: {BASE_CONFIG_OF[name]}) ---\n")

        ok, hours = run_training(name, cfg_path)
        if not ok:
            append(sweep_log, f"{now_iso()} SWEEP HALTED at {name} (training failure)\n")
            sweep_alert.write_text(
                f"Sweep halted at '{name}': training failed. "
                f"See logs/{name}_ALERT.txt and logs/{name}_raw.log\n")
            status = git_commit_and_push(f"Stage 3d sweep HALTED at {name}: training failure (see logs)")
            append(sweep_log, f"{now_iso()} git: {status}\n")
            return

        try:
            run_eval_no_cost(name, cfg_path)
        except subprocess.CalledProcessError as e:
            append(sweep_log, f"{now_iso()} SWEEP HALTED at {name} (eval failure: {e})\n")
            sweep_alert.write_text(
                f"Sweep halted at '{name}': prediction or metrics step "
                f"failed after training completed successfully.\nError: {e}\n")
            status = git_commit_and_push(f"Stage 3d sweep HALTED at {name}: eval failure (see logs)")
            append(sweep_log, f"{now_iso()} git: {status}\n")
            return

        status = git_commit_and_push(f"Stage 3d: complete fresh 30-epoch seed replicate {name}")
        append(sweep_log, f"{now_iso()} {name} complete ({hours:.2f} GPU-h). git: {status}\n")

    total_h = (time.time() - total_start) / 3600
    append(sweep_log, f"{now_iso()} === STAGE 3D SWEEP COMPLETE, {total_h:.2f} GPU-h total ===\n")
    status = git_commit_and_push("Stage 3d sweep complete: all 8 seed-replicate runs")
    append(sweep_log, f"{now_iso()} final git: {status}\n")


if __name__ == "__main__":
    main()
