"""Bounded, monitored child jobs; never attach to or kill unrelated processes."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import time

import psutil

from .runtime import atomic_json, now


@dataclass(frozen=True)
class Limits:
    wall_seconds: float
    memory_bytes: int
    cpu_threads: int
    poll_seconds: float = .1

    def __post_init__(self):
        if (isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float))
                or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0
                or type(self.memory_bytes) is not int or self.memory_bytes <= 0
                or type(self.cpu_threads) is not int or self.cpu_threads < 1):
            raise ValueError("job limits must be positive")
        if isinstance(self.poll_seconds, bool) or not isinstance(self.poll_seconds, (int, float)) or not 0 < self.poll_seconds <= 1:
            raise ValueError("poll interval must be in (0, 1]")


def _terminate_owned(process: subprocess.Popen, identity: float) -> None:
    try:
        parent = psutil.Process(process.pid)
        if abs(parent.create_time() - identity) > .01:
            raise RuntimeError("PID identity changed; refusing termination")
        owned = parent.children(recursive=True) + [parent]
        for proc in reversed(owned):
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=2)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=5)
    except psutil.NoSuchProcess:
        process.wait(timeout=5)


def run_job(command: list[str], cwd: Path, directory: Path, limits: Limits) -> dict:
    """A memory watchdog is sampled, not an OS-level instantaneous allocation cap.

    Callers register the scientific trial before invoking this runner. A successful
    exit provides infrastructure evidence only, never a scientific acceptance.
    """
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(limits.cpu_threads)
    env["CUDA_VISIBLE_DEVICES"] = ""
    start = time.monotonic()
    peak = 0
    cpu_seconds = 0.0
    reason = None
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        identity = psutil.Process(child.pid).create_time()
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    proc = psutil.Process(child.pid)
                    processes = [proc] + proc.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    peak = max(peak, rss)
                    measured_cpu = sum(p.cpu_times().user + p.cpu_times().system for p in processes if p.is_running())
                    cpu_seconds = max(cpu_seconds, measured_cpu)
                except psutil.NoSuchProcess:
                    rss = 0
                atomic_json(directory / "heartbeat.json", {"at": now(), "pid": child.pid,
                            "process_created": identity, "wall_seconds": elapsed, "peak_rss_bytes": peak})
                if elapsed > limits.wall_seconds:
                    reason = "wall_limit"
                elif rss > limits.memory_bytes:
                    reason = "memory_limit"
                if reason:
                    _terminate_owned(child, identity)
                    break
                time.sleep(limits.poll_seconds)
        except BaseException:
            _terminate_owned(child, identity)
            raise
    result = {"command": command, "pid": child.pid, "process_created": identity,
              "exit_code": child.returncode, "termination_reason": reason,
              "wall_seconds": time.monotonic() - start, "peak_rss_bytes": peak,
              "sampled_cpu_seconds": cpu_seconds, "limits": limits.__dict__,
              "memory_enforcement": "sampled process-tree watchdog", "finished_at": now()}
    atomic_json(directory / "process_result.json", result)
    return result
