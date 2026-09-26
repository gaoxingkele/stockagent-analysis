"""Pre-execution source snapshot for the S20 v4 harness (2026-09-24 decision).

Captures every dirty (untracked or modified) research source into a
content-addressed snapshot WITHOUT altering the working tree: no stash,
no reset, no commit. Protected pool configs are copied read-only.

Scope:
  1. Frozen H00 scope, produced by ``Runtime._snapshot`` (research/, src/,
     scripts/, tests/, config/s20*, docs/research/s20*, wiki/smartgoal/,
     CHECKPOINT_S20*, requirements*) -- kept byte-identical to what a
     run-registry snapshot would produce.
  2. Extension: every remaining dirty file the frozen scope does not blob
     (wiki root entries, wiki/README.md, protected config/pool_e*.json),
     recorded in ``source_snapshot_extension.json`` with the same
     content-addressed scheme.

Usage:
    python scripts/snapshot_s20_pre_execution.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.s20_harness.runtime import Runtime, atomic_json, now  # noqa: E402

PLAN = ROOT / "config" / "s20_v4_harness_plan.json"
PROTECTED = ("config/pool_e.json", "config/pool_e_meta.json")
SNAPSHOT_ROOT = ROOT / "output" / "experiments" / "s20_safe_v4" / "pre_execution_snapshots"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def dirty_paths() -> list[str]:
    """Every file that differs from HEAD or is untracked (file-level, not collapsed dirs)."""
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    modified = git("diff", "--name-only", "-z", "HEAD").split("\0")
    return sorted({name for name in untracked + modified if name})


def store_blob(blobs: Path, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    blob = blobs / sha
    if not blob.exists():
        with blob.open("xb") as stream:
            stream.write(data)
    return sha


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = SNAPSHOT_ROOT / stamp
    target.mkdir(parents=True)

    runtime = Runtime(ROOT, PLAN)
    # Frozen H00 semantics: blobs + source_snapshot.json + environment.json +
    # protocol.json + exposure_ledger.json + budget.json. Raises if a protected
    # path changes mid-snapshot.
    runtime._snapshot(target)

    snapshot = json.loads((target / "source_snapshot.json").read_text(encoding="utf-8"))
    covered = {record["path"] for record in snapshot["files"]}
    blobs = target / "source_blobs"

    extra = []
    for name in dirty_paths():
        if name in covered:
            continue
        path = ROOT / name
        if not path.is_file():
            continue  # deletions have no content to preserve
        data = path.read_bytes()
        extra.append({
            "path": name,
            "sha256": store_blob(blobs, data),
            "bytes": len(data),
            "protected": name in PROTECTED,
        })

    atomic_json(target / "source_snapshot_extension.json", {
        "at": now(),
        "reason": (
            "2026-09-24 decision (wiki/smartgoal/2026-09-24_s20_harness_next_step_decision.md): "
            "pre-execution snapshot of every dirty file before P0 work. The frozen H00 scope does "
            "not blob wiki root entries, wiki/README.md, or the protected pool configs; they are "
            "preserved here read-only. Working tree untouched: no stash, no reset, no commit."
        ),
        "files": extra,
        "protected_hashes": snapshot["protected_hashes"],
    })

    print(json.dumps({
        "target": str(target.relative_to(ROOT)),
        "frozen_scope_files": len(snapshot["files"]),
        "extension_files": [record["path"] for record in extra],
        "protected_hashes": snapshot["protected_hashes"],
        "git_status_at_snapshot": snapshot["git_status"].count("\n") + 1,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
