"""Evidence-backed stage runtime. Production paths are never output targets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import psutil

from .errors import DataValidityError

from .contracts import load_plan, validate_plan
from .pipeline import WAITING_STATES


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def atomic_json(path: Path, value) -> None:
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class Runtime:
    def __init__(self, root: Path, plan_path: Path):
        self.root = root.resolve()
        self.plan_path = plan_path.resolve()
        self.plan = load_plan(self.plan_path)
        result = validate_plan(self.plan)
        if not result["valid"]:
            raise ValueError(result["errors"])
        self.protocol_hash = hashlib.sha256(canonical(self.plan)).hexdigest()
        self.base = self.root / "output/experiments/s20_safe_v4" / self.protocol_hash
        # Refuse a redirected output tree, even with unrestricted filesystem access.
        if not self.base.resolve().is_relative_to(self.root):
            raise ValueError("research output escapes repository")
        if os.name == "nt" and not str(self.base).startswith("\\\\?\\"):
            absolute = str(self.base)
            self.base = Path("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                             else "\\\\?\\" + absolute)
        self.stages = {s["id"]: s for s in self.plan["stages"]}

    def connect(self):
        self.base.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.base / "registry.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, stage TEXT NOT NULL, state TEXT NOT NULL,
              started_at TEXT NOT NULL, finished_at TEXT, pid INTEGER NOT NULL,
              process_created REAL NOT NULL, error TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_run ON runs((1))
              WHERE state IN ('RUNNING', 'VERIFYING');
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              at TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
              BEGIN SELECT RAISE(ABORT, 'events append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
              BEGIN SELECT RAISE(ABORT, 'events append-only'); END;
        """)
        return db

    def _event(self, db, run_id, state, payload):
        db.execute("INSERT INTO events(run_id,at,state,payload) VALUES(?,?,?,?)",
                   (run_id, now(), state, json.dumps(payload, sort_keys=True)))

    def _insert_run(self, db, run_id: str, stage: str, track: str):
        """Insert a run row without depending on an optional ``track`` column.

        The canonical registry schema carries eight columns and the track lives in
        the manifest. A registry that already has the optional column is populated
        rather than guessed at, so both shapes stay readable.
        """
        values = (run_id, stage, "RUNNING", now(), None, os.getpid(),
                  psutil.Process().create_time(), None)
        has_track = "track" in {row[1] for row in db.execute("PRAGMA table_info(runs)")}
        if has_track:
            db.execute("INSERT INTO runs(run_id,stage,state,started_at,finished_at,pid,"
                       "process_created,error,track) VALUES(?,?,?,?,?,?,?,?,?)",
                       (*values, track))
        else:
            db.execute("INSERT INTO runs(run_id,stage,state,started_at,finished_at,pid,"
                       "process_created,error) VALUES(?,?,?,?,?,?,?,?)", values)

    def verify(self, run_id: str, require_completion: bool = True, _seen=None) -> dict:
        if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("invalid run id")
        directory = self.base / run_id
        seen = set() if _seen is None else set(_seen)
        if run_id in seen:
            return {"valid": False, "errors": ["cyclic upstream run binding"]}
        seen.add(run_id)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return {"valid": False, "errors": ["manifest missing"]}
        manifest = load_plan(manifest_path)
        errors = []
        if manifest.get("run_id") != run_id:
            errors.append("manifest run identity mismatch")
        if manifest.get("protocol_hash") != self.protocol_hash:
            errors.append("protocol hash mismatch")
        stage = manifest.get("stage")
        if stage not in self.stages:
            errors.append("unknown manifest stage")
        upstream = manifest.get("upstream_runs", {})
        dependencies = self.stages.get(stage, {}).get("depends_on", [])
        track = manifest.get("track", "formal")
        if track not in ("formal", "diagnostic"):
            errors.append("unknown execution track")
        if not isinstance(upstream, dict) or set(upstream) != set(dependencies):
            errors.append("missing or unexpected upstream run bindings")
        else:
            for dependency, binding in upstream.items():
                try:
                    parent_id = binding["run_id"]
                    parent_path = self.base / parent_id / "manifest.json"
                    parent_manifest = load_plan(parent_path)
                    # A diagnostic run may sit on a settled blocker; that blocker is
                    # carried forward instead of passing as a verified prerequisite.
                    need_completion = track != "diagnostic"
                    parent_check = self.verify(parent_id, require_completion=need_completion, _seen=seen)
                    allowed = ("COMPLETED",) if need_completion else ("COMPLETED", "FAILED_VALIDITY", *WAITING_STATES)
                    if (not parent_check["valid"] or parent_manifest.get("terminal_state") not in allowed or
                            digest(parent_path) != binding["manifest_sha256"] or
                            parent_manifest.get("stage") != dependency):
                        errors.append("unverified upstream binding: " + dependency)
                except (ValueError, OSError, KeyError, TypeError):
                    errors.append("unreadable upstream binding: " + dependency)
        artifacts = manifest.get("artifact_hashes", {})
        # A registered gap declares no outputs on purpose and can never count as
        # completion, so it must not be reported as a missing-artifact defect.
        if manifest.get("terminal_state") not in WAITING_STATES:
            for name in self.stages.get(stage, {}).get("outputs", []):
                if name not in artifacts:
                    errors.append("required artifact missing: " + name)
        for name, expected in artifacts.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()):
                errors.append("unsafe artifact path")
            elif not path.is_file() or digest(path) != expected:
                errors.append("artifact mismatch: " + name)
        # Hashing a file-list alone does not prove the source bytes are recoverable.
        snapshot_path = directory / "source_snapshot.json"
        if snapshot_path.is_file():
            snapshot = load_plan(snapshot_path)
            for item in snapshot.get("files", []):
                blob = directory / "source_blobs" / item["sha256"]
                if not blob.is_file() or digest(blob) != item["sha256"]:
                    errors.append("source blob mismatch: " + item["path"])
        if require_completion and manifest.get("terminal_state") != "COMPLETED":
            errors.append("stage not completed")
        if require_completion:
            database = self.base / "registry.sqlite"
            if not database.exists():
                errors.append("completion registry missing")
            else:
                with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
                    row = db.execute("SELECT stage,state FROM runs WHERE run_id=?", (run_id,)).fetchone()
                    event = db.execute("SELECT state,payload FROM events WHERE run_id=? ORDER BY seq DESC LIMIT 1",
                                       (run_id,)).fetchone()
                if row != (stage, "COMPLETED") or not event or event[0] != "COMPLETED":
                    errors.append("completion registry/event mismatch")
                elif json.loads(event[1]).get("manifest_hash") != digest(manifest_path):
                    errors.append("completion event manifest hash mismatch")
        return {"valid": not errors, "errors": errors}

    def status(self) -> dict:
        if not (self.base / "registry.sqlite").exists():
            return {"protocol_hash": self.protocol_hash, "runs": [],
                    "stages": {s: "PLANNED" for s in self.stages}}
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY started_at")]
        state = {s: "PLANNED" for s in self.stages}
        for row in rows:
            if row["state"] in ("RUNNING", "VERIFYING"):
                try:
                    proc = psutil.Process(row["pid"])
                    row["process_live"] = abs(proc.create_time() - row["process_created"]) < .01
                except psutil.NoSuchProcess:
                    row["process_live"] = False
                except psutil.AccessDenied:
                    row["process_live"] = None
            if row["state"] == "COMPLETED":
                row["verification"] = self.verify(row["run_id"])
                state[row["stage"]] = "COMPLETED" if row["verification"]["valid"] else "FAILED_VALIDITY"
            else:
                state[row["stage"]] = row["state"]
        return {"protocol_hash": self.protocol_hash, "runs": rows, "stages": state}

    def run(self, stage: str, *, track: str = "formal", mode: str | None = None) -> dict:
        """Execute one stage.

        ``track`` selects the continuation policy. The frozen ``formal`` track runs
        a stage only when every prerequisite is COMPLETED and is the only track
        that can record formal acceptance. The ``diagnostic`` track may continue
        past settled blockers, inherits them into the manifest, and can never
        record formal acceptance.
        """
        from . import pipeline
        if track not in pipeline.TRACKS:
            raise ValueError("unknown track " + str(track))
        if stage not in self.stages:
            raise ValueError("unknown stage " + stage)
        state = self.status()
        allow_blockers = track == "diagnostic"
        settled = pipeline.SETTLED if allow_blockers else ("COMPLETED",)
        missing = [s for s in self.stages[stage]["depends_on"] if state["stages"][s] not in settled]
        if missing:
            raise ValueError("unverified prerequisites: " + ", ".join(missing))
        if state["stages"][stage] == "COMPLETED":
            raise ValueError("stage already verified; do not silently rerun frozen work")
        route = pipeline.stage_route(stage)
        if route.kind in ("unimplemented", "prospective"):
            raise NotImplementedError(stage + " executor not implemented; register the gap instead")
        chosen = mode or route.default_mode
        if chosen not in ("compute", "intake"):
            raise NotImplementedError(stage + " has no " + str(chosen) + " executor; no completion recorded")
        upstream = {}
        upstream_directories = {}
        for dependency in self.stages[stage]["depends_on"]:
            parent = next(r for r in reversed(state["runs"]) if r["stage"] == dependency)
            parent_id = parent["run_id"]
            if not self.verify(parent_id, require_completion=not allow_blockers)["valid"]:
                raise ValueError("upstream changed before launch: " + dependency)
            upstream[dependency] = {"run_id": parent_id,
                                    "manifest_sha256": digest(self.base / parent_id / "manifest.json")}
            upstream_directories[dependency] = self.base / parent_id
        blockers = {name: state["stages"][name] for name in upstream
                    if state["stages"][name] != "COMPLETED"}
        run_id = stage + "-" + uuid.uuid4().hex
        with self.connect() as db:
            self._insert_run(db, run_id, stage, track)
            self._event(db, run_id, "RUNNING", {"protocol_hash": self.protocol_hash, "upstream_runs": upstream})
        directory = self.base / run_id
        directory.mkdir()
        try:
            self._snapshot(directory)
            if chosen == "compute" and stage in ("H02", "H03", "H04", "H05",
                                                 "H06", "H07", "H08", "H10"):
                terminal, gaps = self._run_compute(stage, directory, upstream_directories)
            elif stage in ("H01", "H02", "H03", "H05"):
                try:
                    if stage == 'H01':
                        from .full_data_audit import audit_dataset
                        audit = audit_dataset(self.root, directory)
                    elif stage == 'H02':
                        from .h02_intake import audit as intake
                        audit = intake(self.root, directory, self.protocol_hash)
                    elif stage == 'H03':
                        from .h03_intake import audit as intake
                        audit = intake(self.root, directory, self.protocol_hash)
                    else:
                        from .h05_intake import audit as intake
                        audit = intake(self.root, directory, self.protocol_hash)
                except (ValueError, AssertionError) as exc:
                    raise DataValidityError(str(exc)) from exc
                terminal = "COMPLETED" if audit["formal_gate_passed"] else "FAILED_VALIDITY"
                gaps = audit["acceptance_gaps"]
            else:
                terminal, gaps = self._resource_check(directory)
            hashes = {name: digest(directory / name) for name in self.stages[stage]["outputs"]}
            for path in directory.rglob("*"):
                if path.is_file() and "source_blobs" not in path.parts:
                    hashes[path.relative_to(directory).as_posix()] = digest(path)
            manifest = {"run_id": run_id, "stage": stage, "protocol_hash": self.protocol_hash,
                        "upstream_runs": upstream,
                        "track": track, "executor_mode": chosen,
                        "inherited_blockers": blockers,
                        "formal_accepted": bool(track == "formal" and terminal == "COMPLETED"),
                        "terminal_state": terminal, "artifact_hashes": hashes,
                        "acceptance_gaps": gaps,
                        "science_state": "INSUFFICIENT_EVIDENCE", "finished_at": now()}
            atomic_json(directory / "manifest.json", manifest)
            check = self.verify(run_id, require_completion=False)
            if not check["valid"]:
                raise ValueError(check["errors"])
            with self.connect() as db:
                db.execute("UPDATE runs SET state='VERIFYING' WHERE run_id=?", (run_id,))
                self._event(db, run_id, "VERIFYING", {
                    "manifest_hash": digest(directory / "manifest.json")})
            with self.connect() as db:
                db.execute("UPDATE runs SET state=?,finished_at=? WHERE run_id=?", (terminal, now(), run_id))
                self._event(db, run_id, terminal, {"manifest_hash": digest(directory / "manifest.json"),
                                                 "acceptance_gaps": manifest["acceptance_gaps"]})
            return manifest
        except Exception as exc:
            failure_state = "FAILED_VALIDITY" if isinstance(exc, DataValidityError) else "FAILED_INFRA"
            failure = {"run_id": run_id, "stage": stage, "at": now(),
                       "protocol_hash": self.protocol_hash, "upstream_runs": upstream,
                       "terminal_state": failure_state, "exception_type": type(exc).__name__,
                       "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
                       "error": str(exc), "formal_stage_completed": False,
                       "automatic_retry_authorized": False}
            # This is failure evidence, not a substitute for required stage outputs.
            failure_hash = None
            try:
                atomic_json(directory / "failure.json", failure)
                failure_hash = digest(directory / "failure.json")
            except OSError:
                # Disk failures must not hide the original error or leave a live run.
                pass
            with self.connect() as db:
                db.execute("UPDATE runs SET state=?,finished_at=?,error=? WHERE run_id=?",
                           (failure_state, now(), str(exc), run_id))
                self._event(db, run_id, failure_state, {"error": str(exc),
                    "exception_type": type(exc).__name__, "failure_sha256": failure_hash,
                    "automatic_retry_authorized": False})
            raise

    def _resource_check(self, directory):
            from .process_runner import Limits, run_job
            budget = load_plan(directory / "budget.json")
            probe_limit = Limits(120, min(budget["memory_limit_bytes"], 2 * 1024**3), 1)
            # Module execution avoids local select.py shadowing the stdlib select.
            probe = run_job([sys.executable, "-c",
                            "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
                            "runpy.run_module('research.s20_harness.resource_probe',run_name='__main__')",
                            str(Path(__file__).resolve().parents[2])],
                            self.root, directory / "resource_probe", probe_limit)
            probe_ok = probe["exit_code"] == 0 and probe["termination_reason"] is None
            budget["resource_probe"] = probe
            budget["training_wall_limit_seconds"] = 3600
            budget["training_wall_limit_basis"] = "conservative policy cap, not a throughput forecast"
            budget["training_authorized"] = False
            budget["required_before_training"] = "H01-H02 data gates and real-data pilot before large fits"
            budget["runtime_enforcement"] = "owned child timeout and sampled aggregate RSS watchdog"
            atomic_json(directory / "budget.json", budget)
            terminal = "COMPLETED" if probe_ok else "FAILED_INFRA"
            return terminal, [] if probe_ok else ["bounded resource probe failed; inspect logs"]

    def _run_compute(self, stage: str, directory: Path, upstream_directories: dict) -> tuple[str, list]:
        """Dispatch a stage to its locally available compute executor.

        The frozen DAG declares only direct dependencies, but a compute executor
        may legitimately need an earlier ancestor (H04 needs H02 labels). The
        lineage is resolved from the recorded manifests rather than by widening
        the frozen dependency list.
        """
        lineage = self._lineage(upstream_directories)
        if stage == "H02":
            from .abcds_labels import build as build_labels
            result = build_labels(self.root, directory)
        elif stage == "H03":
            from .h03_baseline import build as build_baseline
            result = build_baseline(self.root, directory, upstream_directories=lineage)
        elif stage == "H04":
            from .h04_competition import build as build_competition
            result = build_competition(self.root, directory, upstream_directories=lineage)
        elif stage == "H05":
            from .h05_abcds import build as build_calibration
            result = build_calibration(self.root, directory, upstream_directories=lineage)
        elif stage == "H06":
            from .h06_ablation import build as build_ablation
            result = build_ablation(self.root, directory, upstream_directories=lineage)
        elif stage == "H07":
            from .h07_execution import build as build_execution
            result = build_execution(self.root, directory, upstream_directories=lineage)
        elif stage == "H08":
            from .h08_freeze import build as build_freeze
            result = build_freeze(self.root, directory, upstream_directories=lineage)
        elif stage == "H10":
            from .h10_review import build as build_review
            result = build_review(self.root, directory, upstream_directories=lineage)
        else:
            raise NotImplementedError(stage + " has no registered compute executor")
        atomic_json(directory / "executor_result.json", result)
        terminal = "COMPLETED" if result.get("formal_gate_passed") else "FAILED_VALIDITY"
        return terminal, list(result.get("acceptance_gaps") or [])

    def _lineage(self, upstream_directories: dict) -> dict:
        """Map every settled stage to its most recent run directory.

        A stage appears several times in the registry history: an early pass may
        have recorded a gap before the executor existed, and a later pass records
        the real run. Walking only the direct parent's bindings would resolve to
        that earlier gap, so the authoritative run per stage is taken the same way
        :meth:`run` picks its own parents -- latest settled run, in registry order.
        """
        lineage = dict(upstream_directories)
        settled = ("COMPLETED", "FAILED_VALIDITY", *WAITING_STATES)
        for row in self.status()["runs"]:
            if row["state"] not in settled:
                continue
            directory = self.base / row["run_id"]
            if (directory / "manifest.json").is_file():
                lineage[row["stage"]] = directory
        return lineage

    def record_gap(self, stage: str, *, track: str = "diagnostic", kind: str | None = None,
                   reason: str | None = None) -> dict:
        """Register a stage that has no executable route yet.

        This is deliberately not a completion. It records *why* the chain cannot
        advance, binds the settled prerequisites, and leaves the stage in a
        WAITING_* state that can never satisfy a formal gate.
        """
        from . import pipeline
        if track not in pipeline.TRACKS:
            raise ValueError("unknown track " + str(track))
        if stage not in self.stages:
            raise ValueError("unknown stage " + stage)
        route = pipeline.stage_route(stage)
        state = self.status()
        allow_blockers = track == "diagnostic"
        settled = pipeline.SETTLED if allow_blockers else ("COMPLETED",)
        missing = [s for s in self.stages[stage]["depends_on"] if state["stages"][s] not in settled]
        if missing:
            raise ValueError("unverified prerequisites: " + ", ".join(missing))
        if state["stages"][stage] in ("COMPLETED", *WAITING_STATES):
            raise ValueError("stage already settled; do not silently re-record " + stage)
        terminal = "WAITING_MATURITY" if route.kind == "prospective" else "WAITING_IMPLEMENTATION"
        upstream = {}
        for dependency in self.stages[stage]["depends_on"]:
            parent = next(r for r in reversed(state["runs"]) if r["stage"] == dependency)
            parent_id = parent["run_id"]
            if not self.verify(parent_id, require_completion=not allow_blockers)["valid"]:
                raise ValueError("upstream changed before launch: " + dependency)
            upstream[dependency] = {"run_id": parent_id,
                                    "manifest_sha256": digest(self.base / parent_id / "manifest.json")}
        blockers = {name: state["stages"][name] for name in upstream
                    if state["stages"][name] != "COMPLETED"}
        run_id = stage + "-" + uuid.uuid4().hex
        with self.connect() as db:
            self._insert_run(db, run_id, stage, track)
            self._event(db, run_id, "RUNNING", {"protocol_hash": self.protocol_hash,
                                                "upstream_runs": upstream, "gap": route.kind})
        directory = self.base / run_id
        directory.mkdir()
        try:
            self._snapshot(directory)
            gap = {"at": now(), "stage": stage, "track": track, "terminal_state": terminal,
                   "executor_kind": kind or route.kind, "declared_module": route.module,
                   "declared_outputs": list(self.stages[stage]["outputs"]),
                   "reason": reason or route.note,
                   "declared_route_note": route.note,
                   "required_before_this_stage": list(route.requires),
                   "inherited_blockers": blockers,
                   "formal_accepted": False, "completion_claimed": False}
            atomic_json(directory / "gap.json", gap)
            hashes = {}
            for path in directory.rglob("*"):
                if path.is_file() and "source_blobs" not in path.parts:
                    hashes[path.relative_to(directory).as_posix()] = digest(path)
            manifest = {"run_id": run_id, "stage": stage, "protocol_hash": self.protocol_hash,
                        "upstream_runs": upstream, "track": track,
                        "executor_mode": "gap", "inherited_blockers": blockers,
                        "formal_accepted": False, "terminal_state": terminal,
                        "artifact_hashes": hashes,
                        "acceptance_gaps": [reason or route.note] if (reason or route.note) else [],
                        "science_state": "INSUFFICIENT_EVIDENCE", "finished_at": now()}
            atomic_json(directory / "manifest.json", manifest)
            check = self.verify(run_id, require_completion=False)
            if not check["valid"]:
                raise ValueError(check["errors"])
            with self.connect() as db:
                db.execute("UPDATE runs SET state='VERIFYING' WHERE run_id=?", (run_id,))
                self._event(db, run_id, "VERIFYING", {"manifest_hash": digest(directory / "manifest.json")})
            with self.connect() as db:
                db.execute("UPDATE runs SET state=?,finished_at=? WHERE run_id=?", (terminal, now(), run_id))
                self._event(db, run_id, terminal, {"manifest_hash": digest(directory / "manifest.json"),
                                                  "acceptance_gaps": manifest["acceptance_gaps"]})
            return manifest
        except Exception as exc:
            with self.connect() as db:
                db.execute("UPDATE runs SET state=?,finished_at=?,error=? WHERE run_id=?",
                           ("FAILED_INFRA", now(), str(exc), run_id))
                self._event(db, run_id, "FAILED_INFRA", {"error": str(exc)})
            raise

    def _snapshot(self, directory):
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.root)

        protected = ["config/pool_e.json", "config/pool_e_meta.json"]
        before = {p: digest(self.root / p) for p in protected if (self.root / p).is_file()}
        paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
        suffixes = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
        prefixes = ("research/", "src/", "scripts/", "tests/", "config/s20", "docs/research/s20", "wiki/smartgoal/")
        blobs = directory / "source_blobs"
        blobs.mkdir()
        records = []
        for name in sorted(set(paths)):
            path = self.root / name
            wanted = name.startswith(prefixes) or name.startswith(("requirements", "CHECKPOINT_S20"))
            if not wanted or path.suffix not in suffixes or not path.is_file():
                continue
            if not path.resolve().is_relative_to(self.root):
                raise ValueError("source path escapes repository: " + name)
            contents = path.read_bytes()
            sha = hashlib.sha256(contents).hexdigest()
            blob = blobs / sha
            if not blob.exists():
                with blob.open("xb") as stream:
                    stream.write(contents)
            records.append({"path": name, "sha256": sha, "bytes": len(contents)})
        memory = psutil.virtual_memory()
        start = time.perf_counter()
        pilot_bytes = 0
        for record in records:
            pilot_bytes += (blobs / record["sha256"]).stat().st_size
            digest(blobs / record["sha256"])
        elapsed = time.perf_counter() - start
        env = {"at": now(), "python": platform.python_version(), "platform": platform.platform(),
               "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
               "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
               "snapshot_hash_pilot": {"bytes": pilot_bytes, "wall_seconds": elapsed},
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
               "model_training_pilot": "pending before H03; source hashing is not a fit speed estimate"}
        atomic_json(directory / "environment.json", env)
        atomic_json(directory / "protocol.json", self.plan)
        atomic_json(directory / "source_snapshot.json", {"commit": git("rev-parse", "HEAD").decode().strip(),
                    "git_status": git("status", "--porcelain").decode("utf-8"), "files": records,
                    "protected_hashes": before, "scope": "research code and contracts; no secrets or market datasets"})
        atomic_json(directory / "exposure_ledger.json", {"known_exposed_window": self.plan["validation"]["known_exposed_window"],
                    "historical_data_status": "development_only", "all_pre_freeze_dates_sealed_eligible": False,
                    "prospective_start": None, "additional_date_inventory": "required before H08"})
        atomic_json(directory / "budget.json", {"proposal": self.plan["budget_proposal"],
                    "training_jobs_max": 1, "cpu_threads_max": min(8, psutil.cpu_count() or 1),
                    "memory_limit_bytes": int(memory.available * .5), "gpu_enabled": False,
                    "training_authorized": False, "training_wall_limit_seconds": None,
                    "required_before_training": "measured bounded model pilot and enforced wall/CPU limits"})
        after = {p: digest(self.root / p) for p in before}
        if before != after:
            raise ValueError("protected paths changed during snapshot; do not overwrite external changes")
