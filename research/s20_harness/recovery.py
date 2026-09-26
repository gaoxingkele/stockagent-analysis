"""Recover interrupted registry commits without restarting live owners."""
from __future__ import annotations

import json

import psutil

from .runtime import digest, now


def owner_live(pid, created):
    try:
        return abs(psutil.Process(pid).create_time() - created) < .01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return None


def resume(runtime):
    """Reconcile only; no automatic rerun or inferred stage acceptance.

    A dead owner plus a durable VERIFYING hash permits finalizing its already
    verified artifacts. A missing/mismatched anchor is an infrastructure failure,
    not permission to manufacture COMPLETED from a loose manifest.
    """
    if not (runtime.base / "registry.sqlite").exists():
        return {"actions": [], "training_started": False}
    actions = []
    with runtime.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        rows = list(db.execute("SELECT * FROM runs WHERE state IN ('RUNNING','VERIFYING')"))
        for row in rows:
            live = owner_live(row["pid"], row["process_created"])
            if live is not False:
                actions.append({"run_id": row["run_id"],
                                "action": "owner_live" if live else "owner_unknown_no_mutation"})
                continue
            run_id = row["run_id"]
            terminal, error, payload = "FAILED_INFRA", "owner exited without recoverable verified commit", {}
            try:
                check = runtime.verify(run_id, require_completion=False)
                path = runtime.base / run_id / "manifest.json"
                event = db.execute("SELECT * FROM events WHERE run_id=? ORDER BY seq DESC LIMIT 1",
                                   (run_id,)).fetchone()
                if check["valid"] and event and event["state"] == "VERIFYING":
                    anchor = json.loads(event["payload"]).get("manifest_hash")
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    state = manifest.get("terminal_state")
                    if (anchor == digest(path) and manifest.get("run_id") == run_id and
                            manifest.get("stage") == row["stage"] and
                            state in ("COMPLETED", "FAILED_VALIDITY", "FAILED_INFRA")):
                        terminal, error = state, None
                        payload = {"manifest_hash": anchor,
                                   "acceptance_gaps": manifest.get("acceptance_gaps", [])}
            except (ValueError, OSError, KeyError, TypeError) as exc:
                error = "unrecoverable artifact: " + str(exc)
            payload.update({"recovered_dead_owner": True, "error": error,
                            "previous_pid": row["pid"], "previous_process_created": row["process_created"]})
            db.execute("UPDATE runs SET state=?,finished_at=?,error=? WHERE run_id=?",
                       (terminal, now(), error, run_id))
            runtime._event(db, run_id, terminal, payload)
            actions.append({"run_id": run_id, "action": "reconciled", "state": terminal, "error": error})
    return {"actions": actions, "training_started": False,
            "retry_policy": "no automatic retry; failed validity requires new evidence"}
