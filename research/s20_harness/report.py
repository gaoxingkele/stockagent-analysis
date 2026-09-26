"""Read-only stage evidence report; never authorizes training or repairs state."""
from __future__ import annotations

import json
import sqlite3

from .runtime import Runtime, digest, now


def failure_evidence(runtime, row, events):
    """Verify an error receipt without accepting missing stage deliverables."""
    path = runtime.base / row['run_id'] / 'failure.json'
    if not path.exists():
        return {'present': False, 'verified': False}
    errors = []
    try:
        before = digest(path)
        receipt = json.loads(path.read_text(encoding='utf-8'))
        last = events[-1] if events else None
        if (not last or last['state'] != row['state']
                or json.loads(last['payload']).get('failure_sha256') != before):
            errors.append('failure receipt not bound to terminal event')
        if (receipt['run_id'] != row['run_id'] or receipt['stage'] != row['stage']
                or receipt['protocol_hash'] != runtime.protocol_hash
                or receipt['terminal_state'] != row['state']
                or row['state'] not in ('FAILED_VALIDITY', 'FAILED_INFRA')
                or receipt['error'] != row['error']
                or receipt['formal_stage_completed'] is not False
                or receipt['automatic_retry_authorized'] is not False):
            errors.append('failure receipt identity or policy mismatch')
        if digest(path) != before:
            errors.append('failure receipt changed during read')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append('failure receipt unreadable: ' + str(exc))
    return {'present': True, 'verified': not errors, 'errors': errors,
            'scope': 'error receipt only; not complete stage evidence'}


def build_report(runtime: Runtime) -> dict:
    database = runtime.base / "registry.sqlite"
    rows, events = [], {}
    if database.exists():
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY started_at, run_id")]
            for event in db.execute("SELECT * FROM events ORDER BY seq"):
                events.setdefault(event["run_id"], []).append(dict(event))
    history = []
    latest = {}
    for row in rows:
        run_id = row["run_id"]
        check = {"valid": False, "errors": []}
        gaps = []
        science = None
        try:
            check = runtime.verify(run_id, require_completion=False)
            path = runtime.base / run_id / "manifest.json"
            if path.is_file():
                manifest = json.loads(path.read_text(encoding="utf-8"))
                gaps = manifest.get("acceptance_gaps", [])
                science = manifest.get("science_state")
                if manifest.get("run_id") != run_id or manifest.get("stage") != row["stage"]:
                    check["errors"].append("manifest identity differs from registry")
                if manifest.get("terminal_state") != row["state"]:
                    check["errors"].append("manifest state differs from registry")
                recorded = events.get(run_id, [])
                last = recorded[-1] if recorded else None
                if not last or last["state"] != row["state"]:
                    check["errors"].append("terminal event missing or inconsistent")
                elif json.loads(last["payload"]).get("manifest_hash") != digest(path):
                    check["errors"].append("manifest differs from terminal event hash")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            check["errors"].append("evidence unreadable: " + str(exc))
        check["valid"] = not check["errors"]
        item = {"run_id": run_id, "stage": row["stage"], "recorded_state": row["state"],
                "evidence": check, "acceptance_gaps": gaps, "science_state": science,
                "runtime_error": row["error"]}
        item["failure_evidence"] = failure_evidence(runtime, row, events.get(run_id, []))
        history.append(item)
        latest[row["stage"]] = item
    stages = []
    accepted = set()
    # Validated plan is a DAG; derive acceptance independently of declaration order.
    pending = set(runtime.stages)
    while pending:
        ready = [s for s in runtime.stages if s in pending and
                 not (set(runtime.stages[s]["depends_on"]) & pending)]
        if not ready:
            raise ValueError("cyclic stage dependencies")
        for stage in ready:
            spec = runtime.stages[stage]
            run = latest.get(stage)
            missing = [s for s in spec["depends_on"] if s not in accepted]
            valid = bool(run and run["recorded_state"] == "COMPLETED" and
                         run["evidence"]["valid"] and not missing)
            if valid:
                accepted.add(stage)
            stages.append({"stage": stage, "accepted_evidence": valid,
                           "latest_run_id": run["run_id"] if run else None,
                           "recorded_state": run["recorded_state"] if run else "PLANNED",
                           "unverified_dependencies": missing,
                           "executor_implemented": stage in ("H00", "H01", "H02", "H03", "H05"),
                           "executor_scope": "diagnostic intake only; formal acceptance not implemented" if stage in ("H02", "H03", "H05") else None,
                           "required_outputs": spec["outputs"]})
            pending.remove(stage)
    return {"reported_at": now(), "protocol_hash": runtime.protocol_hash,
            "report_kind": "stage_artifact_evidence_not_efficacy",
            "acceptance_gaps_basis": "recorded run-time audit, not a fresh source inventory audit",
            "upstream_run_binding_verification": "required recursively by runtime verifier; legacy dependent runs without pins fail",
            "stages": stages, "run_history": history,
            "formal_training_authorized_by_report": False,
            "production_actions_allowed": False}
