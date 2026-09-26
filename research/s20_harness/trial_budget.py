"""Transactional pre-reservation of fully registered pipeline attempts."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .runtime import digest, now

COUNTERS = ("model_fits", "underlying_fits", "calibrator_fits", "policy_evaluations")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_attempt(root, path, contract_sha, trial_id, attempt_id, input_path, input_sha, artifact, limits):
    """Read an existing budget without initializing or changing its tables."""
    from .bounded_baseline import verify_process
    path=Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('existing workspace budget required')
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        db.execute('BEGIN')
        payload=db.execute('SELECT payload FROM budget_contract WHERE id=1').fetchone()
        records=db.execute('SELECT trial_id,attempt_id,costs,state,reserved_at,result FROM attempts').fetchall()
    if payload is None or hashlib.sha256(payload[0].encode()).hexdigest()!=contract_sha:
        raise ValueError('parent budget contract pin mismatch')
    contract=json.loads(payload[0])
    if canonical(contract)!=payload[0]: raise ValueError('noncanonical budget contract')
    trials={t['trial_id']:t for t in contract['trials']}
    if len(trials)!=len(contract['trials']): raise ValueError('duplicate budget trial')
    used={k:0 for k in COUNTERS};counts={};selected=[]
    for tid,aid,costs,state,reserved,result in records:
        charge=json.loads(costs)
        if tid not in trials or charge!=trials[tid]['costs'] or set(charge)!=set(COUNTERS):
            raise ValueError('budget charge differs from registered trial')
        if any(type(v) is not int or v<0 for v in charge.values()): raise ValueError('invalid budget charge')
        for key in COUNTERS: used[key]+=charge[key]
        counts[tid]=counts.get(tid,0)+1
        if counts[tid]>trials[tid]['max_attempts']: raise ValueError('budget attempt cap exceeded')
        if (tid,aid)==(trial_id,attempt_id): selected.append((charge,state,reserved,result))
    if set(contract['limits'])!=set(COUNTERS) or any(type(v) is not int or v<0 for v in contract['limits'].values()):
        raise ValueError('invalid budget limits')
    if contract['limits']['model_fits']>72 or any(used[k]>contract['limits'][k] for k in COUNTERS):
        raise ValueError('parent budget overspent')
    if len(selected)!=1: raise ValueError('exact parent attempt missing')
    charge,state,reserved,result=selected[0]
    if state!='SUCCEEDED_DIAGNOSTIC' or json.loads(result)!=artifact:
        raise ValueError('parent budget completion/artifact mismatch')
    from .baseline_model import pipeline_costs as baseline_costs
    if digest(Path(input_path))!=input_sha:raise ValueError('parent pipeline input pin mismatch')
    input_plan=json.loads(Path(input_path).read_text(encoding='utf-8'))
    if artifact.get('pipeline_kind')=='joint':
        from .joint_run import pipeline_costs
        expected=pipeline_costs(input_plan)
    else:expected=baseline_costs(input_plan)
    if charge!=expected or trials[trial_id]['input_sha256']!=input_sha:
        raise ValueError('parent pipeline budget/input mismatch')
    verify_process(root,artifact,input_path,input_sha,limits,
                   pipeline=artifact.get('pipeline_kind','baseline'))
    from .label_availability import _instant
    process=json.loads((Path(artifact['process_directory'])/'process_result.json').read_text(encoding='utf-8'))
    if _instant(reserved).timestamp()>process['process_created']:
        raise ValueError('parent budget was not reserved before process creation')
    return dict(budget_path=str(path),budget_sha256=contract_sha,trial_id=trial_id,attempt_id=attempt_id,
        charged_costs=charge,reserved_at=reserved,reserved_counts_at_review=used,recorded_prelaunch_reservation_verified=True,
        recorded_completion_verified=True,external_authenticity_proven=False,formal_training_authorized=False)


class Budget:
    def __init__(self, path, contract):
        if set(contract) != {"budget_id", "limits", "trials"} or not isinstance(contract["budget_id"], str) or not contract["budget_id"]:
            raise ValueError("exact named budget contract required")
        limits = contract["limits"]
        if set(limits) != set(COUNTERS) or any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError("nonnegative integer limits required")
        if limits["model_fits"] > 72:
            raise ValueError("initial model fit cap cannot exceed 72")
        if not isinstance(contract["trials"], list) or not 1 <= len(contract["trials"]) <= 1000:
            raise ValueError("bounded registered trial list required")
        ids = set()
        for trial in contract["trials"]:
            if set(trial) != {"trial_id", "input_sha256", "costs", "max_attempts"}:
                raise ValueError("exact trial contract required")
            if not isinstance(trial["trial_id"], str) or not trial["trial_id"] or trial["trial_id"] in ids:
                raise ValueError("unique trial ID required")
            ids.add(trial["trial_id"])
            if not re.fullmatch(r"[0-9a-f]{64}", trial["input_sha256"] or ""):
                raise ValueError("pinned full pipeline input required")
            if set(trial["costs"]) != set(COUNTERS) or any(type(v) is not int or v < 0 for v in trial["costs"].values()):
                raise ValueError("integer trial costs required")
            if type(trial["max_attempts"]) is not int or not 1 <= trial["max_attempts"] <= 3:
                raise ValueError("bounded registered attempts required")
        self.path = Path(path).resolve()
        self.contract = json.loads(canonical(contract))
        self.serialized = canonical(self.contract)
        self.sha = hashlib.sha256(self.serialized.encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS budget_contract (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (trial_id TEXT, attempt_id TEXT, costs TEXT NOT NULL,
                    state TEXT NOT NULL, reserved_at TEXT NOT NULL, result TEXT, PRIMARY KEY(trial_id,attempt_id));
                CREATE TRIGGER IF NOT EXISTS no_contract_update BEFORE UPDATE ON budget_contract
                    BEGIN SELECT RAISE(ABORT,'immutable budget'); END;
                CREATE TRIGGER IF NOT EXISTS no_contract_delete BEFORE DELETE ON budget_contract
                    BEGIN SELECT RAISE(ABORT,'immutable budget'); END;
                CREATE TRIGGER IF NOT EXISTS no_attempt_delete BEFORE DELETE ON attempts
                    BEGIN SELECT RAISE(ABORT,'attempts retained'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_reservation BEFORE UPDATE ON attempts
                    WHEN NEW.trial_id != OLD.trial_id OR NEW.attempt_id != OLD.attempt_id
                    OR NEW.costs != OLD.costs OR NEW.reserved_at != OLD.reserved_at
                    OR OLD.state != 'RESERVED' OR NEW.state NOT IN ('SUCCEEDED_DIAGNOSTIC','FAILED')
                    BEGIN SELECT RAISE(ABORT,'immutable reservation or terminal attempt'); END;
            """)
            db.execute("INSERT OR IGNORE INTO budget_contract VALUES(1,?)", (self.serialized,))
            if db.execute("SELECT payload FROM budget_contract WHERE id=1").fetchone()[0] != self.serialized:
                raise ValueError("budget contract changed")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def reserve(self, trial_id, attempt_id, input_sha):
        if canonical(self.contract) != self.serialized:
            raise ValueError("in-memory budget contract changed")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("named attempt required")
        trial = next((t for t in self.contract["trials"] if t["trial_id"] == trial_id), None)
        if trial is None or input_sha != trial["input_sha256"]:
            raise ValueError("unregistered trial/input")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT payload FROM budget_contract WHERE id=1").fetchone()[0] != self.serialized:
                raise ValueError("stored budget contract changed")
            existing = db.execute("SELECT state FROM attempts WHERE trial_id=? AND attempt_id=?", (trial_id, attempt_id)).fetchone()
            if existing:
                return {"newly_reserved": False, "state": existing[0], "launch_authorized": False}
            records = db.execute("SELECT trial_id,costs FROM attempts").fetchall()
            if sum(t == trial_id for t, _ in records) >= trial["max_attempts"]:
                raise ValueError("trial attempt cap exceeded")
            used = {c: sum(json.loads(costs)[c] for _, costs in records) for c in COUNTERS}
            if any(used[c] + trial["costs"][c] > self.contract["limits"][c] for c in COUNTERS):
                raise ValueError("pipeline budget exhausted")
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,NULL)",
                       (trial_id, attempt_id, canonical(trial["costs"]), "RESERVED", now()))
        return {"newly_reserved": True, "state": "RESERVED", "launch_authorized": False,
                "budget_reserved_not_formal_training_permission": True}

    def finish(self, trial_id, attempt_id, state, result):
        if state not in {"SUCCEEDED_DIAGNOSTIC", "FAILED"}:
            raise ValueError("terminal diagnostic state required")
        with self.connect() as db:
            cursor = db.execute("UPDATE attempts SET state=?,result=? WHERE trial_id=? AND attempt_id=? AND state='RESERVED'",
                                (state, canonical(result), trial_id, attempt_id))
            if cursor.rowcount != 1:
                raise ValueError("attempt missing or already terminal")

    def status(self):
        with self.connect() as db:
            records = db.execute("SELECT trial_id,attempt_id,costs,state,result FROM attempts ORDER BY reserved_at,trial_id,attempt_id").fetchall()
        return {"budget_sha256": self.sha, "reserved_counts": {c: sum(json.loads(r[2])[c] for r in records) for c in COUNTERS},
                "attempts": [{"trial_id": r[0], "attempt_id": r[1], "costs": json.loads(r[2]), "state": r[3],
                              "result": json.loads(r[4]) if r[4] else None} for r in records],
                "failed_attempts_refunded": False, "formal_training_authorized": False}


def run_baseline(root, input_path, input_sha, budget, trial_id, attempt_id):
    from .baseline_run import build
    from .baseline_model import pipeline_costs
    trial = next((t for t in budget.contract["trials"] if t["trial_id"] == trial_id), None)
    if digest(Path(input_path)) != input_sha:
        raise ValueError("baseline input changed before reservation")
    expected = pipeline_costs(json.loads(Path(input_path).read_text(encoding='utf-8')))
    if trial is None or trial["costs"] != expected:
        raise ValueError("fixed baseline pipeline costs mismatch")
    if digest(Path(input_path)) != input_sha:
        raise ValueError("baseline input changed before reservation")
    ticket = budget.reserve(trial_id, attempt_id, input_sha)
    if not ticket["newly_reserved"]:
        return {"executed": False, "reservation": ticket, "budget": budget.status()}
    try:
        report = build(root, input_path, input_sha)
        summary = Path(report["directory"])/"summary.json"
        budget.finish(trial_id, attempt_id, "SUCCEEDED_DIAGNOSTIC", {"directory": report["directory"], "summary_sha256": digest(summary)})
    except Exception as exc:
        budget.finish(trial_id, attempt_id, "FAILED", {"type": type(exc).__name__, "message": str(exc)})
        raise
    return {"executed": True, "run": report, "budget": budget.status()}
