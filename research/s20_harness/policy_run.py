"""Artifact-bound policy search and held-out application, research only."""
import json
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

from .baseline_run import verify
from .label_availability import _instant
from .policy_search import compare
from .recommendation_policy import apply
from .runtime import atomic_json, digest, now


def build(root, score_directory, score_sha, input_path, input_sha, *, risk_directory=None, risk_sha=None):
    score_directory, input_path = Path(score_directory).resolve(), Path(input_path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha or "") or digest(input_path) != input_sha:
        raise ValueError("policy input pin mismatch")
    raw = input_path.read_bytes()
    payload = json.loads(raw)
    if set(payload) != {"registry", "selection_outcomes", "selection_calendar", "outer_calendar"}:
        raise ValueError("exact policy run schema required")
    if not isinstance(payload["selection_outcomes"], list) or any(
        set(r) != {"sample_id", "target", "label_available_at"} for r in payload["selection_outcomes"]):
        raise ValueError("exact selection outcome records required")
    bindings = [(score_directory, score_sha)]
    verify(score_directory, score_sha)
    plan = json.loads((score_directory/"input.json").read_text(encoding="utf-8"))
    registry = payload["registry"]
    if registry["target_id"] != plan["target_id"]:
        raise ValueError("score target mismatch")
    score = pd.read_parquet(score_directory/"calibrated_predictions.parquet").set_index("sample_id")
    risk = None
    gated = [p for p in registry["policies"] if p["mode"] == "risk_gated"]
    if risk_directory is not None:
        risk_directory = Path(risk_directory).resolve()
        verify(risk_directory, risk_sha)
        bindings.append((risk_directory, risk_sha))
        risk_plan = json.loads((risk_directory/"input.json").read_text(encoding="utf-8"))
        if risk_plan["target_id"] == plan["target_id"]:
            raise ValueError("independent downside target required")
        for field in ("samples", "boundaries", "evidence_mode"):
            if risk_plan[field] != plan[field]:
                raise ValueError("risk/score reference scope mismatch: " + field)
        if any(p["risk_target_id"] != risk_plan["target_id"] for p in gated):
            raise ValueError("risk target mismatch")
        risk = pd.read_parquet(risk_directory/"calibrated_predictions.parquet").set_index("sample_id")
        if not risk.index.equals(score.index) or not risk.segment.equals(score.segment):
            raise ValueError("risk prediction universe mismatch")
    elif gated or risk_sha is not None:
        raise ValueError("gated policy requires pinned downside model")
    samples = pd.DataFrame(plan["samples"])
    meta = samples.set_index("sample_id")

    def candidates(segment):
        scores = score.loc[score.segment.eq(segment)]
        metadata = meta.loc[scores.index]
        return pd.DataFrame({"sample_id": scores.index, "entity_id": metadata.entity_id.to_numpy(),
                             "signal_date": metadata.signal_date.to_numpy(),
                             "prediction_at": metadata.prediction_at.to_numpy(),
                             "score": scores.calibrated_probability.to_numpy(),
                             "risk": risk.loc[scores.index].calibrated_probability.to_numpy() if risk is not None else float("nan")})

    code_pins = {p: digest(p) for p in Path(__file__).parent.glob("*.py")}
    def check():
        for directory, sha in bindings:
            verify(directory, sha)
        if digest(input_path) != input_sha or any(digest(p) != sha for p, sha in code_pins.items()):
            raise ValueError("policy run input/source changed")

    check()
    ledgers, search = compare(samples, candidates("selection-policy"),
                               pd.DataFrame(payload["selection_outcomes"], columns=["sample_id", "target", "label_available_at"]),
                               plan["boundaries"], payload["selection_calendar"], registry)
    outer = candidates("outer-test")
    calendar = payload["outer_calendar"]
    if not calendar or calendar != sorted(set(calendar)) or not set(outer.signal_date).issubset(calendar):
        raise ValueError("complete chronological outer calendar required")
    for date in calendar:
        close = pd.to_datetime(date, format="%Y%m%d").tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        if not _instant(plan["boundaries"][4]["start_at"]) <= close < _instant(plan["boundaries"][4]["end_at"]):
            raise ValueError("calendar outside outer segment")
    chosen = next((p for p in registry["policies"] if p["policy_id"] == search["selected_policy_id"]), None)
    if chosen is None:
        outer["selected"] = False
        outer["reject_reason"] = "no_eligible_policy"
        application = {"status": "NO_POLICY", "outer_rows": len(outer), "selected": 0,
                       "outer_calendar": payload["outer_calendar"], "formal_H05_accepted": False}
    else:
        # Reference decision-time binding, not a claim of a historical receipt.
        locked = dict(chosen, frozen_at=search["label_cutoff"])
        outer, application = apply(outer, locked, payload["outer_calendar"])
        application["registered_policy"] = chosen
        application["selected_at_reference"] = search["label_cutoff"]
    check()
    out = Path(root).resolve()/"output/experiments/s20_safe_v4/sources"/("policy-run-"+uuid.uuid4().hex)
    out.mkdir(parents=True)
    (out/"input.json").write_bytes(raw)
    atomic_json(out/"bindings.json", {"runs": [{"directory": str(p), "summary_sha256": sha} for p, sha in bindings],
                "input_path": str(input_path), "input_sha256": input_sha,
                "code": {str(p): h for p, h in code_pins.items()}})
    atomic_json(out/"search.json", search)
    atomic_json(out/"application.json", application)
    outer.to_parquet(out/"outer_candidates.parquet", index=False)
    for i, policy in enumerate(registry["policies"]):
        ledgers[policy["policy_id"]].to_parquet(out/f"selection_trial_{i:02d}.parquet", index=False)
    check()
    report = {"directory": str(out), "at": now(), "evidence_mode": plan["evidence_mode"],
              "selected_policy_id": search["selected_policy_id"], "policy_evaluations": search["policy_evaluations"],
              "outer_candidates": len(outer), "outer_selected": int(outer.selected.sum()),
              "downside_model_bound": risk is not None, "new_model_fits": 0,
              "outer_outcomes_consumed": False, "historical_decision_receipt_proven": False,
              "semantic_model_replay_performed": False, "formal_H05_accepted": False,
              "artifacts": {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    atomic_json(out/"summary.json", report)
    return report


def verify_policy(directory, summary_sha):
    directory = Path(directory).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", summary_sha or "") or digest(directory/"summary.json") != summary_sha:
        raise ValueError("policy summary pin mismatch")
    report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    count = report.get("policy_evaluations")
    if type(count) is not int or not 1 <= count <= 6:
        raise ValueError("invalid policy trial count")
    required = {"input.json", "bindings.json", "search.json", "application.json", "outer_candidates.parquet"}
    required.update(f"selection_trial_{i:02d}.parquet" for i in range(count))
    if set(report["artifacts"]) != required:
        raise ValueError("incomplete policy artifact set")
    for name, sha in report["artifacts"].items():
        path = (directory/name).resolve()
        if path.parent != directory or digest(path) != sha:
            raise ValueError("policy artifact pin mismatch")
    bindings = json.loads((directory/"bindings.json").read_text(encoding="utf-8"))
    if digest(directory/"input.json") != bindings["input_sha256"]:
        raise ValueError("policy input binding mismatch")
    runs = bindings["runs"]
    if len(runs) not in (1, 2) or report["downside_model_bound"] != (len(runs) == 2):
        raise ValueError("policy model binding count mismatch")
    for run in runs:
        verify(run["directory"], run["summary_sha256"])
    score_directory = Path(runs[0]["directory"])
    plan = json.loads((score_directory/"input.json").read_text(encoding="utf-8"))
    payload = json.loads((directory/"input.json").read_text(encoding="utf-8"))
    search = json.loads((directory/"search.json").read_text(encoding="utf-8"))
    if (search["registry"] != payload["registry"] or search["policy_evaluations"] != count
            or len(payload["registry"]["policies"]) != count
            or search["selected_policy_id"] != report["selected_policy_id"]):
        raise ValueError("policy search binding mismatch")
    scores = pd.read_parquet(score_directory/"calibrated_predictions.parquet")
    ids = scores.loc[scores.segment.eq("outer-test"), "sample_id"].tolist()
    rows = pd.read_parquet(directory/"outer_candidates.parquet")
    if rows.sample_id.tolist() != ids or len(rows) != report["outer_candidates"]:
        raise ValueError("outer candidate universe mismatch")
    if not rows.selected.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("invalid outer selection flags")
    if int(rows.selected.sum()) != report["outer_selected"]:
        raise ValueError("outer selected count mismatch")
    if report["selected_policy_id"] is None and rows.selected.any():
        raise ValueError("selection without eligible policy")
    if plan["evidence_mode"] != report["evidence_mode"] or plan["target_id"] != payload["registry"]["target_id"]:
        raise ValueError("policy evidence/target mismatch")
    return {"artifact_bytes_verified": True, "bound_model_runs": len(runs),
            "outer_universe_verified": True, "semantic_policy_replay_performed": False,
            "formal_H05_accepted": False}
