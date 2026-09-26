"""Read-only model/policy consumers with independently supplied outer outcomes."""
import json
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

from .baseline_evaluation import evaluate
from .policy_run import verify_policy
from .runtime import atomic_json, digest, now


def _evaluation_inputs(directory, payload):
    if set(payload) != {"target_id", "evaluation_at", "outcomes"}:
        raise ValueError("exact outer evaluation payload required")
    if not isinstance(payload["outcomes"], list) or any(not isinstance(r, dict) or
            set(r) != {"sample_id", "target", "label_available_at"} for r in payload["outcomes"]):
        raise ValueError("exact outer outcome records required")
    bindings = json.loads((directory/"bindings.json").read_text(encoding="utf-8"))
    plans = [json.loads((Path(r["directory"])/"input.json").read_text(encoding="utf-8")) for r in bindings["runs"]]
    matches = [i for i, p in enumerate(plans) if p["target_id"] == payload["target_id"]]
    if len(matches) != 1:
        raise ValueError("outer evaluation target mismatch")
    index = matches[0]
    plan = plans[index]
    policy_input = json.loads((directory/"input.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(directory/"outer_candidates.parquet")
    score_field = "score" if index == 0 else "risk"
    if index != 0:
        # Risk reliability must compare downside labels to downside scores, not
        # to safe-profit probabilities. Preserve original selection and scores.
        rows["original_selection_score"] = rows.score
        rows["score"] = rows.risk
    return rows, plan, policy_input, score_field


def build(root, directory, summary_sha, outcome_path, outcome_sha):
    directory, outcome_path = Path(directory).resolve(), Path(outcome_path).resolve()
    validation = verify_policy(directory, summary_sha)
    if not re.fullmatch(r"[0-9a-f]{64}", outcome_sha or "") or digest(outcome_path) != outcome_sha:
        raise ValueError("outer outcome pin mismatch")
    raw = outcome_path.read_bytes()
    payload = json.loads(raw)
    candidates, plan, policy_input, score_field = _evaluation_inputs(directory, payload)
    code_pins = {p: digest(p) for p in Path(__file__).parent.glob("*.py")}
    def check():
        verify_policy(directory, summary_sha)
        if digest(outcome_path) != outcome_sha or any(digest(p) != h for p, h in code_pins.items()):
            raise ValueError("evaluation source changed")
    check()
    rows, metrics = evaluate(candidates,
                             pd.DataFrame(plan["samples"]),
                             pd.DataFrame(payload["outcomes"], columns=["sample_id", "target", "label_available_at"]),
                             evaluation_at=payload["evaluation_at"], calendar=policy_input["outer_calendar"])
    check()
    out = Path(root).resolve()/"output/experiments/s20_safe_v4/sources"/("policy-evaluation-"+uuid.uuid4().hex)
    out.mkdir(parents=True)
    (out/"outcomes.json").write_bytes(raw)
    rows.to_parquet(out/"evaluated_outer.parquet", index=False)
    atomic_json(out/"metrics.json", metrics)
    atomic_json(out/"bindings.json", {"policy_directory": str(directory), "policy_summary_sha256": summary_sha,
                "outcome_path": str(outcome_path), "outcome_sha256": outcome_sha,
                "code": {str(p): h for p, h in code_pins.items()}})
    check()
    report = {"directory": str(out), "at": now(), "target_id": payload["target_id"],
              "evaluated_score_field": score_field,
              "evidence_mode": plan["evidence_mode"], "rows": len(rows), "selected": int(rows.selected.sum()),
              "source_validation": validation, "models_refitted": False, "policy_reselected": False,
              "formal_promotion_authorized": False,
              "artifacts": {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    atomic_json(out/"summary.json", report)
    return report


def verify_evaluation(directory, summary_sha):
    """Recompute maturity, rows and metrics without fitting or policy search."""
    directory = Path(directory).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", summary_sha or "") or digest(directory/"summary.json") != summary_sha:
        raise ValueError("evaluation summary pin mismatch")
    report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    required = {"outcomes.json", "evaluated_outer.parquet", "metrics.json", "bindings.json"}
    if set(report["artifacts"]) != required:
        raise ValueError("evaluation artifact set mismatch")
    def check_bytes():
        if digest(directory/"summary.json") != summary_sha:
            raise ValueError("evaluation summary changed")
        for name, sha in report["artifacts"].items():
            path = (directory/name).resolve()
            if path.parent != directory or digest(path) != sha:
                raise ValueError("evaluation artifact pin mismatch")
    check_bytes()
    binding = json.loads((directory/"bindings.json").read_text(encoding="utf-8"))
    if binding["outcome_sha256"] != digest(directory/"outcomes.json"):
        raise ValueError("evaluation outcome binding mismatch")
    policy_directory = Path(binding["policy_directory"])
    verify_policy(policy_directory, binding["policy_summary_sha256"])
    pins = {}
    for name in ("baseline_evaluation.py", "metrics.py", "label_availability.py"):
        found = [sha for p, sha in binding["code"].items() if Path(p).name == name]
        path = Path(__file__).parent/name
        if len(found) != 1 or digest(path) != found[0]:
            raise ValueError("evaluation computational source mismatch")
        pins[path] = found[0]
    payload = json.loads((directory/"outcomes.json").read_text(encoding="utf-8"))
    candidates, plan, policy_input, score_field = _evaluation_inputs(policy_directory, payload)
    rows, metrics = evaluate(candidates, pd.DataFrame(plan["samples"]),
                             pd.DataFrame(payload["outcomes"], columns=["sample_id", "target", "label_available_at"]),
                             evaluation_at=payload["evaluation_at"], calendar=policy_input["outer_calendar"])
    try:
        saved_rows = pd.read_parquet(directory/"evaluated_outer.parquet")
        # Parquet infers bool for an object column when every outcome is known.
        # Normalize only this declared boolean/unknown field, not arbitrary dtypes.
        if not saved_rows.evaluation_target.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("invalid saved evaluation target")
        saved_rows["evaluation_target"] = pd.Series(
            [None if pd.isna(v) else bool(v) for v in saved_rows.evaluation_target], dtype=object)
        pd.testing.assert_frame_equal(rows, saved_rows, check_exact=True)
    except AssertionError as exc:
        raise ValueError("evaluation semantic row mismatch") from exc
    saved_metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
    if saved_metrics != metrics:
        raise ValueError("evaluation semantic metric mismatch")
    if (report["target_id"] != payload["target_id"] or report["evidence_mode"] != plan["evidence_mode"]
            or report["rows"] != len(rows) or report["selected"] != int(rows.selected.sum())
            or report.get("evaluated_score_field", "score") != score_field):
        raise ValueError("evaluation semantic summary mismatch")
    check_bytes()
    verify_policy(policy_directory, binding["policy_summary_sha256"])
    if any(digest(p) != sha for p, sha in pins.items()):
        raise ValueError("evaluation computational source changed")
    return {"artifact_bytes_verified": True, "maturity_and_metrics_recomputed": True,
            "target_id": payload["target_id"], "evaluated_score_field": score_field,
            "policy_search_semantics_replayed": False, "formal_promotion_authorized": False}
