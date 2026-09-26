"""Recompute a pinned baseline run and compare semantic outputs exactly."""
import json
from pathlib import Path

import pandas as pd

from .baseline_run import build, verify
from .runtime import atomic_json, digest, now


def replay(root, directory, summary_sha):
    directory = Path(directory).resolve()
    verify(directory, summary_sha)
    sources = json.loads((directory/"inputs.json").read_text(encoding="utf-8"))
    # Do not execute arbitrary archived code. Replay only under matching current
    # computational dependencies; runner/verifier changes are recorded separately.
    names = ("baseline_model.py", "calibration_model.py", "feature_pipeline.py", "splits.py",
             "label_availability.py", "recommendation_policy.py", "contracts.py", "runtime.py")
    input_plan = json.loads((directory/"input.json").read_text(encoding="utf-8"))
    if 'anchor_context' in input_plan:
        names += ('anchor_features.py','oof_audit.py')
    if 'anchor_parent_bindings' in input_plan:
        names += ('anchor_parent_receipts.py',)
    if input_plan.get("schema_version") == "2":
        names += ("dependency_receipts.py", "dependency_availability.py")
    if 'policy_control_sources' in input_plan:
        names += ('atr_control_sources.py',)
        if 'vendor_bindings' in input_plan['policy_control_sources']:names += ('atr_vendor_sources.py',)
        if 'dependency_receipts.py' not in names:names += ('dependency_receipts.py',)
    pins = {}
    for name in names:
        matches = [h for p, h in sources.items() if Path(p).name == name]
        path = Path(__file__).parent/name
        if len(matches) != 1 or digest(path) != matches[0]:
            raise ValueError("replay computational source mismatch: " + name)
        pins[path] = matches[0]
    prior = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    result = build(root, directory/"input.json", prior["artifacts"]["input.json"])
    fresh = Path(result["directory"])
    compared = []
    evidence = {"at": now(), "original_directory": str(directory), "original_summary_sha256": summary_sha,
                "replay_directory": str(fresh), "replay_summary_sha256": digest(fresh/"summary.json"),
                "replay_model_fits": result['model_level_fits'], "replay_calibrator_fits": 1,
                "replay_is_new_search_trial": False, "formal_stage_accepted": False,
                "same_implementation_replay_not_independent_algorithm_validation": True}
    try:
        for name in ("baseline_predictions.parquet", "calibrated_predictions.parquet", "candidate_ledger.parquet"):
            pd.testing.assert_frame_equal(pd.read_parquet(directory/name), pd.read_parquet(fresh/name),
                                          check_exact=True, check_like=False)
            compared.append(name)
        for name in ("baseline_card.json", "calibration_card.json", "selection_report.json"):
            old = json.loads((directory/name).read_text(encoding="utf-8"))
            new = json.loads((fresh/name).read_text(encoding="utf-8"))
            if old != new:
                raise ValueError("semantic JSON mismatch: " + name)
            compared.append(name)
        if 'anchor_parent_bindings' in input_plan:
            name='anchor_parent_evidence.json'
            if json.loads((directory/name).read_text())!=json.loads((fresh/name).read_text()):
                raise ValueError('semantic anchor parent evidence mismatch')
            compared.append(name)
        for name in ("status", "evidence_mode", "completed_steps", "rows", "outer_candidates", "selected",
                     "model_level_fits", "calibrator_fits", "risk_model_trained", "efficacy_evaluated",
                     "formal_H03_H05_accepted", "production_eligible"):
            if prior.get(name) != result[name]:
                raise ValueError("semantic summary mismatch: " + name)
        verify(directory, summary_sha)
        verify(fresh, evidence["replay_summary_sha256"])
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("replay computational source changed")
    except Exception as exc:
        atomic_json(fresh/"semantic_replay.json", dict(evidence, status="FAILED", compared=compared,
                    error={"type": type(exc).__name__, "message": str(exc)}))
        raise ValueError("baseline semantic replay failed: " + str(exc)) from exc
    report = dict(evidence, status="VERIFIED_DIAGNOSTIC", compared=compared,
                  semantic_replay_performed=True, performance_gain_proven=False)
    atomic_json(fresh/"semantic_replay.json", report)
    return report
