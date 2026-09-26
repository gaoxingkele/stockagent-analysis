"""Pinned diagnostic baseline/calibration/selection run, not formal H03-H05."""
import argparse
import json
from pathlib import Path
import re
import uuid

import pandas as pd

from .baseline_model import run as fit_baseline, validate_seed
from .calibration_model import run as fit_calibration
from .recommendation_policy import apply as select
from .runtime import atomic_json, digest, now


def build(root, input_path, input_sha):
    input_path = Path(input_path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha or "") or digest(input_path) != input_sha:
        raise ValueError("baseline input pin mismatch")
    raw = input_path.read_bytes()
    plan = json.loads(raw)
    expected = {"schema_version", "evidence_mode", "target_id", "samples", "features",
                "fit_labels", "calibration_labels", "boundaries", "feature_contract", "policy", "calendar"}
    version = plan.get("schema_version")
    if 'random_seed' in plan:
        expected.add('random_seed')
        validate_seed(plan['random_seed'])
    if 'policy_controls' in plan:
        expected.add('policy_controls')
    if 'policy_control_sources' in plan:
        expected.add('policy_control_sources')
        if 'policy_controls' not in plan:
            raise ValueError('control sources require declared derived controls')
    if 'anchor_context' in plan:
        expected.add('anchor_context')
    if 'anchor_parent_bindings' in plan:
        expected.add('anchor_parent_bindings')
        if 'anchor_context' not in plan:
            raise ValueError('anchor parent bindings require context')
    if 'model_family' in plan:
        expected.add('model_family')
        if plan['model_family'] not in {'logistic','shallow_tree','mature_frequency'}:
            raise ValueError('unsupported fixed baseline family')
    if version == "2":
        expected |= {"dependency_contract", "dependency_bindings"}
    if set(plan) != expected or version not in {"1", "2"} or plan["evidence_mode"] not in {"synthetic", "supplied_reference"}:
        raise ValueError("explicit baseline diagnostic schema required")
    if not isinstance(plan["samples"], list) or not 1 <= len(plan["samples"]) <= 100000:
        raise ValueError("bounded nonempty samples required")
    if plan["policy"]["mode"] not in {"score_only_control", "atr_liquidity_control"} or plan["policy"]["target_id"] != plan["target_id"]:
        raise ValueError("single target runner requires matching comparator policy")
    if ('policy_controls' in plan) != (plan['policy']['mode'] == 'atr_liquidity_control'):
        raise ValueError("ATR/liquidity policy and control inputs must be supplied together")
    sources = {input_path: input_sha}
    sources.update({p: digest(p) for p in Path(__file__).parent.glob("*.py")})
    control_evidence=None
    if 'policy_control_sources' in plan:
        from .atr_control_sources import derive
        declared=pd.DataFrame(plan['policy_controls'])
        metadata=pd.DataFrame(plan['samples']).set_index('sample_id').loc[declared.sample_id].reset_index()
        derived,control_evidence=derive(root,metadata[['sample_id','entity_id','signal_date','prediction_at']],
                                        plan['policy_control_sources'])
        pd.testing.assert_frame_equal(declared[derived.columns].reset_index(drop=True),derived,
                                      check_dtype=False,check_exact=True)
        sources.update({Path(p):h for p,h in control_evidence['source_pins'].items()})
    anchor_context=None
    anchor_parent_evidence=None
    if 'anchor_context' in plan:
        from .anchor_features import decode
        anchor_context,anchor_pins=decode(root,plan['anchor_context'])
        sources.update(anchor_pins)
    if 'anchor_parent_bindings' in plan:
        from .anchor_parent_receipts import bind as bind_anchor_parents
        anchor_parent_evidence=bind_anchor_parents(root,anchor_context,plan['anchor_parent_bindings'])
        sources.update({Path(p):h for p,h in anchor_parent_evidence['source_pins'].items()})
    dependency_evidence = None
    if version == "2":
        from .dependency_receipts import bind, KEYS, BINDINGS
        from .dependency_availability import resolve
        from .label_availability import _instant
        for name, fields in (("dependency_contract", KEYS), ("dependency_bindings", KEYS+BINDINGS)):
            rows = plan[name]
            if (not isinstance(rows, list) or len(rows) > 1000000
                    or any(not isinstance(r, dict) or set(r) != set(fields) for r in rows)):
                raise ValueError("exact bounded dependency row schema required")
        bindings = pd.DataFrame(plan["dependency_bindings"], columns=KEYS+BINDINGS)
        contract = pd.DataFrame(plan["dependency_contract"], columns=KEYS)
        receipts, evidence = bind(root, bindings)
        samples = pd.DataFrame(plan["samples"])
        derived, _, timing = resolve(samples, contract, receipts)
        # Keep input.json authoritative for calibration, policy and replay readers.
        # Reject optimistic summaries instead of silently changing only fit inputs.
        for field in ("feature_available_at", "label_available_at"):
            for supplied, actual in zip(samples[field], derived[field]):
                if pd.isna(supplied) != pd.isna(actual) or (pd.notna(supplied) and _instant(supplied) != _instant(actual)):
                    raise ValueError("sample time differs from bound dependency receipts: " + field)
        sources.update({Path(p): h for p,h in evidence["source_pins"].items()})
        dependency_evidence = dict(receipt_evidence=evidence, timing=timing,
                                   formal_data_acceptance=False)

    def check():
        if any(digest(p) != h for p, h in sources.items()):
            raise ValueError("baseline source changed")

    check()
    out = Path(root).resolve()/"output/experiments/s20_safe_v4/sources"/("baseline-run-"+uuid.uuid4().hex)
    out.mkdir(parents=True)
    artifacts = {}
    completed = []

    def checkpoint(state, error=None):
        atomic_json(out/"checkpoint.json", dict(status=state, completed_steps=completed,
                    artifacts=artifacts, error=error, resume_supported=False,
                    formal_training_authorized=False))

    def save(name, value):
        if isinstance(value, pd.DataFrame):
            value.to_parquet(out/name, index=False)
        else:
            atomic_json(out/name, value)
        artifacts[name] = digest(out/name)

    checkpoint("RUNNING")
    try:
        blobs = out/"code_blobs"
        blobs.mkdir()
        for p, h in sources.items():
            if p.suffix == ".py":
                (blobs/h).write_bytes(p.read_bytes())
                if digest(blobs/h) != h:
                    raise ValueError("code snapshot changed")
        (out/"input.json").write_bytes(raw)
        if digest(out/"input.json") != input_sha:
            raise ValueError("input snapshot changed")
        artifacts["input.json"] = input_sha
        save("inputs.json", {str(p): h for p, h in sources.items()})
        if control_evidence is not None:
            save('policy_control_evidence.json',control_evidence)
        if dependency_evidence is not None:
            save("dependency_evidence.json", dependency_evidence)
        if anchor_parent_evidence is not None:
            save('anchor_parent_evidence.json',anchor_parent_evidence)
        samples = pd.DataFrame(plan["samples"])
        features = pd.DataFrame(plan["features"])
        check()
        predictions, baseline = fit_baseline(samples, features, pd.DataFrame(plan["fit_labels"]),
                                             plan["boundaries"], plan["feature_contract"], target_id=plan["target_id"],
                                             model_family=plan.get('model_family','logistic'),anchor_context=anchor_context,
                                             random_seed=plan.get('random_seed',20))
        check()
        save("baseline_predictions.parquet", predictions)
        save("baseline_card.json", baseline)
        completed.append("baseline")
        checkpoint("RUNNING")
        calibrated, calibration = fit_calibration(samples, predictions, pd.DataFrame(plan["calibration_labels"]),
                                                  plan["boundaries"], baseline, target_id=plan["target_id"],anchor_context=anchor_context)
        check()
        save("calibrated_predictions.parquet", calibrated)
        save("calibration_card.json", calibration)
        completed.append("calibration")
        checkpoint("RUNNING")
        outer = calibrated.loc[calibrated.segment.eq("outer-test")].copy()
        metadata = samples.set_index("sample_id").loc[outer.sample_id]
        candidates = pd.DataFrame({"sample_id": outer.sample_id.to_numpy(),
                                   "entity_id": metadata.entity_id.to_numpy(),
                                   "signal_date": metadata.signal_date.to_numpy(),
                                   "prediction_at": metadata.prediction_at.to_numpy(),
                                   "score": outer.calibrated_probability.to_numpy(), "risk": float("nan")})
        controls = pd.DataFrame(plan['policy_controls']) if 'policy_controls' in plan else None
        rows, selection = select(candidates, plan["policy"], plan["calendar"], controls=controls)
        check()
        save("candidate_ledger.parquet", rows)
        save("selection_report.json", selection)
        completed.append("selection")
        checkpoint("COMPLETED_DIAGNOSTIC")
        check()
        if any(digest(out/n) != h for n, h in artifacts.items()):
            raise ValueError("baseline artifact changed")
    except Exception as exc:
        checkpoint("FAILED", {"type": type(exc).__name__, "message": str(exc)})
        raise
    report = dict(directory=str(out), at=now(), status="COMPLETED_DIAGNOSTIC",
                  evidence_mode=plan["evidence_mode"], completed_steps=list(completed),
                  rows=len(samples), outer_candidates=len(rows), selected=int(rows.selected.sum()),
                  model_level_fits=baseline['model_level_fits'], calibrator_fits=1, risk_model_trained=False,
                  efficacy_evaluated=False, formal_H03_H05_accepted=False,
                  production_eligible=False, artifacts=dict(artifacts, **{"checkpoint.json": digest(out/"checkpoint.json")}))
    atomic_json(out/"summary.json", report)
    return report


def verify(directory, summary_sha):
    directory = Path(directory).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", summary_sha or "") or digest(directory/"summary.json") != summary_sha:
        raise ValueError("baseline summary pin mismatch")
    report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    required = {"input.json", "inputs.json", "baseline_predictions.parquet", "baseline_card.json",
                "calibrated_predictions.parquet", "calibration_card.json", "candidate_ledger.parquet",
                "selection_report.json", "checkpoint.json"}
    input_plan = json.loads((directory/"input.json").read_text(encoding="utf-8"))
    if input_plan.get("schema_version") == "2":
        required.add("dependency_evidence.json")
    if 'policy_control_sources' in input_plan:
        required.add('policy_control_evidence.json')
    if 'anchor_parent_bindings' in input_plan:
        required.add('anchor_parent_evidence.json')
    if set(report["artifacts"]) != required or report.get("status") != "COMPLETED_DIAGNOSTIC":
        raise ValueError("baseline artifact set or status invalid")
    for name, sha in report["artifacts"].items():
        path = (directory/name).resolve()
        if path.parent != directory or digest(path) != sha:
            raise ValueError("baseline artifact pin mismatch")
    sources = json.loads((directory/"inputs.json").read_text(encoding="utf-8"))
    if 'policy_control_sources' in input_plan:
        from .atr_control_sources import derive
        declared=pd.DataFrame(input_plan['policy_controls'])
        metadata=pd.DataFrame(input_plan['samples']).set_index('sample_id').loc[declared.sample_id].reset_index()
        derived,evidence=derive(directory.parents[4],metadata[['sample_id','entity_id','signal_date','prediction_at']],
                                input_plan['policy_control_sources'])
        pd.testing.assert_frame_equal(declared[derived.columns].reset_index(drop=True),derived,
                                      check_dtype=False,check_exact=True)
        if evidence!=json.loads((directory/'policy_control_evidence.json').read_text(encoding='utf-8')):
            raise ValueError('control source derivation evidence mismatch')
        if any(sources.get(p)!=h for p,h in evidence['source_pins'].items()):
            raise ValueError('unbound control source evidence')
    if 'anchor_parent_bindings' in input_plan:
        evidence=json.loads((directory/'anchor_parent_evidence.json').read_text(encoding='utf-8'))
        for path,sha in evidence['source_pins'].items():
            if sources.get(path)!=sha or digest(Path(path))!=sha:
                raise ValueError('anchor parent source changed or unbound')
        from .anchor_parent_receipts import verify_budget_binding
        checked=[]
        for binding in input_plan['anchor_parent_bindings']:
            if 'budget_reference' in binding:
                checked.append(dict(model_id=binding['model_id'],**verify_budget_binding(directory.parents[4],binding)))
        if checked!=evidence.get('budget_attempts',[]):
            raise ValueError('anchor parent budget evidence changed')
    for name, sha in sources.items():
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("invalid source digest")
        if Path(name).suffix == ".py" and digest(directory/"code_blobs"/sha) != sha:
            raise ValueError("baseline code blob mismatch")
    checkpoint = json.loads((directory/"checkpoint.json").read_text(encoding="utf-8"))
    if (checkpoint["status"] != "COMPLETED_DIAGNOSTIC"
            or checkpoint["completed_steps"] != ["baseline", "calibration", "selection"]):
        raise ValueError("baseline run incomplete")
    if checkpoint.get("artifacts") != {k: v for k, v in report["artifacts"].items() if k != "checkpoint.json"}:
        raise ValueError("baseline checkpoint artifact binding mismatch")
    return {"artifact_bytes_verified": True, "semantic_replay_performed": False,
            "formal_stage_accepted": False, "artifacts": len(report["artifacts"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2], args.input, args.input_sha256), indent=2))
