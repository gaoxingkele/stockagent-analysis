"""H05 executor: delayed calibration and selective recommendations.

Calibrates each five-class configuration on the mature calibration segment,
then registers a small bounded policy grid. Everything it reports is a
diagnostic reference: a fitted temperature is not evidence that the
probabilities are trustworthy.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .abcds_calibration import run as calibrate
from .abcds_model import CLASS_ORDER, TARGET_ID
from .policy_support import evaluate_selection, gate_threshold
from .runtime import atomic_json, now
from .splits import assign_segments, training_ids

FROZEN_AT = "2024-05-01T00:00:00+08:00"
CALIBRATOR_FIT_CAP = 6
# Registered before evaluation: ungated reference, one gated reference, one
# smaller-cap reference per calibrated configuration.
POLICY_TEMPLATE = (("ungated_top20", False, 20, None),
                   ("q50_top20", True, 20, 0.5),
                   ("q50_top10", True, 10, 0.5))

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; calibration inherits the diagnostic chain",
    "one chronological split and a single seed; no cross-fold robustness",
    "selected-subset reliability is reported, not certified",
    "no mature-feedback update validation and no absolute-probability claim",
    "the independent risk head is reused from H04 rather than re-estimated here",
)


def _load_config_frames(long_frame: pd.DataFrame, config_id: str, samples: pd.DataFrame,
                        assignment: pd.DataFrame) -> pd.DataFrame:
    rows = long_frame[long_frame.config_id.eq(config_id)].copy()
    if rows.empty:
        raise ValueError("no H04 predictions for config " + config_id)
    rows = rows.set_index("sample_id").reindex(samples.sample_id).reset_index()
    for column in ("p_A", "p_B", "p_C", "p_D", "p_S"):
        if column not in rows.columns:
            raise ValueError("H04 long predictions must carry the five-class simplex")
    frame = pd.DataFrame({
        "sample_id": rows.sample_id,
        "segment": assignment.segment.to_numpy(),
        "feature_ready": np.asarray(rows.feature_ready, dtype=bool),
        "prediction_status": rows.prediction_status.to_numpy(),
    })
    for name in CLASS_ORDER:
        frame["p_" + name] = pd.to_numeric(rows["p_" + name], errors="coerce").to_numpy()
    return frame


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    upstream = {name: Path(path) for name, path in upstream_directories.items()}
    for required in ("H02", "H03", "H04"):
        if required not in upstream:
            raise ValueError("H05 requires upstream outputs: " + required)
    labels = pd.read_parquet(upstream["H02"] / "labels.parquet")
    split_manifest = json.loads(
        (upstream["H03"] / "split_manifest.json").read_text(encoding="utf-8"))
    boundaries = split_manifest["segments"]
    long_frame = pd.read_parquet(upstream["H04"] / "oof_predictions.parquet")
    cards = json.loads((upstream["H04"] / "model_cards.json").read_text(encoding="utf-8"))["configs"]

    source = root / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
    samples = pd.read_parquet(source / "samples.parquet")
    resolved = labels[labels.target.notna()].copy()
    keep = set(resolved.sample_id)
    samples = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
    assignment, split = assign_segments(samples, boundaries)
    if split["split_protocol_sha256"] != split_manifest["split_protocol_sha256"]:
        raise ValueError("H05 rebuilt a different split protocol")
    calibration_ids = training_ids(assignment, "calibration")
    calibration_labels = resolved[resolved.sample_id.isin(calibration_ids)][
        ["sample_id", "target"]].copy()
    if set(calibration_labels.sample_id) != set(calibration_ids):
        raise ValueError("H05 calibration labels must cover exactly the eligible rows")

    risk_head = pd.Series(
        pd.to_numeric(long_frame.loc[long_frame.config_id.eq("independent_risk_head"), "risk"],
                      errors="coerce").to_numpy(),
        index=long_frame.loc[long_frame.config_id.eq("independent_risk_head"), "sample_id"].to_numpy())
    calibration_risk = risk_head.reindex(calibration_ids).dropna().to_numpy()

    identity = samples[["sample_id", "entity_id", "signal_date", "prediction_at"]]
    realized = resolved.set_index("sample_id")
    calendar = sorted(samples.signal_date.astype(str).unique().tolist())
    evaluation_segments = {"selection-policy", "outer-test"}
    in_evaluation = assignment.segment.isin(evaluation_segments).to_numpy()
    evaluation_days = int(samples.loc[in_evaluation, "signal_date"].nunique())

    # Only fitted probability models are calibrated. A constant frequency control
    # has nothing to calibrate, and the reused binary reference is not a simplex.
    calibrated_configs = [config_id for config_id, card in cards.items()
                          if tuple(card.get("classes") or ()) == CLASS_ORDER
                          and card.get("model") == "fixed_multinomial_logistic"]
    if not calibrated_configs:
        raise ValueError("no five-class configuration available to calibrate")
    calibrator_fits = 0
    states_path = directory / "calibration_states.jsonl"
    if directory.exists() and any((directory / name).exists() for name in
                                  ("calibration_states.jsonl", "policy_candidates.json",
                                   "risk_coverage.csv", "selected_reliability.csv")):
        raise ValueError("H05 refuses existing outputs")
    directory.mkdir(parents=True, exist_ok=True)
    policy_rows, coverage_rows, reliability_rows = [], [], []
    with states_path.open("w", encoding="utf-8", newline="\n") as handle:
        for config_id in calibrated_configs:
            frame = _load_config_frames(long_frame, config_id, samples, assignment)
            card = cards[config_id]
            output, state = calibrate(samples, frame, calibration_labels, boundaries, card,
                                      target_id=TARGET_ID, class_order=CLASS_ORDER)
            calibrator_fits += int(state["calibrator_fits"])
            if calibrator_fits > CALIBRATOR_FIT_CAP:
                raise ValueError("H05 calibrator budget exceeded")
            handle.write(json.dumps(dict(state, config_id=config_id), ensure_ascii=False,
                                    sort_keys=True) + "\n")
            calibrated = output.set_index("sample_id")
            score = (pd.to_numeric(calibrated["cal_p_A"], errors="coerce")
                     + pd.to_numeric(calibrated["cal_p_B"], errors="coerce"))
            score = score.dropna()
            risk_by_id = pd.Series(risk_head.to_numpy(), index=risk_head.index)
            for policy_name, use_gate, n_cap, quantile in POLICY_TEMPLATE:
                risk_cut = None
                if use_gate:
                    risk_cut = gate_threshold(calibration_risk, quantile)
                policy_id = f"{config_id}::{policy_name}"
                selected, report = evaluate_selection(
                    identity, score, risk_by_id, calendar, policy_id=policy_id,
                    use_gate=use_gate, n_cap=n_cap, frozen_at=FROZEN_AT, risk_cut=risk_cut)
                chosen = realized.reindex(selected)
                chosen = chosen.dropna(subset=["target"])
                date_of = identity.set_index("sample_id").signal_date
                evaluated = [sample for sample in chosen.index
                             if assignment.set_index("sample_id").segment.get(sample)
                             in evaluation_segments]
                evaluated_frame = realized.reindex(evaluated)
                selected_days = int(date_of.reindex(evaluated).nunique()) if evaluated else 0
                policy_rows.append(dict(
                    policy_id=policy_id, config_id=config_id, n_cap=n_cap,
                    use_risk_gate=bool(use_gate), risk_quantile=quantile,
                    risk_threshold=risk_cut, frozen_at=FROZEN_AT,
                    selected_total=int(report["selected"]),
                    selected_in_evaluation=int(len(evaluated)),
                    coverage=float(len(evaluated) / (n_cap * max(evaluation_days, 1))),
                    realized_A=None if not len(evaluated_frame) else
                    float(evaluated_frame.target.eq("A").mean()),
                    realized_hit15=None if not len(evaluated_frame) else
                    float(evaluated_frame.target.isin(["A", "B"]).mean()),
                    realized_risk10=None if not len(evaluated_frame) else
                    float(evaluated_frame.target.isin(["B", "D"]).mean()),
                    active_signal_days=selected_days,
                    note="diagnostic reference policy; not H05 acceptance"))
                coverage_rows.append(dict(
                    policy_id=policy_id, config_id=config_id, n_cap=n_cap,
                    evaluation_signal_days=evaluation_days,
                    denominator=n_cap * max(evaluation_days, 1),
                    selected=int(len(evaluated)),
                    coverage=float(len(evaluated) / (n_cap * max(evaluation_days, 1))),
                    vacancies=int(n_cap * max(evaluation_days, 1) - len(evaluated)),
                    risk_gate_applied=bool(use_gate), risk_threshold=risk_cut))
                if evaluated:
                    predicted = score.reindex(evaluated).dropna()
                    actual = evaluated_frame.target.isin(["A", "B"])
                    reliability_rows.append(dict(
                        policy_id=policy_id, config_id=config_id, n_selected=int(len(evaluated)),
                        mean_predicted_hit15=None if predicted.empty else float(predicted.mean()),
                        realized_hit15=float(actual.mean()),
                        gap=None if predicted.empty else float(predicted.mean() - actual.mean()),
                        subset="selection-policy + outer-test",
                        calibrated=True))

    if calibrator_fits > CALIBRATOR_FIT_CAP:
        raise ValueError("H05 calibrator budget exceeded")
    pd.DataFrame(coverage_rows).to_csv(directory / "risk_coverage.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(directory / "selected_reliability.csv", index=False)
    atomic_json(directory / "policy_candidates.json", {
        "at": now(), "target_id": TARGET_ID, "classes": list(CLASS_ORDER),
        "registered_policies": policy_rows, "calibrator_fit_cap": CALIBRATOR_FIT_CAP,
        "calibrator_fits": calibrator_fits, "calibrated_configs": calibrated_configs,
        "independent_risk_head_source": "reused from H04",
        "formal_H05_accepted": False, "formal_training_authorized": False,
        "production_eligible": False, "absolute_probability_validated": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS)})
    return {
        "formal_gate_passed": False,
        "formal_H05_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "calibrated_configs": calibrated_configs,
        "calibrator_fits": calibrator_fits,
        "calibrator_fit_cap": CALIBRATOR_FIT_CAP,
        "policies_registered": [row["policy_id"] for row in policy_rows],
        "evaluation_signal_days": evaluation_days,
        "selected_reliability": reliability_rows,
        "validation_scope": "single-split diagnostic calibration and policy grid; not formal H05",
    }
