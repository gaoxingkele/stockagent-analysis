"""Separate mature-outcome diagnostic evaluation; never a training consumer."""
import json
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

from .baseline_run import verify
from .label_availability import _instant
from .metrics import binary_bounds
from .runtime import atomic_json, digest, now


def evaluate(candidates, samples, outcomes, *, evaluation_at, calendar):
    cutoff = _instant(evaluation_at)
    if candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError("unique candidate IDs required")
    if not candidates.selected.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("boolean selection required")
    if (not pd.api.types.is_numeric_dtype(candidates.score) or pd.api.types.is_bool_dtype(candidates.score)
            or not candidates.score.dropna().between(0, 1).all()):
        raise ValueError("valid numeric evaluation scores required")
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
        raise ValueError("unique sample IDs required")
    if not set(candidates.sample_id).issubset(samples.sample_id):
        raise ValueError("missing sample horizon metadata")
    if (outcomes.columns.duplicated().any()
            or set(outcomes.columns) != {"sample_id", "target", "label_available_at"}
            or outcomes.sample_id.isna().any() or outcomes.sample_id.duplicated().any()):
        raise ValueError("exact unique outcome schema required")
    if not set(outcomes.sample_id).issubset(candidates.sample_id):
        raise ValueError("outcomes outside evaluation universe")
    if not calendar or list(calendar) != sorted(set(calendar)) or not set(candidates.signal_date).issubset(calendar):
        raise ValueError("complete chronological calendar required")
    for day in calendar:
        close = pd.to_datetime(day, format="%Y%m%d").tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        if close >= cutoff:
            raise ValueError("evaluation precedes calendar close")
    meta = samples.set_index("sample_id")
    labels = outcomes.set_index("sample_id")
    rows = candidates.copy().reset_index(drop=True)
    effective = []
    reasons = []
    for row in rows.itertuples(index=False):
        predicted = _instant(meta.loc[row.sample_id, "prediction_at"])
        horizon = _instant(meta.loc[row.sample_id, "horizon_close_at"])
        if horizon <= predicted or predicted >= cutoff or _instant(row.prediction_at) != predicted:
            raise ValueError("prediction/horizon/evaluation time mismatch")
        value, reason = None, "missing_outcome"
        if row.sample_id in labels.index:
            record = labels.loc[row.sample_id]
            if not pd.isna(record.target) and not isinstance(record.target, (bool, np.bool_)):
                raise ValueError("boolean or unknown target required")
            if pd.isna(record.label_available_at):
                reason = "availability_unknown"
            else:
                available = _instant(record.label_available_at)
                if available < horizon:
                    raise ValueError("outcome available before full horizon")
                if available >= cutoff:
                    reason = "not_mature_at_evaluation"
                elif pd.isna(record.target):
                    reason = "unresolved_outcome"
                else:
                    value, reason = bool(record.target), "mature"
        effective.append(value)
        reasons.append(reason)
    rows["evaluation_target"] = pd.Series(effective, dtype=object)
    rows["evaluation_status"] = reasons

    def summarize(frame):
        known = frame.loc[frame.evaluation_target.notna() & frame.score.notna()]
        if not known.score.between(0, 1).all():
            raise ValueError("invalid evaluation score")
        p = known.score.to_numpy(dtype=float)
        y = known.evaluation_target.to_numpy(dtype=float)
        bins = []
        for i in range(10):
            bucket_all = frame.loc[frame.score.ge(i / 10) & (frame.score.lt((i + 1) / 10) if i < 9 else frame.score.le(1))]
            bucket = bucket_all.loc[bucket_all.evaluation_target.notna()]
            bounds = binary_bounds(bucket_all.evaluation_target.tolist())
            mean_all = float(bucket_all.score.mean()) if len(bucket_all) else None
            bins.append({"lower": i / 10, "upper": (i + 1) / 10, "known_count": len(bucket),
                         "mean_prediction": float(bucket.score.mean()) if len(bucket) else None,
                         "observed_rate": float(bucket.evaluation_target.astype(float).mean()) if len(bucket) else None,
                         "scored_count": len(bucket_all), "event_bounds": bounds,
                         "mean_prediction_all_scored": mean_all,
                         "known_fraction": len(bucket) / len(bucket_all) if len(bucket_all) else None,
                         "prediction_minus_rate_lower": mean_all - bounds['rate_upper'] if mean_all is not None else None,
                         "prediction_minus_rate_upper": mean_all - bounds['rate_lower'] if mean_all is not None else None,
                         "legacy_mean_and_observed_are_known_only": True})
        return {"event_bounds": binary_bounds(frame.evaluation_target.tolist()),
                "scored_rows": int(frame.score.notna().sum()),
                "missing_prediction_rows": int(frame.score.isna().sum()),
                "scored_mature_rows": len(known),
                "brier_known_only": float(np.mean((p - y) ** 2)) if len(known) else None,
                "known_only_diagnostic": True, "reliability_bins": bins,
                "bin_rule": "fixed_tenths_left_closed_right_open_except_final_closed",
                "reliability_bounds_are_confidence_intervals": False}

    picked = rows.loc[rows.selected]
    return rows, {"evaluation_at": cutoff.isoformat(), "all_candidates": summarize(rows),
                  "selected_candidates": summarize(picked),
                  "daily": [{"signal_date": day, "selected": summarize(picked.loc[picked.signal_date.eq(day)])}
                            for day in calendar],
                  "status_counts": rows.evaluation_status.value_counts().to_dict(),
                  "all_candidates_retained": True, "confidence_intervals_computed": False,
                  "selected_probability_reliability_proven": False, "executed_returns_evaluated": False,
                  "formal_promotion_authorized": False}


def build(root, run_directory, run_sha, outcome_path, outcome_sha):
    run_directory, outcome_path = Path(run_directory).resolve(), Path(outcome_path).resolve()
    validation = verify(run_directory, run_sha)
    if not re.fullmatch(r"[0-9a-f]{64}", outcome_sha or "") or digest(outcome_path) != outcome_sha:
        raise ValueError("outcome input pin mismatch")
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    if set(payload) != {"target_id", "evaluation_at", "outcomes"}:
        raise ValueError("exact evaluation payload required")
    if not isinstance(payload["outcomes"], list) or any(
            not isinstance(row, dict) or set(row) != {"sample_id", "target", "label_available_at"}
            for row in payload["outcomes"]):
        raise ValueError("exact outcome records required")
    plan = json.loads((run_directory/"input.json").read_text(encoding="utf-8"))
    if payload["target_id"] != plan["target_id"]:
        raise ValueError("evaluation target mismatch")
    candidates = pd.read_parquet(run_directory/"candidate_ledger.parquet")
    rows, metrics = evaluate(candidates, pd.DataFrame(plan["samples"]),
                             pd.DataFrame(payload["outcomes"], columns=["sample_id", "target", "label_available_at"]),
                             evaluation_at=payload["evaluation_at"], calendar=plan["calendar"])
    verify(run_directory, run_sha)
    if digest(outcome_path) != outcome_sha:
        raise ValueError("outcome changed during evaluation")
    out = Path(root).resolve()/"output/experiments/s20_safe_v4/sources"/("baseline-evaluation-"+uuid.uuid4().hex)
    out.mkdir(parents=True)
    rows.to_parquet(out/"evaluated_candidates.parquet", index=False)
    atomic_json(out/"metrics.json", metrics)
    report = {"at": now(), "directory": str(out), "run_directory": str(run_directory),
              "run_summary_sha256": run_sha, "outcome_path": str(outcome_path), "outcome_sha256": outcome_sha,
              "target_id": payload["target_id"], "evidence_mode": plan["evidence_mode"],
              "source_validation": validation, "rows": len(rows), "selected": int(rows.selected.sum()),
              "code_sha256": digest(Path(__file__)), "semantic_prediction_replay_performed": False,
              "formal_promotion_authorized": False,
              "artifacts": {n: digest(out/n) for n in ("evaluated_candidates.parquet", "metrics.json")}}
    atomic_json(out/"summary.json", report)
    return report
