"""Retrospective factor dependency validity, never an ex-ante stock filter."""
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, now


def assess(dependencies, anomalies):
    required = {"sample_id", "dependency_id", "event_codes", "window_start", "window_end", "factor_usage"}
    if not required.issubset(dependencies):
        raise ValueError("missing dependency contract")
    if dependencies[["sample_id", "dependency_id"]].isna().any().any() or dependencies.duplicated(["sample_id", "dependency_id"]).any():
        raise ValueError("unique dependency identity required")
    if not {"ts_code", "event_date", "evidence_id"}.issubset(anomalies) or anomalies[["ts_code", "event_date", "evidence_id"]].isna().any().any():
        raise ValueError("missing anomaly identity")
    if anomalies.duplicated(["ts_code", "event_date"]).any():
        raise ValueError("duplicate anomaly")
    for date in anomalies.event_date:
        pd.to_datetime(date, format="%Y%m%d", errors="raise")
    result = dependencies.copy()
    hits, states = [], []
    for r in dependencies.itertuples():
        if not isinstance(r.event_codes, (list, tuple)) or not r.event_codes:
            raise ValueError("explicit identity codes required")
        if r.factor_usage not in {"none_raw_economic", "within_window_ratios", "absolute_factor_levels"}:
            raise ValueError("unsupported factor dependency")
        for date in [r.window_start, r.window_end]:
            pd.to_datetime(date, format="%Y%m%d", errors="raise")
        if r.window_start > r.window_end:
            raise ValueError("reversed dependency window")
        selected = anomalies.loc[anomalies.ts_code.isin(r.event_codes)]
        if r.factor_usage == "none_raw_economic":
            selected = selected.iloc[:0]
        elif r.factor_usage == "within_window_ratios":
            # A persistent common factor scale cancels in ratios if both
            # endpoints are after the transition. Start is exclusive.
            selected = selected.loc[selected.event_date.gt(r.window_start) & selected.event_date.le(r.window_end)]
        else:
            # Absolute levels retain all earlier unresolved scale changes.
            selected = selected.loc[selected.event_date.le(r.window_end)]
        ids = sorted(set(selected.evidence_id))
        hits.append(ids)
        states.append("unresolved_factor_dependency" if ids else "no_registered_factor_conflict")
    result["factor_conflict_evidence_ids"] = hits
    result["factor_dependency_status"] = states
    result["recommendation_kept"] = True
    result["formal_training_eligible"] = False
    return result


def case_impact(root, calendar_path, calendar_sha, case_path, case_sha):
    import json
    pins = [(Path(calendar_path), calendar_sha), (Path(case_path), case_sha)]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("source pin mismatch")
    config = json.loads(pins[1][0].read_text(encoding="utf-8"))
    case = next(c for c in config["remaining_unexplained_reference_cases"] if c["ts_code"] == "603081.SH")
    cal = pd.read_parquet(calendar_path)
    dates = sorted(cal.loc[cal.exchange.eq("SSE") & cal.is_open.astype(int).eq(1), "cal_date"].astype(str))
    dependencies = []
    for i in range(60, len(dates) - 20):
        for name, start, end, usage in [("feature_60_return", dates[i-60], dates[i], "within_window_ratios"),
                                        ("factor_20_label", dates[i+1], dates[i+20], "within_window_ratios"),
                                        ("raw_economic_P", dates[i+1], dates[i+20], "none_raw_economic")]:
            dependencies.append(dict(sample_id=dates[i] + ":603081.SH", dependency_id=name, event_codes=["603081.SH"],
                                     window_start=start, window_end=end, factor_usage=usage))
    anomaly = pd.DataFrame([dict(ts_code=case["ts_code"], event_date=case["event_date"], evidence_id=case_sha)])
    table = assess(pd.DataFrame(dependencies), anomaly)
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("source changed")
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("factor-dependencies-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    table.to_parquet(out / "dependencies.parquet", index=False)
    atomic_json(out / "inputs.json", [{"path": str(p.resolve()), "sha256": sha} for p, sha in pins])
    affected = table.loc[table.factor_dependency_status.eq("unresolved_factor_dependency")]
    report = dict(directory=str(out), at=now(), dependency_rows=len(table), samples=int(table.sample_id.nunique()),
                  affected_by_dependency=affected.dependency_id.value_counts().to_dict(), formal_training_authorized=False,
                  scope="603081 registered case; hypothetical 60-day feature and 20-day label contracts, not actual model dependencies",
                  warning="No registered factor conflict does not prove event/PIT/price completeness. No candidate excluded.",
                  table_sha256=digest(out / "dependencies.parquet"), inputs_sha256=digest(out / "inputs.json"), code_sha256=digest(Path(__file__)))
    atomic_json(out / "summary.json", report)
    return report
