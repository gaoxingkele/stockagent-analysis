"""Provider-day evidence overlay; does not overwrite exchange evidence or gaps."""
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, now
from .label_availability import _instant


def overlay(gaps, daily, committed_dates):
    keys = ["ts_code", "trade_date"]
    if gaps[keys].isna().any().any() or gaps.duplicated(keys).any():
        raise ValueError("unique nonmissing gap keys required")
    if len(committed_dates) != len(set(committed_dates)):
        raise ValueError("duplicate committed dates")
    required = {*keys, "suspend_type", "suspend_timing", "quote_present", "receipt_sha256", "received_at"}
    if not required.issubset(daily):
        raise ValueError("missing daily evidence fields")
    if not set(daily.trade_date).issubset(committed_dates):
        raise ValueError("daily evidence outside committed coverage")
    if not daily.suspend_type.isin(["S", "R"]).all():
        raise ValueError("invalid event types")
    if daily[[*keys, "receipt_sha256", "received_at", "quote_present"]].isna().any().any():
        raise ValueError("missing evidence identity/provenance")
    if not daily.quote_present.map(lambda v: isinstance(v, bool)).all():
        raise ValueError("quote presence must be boolean")
    for value in daily.received_at:
        _instant(value)
    grouped = []
    for (code, date), group in daily.groupby(keys, sort=False):
        full = group.suspend_type.eq("S") & group.suspend_timing.isna()
        if group.quote_present.any():
            state = "quote_conflict_with_original_gap"
        elif len(group) != 1:
            state = "multiple_provider_events_review"
        elif full.any():
            state = "provider_full_day_candidate"
        elif group.suspend_type.eq("R").any():
            state = "provider_resumption_but_gap"
        else:
            state = "provider_intraday_or_unknown_timing"
        grouped.append({"ts_code": code, "trade_date": date, "daily_evidence_state": state,
                        "daily_receipt_hashes": sorted(set(group.receipt_sha256)),
                        "daily_received_at": sorted(set(group.received_at))})
    evidence = pd.DataFrame(grouped, columns=[*keys, "daily_evidence_state", "daily_receipt_hashes", "daily_received_at"])
    if set(evidence.columns) & (set(gaps.columns) - set(keys)):
        raise ValueError("overlay already present")
    result = gaps.merge(evidence, on=keys, how="left", sort=False, validate="one_to_one")
    absent = result.daily_evidence_state.isna()
    result.loc[absent & result.trade_date.isin(committed_dates), "daily_evidence_state"] = "no_provider_record_on_queried_date"
    result.loc[absent & ~result.trade_date.isin(committed_dates), "daily_evidence_state"] = "date_not_yet_queried"
    if result[keys].values.tolist() != gaps[keys].values.tolist():
        raise ValueError("gap order/denominator changed")
    return result, {"gap_stock_dates": len(gaps), "committed_dates": len(committed_dates),
                    "daily_states": result.daily_evidence_state.value_counts().to_dict(),
                    "original_gaps_preserved": True, "formal_training_authorized": False}


def build(root, gaps_path, gaps_sha, audit_directory, audit_summary_sha):
    import json
    root, audit_directory = Path(root).resolve(), Path(audit_directory)
    summary_path = audit_directory / "summary.json"
    if digest(summary_path) != audit_summary_sha or digest(Path(gaps_path)) != gaps_sha:
        raise ValueError("input pin mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows_path, inputs_path = audit_directory / "rows.parquet", audit_directory / "inputs.json"
    if digest(rows_path) != summary["rows_sha256"] or digest(inputs_path) != summary["inputs_sha256"]:
        raise ValueError("audit artifact mismatch")
    receipts = json.loads(inputs_path.read_text(encoding="utf-8"))
    if len(receipts) != summary["committed_dates"]:
        raise ValueError("audit receipt count mismatch")
    gaps, rows = pd.read_parquet(gaps_path), pd.read_parquet(rows_path)
    if len(rows) != summary["rows"]:
        raise ValueError("audit row count mismatch")
    receipt_by_date = {r["date"]: r["receipt_sha256"] for r in receipts}
    if not rows.receipt_sha256.eq(rows.trade_date.map(receipt_by_date)).all():
        raise ValueError("row receipt lineage mismatch")
    joined, report = overlay(gaps, rows, [r["date"] for r in receipts])
    for path, sha in [(summary_path, audit_summary_sha), (Path(gaps_path), gaps_sha),
                      (rows_path, summary["rows_sha256"]), (inputs_path, summary["inputs_sha256"])]:
        if digest(path) != sha:
            raise ValueError("input changed during overlay")
    output = root / "output/experiments/s20_safe_v4/sources" / ("daily-gap-join-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    joined.to_parquet(output / "gaps.parquet", index=False)
    report.update(directory=str(output), at=now(), gaps_sha256=gaps_sha, audit_summary_sha256=audit_summary_sha,
                  audit_directory=str(audit_directory.resolve()), code_sha256=digest(Path(__file__)),
                  table_sha256=digest(output / "gaps.parquet"))
    atomic_json(output / "summary.json", report)
    return report
