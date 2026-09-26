"""Independent exchange-interval versus provider-day retrospective crosscheck."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .daily_gap_join import overlay
from .runtime import atomic_json, digest, now


def compare(exchange, daily, committed_dates):
    required = {"ts_code", "trade_date", "source_sha256", "quote_present", "status"}
    if not required.issubset(exchange) or exchange[list(required)].isna().any().any():
        raise ValueError("missing exchange provenance")
    if not exchange.status.eq("missing_consistent_with_reported_suspension").all():
        raise ValueError("only independently reconciled full-session rows supported")
    if not exchange.quote_present.eq(False).all():
        raise ValueError("exchange evidence has quote conflict")
    # overlay retains each exchange row and its provenance, never promoting
    # unobserved vendor-only dates to exchange-confirmed truth.
    result, _ = overlay(exchange, daily, committed_dates)
    result["crosscheck_consistent"] = result.daily_evidence_state.eq("provider_full_day_candidate")
    return result, {
        "exchange_stock_dates": len(exchange),
        "consistent_stock_dates": int(result.crosscheck_consistent.sum()),
        "states": result.daily_evidence_state.value_counts().to_dict(),
        "scope": "selected closed exchange intervals only; not exhaustive history",
        "historical_prediction_availability_proven": False,
        "formal_training_authorized": False,
    }


def build(root, exchange_path, exchange_sha, audit_directory, audit_summary_sha):
    exchange_path, audit_directory = Path(exchange_path), Path(audit_directory)
    summary_path = audit_directory / "summary.json"
    pins = [(exchange_path, exchange_sha), (summary_path, audit_summary_sha)]
    for path, sha in pins:
        if digest(path) != sha:
            raise ValueError("input pin mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows_path, inputs_path = audit_directory / "rows.parquet", audit_directory / "inputs.json"
    pins += [(rows_path, summary["rows_sha256"]), (inputs_path, summary["inputs_sha256"])]
    for path, sha in pins:
        if digest(path) != sha:
            raise ValueError("audit artifact mismatch")
    receipts = json.loads(inputs_path.read_text(encoding="utf-8"))
    daily = pd.read_parquet(rows_path)
    if len(daily) != summary["rows"] or len(receipts) != summary["committed_dates"]:
        raise ValueError("audit count mismatch")
    receipt_by_date = {r["date"]: r["receipt_sha256"] for r in receipts}
    if len(receipt_by_date) != len(receipts) or not daily.receipt_sha256.eq(daily.trade_date.map(receipt_by_date)).all():
        raise ValueError("audit receipt lineage mismatch")
    table, report = compare(pd.read_parquet(exchange_path), daily, list(receipt_by_date))
    for path, sha in pins:
        if digest(path) != sha:
            raise ValueError("input changed during crosscheck")
    output = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("suspension-crosscheck-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    table.to_parquet(output / "stock_dates.parquet", index=False)
    atomic_json(output / "inputs.json", [{"path": str(p.resolve()), "sha256": s} for p, s in pins])
    report.update(directory=str(output), at=now(), code_sha256=digest(Path(__file__)),
                  overlay_code_sha256=digest(Path(__file__).with_name("daily_gap_join.py")),
                  table_sha256=digest(output / "stock_dates.parquet"), inputs_sha256=digest(output / "inputs.json"))
    atomic_json(output / "summary.json", report)
    return report
