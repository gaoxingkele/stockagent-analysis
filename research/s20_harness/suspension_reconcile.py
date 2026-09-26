"""Closed reported suspension intervals versus observed daily rows."""
from pathlib import Path
import uuid

import pandas as pd

from .label_availability import _instant
from .runtime import atomic_json, digest, now


def reconcile(intervals, observed, calendar):
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("unique ordered calendar required")
    if not {"ts_code", "trade_date"}.issubset(observed):
        raise ValueError("observation identity missing")
    if observed[["ts_code", "trade_date"]].isna().any().any() or observed.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("invalid observed identities")
    if not set(observed.trade_date).issubset(calendar):
        raise ValueError("observations outside calendar")
    sessions = [(date, pd.Timestamp(date + " 09:30", tz="Asia/Shanghai"),
                 pd.Timestamp(date + " 15:00", tz="Asia/Shanghai")) for date in calendar]
    seen = set(zip(observed.ts_code, observed.trade_date))
    events, checks = [], []
    for index, row in enumerate(intervals.to_dict("records")):
        start = _instant(row["suspended_at"])
        event = {"event_row": index, "ts_code": row["ts_code"], "source_sha256": row["source_sha256"],
                 "status": "unknown_end_not_used", "full_sessions": 0, "partial_sessions": 0}
        events.append(event)
        if row["interval_status"] == "unknown_end":
            if pd.notna(row["resumed_at"]):
                raise ValueError("unknown end contradicts resumption")
            continue
        if row["interval_status"] != "closed_reported" or pd.isna(row["resumed_at"]):
            raise ValueError("invalid interval state")
        end = _instant(row["resumed_at"])
        if end <= start:
            raise ValueError("reversed interval")
        for date, opening, closing in sessions:
            # Half-open suspension [start,end). Resuming at close is partial,
            # not proof that a daily quote should be absent.
            if start <= opening and end > closing:
                present = (row["ts_code"], date) in seen
                checks.append({"event_row": index, "ts_code": row["ts_code"], "trade_date": date,
                               "source_sha256": row["source_sha256"], "quote_present": present,
                               "status": "quote_conflicts_full_suspension" if present else "missing_consistent_with_reported_suspension"})
                event["full_sessions"] += 1
            elif start < closing and end > opening:
                event["partial_sessions"] += 1
        event["status"] = "closed_interval_compared"
    columns = ["event_row", "ts_code", "trade_date", "source_sha256", "quote_present", "status"]
    detail = pd.DataFrame(checks, columns=columns)
    return pd.DataFrame(events), detail, {
        "events": len(events), "unknown_end_events": sum(e["status"] == "unknown_end_not_used" for e in events),
        "full_session_evidence_rows": len(detail),
        "unique_full_stock_dates": len(detail.drop_duplicates(["ts_code", "trade_date"])),
        "conflicting_quote_rows": int(detail.quote_present.sum()) if len(detail) else 0,
        "partial_session_intersections": sum(e["partial_sessions"] for e in events),
        "formal_coverage_verified": False,
        "interpretation": "retrospective source consistency, not historical prediction-time eligibility"}


def build(root, intervals_path, expected_sha256, calendar_path, calendar_sha256):
    root = Path(root).resolve()
    inputs = []
    def read(path, expected=None, columns=None):
        path = Path(path)
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError("input pin mismatch")
        frame = pd.read_parquet(path, columns=columns)
        if digest(path) != sha:
            raise ValueError("input changed during read")
        inputs.append({"path": str(path.resolve()), "sha256": sha})
        return frame
    intervals = read(intervals_path, expected_sha256)
    cal = read(calendar_path, calendar_sha256)
    dates = sorted(cal.loc[cal.exchange.eq("SZSE") & cal.is_open.astype(int).eq(1), "cal_date"].astype(str))
    needed = set(intervals.ts_code)
    frames = []
    for date in dates:
        frame = read(root / "output/tushare_cache/daily" / (date + ".parquet"), columns=["ts_code", "trade_date"])
        frame["trade_date"] = frame.trade_date.astype(str)
        if not frame.trade_date.eq(date).all():
            raise ValueError("daily date mismatch")
        frames.append(frame.loc[frame.ts_code.isin(needed)])
    events, detail, report = reconcile(intervals, pd.concat(frames, ignore_index=True), dates)
    output = root / "output/experiments/s20_safe_v4/sources" / ("suspension-reconcile-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    events.to_parquet(output / "events.parquet", index=False)
    detail.to_parquet(output / "stock_dates.parquet", index=False)
    atomic_json(output / "inputs.json", inputs)
    report.update(directory=str(output), at=now(), calendar_days=len(dates),
                  code_sha256=digest(Path(__file__)), inputs_sha256=digest(output / "inputs.json"),
                  events_sha256=digest(output / "events.parquet"), stock_dates_sha256=digest(output / "stock_dates.parquet"))
    atomic_json(output / "summary.json", report)
    return report
