"""Retrospective listing coverage audit; never an ex-ante universe filter."""
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, now


def listing_coverage(basic, observed, calendar):
    required = {"ts_code", "list_status", "list_date", "delist_date"}
    if not required.issubset(basic) or not {"ts_code", "trade_date"}.issubset(observed):
        raise ValueError("missing listing or observation fields")
    if basic.ts_code.isna().any() or basic.ts_code.duplicated().any():
        raise ValueError("unique metadata codes required")
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("nonempty unique ordered calendar required")
    for date in calendar:
        pd.to_datetime(date, format="%Y%m%d", errors="raise")
    if observed[["ts_code", "trade_date"]].isna().any().any():
        raise ValueError("missing observed identity/date")
    if not set(observed.trade_date).issubset(calendar):
        raise ValueError("observation outside calendar")
    if observed.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate observed code/date")
    by_code = {code: set(rows.trade_date) for code, rows in observed.groupby("ts_code")}
    rows = []
    for record in basic.to_dict("records"):
        code = record["ts_code"]
        dates = by_code.get(code, set())
        row = {**record, "observed_days": len(dates), "observed_first": min(dates) if dates else None,
               "observed_last": max(dates) if dates else None,
               "expected_listing_days": None, "unobserved_listing_days": None,
               "outside_listing_days": None, "status": "unknown_listing_interval"}
        start, end = record["list_date"], record["delist_date"]
        start = None if pd.isna(start) or start == "" else str(start)
        end = None if pd.isna(end) or end == "" else str(end)
        if start is not None and not (record["list_status"] == "D" and end is None):
            pd.to_datetime(start, format="%Y%m%d", errors="raise")
            if end is not None:
                pd.to_datetime(end, format="%Y%m%d", errors="raise")
                if end < start:
                    raise ValueError("reversed listing interval")
            # Delisting date is an exclusive boundary; status-specific last
            # tradable day still needs separate evidence. Missing != suspended.
            expected = {date for date in calendar if date >= start and (end is None or date < end)}
            row.update(expected_listing_days=len(expected),
                       unobserved_listing_days=len(expected - dates),
                       outside_listing_days=len(dates - expected),
                       status=("outside_audit_window" if not expected else
                               "no_observations_in_listing_window" if not dates & expected else
                               "observed_with_gaps" if expected - dates else "observed_all_listing_days"))
        rows.append(row)
    result = pd.DataFrame(rows)
    return result, {"metadata_rows": len(basic), "observed_codes": len(by_code),
                    "observed_codes_missing_metadata": sorted(set(by_code) - set(basic.ts_code)),
                    "status_counts": result.status.value_counts().to_dict(),
                    "formal_universe_verified": False,
                    "interpretation": "retrospective current metadata; gaps are not proven suspensions or data errors"}


def build(root, basic_path, calendar_path):
    root, basic_path, calendar_path = map(lambda p: Path(p).resolve(), (root, basic_path, calendar_path))
    receipts = []

    def read(path, columns=None):
        sha = digest(path)
        frame = pd.read_parquet(path, columns=columns)
        if digest(path) != sha:
            raise ValueError("source changed while reading")
        receipts.append({"path": str(path), "sha256": sha, "rows": len(frame)})
        return frame

    basic = read(basic_path)
    cal = read(calendar_path)
    calendars = [sorted(cal.loc[cal.exchange.eq(ex) & cal.is_open.astype(int).eq(1), "cal_date"].astype(str))
                 for ex in ("SSE", "SZSE")]
    if calendars[0] != calendars[1]:
        raise ValueError("SH/SZ calendars differ")
    files = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
    if not files:
        raise ValueError("no daily partitions")
    frames = []
    for path in files:
        frame = read(path, ["ts_code", "trade_date"])
        frame["trade_date"] = frame.trade_date.astype(str)
        if not frame.trade_date.eq(path.stem).all():
            raise ValueError("daily filename mismatch")
        frames.append(frame.loc[frame.ts_code.str.endswith((".SH", ".SZ"))])
    dates = [d for d in calendars[0] if files[0].stem <= d <= files[-1].stem]
    if dates != [p.stem for p in files]:
        raise ValueError("daily partitions do not cover verified SH/SZ calendar")
    table, summary = listing_coverage(basic.loc[basic.ts_code.str.endswith((".SH", ".SZ"))],
                                      pd.concat(frames, ignore_index=True), dates)
    output = root / "output/experiments/s20_safe_v4/sources" / ("universe-coverage-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    table.to_parquet(output / "listing_coverage.parquet", index=False)
    atomic_json(output / "inputs.json", receipts)
    summary.update(at=now(), directory=str(output), calendar_days=len(dates),
                   daily_files=len(files), code_sha256=digest(Path(__file__)),
                   table_sha256=digest(output / "listing_coverage.parquet"),
                   inputs_sha256=digest(output / "inputs.json"),
                   scope="SH/SZ only; code aliases not merged; metadata snapshot is not exhaustive master")
    atomic_json(output / "summary.json", summary)
    return summary
