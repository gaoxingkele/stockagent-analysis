"""Missing listing-edge sessions outside the original between-quote gap audit."""
from pathlib import Path
import json
import uuid

import pandas as pd

from .daily_gap_join import overlay
from .runtime import atomic_json, digest, now


def expand(coverage, calendar):
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("ordered unique calendar required")
    if coverage.ts_code.isna().any() or coverage.ts_code.duplicated().any():
        raise ValueError("unique coverage codes required")
    rows, unknown = [], []
    for c in coverage.itertuples():
        if pd.isna(c.expected_listing_days):
            unknown.append(c.ts_code)
            continue
        start, end = c.list_date, c.delist_date
        end = None if pd.isna(end) or end == "" else str(end)
        dates = [d for d in calendar if d >= str(start) and (end is None or d < end)]
        if len(dates) != c.expected_listing_days:
            raise ValueError("calendar/listing count disagreement")
        first, last = c.observed_first, c.observed_last
        if c.observed_days == 0:
            if not pd.isna(first) or not pd.isna(last):
                raise ValueError("empty observation bounds inconsistent")
            missing = [(d, "entire_listing_window_unobserved") for d in dates]
        else:
            if pd.isna(first) or pd.isna(last) or first > last or first not in calendar or last not in calendar:
                raise ValueError("invalid observation bounds")
            missing = [(d, "before_first_quote" if d < first else "after_last_quote")
                       for d in dates if d < first or d > last]
        if len(missing) > c.unobserved_listing_days:
            raise ValueError("edges exceed all missing listing sessions")
        rows.extend(dict(ts_code=c.ts_code, trade_date=d, edge_kind=kind) for d, kind in missing)
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "edge_kind"]), unknown


def build(root, coverage_path, coverage_sha, calendar_path, calendar_sha, daily_path, daily_sha, *, audit_summary_sha):
    pins = [(Path(coverage_path), coverage_sha), (Path(calendar_path), calendar_sha), (Path(daily_path), daily_sha)]
    summary_path = Path(daily_path).parent / "summary.json"
    if digest(summary_path) != audit_summary_sha:
        raise ValueError("daily audit pin mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inputs_path = summary_path.parent / "inputs.json"
    pins += [(summary_path, audit_summary_sha), (inputs_path, summary["inputs_sha256"])]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input pin mismatch")
    coverage, cal, daily = [pd.read_parquet(p) for p, _ in pins[:3]]
    receipts = json.loads(inputs_path.read_text(encoding="utf-8"))
    committed = {r["date"]: r["receipt_sha256"] for r in receipts}
    if summary["rows_sha256"] != daily_sha or len(daily) != summary["rows"] or len(receipts) != summary["committed_dates"] or len(committed) != len(receipts):
        raise ValueError("daily audit count/hash mismatch")
    if not daily.receipt_sha256.eq(daily.trade_date.map(committed)).all():
        raise ValueError("daily receipt lineage mismatch")
    calendars = [sorted(cal.loc[cal.exchange.eq(ex) & cal.is_open.astype(int).eq(1), "cal_date"].astype(str)) for ex in ["SSE", "SZSE"]]
    if calendars[0] != calendars[1]:
        raise ValueError("exchange calendars disagree")
    edges, unknown = expand(coverage, calendars[0])
    # Receipt coverage, not rows or calendar membership, establishes querying.
    joined, _ = overlay(edges, daily, list(committed))
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input changed during edge audit")
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("listing-edges-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    joined.to_parquet(out / "edges.parquet", index=False)
    atomic_json(out / "inputs.json", [{"path": str(p.resolve()), "sha256": sha} for p, sha in pins])
    report = dict(directory=str(out), at=now(), edge_stock_dates=len(joined), edge_codes=int(joined.ts_code.nunique()),
                  edge_kinds=joined.edge_kind.value_counts().to_dict(), daily_states=joined.daily_evidence_state.value_counts().to_dict(),
                  unknown_listing_codes=unknown, formal_training_authorized=False,
                  scope="current metadata listing edges, raw codes; no alias merging or PIT universe assertion",
                  table_sha256=digest(out / "edges.parquet"), inputs_sha256=digest(out / "inputs.json"), code_sha256=digest(Path(__file__)))
    atomic_json(out / "summary.json", report)
    return report
