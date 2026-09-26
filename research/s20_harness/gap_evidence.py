"""Expand original path gaps and attach retrospective suspension evidence."""
from pathlib import Path
import re
import uuid

import pandas as pd

from .runtime import atomic_json, digest, now


def join_gaps(anomalies, evidence, calendar):
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("unique ordered calendar required")
    positions = {d: i for i, d in enumerate(calendar)}
    rows = []
    for index, row in anomalies.loc[anomalies.issue.eq("missing_observed_sessions")].iterrows():
        match = re.fullmatch(r"previous=(\d{8});count=(\d+);not_proven_suspension", row.detail)
        if not match or match[1] not in positions or row.trade_date not in positions:
            raise ValueError("gap format or calendar endpoints invalid")
        missing = calendar[positions[match[1]] + 1:positions[row.trade_date]]
        if not missing or len(missing) != int(match[2]):
            raise ValueError("gap count disagrees with verified calendar")
        rows.extend({"anomaly_row": index, "ts_code": row.ts_code, "trade_date": d} for d in missing)
    expanded = pd.DataFrame(rows, columns=["anomaly_row", "ts_code", "trade_date"])
    if expanded.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("overlapping original gaps")
    required = {"ts_code", "trade_date", "status", "source_sha256", "quote_present"}
    if not required.issubset(evidence):
        raise ValueError("missing evidence fields")
    if evidence[list(required)].isna().any().any():
        raise ValueError("missing evidence values")
    if not evidence.quote_present.map(lambda v: isinstance(v, bool)).all():
        raise ValueError("quote presence must be boolean")
    allowed = {"missing_consistent_with_reported_suspension", "quote_conflicts_full_suspension"}
    if not evidence.status.isin(allowed).all() or not evidence.quote_present.eq(evidence.status.eq("quote_conflicts_full_suspension")).all():
        raise ValueError("evidence status contradicts quote observation")
    grouped = []
    for (code, date), group in evidence.groupby(["ts_code", "trade_date"], sort=False):
        grouped.append({"ts_code": code, "trade_date": date,
                        "evidence_state": "conflicting_quote" if group.quote_present.any() else "reported_suspension_consistent",
                        "source_hashes": sorted(set(group.source_sha256))})
    grouped = pd.DataFrame(grouped, columns=["ts_code", "trade_date", "evidence_state", "source_hashes"])
    joined = expanded.merge(grouped, how="left", on=["ts_code", "trade_date"], validate="one_to_one", sort=False)
    joined["evidence_state"] = joined.evidence_state.fillna("unresolved")
    original = set(zip(expanded.ts_code, expanded.trade_date))
    source_keys = set(zip(grouped.ts_code, grouped.trade_date))
    return joined, {"original_gap_events": int(anomalies.issue.eq("missing_observed_sessions").sum()),
                    "expanded_gap_stock_dates": len(expanded),
                    "states": joined.evidence_state.value_counts().to_dict(),
                    "evidence_stock_dates_outside_original_gaps": len(source_keys - original),
                    "original_anomalies_preserved": True, "formal_training_authorized": False,
                    "scope": "between-observation gaps only; not pre-first/post-last/delisted or complete universe coverage"}


def build(root, *, anomalies_path, anomalies_sha, evidence_path, evidence_sha, calendar_path, calendar_sha):
    root = Path(root).resolve()
    receipts = []
    def read(path, expected):
        path = Path(path)
        if digest(path) != expected:
            raise ValueError("input pin mismatch")
        frame = pd.read_parquet(path)
        if digest(path) != expected:
            raise ValueError("input changed")
        receipts.append({"path": str(path.resolve()), "sha256": expected})
        return frame
    anomalies, evidence, cal = read(anomalies_path, anomalies_sha), read(evidence_path, evidence_sha), read(calendar_path, calendar_sha)
    calendars = [sorted(cal.loc[cal.exchange.eq(ex) & cal.is_open.astype(int).eq(1), "cal_date"].astype(str)) for ex in ("SSE", "SZSE")]
    if calendars[0] != calendars[1]:
        raise ValueError("SH/SZ calendar mismatch")
    # Raw H01 includes BJ; keep it in a distinct unsupported slice.
    supported = anomalies.loc[anomalies.ts_code.str.endswith((".SH", ".SZ"))]
    joined, report = join_gaps(supported, evidence, calendars[0])
    output = root / "output/experiments/s20_safe_v4/sources" / ("gap-evidence-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    joined.to_parquet(output / "gap_stock_dates.parquet", index=False)
    atomic_json(output / "inputs.json", receipts)
    report.update(directory=str(output), at=now(), code_sha256=digest(Path(__file__)),
                  table_sha256=digest(output / "gap_stock_dates.parquet"), inputs_sha256=digest(output / "inputs.json"),
                  unsupported_market_gap_events=int((~anomalies.ts_code.str.endswith((".SH", ".SZ")) & anomalies.issue.eq("missing_observed_sessions")).sum()))
    atomic_json(output / "summary.json", report)
    return report
