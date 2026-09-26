"""Declared cash vs ex-reference consistency; never substitutes investor cash."""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, load_plan, now


def compare(events, quotes, calendar):
    if quotes.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate stock/date quote keys")
    if events.normalized_event_id.isna().any() or events.normalized_event_id.duplicated().any():
        raise ValueError("duplicate/missing normalized event identity")
    indexed = quotes.set_index(["ts_code", "trade_date"])
    previous = dict(zip(calendar[1:], calendar[:-1]))
    rows = []
    for e in events.itertuples():
        record = {"event_id": e.normalized_event_id, "ts_code": e.ts_code, "record_date": e.record_date,
                  "ex_date": e.ex_date, "declared_gross_cash": e.cash_div_tax,
                  "status": "unsupported_terms_or_market", "formal_eligible": False}
        supported = (e.ts_code.endswith((".SH", ".SZ")) and
                     e.event_terms_usable_for_gross_reference_diagnostic == True and
                     e.conflicting_variants_same_identity == False and e.stk_div == 0 and
                     pd.notna(e.cash_div_tax) and math.isfinite(e.cash_div_tax) and e.cash_div_tax > 0)
        if supported:
            if previous.get(e.ex_date) != e.record_date:
                record["status"] = "record_ex_not_adjacent_market_dates"
            elif (e.ts_code, e.record_date) not in indexed.index or (e.ts_code, e.ex_date) not in indexed.index:
                record["status"] = "missing_quote"
            else:
                close = float(indexed.loc[(e.ts_code, e.record_date), "close"])
                reference = float(indexed.loc[(e.ts_code, e.ex_date), "pre_close"])
                if not all(math.isfinite(x) and x > 0 for x in (close, reference)):
                    record["status"] = "invalid_quote"
                else:
                    reduction = close - reference
                    # Fixed quote-rounding allowance, not a fitted criterion.
                    scales = [scale for scale in (.1, 1., 10.) if abs(reduction - e.cash_div_tax * scale) <= .011]
                    record.update(reference_reduction=reduction, matching_cash_scales=scales,
                                  tolerance=.011, observed_record_close=close, ex_pre_close=reference)
                    record["status"] = ("consistent_declared_rate" if scales == [1.] else
                                        "rounding_scale_ambiguous" if len(scales) > 1 else
                                        "possible_unit_scale_mismatch" if scales else "reference_difference_needs_explanation")
        rows.append(record)
    return pd.DataFrame(rows)


def run(root, normalized_path, expected_hash):
    root, normalized_path = Path(root).resolve(), Path(normalized_path).resolve()
    if digest(normalized_path) != expected_hash:
        raise ValueError("normalized input hash mismatch")
    events = pd.read_parquet(normalized_path)
    inventory_path = root / "config/s20_v4_data_sources.json"
    inventory_hash = digest(inventory_path)
    inventory = load_plan(inventory_path)
    spec = next(s for s in inventory["sources"] if s["role"] == "exchange_calendar" and s.get("sha256"))
    calendar_path = root / spec["path"]
    if digest(calendar_path) != spec["sha256"]:
        raise ValueError("calendar hash mismatch")
    cal = pd.read_parquet(calendar_path)
    calendar = sorted(cal.loc[cal.exchange.eq("SSE") & cal.is_open.eq(1), "cal_date"].astype(str))
    sz = sorted(cal.loc[cal.exchange.eq("SZSE") & cal.is_open.eq(1), "cal_date"].astype(str))
    if calendar != sz:
        raise ValueError("market calendars differ")
    frames, inputs = [], {str(normalized_path): expected_hash, str(calendar_path): spec["sha256"],
                          str(inventory_path): inventory_hash}
    wanted = set(events.record_date.dropna()) | set(events.ex_date.dropna())
    for path in sorted((root / "output/tushare_cache/daily").glob("*.parquet")):
        if path.stem not in wanted:
            continue
        sha = digest(path)
        frame = pd.read_parquet(path, columns=["ts_code", "trade_date", "close", "pre_close"])
        if digest(path) != sha:
            raise ValueError("quotes changed during audit")
        inputs[str(path)] = sha
        frames.append(frame)
    result = compare(events, pd.concat(frames, ignore_index=True), calendar)
    for name, sha in inputs.items():
        if digest(Path(name)) != sha:
            raise ValueError("input changed before audit publication")
    output = root / "output/experiments/s20_safe_v4/sources" / ("cash-rates-" + uuid.uuid4().hex)
    if not output.resolve().is_relative_to(root):
        raise ValueError("output escapes repository")
    output.mkdir(parents=True)
    result.to_parquet(output / "cash_rate_checks.parquet", index=False)
    atomic_json(output / "inputs.json", inputs)
    report = {"at": now(), "directory": str(output), "event_rows": len(result),
              "status_counts": result.status.value_counts().to_dict(), "code_sha256": digest(Path(__file__)),
              "artifacts": {name: digest(output / name) for name in ("cash_rate_checks.parquet", "inputs.json")},
              "formal_training_authorized": False,
              "interpretation": "reference consistency is corroboration, not beneficiary proof; repurchase exclusions can legitimately differ"}
    atomic_json(output / "summary.json", report)
    return report
