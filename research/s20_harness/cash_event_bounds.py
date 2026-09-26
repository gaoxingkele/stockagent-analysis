"""Partial-identification diagnostic conditional on supplied cash-event terms.

Zero-to-declared-gross cash is a scenario envelope, not proof of entitlement,
units, missing-event coverage, or an approved replacement label.
"""
from __future__ import annotations

import math

import pandas as pd

from .labels import _invalid_quote_dates


def evaluate(raw, supplied_events, *, buy_cost=.001, sell_cost=.0015):
    if raw.empty or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("chronological nonempty window required")
    if _invalid_quote_dates(raw):
        return {"status": "unknown_quotes", "p_class": None}
    if not math.isfinite(buy_cost) or buy_cost < 0 or not math.isfinite(sell_cost) or not 0 <= sell_cost < 1:
        raise ValueError("invalid costs")
    start, end = str(raw.index[0]), str(raw.index[-1])
    dates = pd.to_datetime(supplied_events.record_date, format="%Y%m%d", errors="coerce")
    if dates.isna().any():
        return {"status": "unknown_record_date", "p_class": None}
    events = supplied_events.loc[supplied_events.record_date.between(start, end)]
    if events.empty:
        return {"status": "no_relevant_supplied_event", "p_class": None}
    valid = (events.event_terms_usable_for_gross_reference_diagnostic.eq(True) &
             events.conflicting_variants_same_identity.eq(False) & events.stk_div.eq(0) &
             events.cash_div_tax.gt(0) & events.cash_div_tax.map(math.isfinite))
    if not valid.all() or events.normalized_event_id.duplicated().any():
        return {"status": "unsupported_or_unresolved_event_terms", "p_class": None}
    if not events.record_date.isin(raw.index).all():
        return {"status": "unknown_record_holdings", "p_class": None}
    upper_cash = raw.close * 0.
    for event in events.itertuples():
        upper_cash.loc[upper_cash.index >= event.ex_date] += float(event.cash_div_tax)
    entry = float(raw.open.iloc[0]) * (1 + buy_cost)
    net_low = float(raw.close.iloc[-1]) * (1 - sell_cost) / entry - 1
    net_high = (float(raw.close.iloc[-1]) * (1 - sell_cost) + float(upper_cash.iloc[-1])) / entry - 1
    possible = bool(raw.low.min() < entry * .95)
    certain = bool((raw.low + upper_cash).min() < entry * .95)
    ups = [True] if net_low > 0 else [False] if net_high <= 0 else [False, True]
    risks = [True] if certain else [False] if not possible else [False, True]
    classes = sorted({("B" if risk else "A") if up else ("D" if risk else "C") for up in ups for risk in risks})
    return {"status": "conditional_cash_envelope", "p_class": classes[0] if len(classes) == 1 else None,
            "class_outer_set": classes, "terminal_net_lower": net_low, "terminal_net_upper": net_high,
            "b5_certain": certain, "b5_possible": possible, "declared_terminal_cash_upper": float(upper_cash.iloc[-1]),
            "event_ids": events.normalized_event_id.tolist(), "formal_training_eligible": False,
            "assumptions": ["only supplied ordinary cash events affect wealth", "declared cash units/rates are correct",
                            "net cash entitlement between zero and declared gross", "reference entry and exit"],
            "event_coverage_proven": False, "beneficiary_verified": False}


def study(root, partition, summary_sha256):
    import json
    from pathlib import Path
    import uuid
    from .label_partition_verify import verify
    from .labels import _align_window
    from .runtime import load_plan, digest, atomic_json, now

    root, partition = Path(root).resolve(), Path(partition).resolve()
    labels, verification = verify(partition, summary_sha256)
    inputs = load_plan(partition / "inputs.json")
    paths = [Path(name) for name in inputs]
    normalized = [p for p in paths if p.name == "normalized_distributions.parquet"]
    if len(normalized) != 1:
        raise ValueError("exactly one pinned normalized source required")
    distributions = pd.read_parquet(normalized[0])
    quotes = pd.concat([pd.read_parquet(p) for p in paths if p.suffix == ".parquet" and p.parent.name == "daily"])
    if quotes.duplicated(["entity_id", "trade_date"]).any():
        raise ValueError("duplicate quote identity")
    candidates = pd.read_parquet(partition / "candidates.parquet").set_index("sample_id")
    groups = {key: group for key, group in quotes.groupby("entity_id", sort=False)}
    dates = sorted(quotes.trade_date.astype(str).unique())
    rows = []
    for label in labels.itertuples():
        result = {"status": "not_unknown_event_candidate", "p_class": None}
        if label.label_status == "unknown_event_terms":
            codes = candidates.loc[label.sample_id, "event_codes"]
            events = distributions.loc[distributions.ts_code.isin(codes)]
            window = [d for d in dates if label.entry_date <= d <= label.horizon_end]
            if len(window) != 20:
                raise ValueError("incomplete market-date window")
            raw = _align_window(groups[label.entity_id], window)
            result = evaluate(raw, events)
        rows.append({"sample_id": label.sample_id, "original_p_class": label.p_class,
                     "status": result["status"], "conditional_p_class": result["p_class"],
                     "payload_json": json.dumps(result, sort_keys=True), "formal_training_eligible": False})
    output = root / "output/experiments/s20_safe_v4/sources" / ("cash-bounds-" + uuid.uuid4().hex)
    if not output.resolve().is_relative_to(root):
        raise ValueError("output escapes repository")
    output.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "bounds.parquet", index=False)
    report = {"at": now(), "directory": str(output), "rows": len(frame),
              "status_counts": frame.status.value_counts().to_dict(),
              "conditional_class_counts": frame.conditional_p_class.fillna("UNKNOWN_OR_NOT_STUDIED").value_counts().to_dict(),
              "source_partition": str(partition), "source_summary_sha256": summary_sha256,
              "source_verification": verification, "bounds_sha256": digest(output / "bounds.parquet"),
              "code_sha256": digest(Path(__file__)), "formal_training_authorized": False,
              "original_labels_unchanged": True}
    atomic_json(output / "summary.json", report)
    return report
