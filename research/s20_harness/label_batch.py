"""Denominator-preserving P-label materialization; diagnostic until H02 passes."""
from __future__ import annotations

import json

import pandas as pd

from .event_index import EventIndex
from .execution import market_horizon
from .labels import label_p_track


def materialize(candidates, quotes, calendar, distributions, events, decisions, *, tax_contexts=None):
    required = {"sample_id", "entity_id", "signal_date", "event_codes"}
    if not required.issubset(candidates.columns):
        raise ValueError("missing candidate contract fields")
    if candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError("duplicate/missing candidate sample identity")
    if not {"entity_id", "trade_date", "open", "high", "low", "close"}.issubset(quotes.columns):
        raise ValueError("missing quote contract fields")
    if quotes.duplicated(["entity_id", "trade_date"]).any():
        raise ValueError("duplicate entity/date quotes")
    groups = {key: group.assign(ts_code=key) for key, group in quotes.groupby("entity_id", sort=False)}
    empty = quotes.iloc[:0].assign(ts_code=pd.Series(dtype=str))
    event_index = EventIndex(distributions, events, decisions)
    results = []
    for candidate in candidates.itertuples(index=False):
        if not isinstance(candidate.event_codes, (list, tuple)) or not candidate.event_codes:
            raise ValueError("event codes must explicitly cover the candidate identity")
        horizon = market_horizon(calendar, str(candidate.signal_date), horizon=20)
        selected = event_index.window(candidate.event_codes, horizon["entry_date"], horizon["horizon_end"])
        if selected["unresolved_event_ids"]:
            payload = {"p_class": None, "label_realized": False, "label_status": "unknown_event_terms",
                       "unresolved_event_ids": selected["unresolved_event_ids"]}
        else:
            payload = label_p_track(groups.get(candidate.entity_id, empty), calendar,
                                    str(candidate.signal_date), candidate.entity_id,
                                    distributions=selected["events"])
        row = {"sample_id": candidate.sample_id, "entity_id": candidate.entity_id,
                        "signal_date": str(candidate.signal_date), "entry_date": horizon["entry_date"],
                        "horizon_end": horizon["horizon_end"], "p_class": payload.get("p_class"),
                        "label_realized": payload.get("label_realized", False),
                        "label_status": payload.get("label_status", payload.get("fill_status", "unknown")),
                        "recommendation_kept": True, "formal_training_eligible": False,
                        "factor_usage": "none_raw_economic",
                        "event_coverage_proven": False, "payload_json": json.dumps(payload, sort_keys=True)}
        if tax_contexts is not None:
            taxed = {"p_class": None, "formal_training_eligible": False}
            if not payload.get("label_realized"):
                taxed["status"] = "upstream_outcome_unresolved"
            elif candidate.sample_id not in tax_contexts:
                taxed["status"] = "tax_context_missing"
            elif any(event.bonus_per_share for event in selected["events"]):
                taxed["status"] = "bonus_tax_policy_unsupported"
            else:
                from .taxed_window import safe_profit_bounds
                from .labels import _align_window
                raw = _align_window(groups[candidate.entity_id], horizon["window"])
                try:
                    taxed = safe_profit_bounds(raw, selected["events"], **tax_contexts[candidate.sample_id])
                    taxed["status"] = "tax_bounds_diagnostic"
                except ValueError as exc:
                    taxed.update(status="tax_context_or_terms_invalid", reason=str(exc))
            row.update(taxed_p_class=taxed.get("p_class"), taxed_status=taxed["status"],
                       taxed_payload_json=json.dumps(taxed, sort_keys=True))
        results.append(row)
    output = pd.DataFrame(results)
    return output, {"candidate_rows": len(candidates), "output_rows": len(output),
                    "candidate_order_and_denominator_preserved": output.sample_id.tolist() == candidates.sample_id.tolist() if len(output) else len(candidates) == 0,
                    "class_counts": output.p_class.fillna("UNKNOWN").value_counts().to_dict() if len(output) else {},
                    "formal_training_authorized": False,
                    "tax_bounds_requested": tax_contexts is not None,
                    "taxed_status_counts": output.taxed_status.value_counts().to_dict() if tax_contexts is not None and len(output) else {},
                    "limitations": ["supplied event coverage unverified", "reference fills not tradability proof",
                                    "label availability not established", "not a full H02 stage"]}
