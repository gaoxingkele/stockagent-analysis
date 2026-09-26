"""Semantic corporate-action audit. Unknown event terms never become zeros."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .runtime import atomic_json, digest, load_plan, now


def audit_events(frame):
    data = frame.copy().reset_index(drop=True)
    reasons = [[] for _ in range(len(data))]

    def flag(mask, reason):
        for index in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[index].append(reason)

    dates = {}
    for column in ("record_date", "ex_date", "pay_date", "div_listdate", "imp_ann_date", "ann_date"):
        dates[column] = pd.to_datetime(data[column].astype("string"), format="%Y%m%d", errors="coerce")
    cash = pd.to_numeric(data.cash_div_tax, errors="coerce")
    bonus = pd.to_numeric(data.stk_div, errors="coerce")
    flag(~np.isfinite(cash) | cash.lt(0), "cash_rate_unknown_or_invalid")
    flag(~np.isfinite(bonus) | bonus.lt(0), "bonus_rate_unknown_or_invalid")
    flag(data.div_proc.ne("实施"), "not_confirmed_implemented")
    flag(dates["record_date"].isna() | dates["ex_date"].isna(), "missing_record_or_ex_date")
    flag(dates["record_date"].ge(dates["ex_date"]), "record_not_before_ex_date")
    flag(dates["imp_ann_date"].isna(), "implementation_announcement_unknown")
    flag(dates["imp_ann_date"].gt(dates["record_date"]), "implementation_announcement_after_record")
    flag(cash.gt(0) & (dates["pay_date"].isna() | dates["pay_date"].lt(dates["ex_date"])), "payment_date_unknown_or_invalid")
    flag(bonus.gt(0) & (dates["div_listdate"].isna() | dates["div_listdate"].lt(dates["ex_date"])), "bonus_listing_unknown_or_invalid")
    flag(data.duplicated(["ts_code", "ex_date"], keep=False), "multiple_events_same_stock_ex_date")
    flag(data.ts_code.isna() | ~data.ts_code.astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)"), "invalid_code")
    data["audit_reasons"] = [";".join(items) for items in reasons]
    data["event_terms_usable_for_gross_reference_diagnostic"] = [not items for items in reasons]
    summary = {"rows": len(data), "term_checks_passed_rows": int(data.event_terms_usable_for_gross_reference_diagnostic.sum()),
               "reason_counts": pd.Series([reason for items in reasons for reason in items], dtype="string").value_counts().to_dict(),
               "cash_tax_policy_validated": False, "historical_feed_revision_policy_validated": False,
               "full_corporate_action_coverage_validated": False,
               "warning": "passing event checks permits gross reference diagnostics only; no net-profit/PIT/rights-issue claim"}
    return data, summary


def compare_reference_transitions(transitions, audited_events):
    """Conservative diagnostic for simple cash/bonus distributions only."""
    keys = ["ts_code", "ex_date"]
    counts = audited_events.groupby(keys, dropna=False).size().rename("event_count").reset_index()
    singles = audited_events.loc[~audited_events.duplicated(keys, keep=False),
        keys + ["cash_div_tax", "stk_div", "event_terms_usable_for_gross_reference_diagnostic"]]
    out = transitions.merge(counts, left_on=["ts_code", "trade_date"], right_on=keys, how="left")
    out = out.merge(singles, on=keys, how="left", validate="many_to_one")
    cash = pd.to_numeric(out.cash_div_tax, errors="coerce")
    bonus = pd.to_numeric(out.stk_div, errors="coerce")
    out["simple_distribution_reference"] = (out.previous_close - cash) / (1 + bonus)
    difference = (out.pre_close - out.simple_distribution_reference).abs()
    tolerance = np.maximum(.011, out.previous_close.abs() * .001)
    out["event_reference_status"] = np.select(
        [out.event_count.isna(), out.event_count.gt(1),
         ~out.event_terms_usable_for_gross_reference_diagnostic.eq(True), difference.le(tolerance)],
        ["no_matching_distribution", "multiple_events_unresolved", "event_terms_unresolved", "consistent_with_simple_gross_distribution"],
        default="distribution_does_not_explain_reference")
    return out


def run(root, source_id, factor_summary=None):
    if Path(source_id).name != source_id or not source_id.startswith("dividends-"):
        raise ValueError("invalid source id")
    base = root / "output/experiments/s20_safe_v4/sources" / source_id
    summary_path = base / "collection_summary.json"
    summary = load_plan(summary_path)
    if not summary["complete_acquisition"] or not summary["all_schema_valid"]:
        raise ValueError("full schema-valid collection required")
    plan = load_plan(base / "collection_plan.json")
    frames, inputs = [], []
    for date in plan["dates"]:
        path = base / f"{date}.parquet"
        receipt_path = base / f"{date}.json"
        receipt = load_plan(receipt_path)
        if not receipt["validation"]["valid"] or digest(path) != receipt["sha256"]:
            raise ValueError("event source invalid or mutated")
        frames.append(pd.read_parquet(path))
        inputs.append({"file": path.name, "sha256": receipt["sha256"], "receipt_sha256": digest(receipt_path)})
    events, audit = audit_events(pd.concat(frames, ignore_index=True))
    observed_codes = set()
    daily_inputs = []
    for path in sorted((root / "output/tushare_cache/daily").glob("*.parquet")):
        before = digest(path)
        observed_codes.update(pd.read_parquet(path, columns=["ts_code"]).ts_code.astype(str))
        if digest(path) != before:
            raise ValueError("daily source changed during scope audit")
        daily_inputs.append({"path": path.relative_to(root).as_posix(), "sha256": before})
    events["in_observed_daily_codes"] = events.ts_code.isin(observed_codes)
    audit["scope_diagnostics"] = {
        "observed_daily_code_events": int(events.in_observed_daily_codes.sum()),
        "outside_observed_daily_code_events": int((~events.in_observed_daily_codes).sum()),
        "in_scope_reason_counts": events.loc[events.in_observed_daily_codes, "audit_reasons"].str.split(";").explode().loc[lambda x: x.ne("")].value_counts().to_dict(),
        "not_an_ex_ante_universe_filter": True}
    output = base / ("audit-" + uuid.uuid4().hex)
    output.mkdir()
    events.to_parquet(output / "event_audit.parquet", index=False)
    reference_input = None
    if factor_summary is not None:
        factor_path = (root / factor_summary).resolve()
        if not factor_path.is_relative_to(root.resolve()):
            raise ValueError("reference report must be inside research repository")
        previous = load_plan(factor_path)
        transitions = pd.DataFrame(previous["unexplained_transitions"])
        if not transitions.empty:
            comparison = compare_reference_transitions(transitions, events)
            comparison.to_parquet(output / "unexplained_reference_comparison.parquet", index=False)
            audit["reference_comparison_counts"] = comparison.event_reference_status.value_counts().to_dict()
        reference_input = {"path": str(factor_path), "sha256": digest(factor_path)}
    audit.update({"source_id": source_id, "at": now(), "directory": str(output), "formal_H01_gate_passed": False})
    atomic_json(output / "summary.json", audit)
    atomic_json(output / "inputs.json", {"files": inputs, "code_hash": digest(Path(__file__)),
                "daily_scope_inputs": daily_inputs,
                "reference_report": reference_input,
                "collection_plan_hash": digest(base / "collection_plan.json"), "collection_summary_hash": digest(summary_path)})
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--factor-summary")
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[2], args.source_id, args.factor_summary), ensure_ascii=False, indent=2))
