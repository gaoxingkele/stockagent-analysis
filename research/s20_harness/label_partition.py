"""Persist a complete observed SH/SZ signal-day diagnostic label partition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import pandas as pd

from .distribution_adapter import adapt
from .label_batch import materialize
from .runtime import atomic_json, digest, load_plan, now


def event_codes(entity, trading_code, aliases):
    matches = [a for a in aliases if a["entity_id"] == entity]
    if len(matches) > 1:
        raise ValueError("multiple alias definitions for entity")
    if not matches:
        return [trading_code]
    alias = matches[0]
    codes = sorted({alias["old_code"], alias["new_code"]})
    if alias["share_conversion_ratio"] != 1 or trading_code not in codes:
        raise ValueError("inconsistent identity event scope")
    return codes


def review_input(root, review_path=None, review_sha256=None):
    if (review_path is None) != (review_sha256 is None):
        raise ValueError("custom review path and expected hash required together")
    path = Path(review_path).resolve() if review_path is not None else Path(root).resolve() / "config/s20_v4_distribution_reviews.json"
    if review_sha256 is not None and digest(path) != review_sha256:
        raise ValueError("custom review input hash mismatch")
    return path, review_sha256


def build(root, signal_date, normalized_path, normalized_sha256, *, tax_bounds=False,
          review_path=None, review_sha256=None):
    root = Path(root).resolve()
    selected_review, expected_review = review_input(root, review_path, review_sha256)
    sources = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and expected != value:
            raise ValueError("input hash mismatch: " + str(path))
        if str(path) in sources and sources[str(path)] != value:
            raise ValueError("input changed during build: " + str(path))
        sources[str(path)] = value
        return path

    inventory = load_plan(pin(root / "config/s20_v4_data_sources.json"))
    panel_spec = next(s for s in inventory["sources"] if s["role"] == "identity_aware_daily_panel")
    panel = root / panel_spec["path"]
    summary = load_plan(pin(panel / "summary.json", panel_spec["summary_sha256"]))
    receipts = json.loads(pin(panel / "inputs_outputs.json", summary["receipt_sha256"]).read_text(encoding="utf-8"))
    aliases = load_plan(pin(root / "config/s20_v4_security_aliases.json", summary["config_sha256"]))["aliases"]
    cal_spec = next(s for s in inventory["sources"] if s["role"] == "exchange_calendar" and s.get("sha256"))
    cal = pd.read_parquet(pin(root / cal_spec["path"], cal_spec["sha256"]))
    dates = sorted(cal.loc[cal.exchange.eq("SSE") & cal.is_open.eq(1), "cal_date"].astype(str).unique())
    sz_dates = sorted(cal.loc[cal.exchange.eq("SZSE") & cal.is_open.eq(1), "cal_date"].astype(str).unique())
    if dates != sz_dates:
        raise ValueError("SH/SZ calendar mismatch requires market-specific partitions")
    start = dates.index(signal_date)
    window = dates[start:start + 21]
    if len(window) != 21:
        raise ValueError("complete label horizon not available")
    frames, observed = [], set()
    for receipt in receipts:
        path = panel / receipt["canonical"]
        if not path.resolve().is_relative_to(panel.resolve()):
            raise ValueError("partition path escapes panel")
        if path.stem in window:
            if path.stem in observed:
                raise ValueError("duplicate date in panel receipts")
            observed.add(path.stem)
            frame = pd.read_parquet(pin(path, receipt["canonical_sha256"]))
            if not frame.trade_date.astype(str).eq(path.stem).all():
                raise ValueError("partition date mismatch")
            frames.append(frame)
    if observed != set(window):
        raise ValueError("missing market-date partition")
    quotes = pd.concat(frames, ignore_index=True)
    candidates = quotes.loc[quotes.trade_date.astype(str).eq(signal_date) &
                            quotes.trading_code.str.endswith((".SH", ".SZ")),
                            ["entity_id", "trading_code"]].copy()
    candidates["sample_id"] = signal_date + ":" + candidates.entity_id
    candidates["signal_date"] = signal_date
    candidates["event_codes"] = [event_codes(r.entity_id, r.trading_code, aliases) for r in candidates.itertuples()]
    path = pin(normalized_path, normalized_sha256)
    normalization = load_plan(pin(path.parent / "summary.json"))
    if normalization["output_sha256"] != normalized_sha256:
        raise ValueError("normalization manifest mismatch")
    distributions = pd.read_parquet(path)
    reviews = load_plan(pin(selected_review, expected_review))["reviews"]
    unit_reviews = load_plan(pin(root / "config/s20_v4_cash_unit_reviews.json"))["reviews"]
    unit_cases = load_plan(pin(root / "config/s20_v4_cash_unit_cases.json"))["cases"]
    instrument_overrides = {case["ts_code"]: case["instrument_type"] for case in unit_cases if case.get("instrument_type")}
    # Preserve this materializer and all local harness dependencies as code blobs.
    for code in sorted(Path(__file__).parent.glob("*.py")):
        pin(code)
    pin(root / "src/stockagent_analysis/s20_v3.py")
    output = root / "output/experiments/s20_safe_v4/sources" / ("labels-" + uuid.uuid4().hex)
    if not output.resolve().is_relative_to(root):
        raise ValueError("output escapes repository")
    output.mkdir(parents=True)
    atomic_json(output / "state.json", {"state": "RUNNING", "at": now(), "signal_date": signal_date})
    try:
        events, decisions = adapt(distributions, reviews, rate_policy="gross_reference_diagnostic", unit_reviews=unit_reviews)
        contexts = None
        if tax_bounds:
            contexts = {r.sample_id: {"acquisition_date": window[1], "transfer_settlement_date": None,
                        "investor_scope": "personal_public_market_unrestricted_single_lot",
                        "instrument_type": instrument_overrides.get(r.trading_code, "ordinary_share"),
                        "market": r.trading_code[-2:], "cash_rate_basis": "gross_per_share"}
                        for r in candidates.itertuples()}
        result, report = materialize(candidates, quotes, dates, distributions, events, decisions,
                                     tax_contexts=contexts)
        report["tax_context_basis"] = "conditional reference entry, unknown transfer settlement; not actual lot evidence" if tax_bounds else None
        report["instrument_scope_basis"] = "reviewed exceptional-instrument overrides; default ordinary-share assumption is not a complete security master"
        for name, expected in sources.items():
            if digest(Path(name)) != expected:
                raise ValueError("source changed before publication: " + name)
        candidates.to_parquet(output / "candidates.parquet", index=False)
        result.to_parquet(output / "labels.parquet", index=False)
        if pd.read_parquet(output / "labels.parquet").sample_id.tolist() != candidates.sample_id.tolist():
            raise ValueError("persisted denominator mismatch")
        blobs = output / "code_blobs"
        blobs.mkdir()
        for name, expected in sources.items():
            if Path(name).suffix == ".py":
                content = Path(name).read_bytes()
                import hashlib
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("code changed during snapshot")
                (blobs / expected).write_bytes(content)
        atomic_json(output / "inputs.json", sources)
        report.update({"at": now(), "directory": str(output), "signal_date": signal_date,
                       "review_path": str(selected_review), "review_sha256": sources[str(selected_review)],
                       "candidate_scope": "observed canonical SH/SZ quotes on signal date, not verified PIT universe",
                       "alias_event_codes_are_accounting_keys_not_predictive_features": True,
                       "status_counts": result.label_status.value_counts().to_dict(),
                       "artifact_hashes": {name: digest(output / name) for name in
                                           ("candidates.parquet", "labels.parquet", "inputs.json")}})
        atomic_json(output / "summary.json", report)
        atomic_json(output / "state.json", {"state": "COMPLETED_DIAGNOSTIC", "at": now(),
                                             "summary_sha256": digest(output / "summary.json")})
        return report
    except Exception as exc:
        atomic_json(output / "state.json", {"state": "FAILED", "at": now(), "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--normalized-path", type=Path, required=True)
    parser.add_argument("--normalized-sha256", required=True)
    parser.add_argument("--tax-bounds", action="store_true")
    parser.add_argument("--review-path", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2], args.signal_date,
                           args.normalized_path, args.normalized_sha256, tax_bounds=args.tax_bounds,
                           review_path=args.review_path, review_sha256=args.review_sha256), indent=2))
