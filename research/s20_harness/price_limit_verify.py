"""Reconstruct complete acquired limit coverage; exceptions are not no-limit permission."""
from pathlib import Path
import json

import pandas as pd

from . import price_limit_audit
from .runtime import digest, load_plan


def verify(root, directory, summary_sha, expected_source):
    root, directory, source = Path(root).resolve(), Path(directory).resolve(), Path(expected_source).resolve()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("limit evidence escapes root")
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError("limit evidence hash mismatch")
        if path in pins and pins[path] != sha:
            raise ValueError("limit evidence changed")
        pins[path] = sha
        return path

    summary = load_plan(pin(directory / "summary.json", summary_sha))
    if Path(summary["source"]).resolve() != source:
        raise ValueError("limit source mismatch")
    code = Path(price_limit_audit.__file__).resolve()
    if digest(code) != summary["code_sha256"]:
        raise ValueError("limit audit code changed")
    pins[code] = summary["code_sha256"]
    inputs = json.loads(pin(directory / "inputs.json", summary["inputs_sha256"]).read_text(encoding="utf-8"))
    saved_diagnostics = json.loads(pin(directory / "date_diagnostics.json").read_text(encoding="utf-8"))
    plan = load_plan(pin(source / "collection_plan.json", summary["collection_plan_sha256"]))
    if plan["api"] != "stk_limit" or not plan["daily"]:
        raise ValueError("complete stk_limit acquisition required")
    days = plan["daily"]
    dates = [d["date"] for d in days]
    if dates != sorted(set(dates)) or len(inputs) != len(days):
        raise ValueError("invalid limit date inventory")
    daily_files = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
    if [p.stem for p in daily_files] != dates or {p.name for p in source.glob("*.parquet")} != {d + ".parquet" for d in dates}:
        raise ValueError("limit or daily file inventory changed")
    diagnostics = []
    for day, binding, daily in zip(days, inputs, daily_files):
        if (Path(binding["daily_path"]).resolve() != daily.resolve()
                or (root / day["path"]).resolve() != daily.resolve()
                or binding["daily_sha256"] != day["sha256"]):
            raise ValueError("limit daily binding mismatch")
        codes = set(pd.read_parquet(pin(daily, day["sha256"]), columns=["ts_code"]).ts_code.astype(str))
        date = day["date"]
        receipt = load_plan(pin(source / (date + ".json"), binding["receipt_sha256"]))
        path = source / (date + ".parquet")
        if (Path(binding["limit_path"]).resolve() != path.resolve() or receipt["date"] != date
                or receipt["sha256"] != binding["limit_sha256"] or receipt["possible_truncation"] is not False):
            raise ValueError("limit receipt binding mismatch")
        requested, received = pd.Timestamp(receipt["requested_at"]), pd.Timestamp(receipt["received_at"])
        if (pd.isna(requested) or pd.isna(received) or requested.tzinfo is None
                or received.tzinfo is None or received < requested
                or any(receipt[k] != binding[k] for k in ("requested_at", "received_at"))):
            raise ValueError("limit receipt timing invalid")
        frame = pd.read_parquet(pin(path, receipt["sha256"]))
        check = price_limit_audit.inspect_limits(frame, date, codes)
        if len(frame) >= plan["row_cap"] or check != receipt["validation"]:
            raise ValueError("limit receipt validation mismatch")
        diagnostics.append(dict(check, status="inspected"))
    if diagnostics != saved_diagnostics:
        raise ValueError("limit reconstructed date diagnostics mismatch")
    rebuilt = {"daily_dates": len(days), "limit_files": len(days), "missing_dates": [], "extra_dates": [],
               "acquisition_receipts_verified": True,
               "expected_stock_dates": sum(d["expected_codes"] for d in diagnostics),
               "usable_stock_dates": sum(d["usable_expected_codes"] for d in diagnostics)}
    for key in ("invalid_limit_rows", "duplicate_code_rows", "wrong_date_rows"):
        rebuilt[key] = sum(d[key] for d in diagnostics)
    if any(summary[k] != v for k, v in rebuilt.items()):
        raise ValueError("limit reconstructed summary mismatch")
    for path, sha in pins.items():
        if digest(path) != sha:
            raise ValueError("limit evidence changed during reconstruction")
    return dict(rebuilt, full_coverage_reconstructed=True, effective_rule_semantics_proven=False,
                historical_availability_proven=False, auction_fill_proven=False, formal_H01_gate_passed=False,
                evidence_files=[{"path": str(p), "sha256": sha, "role": "fresh_limit_coverage_reconstruction"}
                                for p, sha in pins.items()])
