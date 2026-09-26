"""Fresh factor-ratio reconstruction, never a cash-ledger or PIT certificate."""
from pathlib import Path

import pandas as pd

from . import adjustment_reconcile
from .adjustment_source import validate_factors
from .runtime import digest, load_plan


def verify(root, directory, summary_sha):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    source = directory.parent
    pins = {}

    def pin(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("adjustment evidence escapes root")
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError("adjustment evidence hash mismatch")
        if path in pins and pins[path] != sha:
            raise ValueError("adjustment evidence changed")
        pins[path] = sha
        return path

    summary = load_plan(pin(directory / "summary.json", summary_sha))
    inputs = load_plan(pin(directory / "inputs.json"))
    code = Path(adjustment_reconcile.__file__).resolve()
    if digest(code) != inputs["code_hash"]:
        raise ValueError("adjustment reconstruction code changed")
    pins[code] = inputs["code_hash"]
    plan = load_plan(pin(source / "collection_plan.json"))
    if summary["source_id"] != source.name or plan["source_id"] != source.name:
        raise ValueError("adjustment source identity mismatch")
    days = plan["daily"]
    if not days or len({x["date"] for x in days}) != len(days):
        raise ValueError("empty or duplicate adjustment dates")
    if [x["daily"] for x in inputs["files"]] != days:
        raise ValueError("adjustment input inventory mismatch")
    actual = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
    if {p.resolve() for p in actual} != {(root / x["path"]).resolve() for x in days}:
        raise ValueError("adjustment daily coverage changed")
    frames, missing = [], 0
    for day, binding in zip(days, inputs["files"]):
        daily = pin(root / day["path"], day["sha256"])
        if daily.stem != day["date"]:
            raise ValueError("adjustment daily date mismatch")
        receipt = load_plan(pin(source / (day["date"] + ".json"), binding["receipt_sha256"]))
        if (receipt["date"] != day["date"] or receipt["file"] != binding["factor_file"]
                or Path(receipt["file"]).name != receipt["file"]
                or receipt["sha256"] != binding["factor_sha256"]):
            raise ValueError("adjustment receipt binding mismatch")
        factors = pd.read_parquet(pin(source / receipt["file"], receipt["sha256"]))
        prices = pd.read_parquet(daily, columns=["ts_code", "trade_date", "close", "pre_close"])
        prices["trade_date"] = prices.trade_date.astype(str)
        factors["trade_date"] = factors.trade_date.astype(str)
        if not prices.trade_date.eq(day["date"]).all():
            raise ValueError("adjustment price date mismatch")
        check = validate_factors(factors, day["date"], prices.ts_code)
        if not check["valid"]:
            raise ValueError("invalid adjustment factors")
        missing += len(check["missing_daily_codes"])
        frames.append(prices.merge(factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one"))
    triggers, rebuilt = adjustment_reconcile.reconcile_panel(pd.concat(frames, ignore_index=True))
    saved = pd.read_parquet(pin(directory / "reference_reconciliation.parquet"))
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), triggers.reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    for key in ("rows", "raw_discontinuities", "trigger_resolution_counts", "missing_factor_rows",
                "all_transition_status_counts", "interpretation"):
        if summary[key] != rebuilt[key]:
            raise ValueError("adjustment reconstructed summary mismatch: " + key)
    pd.testing.assert_frame_equal(pd.DataFrame(summary["unexplained_transitions"]).sort_index(axis=1),
                                  pd.DataFrame(rebuilt["unexplained_transitions"]).sort_index(axis=1),
                                  check_dtype=False, check_exact=True)
    market = {m: f.factor_reconciliation.value_counts().to_dict()
              for m, f in triggers.groupby(triggers.ts_code.str.rsplit(".", n=1).str[-1])}
    if summary["by_market"] != market:
        raise ValueError("adjustment market summary mismatch")
    for path, sha in pins.items():
        if digest(path) != sha:
            raise ValueError("adjustment source changed during reconstruction")
    return {"semantic_reconstruction_verified": True, "dates": len(days),
            "rows": rebuilt["rows"], "raw_discontinuities": rebuilt["raw_discontinuities"],
            "trigger_resolution_counts": rebuilt["trigger_resolution_counts"],
            "unexplained_transitions": len(rebuilt["unexplained_transitions"]),
            "missing_daily_code_rows": missing, "PIT_verified": False,
            "economic_cash_ledger_proven": False, "formal_H01_gate_passed": False,
            "evidence_files": [{"path": str(p), "sha256": sha,
                                "role": "fresh_adjustment_ratio_reconstruction"} for p, sha in pins.items()]}
