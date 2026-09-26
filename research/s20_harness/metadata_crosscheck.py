"""Targeted single-code checks against a frozen global metadata snapshot."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import pandas as pd

from .metadata_source import NAME_FIELDS
from .runtime import atomic_json, digest, now


def run(root, query):
    source = root / "output/experiments/s20_safe_v4/sources/metadata-a8335d9147fa4cfba794a40a81d6669d/namechange.parquet"
    source_hash = digest(source)
    global_frame = pd.read_parquet(source)
    output = source.parent / ("crosscheck-" + uuid.uuid4().hex)
    output.mkdir()
    results = []
    # Missing codes, reviewed alias, and two unrelated controls. No selection by
    # future return. This sample cannot prove the whole endpoint complete.
    for code in ("300114.SZ", "302132.SZ", "688208.SH", "000001.SZ", "600000.SH"):
        started = now()
        try:
            frame = query(code)
        except Exception as exc:
            atomic_json(output / "failure.json", {"code": code, "at": now(), "error_type": type(exc).__name__})
            raise RuntimeError("single-code query failed; sanitized receipt saved") from None
        if not set(NAME_FIELDS.split(",")).issubset(frame.columns) or not frame.ts_code.eq(code).all() or len(frame) >= 1000:
            raise ValueError("invalid/truncated single-code response")
        path = output / f"{code}.parquet"
        frame.to_parquet(path, index=False)
        baseline = global_frame.loc[global_frame.ts_code.eq(code), NAME_FIELDS.split(",")]
        def keys(data):
            return set(data[NAME_FIELDS.split(",")].apply(lambda row: row.to_json(force_ascii=True), axis=1))
        observed_keys, baseline_keys = keys(frame), keys(baseline)
        results.append({"ts_code": code, "requested_at": started, "received_at": now(), "rows": len(frame),
                        "file": path.name, "sha256": digest(path), "global_rows": len(baseline),
                        "single_only_rows": len(observed_keys - baseline_keys),
                        "global_only_rows": len(baseline_keys - observed_keys),
                        "ann_date_missing": int(frame.ann_date.isna().sum())})
        atomic_json(output / "receipts.json", results)
    report = {"at": now(), "directory": str(output), "source_hash": source_hash,
              "checks": results, "global_coverage_proven": False, "formal_H01_gate_passed": False,
              "note": "empty single-code result is not proof of no historical ST; changes require a new source version"}
    atomic_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    from dotenv import load_dotenv
    import requests
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN unavailable")
    def query(code):
        response = requests.post("https://api.tushare.pro", json={"api_name": "namechange", "token": token,
                                  "params": {"ts_code": code, "limit": 1000}, "fields": NAME_FIELDS}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(data["data"]["items"], columns=data["data"]["fields"])
    print(json.dumps(run(root, query), indent=2))
