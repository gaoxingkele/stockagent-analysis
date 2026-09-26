"""Read-only provider metadata acquisition; coverage is audited, not assumed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, now

BASIC_FIELDS = "ts_code,symbol,name,market,exchange,curr_type,list_status,list_date,delist_date"
NAME_FIELDS = "ts_code,name,start_date,end_date,ann_date,change_reason"


def collect(root, query, page_size=1000, page_cap=30):
    if not 1 <= page_size <= 1000 or not 1 <= page_cap <= 30:
        raise ValueError("invalid pagination budget")
    base = root / "output/experiments/s20_safe_v4/sources" / ("metadata-" + uuid.uuid4().hex)
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe output")
    base.mkdir(parents=True)
    receipts = []

    def request(api, fields, params, name):
        started = now()
        try:
            frame = query(api, fields, params)
        except Exception as exc:
            atomic_json(base / "failure.json", {"api": api, "params": params, "at": now(), "error_type": type(exc).__name__})
            raise RuntimeError("metadata provider failed; sanitized receipt retained") from None
        if not set(fields.split(",")).issubset(frame.columns):
            raise ValueError("metadata schema mismatch")
        path = base / name
        frame.to_parquet(path, index=False)
        receipts.append({"api": api, "params": params, "requested_at": started, "received_at": now(),
                         "file": name, "rows": len(frame), "sha256": digest(path)})
        atomic_json(base / "receipts.json", receipts)
        return frame

    basics = []
    for status in ("L", "D", "P"):
        frame = request("stock_basic", BASIC_FIELDS, {"list_status": status}, f"stock_basic_{status}.parquet")
        if len(frame) >= 6000 or not frame.list_status.eq(status).all():
            raise ValueError("stock list truncated or status mismatch")
        basics.append(frame)
    names = []
    fingerprints = set()
    pagination_finished = False
    for page in range(page_cap):
        frame = request("namechange", NAME_FIELDS, {"limit": page_size, "offset": page * page_size}, f"namechange_{page:03d}.parquet")
        key = frame.to_json(orient="split", index=False)
        if len(frame) and key in fingerprints:
            raise ValueError("repeated page; provider may ignore offset")
        fingerprints.add(key)
        names.append(frame)
        if len(frame) < page_size:
            pagination_finished = True
            break
    basic = pd.concat(basics, ignore_index=True)
    name = pd.concat(names, ignore_index=True)
    basic.to_parquet(base / "stock_basic.parquet", index=False)
    name.to_parquet(base / "namechange.parquet", index=False)
    observed = set()
    daily_hashes = []
    for path in sorted((root / "output/tushare_cache/daily").glob("*.parquet")):
        before = digest(path)
        observed.update(pd.read_parquet(path, columns=["ts_code"]).ts_code.astype(str))
        if digest(path) != before:
            raise ValueError("daily data changed during metadata audit")
        daily_hashes.append({"path": path.relative_to(root).as_posix(), "sha256": before})
    result = {"directory": str(base), "at": now(), "stock_basic_rows": len(basic),
              "basic_duplicate_codes": int(basic.ts_code.duplicated().sum()),
              "observed_codes_missing_basic": sorted(observed - set(basic.ts_code)),
              "name_rows": len(name), "name_unique_codes": int(name.ts_code.nunique()),
              "name_missing_announcement_dates": int(name.ann_date.isna().sum()),
              "name_duplicate_rows": int(name.duplicated().sum()),
              "observed_codes_without_name_rows": sorted(observed - set(name.ts_code)),
              "pagination_finished": pagination_finished,
              "global_endpoint_coverage_proven": False,
              "known_provider_issue_to_crosscheck": "https://github.com/waditu/tushare/issues/1901",
              "no_name_rows_means": "unknown coverage, not automatically never-ST",
              "current_metadata_is_historical_features": False,
              "formal_H01_gate_passed": False, "historical_revision_availability_proven": False}
    atomic_json(base / "summary.json", result)
    atomic_json(base / "inputs.json", {"daily": daily_hashes, "code_hash": digest(Path(__file__)),
                "basic_hash": digest(base / "stock_basic.parquet"), "names_hash": digest(base / "namechange.parquet")})
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv
    import requests
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN unavailable")
    session = requests.Session()
    def query(api, fields, params):
        response = session.post("https://api.tushare.pro", json={"api_name": api, "token": token,
                                "params": params, "fields": fields}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
    result = collect(root, query)
    print(json.dumps({key: value if not isinstance(value, list) else {"count": len(value), "first_10": value[:10]}
                      for key, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
