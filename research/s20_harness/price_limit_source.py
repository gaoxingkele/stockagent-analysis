"""Bounded daily-limit acquisition, isolated from production and sibling caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import pandas as pd
import psutil

from .price_limit_audit import inspect_limits
from .runtime import atomic_json, digest, load_plan, now

FIELDS = "trade_date,ts_code,pre_close,up_limit,down_limit"
ROW_CAP = 5800


def collect(root, query, source_id=None, max_requests=700, wall_seconds=1800, pause=.6):
    root = Path(root).resolve()
    if not 1 <= max_requests <= 700 or not 0 < wall_seconds <= 1800 or not 0 <= pause <= 10:
        raise ValueError("invalid request budget")
    source_id = source_id or "limits-" + uuid.uuid4().hex
    if Path(source_id).name != source_id or not source_id.startswith("limits-"):
        raise ValueError("invalid source id")
    output = root / "output/experiments/s20_safe_v4/sources" / source_id
    if not output.resolve().is_relative_to(root):
        raise ValueError("unsafe output")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "active.json"
    identity = {"pid": os.getpid(), "process_created": psutil.Process().create_time()}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(identity, at=now()), stream)
    started = time.monotonic()
    requests_used, records = 0, []
    try:
        plan_path = output / "collection_plan.json"
        if plan_path.exists():
            plan = load_plan(plan_path)
            if plan["code_sha256"] != digest(Path(__file__)):
                raise ValueError("collector changed; new source required")
        else:
            sources = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
            if not sources:
                raise ValueError("no reference daily data")
            plan = {"at": now(), "code_sha256": digest(Path(__file__)), "api": "stk_limit",
                    "documentation": "https://tushare.pro/document/2?doc_id=183",
                    "row_cap": ROW_CAP, "daily": [{"date": p.stem, "path": p.relative_to(root).as_posix(),
                                                     "sha256": digest(p)} for p in sources]}
            atomic_json(plan_path, plan)
        for item in plan["daily"]:
            source = (root / item["path"]).resolve()
            if not source.is_relative_to(root) or digest(source) != item["sha256"]:
                raise ValueError("reference input changed or unsafe")
            date = item["date"]
            path, receipt_path = output / (date + ".parquet"), output / (date + ".json")
            if receipt_path.exists():
                receipt = load_plan(receipt_path)
                if receipt["date"] != date or digest(path) != receipt["sha256"]:
                    raise ValueError("cached limit receipt mismatch")
                if receipt["possible_truncation"]:
                    raise ValueError("truncated response requires separate collection plan")
                records.append(receipt)
                continue
            if path.exists():
                raise ValueError("uncommitted file; no overwrite")
            if requests_used >= max_requests or time.monotonic() - started >= wall_seconds:
                break
            requested_at = now()
            requests_used += 1
            try:
                frame = query(date)
            except Exception as exc:
                atomic_json(output / ("failure-" + uuid.uuid4().hex + ".json"),
                            {"at": now(), "date": date, "error_type": type(exc).__name__})
                raise RuntimeError("provider request failed; sanitized evidence saved") from None
            codes = set(pd.read_parquet(source, columns=["ts_code"]).ts_code.astype(str))
            check = inspect_limits(frame, date, codes)
            if digest(source) != item["sha256"]:
                raise ValueError("reference changed while querying")
            frame.to_parquet(path, index=False)
            receipt = {"date": date, "requested_at": requested_at, "received_at": now(),
                       "sha256": digest(path), "validation": check, "possible_truncation": len(frame) >= ROW_CAP}
            atomic_json(receipt_path, receipt)
            if receipt["possible_truncation"]:
                raise ValueError("possible truncated response; evidence retained, not accepted")
            records.append(receipt)
            heartbeat = dict(identity, at=now(), source_id=source_id, completed_dates=len(records), total_dates=len(plan["daily"]))
            atomic_json(output / "heartbeat.json", heartbeat)
            if requests_used == 1 or requests_used % 25 == 0:
                print(json.dumps(heartbeat), flush=True)
            if pause:
                time.sleep(pause)
        summary = {"at": now(), "source_id": source_id, "completed_dates": len(records),
                   "total_dates": len(plan["daily"]), "complete_acquisition": len(records) == len(plan["daily"]),
                   "requests_used": requests_used, "formal_H01_gate_passed": False,
                   "invalid_limit_rows": sum(r["validation"]["invalid_limit_rows"] for r in records),
                   "usable_stock_dates": sum(r["validation"]["usable_expected_codes"] for r in records),
                   "availability": "retrieved now; historical receipt availability not inferred",
                   "semantics": "zero/missing limit values retained as unknown, not assumed unrestricted trading"}
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id")
    parser.add_argument("--max-requests", type=int, default=700)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv
    import requests
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN unavailable")
    session = requests.Session()
    def query(date):
        response = session.post("https://api.tushare.pro", json={"api_name": "stk_limit", "token": token,
                                "params": {"trade_date": date}, "fields": FIELDS}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
    print(json.dumps(collect(root, query, args.source_id, args.max_requests), indent=2))
