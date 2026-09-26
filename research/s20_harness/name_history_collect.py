"""Bounded per-stock name-history acquisition after verified global omissions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import pandas as pd
import psutil

from .metadata_source import NAME_FIELDS
from .runtime import atomic_json, digest, load_plan, now


def collect(root, query, source_id=None, max_requests=6000, wall_seconds=1800, pause=.2):
    if not 1 <= max_requests <= 6000 or not 0 < wall_seconds <= 1800:
        raise ValueError("invalid collection budget")
    source_id = source_id or "namehistory-" + uuid.uuid4().hex
    if Path(source_id).name != source_id or not source_id.startswith("namehistory-"):
        raise ValueError("invalid source id")
    base = root / "output/experiments/s20_safe_v4/sources" / source_id
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe output")
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "active.json"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "process_created": psutil.Process().create_time(), "at": now()}, stream)
    started = time.monotonic()
    count = 0
    records = []
    try:
        plan_path = base / "collection_plan.json"
        if plan_path.exists():
            plan = load_plan(plan_path)
            if plan["code_hash"] != digest(Path(__file__)):
                raise ValueError("collector changed; new source version required")
        else:
            codes, inputs = set(), []
            for path in sorted((root / "output/tushare_cache/daily").glob("*.parquet")):
                before = digest(path)
                codes.update(pd.read_parquet(path, columns=["ts_code"]).ts_code.astype(str))
                if digest(path) != before:
                    raise ValueError("input changed")
                inputs.append({"path": path.relative_to(root).as_posix(), "sha256": before})
            if not codes:
                raise ValueError("no input universe")
            plan = {"at": now(), "codes": sorted(codes), "inputs": inputs, "api": "namechange",
                    "code_hash": digest(Path(__file__)), "request_cap": max_requests, "wall_seconds": wall_seconds,
                    "reason": "global endpoint omissions verified by independent per-code query",
                    "empty_response_semantics": "unknown; never automatically normal/ST-free"}
            atomic_json(plan_path, plan)
        for code in plan["codes"]:
            path = base / f"{code}.parquet"
            receipt_path = base / f"{code}.json"
            if receipt_path.exists():
                record = load_plan(receipt_path)
                if not path.is_file() or digest(path) != record["sha256"]:
                    raise ValueError("cached history hash mismatch")
                records.append(record)
                continue
            if path.exists():
                raise ValueError("uncommitted history file; no overwrite")
            if count >= max_requests or time.monotonic() - started >= wall_seconds:
                break
            requested = now()
            count += 1
            try:
                frame = query(code)
            except Exception as exc:
                atomic_json(base / ("failure-" + uuid.uuid4().hex + ".json"), {"at": now(), "code": code, "error_type": type(exc).__name__})
                raise RuntimeError("history request failed; sanitized receipt retained") from None
            if not set(NAME_FIELDS.split(",")).issubset(frame.columns) or not frame.ts_code.eq(code).all() or len(frame) >= 1000:
                raise ValueError("invalid or potentially truncated single-stock response")
            frame.to_parquet(path, index=False)
            record = {"code": code, "requested_at": requested, "received_at": now(), "rows": len(frame),
                      "missing_ann_date": int(frame.ann_date.isna().sum()), "sha256": digest(path)}
            atomic_json(receipt_path, record)
            records.append(record)
            heartbeat = {"at": now(), "pid": os.getpid(), "process_created": psutil.Process().create_time(),
                         "source_id": source_id, "completed_codes": len(records), "total_codes": len(plan["codes"])}
            atomic_json(base / "heartbeat.json", heartbeat)
            if count == 1 or count % 100 == 0:
                print(json.dumps(heartbeat), flush=True)
            if pause:
                time.sleep(pause)
        result = {"at": now(), "source_id": source_id, "complete_acquisition": len(records) == len(plan["codes"]),
                  "completed_codes": len(records), "total_codes": len(plan["codes"]), "requests": count,
                  "empty_codes": [r["code"] for r in records if not r["rows"]],
                  "missing_ann_date_rows": sum(r["missing_ann_date"] for r in records),
                  "formal_H01_gate_passed": False, "historical_revision_availability_proven": False}
        atomic_json(base / "summary.json", result)
        return result
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id")
    parser.add_argument("--max-requests", type=int, default=6000)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from dotenv import load_dotenv
    import requests
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN unavailable")
    session = requests.Session()
    def query(code):
        response = session.post("https://api.tushare.pro", json={"api_name": "namechange", "token": token,
                               "params": {"ts_code": code, "limit": 1000}, "fields": NAME_FIELDS}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError("provider rejected request")
        return pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
    print(json.dumps(collect(root, query, args.source_id, args.max_requests, args.wall_seconds), indent=2))
