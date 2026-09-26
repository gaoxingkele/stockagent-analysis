"""Small real-provider feasibility probe, not a complete history collector."""
from pathlib import Path
import os
import uuid

import pandas as pd
import requests
from dotenv import load_dotenv

from .runtime import atomic_json, digest, now


def main():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("provider credentials unavailable")
    output = root / "output/experiments/s20_safe_v4/sources" / ("suspension-daily-probe-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    receipts = []
    for date in ("20240102", "20250506", "20260911"):
        receipt = {"date": date, "requested_at": now(), "api": "suspend_d"}
        try:
            response = requests.post("https://api.tushare.pro", json={"api_name": "suspend_d", "token": token,
                "params": {"trade_date": date}, "fields": "ts_code,trade_date,suspend_timing,suspend_type"}, timeout=25)
            response.raise_for_status()
            payload = response.json()
            receipt["received_at"] = now()
            if payload.get("code") != 0:
                raise ValueError("provider rejected request")
            frame = pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
            destination = output / (date + ".parquet")
            frame.to_parquet(destination, index=False)
            receipt.update(rows=len(frame), sha256=digest(destination),
                           dates_match=bool(frame.trade_date.astype(str).eq(date).all()),
                           type_counts=frame.suspend_type.value_counts().to_dict(),
                           partial_timing_rows=int(frame.suspend_timing.notna().sum()), status="ACQUIRED_NOT_APPROVED")
        except Exception as exc:
            receipt.update(status="FAILED", error_type=type(exc).__name__)
        receipts.append(receipt)
        atomic_json(output / "receipts.json", receipts)
    result = {"directory": str(output), "receipts": receipts,
              "formal_training_authorized": False, "code_sha256": digest(Path(__file__)),
              "documentation": "https://tushare.pro/document/2?doc_id=214",
              "limits": "three-date feasibility only; no full coverage, PIT or null timing acceptance"}
    atomic_json(output / "summary.json", result)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
