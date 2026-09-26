"""Bounded fresh-source probe of the unresolved 603081 factor transition."""
import json
import os
from pathlib import Path
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
        raise ValueError("provider credential unavailable")
    out = root / "output/experiments/s20_safe_v4/sources" / ("factor-case-probe-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    receipts = []
    for api, fields in [("adj_factor", "ts_code,trade_date,adj_factor"),
                        ("daily", "ts_code,trade_date,open,high,low,close,pre_close,pct_chg")]:
        params = dict(ts_code="603081.SH", start_date="20240617", end_date="20240705")
        receipt = dict(api=api, params=params, requested_at=now())
        try:
            response = requests.post("https://api.tushare.pro", json=dict(api_name=api, token=token, params=params, fields=fields), timeout=25)
            response.raise_for_status()
            data = response.json()
            receipt["received_at"] = now()
            if data.get("code") != 0:
                raise ValueError("provider rejected request")
            frame = pd.DataFrame(data["data"]["items"], columns=data["data"]["fields"])
            if frame.empty or len(frame) > 20 or not frame.ts_code.eq("603081.SH").all() or not frame.trade_date.between("20240617", "20240705").all() or frame.trade_date.duplicated().any():
                raise ValueError("unexpected probe scope")
            p = out / (api + ".parquet")
            frame.to_parquet(p, index=False)
            receipt.update(status="ACQUIRED_NOT_APPROVED", rows=len(frame), sha256=digest(p))
        except Exception as exc:
            receipt.update(status="FAILED", error_type=type(exc).__name__)
        receipts.append(receipt)
        atomic_json(out / "receipts.json", receipts)
    report = dict(directory=str(out), receipts=receipts, code_sha256=digest(Path(__file__)),
                  formal_training_authorized=False, historical_source_overwritten=False)
    atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
