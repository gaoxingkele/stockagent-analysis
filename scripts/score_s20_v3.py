"""Score a research factor table using an explicitly selected S20-v3 bundle."""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.s20_v3 import probability_outputs


def score_frame(frame, bundle, risk_weight=1.0):
    schema = json.loads((bundle / "schema.json").read_text(encoding="utf-8"))
    features = schema["features"]
    missing = set(["ts_code", "trade_date", *features]) - set(frame.columns)
    if missing:
        raise ValueError(f"missing input fields: {sorted(missing)}")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate stock-date rows")
    model = lgb.Booster(model_file=str(bundle / "model.txt"))
    raw = model.predict(frame[features], num_threads=8)
    logits = (np.log(np.clip(raw, 1e-8, 1)) @ np.asarray(schema["calibration_coefficients"]).T
              + np.asarray(schema["calibration_intercepts"]))
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    outputs = probability_outputs(p, risk_weight=risk_weight)
    result = pd.concat([frame[["ts_code", "trade_date"]].reset_index(drop=True),outputs],axis=1)
    result = result.sort_values(["trade_date", "score", "ts_code"],ascending=[True,False,True])
    result["daily_rank"] = result.groupby("trade_date").cumcount() + 1
    result["research_only"] = True
    result["probability_status"] = "calibration_not_validated"
    result["risk_weight"] = risk_weight
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input,dtype={"ts_code":str,"trade_date":str})
    result = score_frame(frame,args.bundle,args.risk_weight)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.suffix == ".parquet":
        result.to_parquet(args.output,index=False)
    else:
        result.to_csv(args.output,index=False,encoding="utf-8-sig")
    print(f"scored {len(result)} research rows -> {args.output}")


if __name__ == "__main__":
    main()
