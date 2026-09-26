"""Build fresh six-class S20 opportunity/risk labels from cached price paths."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_s20_first_passage_labels import load_daily
from stockagent_analysis.s20_v3 import daily_labels


def main():
    out = ROOT / "output/experiments/s20_v3"
    out.mkdir(parents=True, exist_ok=True)
    daily = load_daily(ROOT / "output/tushare_cache/daily", "20240101", "20260911")
    parts = []
    for i, (_, group) in enumerate(daily.groupby("ts_code", sort=True), 1):
        result = daily_labels(group)
        if len(result):
            parts.append(result)
        if i % 500 == 0:
            print(f"labeled {i} stocks", flush=True)
    labels = pd.concat(parts, ignore_index=True)
    assert not labels.duplicated(["ts_code", "trade_date"]).any()
    labels.to_parquet(out / "labels.parquet", index=False)
    audit = {"rows": len(labels), "symbols": labels.ts_code.nunique(),
             "date_min": labels.trade_date.min(), "date_max": labels.trade_date.max(),
             "source_end": daily.trade_date.max(), "counts": labels.reason.value_counts().to_dict()}
    (out / "label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
