"""Local cache inventory. Missing files are reported; market data is not invented."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DAILY_DIR = Path("output/tushare_cache/daily")
BASIC_FILE = Path("output/tushare_cache/stock_basic.parquet")


def discover_market_files(root: str | Path) -> dict:
    root = Path(root)
    daily = root / DAILY_DIR
    files = sorted(daily.glob("*.parquet")) if daily.exists() else []
    return {
        "daily_dir": str(daily),
        "daily_dir_exists": daily.exists(),
        "n_daily_files": len(files),
        "first_date": files[0].stem if files else None,
        "last_date": files[-1].stem if files else None,
        "stock_basic": str(root / BASIC_FILE),
        "stock_basic_exists": (root / BASIC_FILE).exists(),
    }


def audit_local_cache(root: str | Path, sample: int = 3) -> dict:
    root = Path(root)
    discovered = discover_market_files(root)
    if not discovered["daily_dir_exists"] or discovered["n_daily_files"] == 0:
        return {
            **discovered,
            "ran": False,
            "reason": "local daily/calendar files missing; not invented",
            "unexplained_lookahead": None,
            "coverage_dates": 0,
        }
    daily = root / DAILY_DIR
    files = sorted(daily.glob("*.parquet"))
    picks = [files[0], files[len(files) // 2], files[-1]][: max(sample, 1)]
    unexplained = 0
    rows = 0
    for path in picks:
        frame = pd.read_parquet(path)
        rows += int(len(frame))
        if "trade_date" in frame.columns:
            dates = frame["trade_date"].map(lambda x: str(int(x)) if str(x).isdigit() else str(x))
            if (dates != path.stem).any():
                unexplained += int((dates != path.stem).sum())
    basic_note = None
    if discovered["stock_basic_exists"]:
        basic_note = "stock_basic.parquet is a single snapshot; treat as current metadata until as-of history exists"
    return {
        **discovered,
        "ran": True,
        "sampled_files": [p.stem for p in picks],
        "sampled_rows": rows,
        "unexplained_lookahead": unexplained,
        "coverage_dates": discovered["n_daily_files"],
        "stock_basic_pit": basic_note,
    }
