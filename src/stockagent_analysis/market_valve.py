"""Market-wide sell-off warning valve (monitor mode).

Design record: wiki/2026-09-29_market-valve-monitor.md.

Signal: limit-down closes summed over the last 5 sessions (SH/SZ A shares,
10% boards at -9.5%, 20% boards at -19.5%). Broad sell-offs cluster, so a high
count raises the chance of another broad sell-off session within 5 sessions.
Levels use fixed cut-offs frozen from the 2024-04..2026-01 distribution.

The valve does not change any list on its own: in the development window no
action (skip, half size, switch to the safe list, tighter stop) improved the
20-session sleeve results, because the highest readings often mark the end of a
panic. The pre-registered actions are tracked as counterfactuals and only
enabled after they win on data after the freeze.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VALVE_CONTRACT_PATH = ROOT / "config/s20_pure_valve_v1.json"
LEVELS = ("green", "yellow", "orange", "red")
LEVEL_CN = {"green": "绿", "yellow": "黄", "orange": "橙", "red": "红"}


@dataclass(frozen=True)
class ValveConfig:
    window: int = 5
    yellow_at: int = 47      # dev-window 50th percentile of the 5-session limit-down sum
    orange_at: int = 102     # 80th percentile
    red_at: int = 154        # 90th percentile
    mode: str = "monitor"    # no automatic action until a pre-registered action wins

    def to_dict(self) -> dict:
        return asdict(self)


def daily_breadth(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-date limit-down count from a long daily table (ts_code, trade_date, pct_chg)."""
    d = daily[daily["ts_code"].str.endswith((".SH", ".SZ"))]
    board20 = d["ts_code"].str[:3].isin(["300", "301", "688", "689"])
    down = d["pct_chg"] <= np.where(board20, -19.5, -9.5)
    return down.groupby(d["trade_date"].astype(str)).sum().rename("limit_down").reset_index()


def valve_levels(breadth: pd.DataFrame, cfg: ValveConfig = ValveConfig()) -> pd.DataFrame:
    """breadth: columns date/trade_date + limit_down. Adds limit_down_5d and level."""
    b = breadth.rename(columns={"trade_date": "date"}).sort_values("date").reset_index(drop=True)
    b["limit_down_5d"] = b["limit_down"].rolling(cfg.window).sum()
    x = b["limit_down_5d"]
    b["level"] = np.select([x >= cfg.red_at, x >= cfg.orange_at, x >= cfg.yellow_at, x.notna()],
                           ["red", "orange", "yellow", "green"], default="unknown")
    return b


def load_contract() -> dict:
    return json.loads(VALVE_CONTRACT_PATH.read_text(encoding="utf-8"))
