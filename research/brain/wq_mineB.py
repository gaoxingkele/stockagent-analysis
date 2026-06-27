# -*- coding: utf-8 -*-
"""BRAIN Phase B 多信号组合 — 修正动量(0.93) + value(0.44) + volprem(0.41) 组合冲 1.25.

Phase A 实证: 正交族弱 (value +0.44 / volprem +0.41 是仅有正向, 质量类全负)。
理论无相关上限 √(0.93²+0.44²+0.41²)≈1.11 < 1.25。本批实测组合真实天花板:
rank-sum 组合 (避开 z-score 对修正动量的惩罚) × 权重 × 中性化变体。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wq_mineA import run

EM = "anl4_afv4_eps_mean"
REV = f"ts_delta({EM}, 120)"
REV90 = f"ts_delta({EM}, 90)"
VAL = "fscore_bfl_value"
VP = "(implied_volatility_call_60 - historical_volatility_60)"
LV = "(-historical_volatility_60)"

CANDIDATES = [
    ("rev_val",        f"rank({REV}) + rank({VAL})", 12, "SUBINDUSTRY"),
    ("rev2_val",       f"2*rank({REV}) + rank({VAL})", 12, "SUBINDUSTRY"),
    ("rev3_val",       f"3*rank({REV}) + rank({VAL})", 12, "SUBINDUSTRY"),
    ("rev_val_ind",    f"rank({REV}) + rank({VAL})", 12, "INDUSTRY"),
    ("rev_val_sec",    f"rank({REV}) + rank({VAL})", 12, "SECTOR"),
    ("rev_val_mkt",    f"rank({REV}) + rank({VAL})", 12, "MARKET"),
    ("rev90_val",      f"rank({REV90}) + rank({VAL})", 10, "SUBINDUSTRY"),
    ("rev_val_vp",     f"rank({REV}) + rank({VAL}) + rank({VP})", 12, "SUBINDUSTRY"),
    ("rev2_val_vp",    f"2*rank({REV}) + rank({VAL}) + rank({VP})", 12, "SUBINDUSTRY"),
    ("rev_val_lv",     f"rank({REV}) + rank({VAL}) + rank({LV})", 12, "SUBINDUSTRY"),
    ("rev2_val_vp_lv", f"2*rank({REV}) + rank({VAL}) + rank({VP}) + rank({LV})", 12, "SUBINDUSTRY"),
    ("rev_valz",       f"rank(winsorize(ts_zscore({REV},250),std=4) + winsorize(ts_zscore({VAL},250),std=4))", 12, "SUBINDUSTRY"),
]

if __name__ == "__main__":
    run(CANDIDATES, Path(__file__).resolve().parent / "mineB_results.json", "Phase B 多信号组合")
