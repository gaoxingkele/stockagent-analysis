#!/usr/bin/env python
"""Freeze the market sell-off valve v1 (monitor mode) and its pre-registered actions.

Evidence inputs (research/jev_market/valve_study.py):
  output/jev_market/valve_risk_history.csv  expanding-window risk scores per day
  output/jev_market/valve_policies.csv      sleeve results of every action x threshold
Cut-offs = dev-window (<= 20260131) percentiles 50/80/90 of the 5-session
limit-down sum; the config dataclass must match them exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.market_valve import VALVE_CONTRACT_PATH, ValveConfig  # noqa: E402

OUT = ROOT / "output/jev_market"


def main() -> int:
    cfg = ValveConfig()
    risk = pd.read_csv(OUT / "valve_risk_history.csv", dtype={"date": str})
    dev = risk.loc[risk.date <= "20260131", "rule_ld5"].dropna()
    cut = {q: float(dev.quantile(q)) for q in (0.5, 0.8, 0.9)}
    assert (cfg.yellow_at, cfg.orange_at, cfg.red_at) == (round(cut[0.5]), round(cut[0.8]), round(cut[0.9])), cut
    pol = pd.read_csv(OUT / "valve_policies.csv")
    keep = pol[(pol.policy == "none") & (pol.thr > 1) | (pol.thr == 0.9)]
    contract = {
        "name": "s20_pure_valve_v1",
        "frozen_on": "2026-09-29",
        "status": "monitor_preregistered+B_enabled_by_user",
        "rationale": ["wiki/2026-09-29_jev-market-selloff-probe.md", "wiki/2026-09-29_market-valve-monitor.md"],
        "signal": "sum of SH/SZ limit-down closes over the last 5 sessions (10% boards <= -9.5%, 20% boards <= -19.5%)",
        "levels": {"green": f"< {cfg.yellow_at}", "yellow": f"{cfg.yellow_at}-{cfg.orange_at - 1}",
                   "orange": f"{cfg.orange_at}-{cfg.red_at - 1}", "red": f">= {cfg.red_at}"},
        "config": cfg.to_dict(),
        "target": "any broad sell-off session (median return <= -2.5% or >= 100 limit-down) in the next 5 sessions",
        "evidence": {
            "auc_expanding_window": {"dev_2024_10_2026_01": 0.635, "2026_02_09": 0.587,
                                     "note": "90-day stratified probe gave 0.72; the full-day estimate is lower"},
            "why_no_action_now": "on dev, no action at any threshold beat doing nothing on mean sleeve return; "
                                 "the top-decile readings often mark the end of a panic (dev red-bucket sleeves +6.7% v1)",
            "oracle": "sleeves entered when a sell-off did follow: v1 -3.3% (dev) / -2.1% (confirm) vs +3.2% / +3.6% otherwise",
            "policy_table_red_and_none": keep.to_dict(orient="records"),
        },
        "action_now": "show the level on every daily list; B_safe_red is active (user amendment 2026-09-30); "
                      "A and C remain monitor-only counterfactuals",
        "amendments": [{
            "date": "2026-09-30", "by": "user", "change": "enable B_safe_red",
            "basis": "risk preference (avoid sell-off streaks), not an evidence promotion: on dev B lowered the "
                     "aggressive list's mean sleeve return; in the 2026 window it protected against repeated sell-offs",
            "still_recorded": "the 'none' counterfactual for v1, so the choice can be reviewed at 60 matured days",
        }],
        "preregistered_actions": {
            "A_skip_red": "red: open no new sleeve that day (both lists)",
            "B_safe_red": "red: v1 users receive the v1.1 safe list instead",
            "C_tight_orange": "orange or red: stop at -8% instead of -10% for that day's sleeve",
        },
        "promotion_rule": "after >= 60 matured signal days from 20260806, an action is enabled only if it beats "
                          "'none' on mean sleeve return AND worst month for the list it applies to",
    }
    VALVE_CONTRACT_PATH.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"frozen -> {VALVE_CONTRACT_PATH}  cut-offs {cut}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
