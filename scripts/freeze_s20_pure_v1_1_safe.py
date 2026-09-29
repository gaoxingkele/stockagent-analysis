#!/usr/bin/env python
"""Freeze S20-Pure v1.1-safe ("safe and up") before any new-window evaluation.

Parameters come from the dev-only selection rule written into
scripts/analyze_s20_pure_r13_vol_band.py (bands whose dev bad and crash15 rates
are below the universe; maximise band expectancy). Re-computes the dev and
(consumed, descriptive) confirm evidence with the production select_safe().
Reuses the frozen stage1 artifacts of v1 unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20_pure import CONTRACT_PATH, SAFE_CONTRACT_PATH, SafeConfig, select_safe  # noqa: E402
from analyze_s20_pure_r11_safe_up import load, metrics  # noqa: E402


def clean(d: dict) -> dict:
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in d.items()}


def main() -> int:
    cfg = SafeConfig()
    p = load()
    L = select_safe(p, cfg)
    evidence = {per: clean(metrics(g)) for per, g in L.groupby("period")}
    universe = {per: clean(metrics(g)) for per, g in p.groupby("period")}
    v1 = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract = {
        "name": "s20_pure_v1_1_safe",
        "frozen_on": "2026-09-29",
        "status": "shadow_preregistered",
        "objective": "safe and up: avoid the -D crash line first, most names rise, small gains count",
        "rationale": ["wiki/2026-09-29_s20-pure-r11-safe-up-band.md",
                      "wiki/2026-09-29_s20-pure-r12-safe-model.md",
                      "wiki/2026-09-29_s20-pure-r13-vol-band.md"],
        "stage1": {"reuses": "config/s20_pure_v1.json", "artifacts": v1["stage1"]["artifacts"]},
        "list": cfg.to_dict(),
        "outcome_definition": {
            "success": f"+{cfg.band_low:g}% before -{cfg.crash_line:g}%, or neither and day-{cfg.horizon} close > 0",
            "bad": f"-{cfg.crash_line:g}% touched before +{cfg.band_low:g}% (stop fills at the line or the gap open)",
            "exit": f"half at +{cfg.band_low:g}% with stop moved to entry; rest at +{cfg.band_high:g}%, "
                    f"back at entry, or day-{cfg.horizon} close; full exit at -{cfg.crash_line:g}%",
        },
        "selection_rule": "dev only: among natr bands with dev bad and crash15 below the universe, "
                          "maximise band expectancy; ties -> higher success",
        "evidence_dev": evidence["dev"],
        "universe_dev": universe["dev"],
        "evidence_confirm_consumed_descriptive": evidence["confirm"],
        "universe_confirm": universe["confirm"],
        "known_limits": [
            "stop-outs gap through the -10% line 13.5% of the time (mean fill -10.55%, p5 -12.1%, worst -34.9%)",
            "in the consumed confirm window (high-volatility leadership) the list trailed the universe on success/expectancy",
            "day-level (market-wide) crashes cannot be removed by stock selection; see wiki R06/R10",
        ],
        "evaluation": {"window_start": "20260806", "min_matured_days_for_review": 60,
                       "baselines": ["same-rule universe", "S20-Pure v1 (aggressive)"]},
    }
    SAFE_CONTRACT_PATH.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: contract[k] for k in ("evidence_dev", "universe_dev")}, indent=1, ensure_ascii=False))
    print(f"frozen -> {SAFE_CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
