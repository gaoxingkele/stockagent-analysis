#!/usr/bin/env python
"""Freeze S20-Pure v1: stage1 scorer artifacts + funnel contract.

stage1 (S20-20R portable residual) was confirmed on 2026-09-07, but only its
residual boosters were saved; the R20 anchor was refit in memory. This script
refits the anchor with the exact recipe of confirm_s20_20r_portable.py
(same 50% sample, seed, early-stopped iteration count), saves it next to the
residual boosters, and proves reproduction by re-scoring the confirmation
window and comparing with the saved stage1_probability.

No new modelling decision is made here. The funnel parameters are the ones
pre-registered in wiki/2026-09-28_s20-pure-r10-synthesis.md.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

sys.argv = sys.argv[:1]
import confirm_s20_20r_portable as C  # noqa: E402
from stockagent_analysis.s20_pure import (  # noqa: E402
    FROZEN_DIR,
    CONTRACT_PATH,
    PureConfig,
    stage1_probability,
)

RUN_V1 = ROOT / "output/experiments/s20_20r_confirmation/run_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = C.parse_args()
    seeds = tuple(int(v) for v in args.seeds.split(","))
    development, features, audit = C._prepare(args)
    features = [f for f in features if f not in C.PORTABLE_EXCLUDED_FEATURES]
    dates = development["trade_date"].astype(str)
    ends = development["horizon_end_date"].astype(str)
    mature = ends < C.CONFIRMATION_START
    anchor_fit = development[(dates <= "20250630") & (ends < "20250701")]
    anchor_tune = development[dates.between("20250701", "20250831") & (ends < "20250901")]
    early = C._fit_anchor(anchor_fit, anchor_tune, features, seeds[0] - 503, args.num_threads)
    final_anchor = C._refit_anchor(development[mature], features, seeds[0] - 503,
                                   args.num_threads, early.best_iteration)
    print(f"anchor: early best_iteration={early.best_iteration}, refit rows={int(mature.sum()):,}", flush=True)

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    final_anchor.save_model(str(FROZEN_DIR / "anchor.txt"))
    for seed in seeds:
        shutil.copy2(RUN_V1 / f"residual_seed{seed}.txt", FROZEN_DIR / f"residual_seed{seed}.txt")
    (FROZEN_DIR / "features.json").write_text(json.dumps(features, indent=1), encoding="utf-8")

    # reproduction check on the (already consumed) confirmation window
    confirmation = C._load_confirmation(args, features)
    confirmation["stage1_repro"] = stage1_probability(confirmation)
    saved = pd.read_parquet(RUN_V1 / "predictions.parquet", columns=["ts_code", "trade_date", "stage1_probability"])
    m = confirmation.merge(saved, on=["ts_code", "trade_date"], how="inner")
    diff = (m.stage1_repro - m.stage1_probability).abs()
    rank_corr = m.groupby("trade_date").apply(
        lambda g: g.stage1_repro.corr(g.stage1_probability, method="spearman")).mean()
    top20_overlap = m.groupby("trade_date").apply(
        lambda g: len(set(g.nlargest(20, "stage1_repro").ts_code) & set(g.nlargest(20, "stage1_probability").ts_code)) / 20
    ).mean()
    repro = {"rows": len(m), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
             "daily_spearman_mean": float(rank_corr), "daily_top20_overlap_mean": float(top20_overlap)}
    print(repro, flush=True)
    if repro["daily_top20_overlap_mean"] < 0.95:
        raise SystemExit("stage1 reproduction failed; not freezing")

    cfg = PureConfig()
    contract = {
        "name": "s20_pure_v1",
        "frozen_on": "2026-09-29",
        "status": "shadow_preregistered",
        "rationale": "wiki/2026-09-28_s20-pure-chain.md (R01-R10)",
        "stage1": {
            "source": "S20-20R portable residual (confirm_s20_20r_portable.py recipe)",
            "anchor_early_best_iteration": int(early.best_iteration),
            "anchor_refit_rows": int(mature.sum()),
            "development_sample_bps": args.sample_bps,
            "reproduction_vs_run_v1": repro,
            "artifacts": {p.name: sha256(p) for p in sorted(FROZEN_DIR.glob("*.txt"))},
            "feature_count": len(features),
        },
        "funnel": cfg.to_dict(),
        "evaluation": {
            "window_start": "20260806",
            "note": "2026-01-27..08-05 is consumed; only signal dates >= 20260806 with a fully "
                    "matured 20-session horizon count. Promotion review needs >= 60 matured days.",
            "min_matured_days_for_review": 60,
            "baselines": ["same-rule universe", "stage1 Top20 without amplitude cap"],
            "metrics": ["exit win rate", "exit mean per trade (0.3% cost)", "pure_up share",
                        "pure_down share", "worst month", "realised failure rate of the list"],
        },
    }
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"frozen -> {CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
