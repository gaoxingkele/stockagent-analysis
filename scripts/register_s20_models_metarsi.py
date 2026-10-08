#!/usr/bin/env python
"""Register the S20 stage1 models and their block results in the metaRSI ledger, then audit them.

Reads output/experiments/s20_model_compare_20261005/facts.json (written by compare_s20_models.py).
Everything submitted here is a measured fact: tree counts from the saved model files, fit-window
lengths from the trading calendar, block results and the selection profile from the comparison.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/aicoding/mylib/skills/meta-rsi/scripts")
from metarsi import main as metarsi  # noqa: E402
from s20_frames import MARKET_FEATURES  # noqa: E402

FACTS = ROOT / "output/experiments/s20_model_compare_20261005/facts.json"
CAL = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))


def run(*argv: str) -> None:
    metarsi(["--project", str(ROOT), *argv])


def sessions(start: str, end: str) -> int:
    return sum(1 for d in CAL if start <= d <= end)


def trees(pattern: str) -> list[int]:
    return [lgb.Booster(model_file=f).num_trees() for f in sorted(glob.glob(str(ROOT / pattern)))]


def market_gain(pattern: str) -> float | None:
    total = None
    for f in sorted(glob.glob(str(ROOT / pattern))):
        b = lgb.Booster(model_file=f)
        g = pd.Series(b.feature_importance("gain"), index=b.feature_name())
        total = g if total is None else total.add(g, fill_value=0)
    if total is None:
        return None
    return round(float(total[[k for k in total.index if k in MARKET_FEATURES]].sum() / total.sum()), 3)


def main() -> int:
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    frozen_dir = "output/production/s20_pure_v1"
    folds_dir = "output/experiments/s20_20r_residual_portable"
    models = {
        "s20_frozen_stage1": dict(
            key="frozen", role="shipped",
            recipe="R20 锚模型（早停）加有界残差（早停）；残差拟合 2025-09-01..10-31，调参 2025-11-01..2026-01-26；之后参数固定",
            trees=trees(f"{frozen_dir}/anchor.txt") + trees(f"{frozen_dir}/residual_seed*.txt"),
            gain=market_gain(f"{frozen_dir}/residual_seed*.txt"), fit_days=sessions("20250901", "20251031"),
            relation={"2025-09..11": "fit", "2025-12..2026-02": "tune"}),
        "s20_fold_models": dict(
            key="folds", role="reference",
            recipe="同一份脚本的三折：wf1 残差拟合 2024-07..10，wf2 2024-11..2025-02，wf3 2025-03..06；各自早停",
            trees=trees(f"{folds_dir}/wf*/r20_anchor_refit.txt") + trees(f"{folds_dir}/wf*/s20_residual_seed*.txt"),
            gain=market_gain(f"{folds_dir}/wf*/s20_residual_seed*.txt"),
            fit_days=min(sessions("20240701", "20241031"), sessions("20241101", "20250228"), sessions("20250301", "20250630")),
            relation={}),
        "s20_unified_v1": dict(
            key="unified", role="candidate",
            recipe="单一分类器学 positive20；2024-01-02 起扩展窗口；固定 300 棵树；价格和成交量单位的特征去量纲；每季度重训，区块只由之前的模型打分",
            trees=trees("output/experiments/s20_unified_v1_20261005/unified_v1_*.txt"),
            gain=market_gain("output/experiments/s20_unified_v1_20261005/unified_v1_*.txt"), fit_days=sessions("20240102", "20250123"),
            relation={}),
        "s20_unified_v2": dict(
            key="unified_v2", role="candidate",
            recipe="按日排序（LambdaRank，每个交易日一组）学 positive20；去掉市场状态特征；个股特征去量纲后取日内分位；其余同 v1",
            trees=trees("output/experiments/s20_unified_v2_20261005/unified_v2_*.txt"),
            gain=market_gain("output/experiments/s20_unified_v2_20261005/unified_v2_*.txt"), fit_days=sessions("20240102", "20250123"),
            relation={}),
    }
    for name, m in models.items():
        if m["key"] not in facts or not m["trees"]:
            print(f"skip {name}: no facts or model files yet")
            continue
        argv = ["model", "register", "--name", name, "--role", m["role"], "--recipe", m["recipe"], "--trees", ",".join(map(str, m["trees"]))]
        if m["gain"] is not None:
            argv += ["--market-gain", str(m["gain"])]
        run(*argv)
        for block, v in facts[m["key"]].items():
            run("model", "block", "--name", name, "--block", block, "--relation", m["relation"].get(block, "oos"),
                "--metric-name", "进攻版每笔(名单口径)", "--value", str(v["offensive"]), "--days", str(v["days"]),
                "--fit-days", str(m["fit_days"]), "--profile", f"ma20_up={v['ma20_up']}", "--profile", f"ma20_up_market={v['ma20_up_market']}")
    # the evidence that used to be quoted for the frozen model was scored by the fold models
    if "folds" in facts:
        days = sum(v["days"] for v in facts["folds"].values())
        mean = sum(v["offensive"] * v["days"] for v in facts["folds"].values()) / days
        run("model", "block", "--name", "s20_frozen_stage1", "--block", "原先引用的开发期成绩", "--relation", "oos",
            "--metric-name", "进攻版每笔(名单口径)", "--value", f"{mean:.2f}", "--days", str(days), "--scored-by", "s20_fold_models")
    run("model", "audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
