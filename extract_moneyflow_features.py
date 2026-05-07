#!/usr/bin/env python3
"""批量拉取所有股票 3 年 moneyflow + 计算 13 个资金分层特征.

输出:
  output/moneyflow/cache/{ts_code}.parquet  (raw 缓存, 可断点续跑)
  output/moneyflow/features.parquet         (合并后所有股票特征)

可用于:
  1. 重训 lgbm_uptrend / lgbm_risk
  2. 未来选股系统的独立资金分层信号
"""
from __future__ import annotations
import logging, os, time
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.stockagent_analysis.moneyflow.extractor import batch_fetch
from src.stockagent_analysis.moneyflow.features import compute_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mf_main")

PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR = ROOT / "output" / "moneyflow"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

START = "20230101"
END = "20260420"


def main():
    t0 = time.time()
    # 设置 token
    if not os.environ.get("TUSHARE_TOKEN"):
        env = ROOT / ".env"
        if env.exists():
            try:
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.startswith("TUSHARE_TOKEN="):
                        os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
                        break
            except Exception:
                pass

    log.info("加载股票列表 (从 factor_lab_3y)...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code"])
        parts.append(df)
    ts_codes = sorted(pd.concat(parts)["ts_code"].unique())
    log.info("股票数: %d", len(ts_codes))

    log.info("批量拉 moneyflow (含缓存) %s → %s", START, END)
    raw = batch_fetch(ts_codes, START, END,
                       cache_dir=CACHE_DIR, sleep_sec=0.05, log_every=300)
    log.info("总 raw 数据: %d 行", len(raw))

    if raw.empty:
        log.error("没拉到任何数据!")
        return

    # 计算特征
    log.info("计算特征 (按股分组)...")
    feat_dfs = []
    for ts, group in raw.groupby("ts_code"):
        feat = compute_features(group)
        if not feat.empty:
            feat_dfs.append(feat)

    feat = pd.concat(feat_dfs, ignore_index=True)
    out_path = OUT_DIR / "features.parquet"
    feat.to_parquet(out_path, index=False)
    log.info("写出 %s (%d 行, %d 股)", out_path, len(feat), feat["ts_code"].nunique())

    # 分布检查
    print("\n=== 特征分布 ===")
    feat_cols = [c for c in feat.columns if c not in ("ts_code", "trade_date")]
    print(feat[feat_cols].describe(percentiles=[0.05, 0.5, 0.95]).round(3).to_string())

    log.info("总耗时: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
