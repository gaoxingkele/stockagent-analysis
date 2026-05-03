#!/usr/bin/env python3
"""市场状态额外特征 — regime 持续天数 + 强度 + 大盘斜率 z-score 等.

新增 5 个市场级特征 (每个交易日一个值, merge 到全部股票):
  regime_days_in:    当前 regime 已持续天数 (区分初期/后期)
  regime_intensity:  当前 regime 内累计涨跌% (区分强弱程度)
  hs300_ma60_z60:    沪深300 ma60 斜率相对 60 日历史 z-score
  cyb_rel_strength:  创业板 20 日相对沪深300 超额%
  zz500_rel_strength: 中证500 20 日相对沪深300 超额%

输出: output/regime_extra/regime_extra.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
REGIME_PATH = ROOT / "output" / "regimes" / "daily_regime.parquet"
OUT_DIR = ROOT / "output" / "regime_extra"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print("加载 daily_regime...")
    df = pd.read_parquet(REGIME_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 1. regime_days_in: 当前 regime 持续天数
    df["_block"] = (df["regime"] != df["regime"].shift()).cumsum()
    df["regime_days_in"] = df.groupby("_block").cumcount()

    # 2. regime_intensity: 当前 regime 块内累计涨跌
    # 用 ret_5d * regime_days_in / 5 近似 (每天 ret_5d 累加近似)
    # 更精确: 用 group block 内累计 ret_1d
    # 由于 daily_regime 已经有 hs300 close 衍生, 用 ret_5d 作代理
    df["regime_intensity"] = df.groupby("_block")["ret_5d"].transform(
        lambda x: x.fillna(0).cumsum() * 100 / 5  # 近似累计
    )

    # 3. hs300 ma60 斜率 z-score (用 ret_60d 已经有)
    # ret_60d 60 日 z-score
    df["hs300_ret60_z60"] = (df["ret_60d"] - df["ret_60d"].rolling(60).mean()) / \
                              df["ret_60d"].rolling(60).std()

    # 4. 创业板相对沪深300 超额 (20 日)
    df["cyb_rel_strength"] = df["cyb_ret_20d"] - df["ret_20d"]
    df["zz500_rel_strength"] = df["zz500_ret_20d"] - df["ret_20d"]

    # 输出
    out = df[["trade_date", "regime_days_in", "regime_intensity",
              "hs300_ret60_z60", "cyb_rel_strength", "zz500_rel_strength"]].copy()
    # 填充 inf/nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out.to_parquet(OUT_DIR / "regime_extra.parquet", index=False)

    print(f"\n=== 5 个市场状态特征统计 ===")
    print(out[["regime_days_in", "regime_intensity", "hs300_ret60_z60",
                "cyb_rel_strength", "zz500_rel_strength"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())
    print(f"\n写出 {OUT_DIR / 'regime_extra.parquet'}")


if __name__ == "__main__":
    main()
