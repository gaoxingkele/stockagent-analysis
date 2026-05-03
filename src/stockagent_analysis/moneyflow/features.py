"""Moneyflow 特征计算 — 12 个资金分层特征."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger("stockagent.moneyflow.features")

# 单位换算: tushare amount 是千元, /1e5 转为亿元
THOUSAND_TO_YI = 1.0 / 1e5


def compute_features(daily_mf: pd.DataFrame) -> pd.DataFrame:
    """从单股 raw moneyflow 计算 12 个特征.

    输入: daily_mf DataFrame, 必须按 trade_date 升序, 含 RAW_FIELDS.
    输出: DataFrame, 列:
      ts_code, trade_date,
      sm_net_5d, mid_net_5d, lg_net_5d, elg_net_5d, main_net_5d,
      lg_net_20d, elg_net_20d, main_net_20d,
      main_consec_in, main_consec_out,
      dispersion, elg_ratio, buy_sell_imb_5d
    """
    if daily_mf.empty: return pd.DataFrame()
    df = daily_mf.copy().sort_values("trade_date").reset_index(drop=True)
    # 按当日单层级净流入 (单位: 亿元)
    df["sm_net"]  = (df["buy_sm_amount"]  - df["sell_sm_amount"])  * THOUSAND_TO_YI
    df["mid_net"] = (df["buy_md_amount"]  - df["sell_md_amount"])  * THOUSAND_TO_YI
    df["lg_net"]  = (df["buy_lg_amount"]  - df["sell_lg_amount"])  * THOUSAND_TO_YI
    df["elg_net"] = (df["buy_elg_amount"] - df["sell_elg_amount"]) * THOUSAND_TO_YI
    df["main_net"] = df["lg_net"] + df["elg_net"]
    df["total_amount"] = (df["buy_sm_amount"] + df["sell_sm_amount"] +
                           df["buy_md_amount"] + df["sell_md_amount"] +
                           df["buy_lg_amount"] + df["sell_lg_amount"] +
                           df["buy_elg_amount"] + df["sell_elg_amount"]) * THOUSAND_TO_YI

    # 滚动 5 日 / 20 日累计
    for col in ["sm_net", "mid_net", "lg_net", "elg_net", "main_net"]:
        df[f"{col}_5d"]  = df[col].rolling(5,  min_periods=3).sum()
        df[f"{col}_20d"] = df[col].rolling(20, min_periods=10).sum()

    # 主力连续流入/流出天数
    main_pos = (df["main_net"] > 0).astype(int)
    main_neg = (df["main_net"] < 0).astype(int)
    # 累加直到符号变化
    consec_in  = []
    consec_out = []
    cnt_in = cnt_out = 0
    for i in range(len(df)):
        if main_pos.iloc[i]:
            cnt_in += 1; cnt_out = 0
        elif main_neg.iloc[i]:
            cnt_out += 1; cnt_in = 0
        else:
            cnt_in = cnt_out = 0
        consec_in.append(cnt_in)
        consec_out.append(cnt_out)
    df["main_consec_in"] = consec_in
    df["main_consec_out"] = consec_out

    # 主力 vs 散户分歧度 (5 日)
    df["dispersion_5d"] = np.sign(df["main_net_5d"]) - np.sign(df["sm_net_5d"])
    # +2 = 主力流入, 散户流出 (分歧最大, 也是经典"主力派发或主力建仓"形态)
    # -2 = 主力流出, 散户流入
    # 0 = 同向
    df["dispersion_5d"] = df["dispersion_5d"].fillna(0).astype(float)

    # 超大单占主力比 (大资金占比, 0-1+, 可能为负)
    df["elg_ratio_5d"] = df["elg_net_5d"] / df["main_net_5d"].replace(0, np.nan)
    df["elg_ratio_5d"] = df["elg_ratio_5d"].clip(-2, 3).fillna(0)

    # 买卖盘失衡度 (买盘 - 卖盘) / 总成交额
    df["buy_total"] = (df["buy_sm_amount"] + df["buy_md_amount"] +
                        df["buy_lg_amount"] + df["buy_elg_amount"]) * THOUSAND_TO_YI
    df["sell_total"] = (df["sell_sm_amount"] + df["sell_md_amount"] +
                         df["sell_lg_amount"] + df["sell_elg_amount"]) * THOUSAND_TO_YI
    df["buy_sell_imb"] = (df["buy_total"] - df["sell_total"]) / df["total_amount"].replace(0, np.nan)
    df["buy_sell_imb_5d"] = df["buy_sell_imb"].rolling(5, min_periods=3).mean()

    keep_cols = [
        "ts_code", "trade_date",
        "sm_net_5d", "mid_net_5d", "lg_net_5d", "elg_net_5d", "main_net_5d",
        "lg_net_20d", "elg_net_20d", "main_net_20d",
        "main_consec_in", "main_consec_out",
        "dispersion_5d", "elg_ratio_5d", "buy_sell_imb_5d",
    ]
    return df[keep_cols].copy()


def merge_to_parquet(features_df: pd.DataFrame, target_parquet: Path) -> int:
    """合并 moneyflow 特征到目标 parquet (按 ts_code+trade_date)."""
    base = pd.read_parquet(target_parquet)
    base["trade_date"] = base["trade_date"].astype(str)
    features_df = features_df.copy()
    features_df["trade_date"] = features_df["trade_date"].astype(str)
    merged = base.merge(features_df, on=["ts_code", "trade_date"], how="left")
    return len(merged)
