"""龙虎榜因子工程: 从 top_list + top_inst 构建日级 (ts_code, trade_date) 因子.

输入:
  output/tushare_cache/top_list.parquet  上榜个股明细 (61,548 行)
  output/tushare_cache/top_inst.parquet  营业部/机构明细 (633,613 行)

输出:
  output/top_list_features/features.parquet  (ts_code, trade_date) 级 12 个因子

因子设计 (跨股票分布, 按 ts_code × trade_date 唯一):
  A 龙虎榜基础:
    1. tl_in_list: 当日上龙虎榜 (0/1)
    2. tl_net_rate: 净买入率 (机构净买入额 / 总成交)
    3. tl_amount_rate: 龙虎榜成交占当日总成交比
    4. tl_reason_buy: reason 含"涨幅" / "买入" (上涨型上榜)
    5. tl_reason_sell: reason 含"跌幅" / "卖出" (下跌型上榜)

  B 营业部/机构:
    6. ti_inst_net_buy: 当日"机构专用"席位净买入额
    7. ti_inst_count: 机构席位数量
    8. ti_hot_seat_net_buy: 知名游资席位 (含"拉萨""陆家嘴""陈小群")净买入

  C 历史累计:
    9. tl_count_30d: 近 30 交易日上榜次数
    10. tl_net_buy_30d_sum: 近 30 日累计净买入
    11. days_since_last_tl: 距上次上榜天数 (热度时效, 用大数填非上榜)
    12. ti_inst_net_buy_30d: 近 30 日机构净买入累计

注意:
  - 非上榜日所有因子为 0 (不是 NaN), 让模型可以用 "is_in_list=0" 学到信号
  - 跨股票合并到主因子 pipeline
"""
from __future__ import annotations
import time
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "output" / "tushare_cache"
OUT_DIR = ROOT / "output" / "top_list_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 知名游资席位关键词 (中信证券拉萨 / 国泰君安拉萨 / 银河证券绍兴 / 华泰证券深圳益田 / 陆家嘴 / 等)
HOT_SEAT_KW = ["拉萨", "陆家嘴", "深圳益田", "绍兴", "成都北一环", "陈小群"]


def main():
    t0 = time.time()
    print(f"\n=== 龙虎榜因子工程 ===\n", flush=True)

    tl = pd.read_parquet(CACHE / "top_list.parquet")
    ti = pd.read_parquet(CACHE / "top_inst.parquet")
    print(f"  top_list: {len(tl):,}, top_inst: {len(ti):,}", flush=True)
    tl["trade_date"] = tl["trade_date"].astype(str)
    ti["trade_date"] = ti["trade_date"].astype(str)

    # === A 上榜基础 (top_list 每股每日 1 行) ===
    print(f"\n[A] 上榜基础因子 ...", flush=True)
    tl["tl_in_list"] = 1
    tl["tl_net_rate"] = tl["net_rate"].fillna(0)
    tl["tl_amount_rate"] = tl["amount_rate"].fillna(0)
    # reason 解析
    tl["tl_reason_buy"] = tl["reason"].fillna("").apply(
        lambda x: int(any(k in x for k in ["涨幅", "异常", "买入", "上榜"]))
    )
    tl["tl_reason_sell"] = tl["reason"].fillna("").apply(
        lambda x: int(any(k in x for k in ["跌幅", "卖出", "跌", "降"]))
    )

    base_cols = ["ts_code", "trade_date", "tl_in_list", "tl_net_rate", "tl_amount_rate",
                   "tl_reason_buy", "tl_reason_sell"]
    a_df = tl[base_cols].drop_duplicates(["ts_code", "trade_date"], keep="first")
    print(f"  A 因子: {len(a_df):,} 行 (每股每日 1 行)", flush=True)

    # === B 营业部 / 机构 (top_inst 每股每日多行 → 聚合) ===
    print(f"\n[B] 机构/游资因子 ...", flush=True)
    ti["is_inst"] = ti["exalter"].fillna("").str.contains("机构专用", regex=False).astype(int)
    ti["is_hot_seat"] = ti["exalter"].fillna("").apply(
        lambda x: int(any(k in x for k in HOT_SEAT_KW))
    )
    ti["net_buy"] = ti["net_buy"].fillna(0)

    b_df = ti.groupby(["ts_code", "trade_date"]).agg(
        ti_inst_net_buy=("net_buy", lambda x: x[ti.loc[x.index, "is_inst"] == 1].sum()),
        ti_inst_count=("is_inst", "sum"),
        ti_hot_seat_net_buy=("net_buy", lambda x: x[ti.loc[x.index, "is_hot_seat"] == 1].sum()),
    ).reset_index()
    print(f"  B 因子: {len(b_df):,} 行", flush=True)

    # 合并 A + B
    merged = a_df.merge(b_df, on=["ts_code", "trade_date"], how="left")
    for c in ["ti_inst_net_buy", "ti_inst_count", "ti_hot_seat_net_buy"]:
        merged[c] = merged[c].fillna(0)
    print(f"  A+B merge: {len(merged):,}", flush=True)

    # === C 历史累计 (按 ts_code rolling) ===
    print(f"\n[C] 历史累计因子 (30 日 rolling) ...", flush=True)
    # 注意: 非上榜日没有 row, 需要先 reindex 出 full daily
    # 简化: 直接对上榜行做 rolling count, 时间是稀疏的
    # 改用方法: 对每只股, 按 trade_date 升序排, 算"距上次"和"近 30 上榜数"

    merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    merged["trade_date_dt"] = pd.to_datetime(merged["trade_date"])

    # tl_count_30d: 当前日往前 30 天内, 该股上榜几次 (含当日)
    # 用 rolling 但 trade_date 是日历日, 用 expanding 太慢. 简化: rolling(window=30, on='date_idx')
    # 用 numeric index per ts_code 替代日历日 rolling

    # 直接用 sort + groupby transform/cumcount, 避免 apply 丢列
    print(f"  rolling 30d (用 groupby + rolling transform) ...", flush=True)
    merged = merged.set_index("trade_date_dt")
    # tl_count_30d
    merged["tl_count_30d"] = (
        merged.groupby("ts_code")["tl_in_list"]
              .transform(lambda x: x.rolling("30D").sum())
    )
    # tl_net_buy_30d_sum
    merged["tl_net_buy_30d_sum"] = (
        merged.groupby("ts_code")["ti_inst_net_buy"]
              .transform(lambda x: x.rolling("30D").sum())
    )
    merged = merged.reset_index()
    # days_since_last_tl
    merged = merged.sort_values(["ts_code", "trade_date_dt"]).reset_index(drop=True)
    merged["prev_date"] = merged.groupby("ts_code")["trade_date_dt"].shift(1)
    merged["days_since_last_tl"] = (merged["trade_date_dt"] - merged["prev_date"]).dt.days.fillna(999)
    merged["ti_inst_net_buy_30d"] = merged["tl_net_buy_30d_sum"]
    print(f"  rolling 完成 {time.time()-t0:.0f}s", flush=True)

    # 输出
    out_cols = ["ts_code", "trade_date",
                  "tl_in_list", "tl_net_rate", "tl_amount_rate",
                  "tl_reason_buy", "tl_reason_sell",
                  "ti_inst_net_buy", "ti_inst_count", "ti_hot_seat_net_buy",
                  "tl_count_30d", "tl_net_buy_30d_sum", "days_since_last_tl",
                  "ti_inst_net_buy_30d"]
    out = merged[out_cols].copy()
    out_p = OUT_DIR / "features.parquet"
    out.to_parquet(out_p, index=False)
    print(f"\n输出: {out_p}", flush=True)
    print(f"  行数: {len(out):,}", flush=True)
    print(f"  覆盖股: {out['ts_code'].nunique():,}, 上榜日: {out['trade_date'].nunique():,}",
           flush=True)
    print(f"  列: {out.columns.tolist()}", flush=True)

    # 简要统计
    print(f"\n--- 因子统计 (上榜日) ---")
    for c in out_cols[2:]:
        if c in out.columns and out[c].dtype != "object":
            print(f"  {c:25s}: 均 {out[c].mean():+.2e}, std {out[c].std():.2e}, "
                   f"非零 {(out[c] != 0).mean()*100:.1f}%", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
