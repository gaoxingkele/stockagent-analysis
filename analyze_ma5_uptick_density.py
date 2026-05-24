"""验证 MA5 上拐作为启动子触发信号的有效性.

MA5 上拐定义 (严格):
  MA5[t] > MA5[t-1]  AND  MA5[t-1] <= MA5[t-2]
  即 MA5 三日 V 形, 拐头向上

要算 3 个关键指标:
  1. MA5 上拐时点占总 bar 比例 (估 10-20%)
  2. MA5 上拐时点的"启动子"占比 (启动子 = 5 日涨 ≥10% & 回撤 ≤5%)
  3. 启动子的"MA5 上拐覆盖率" (有多少启动子真的在 MA5 上拐 ±1 bar 内触发)

如果 #2 显著高于 12% (无 MA5 滤的基线), 且 #3 > 70%, 则方案成立.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    print(f"=== MA5 上拐 + 启动子 联合分析 ===\n", flush=True)

    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    print(f"daily 文件: {len(files)} 个", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # ST 排除
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name"]].drop_duplicates("ts_code")
    st_set = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    big = big[~big["ts_code"].isin(st_set)].reset_index(drop=True)
    print(f"排 ST 后: {len(big):,} bar, {big['ts_code'].nunique()} 股", flush=True)

    # MA5 + 上拐信号
    print(f"\n[1] 计算 MA5 + 上拐信号 ...", flush=True)
    big["ma5"] = big.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    big["ma5_prev"] = big.groupby("ts_code")["ma5"].shift(1)
    big["ma5_prev2"] = big.groupby("ts_code")["ma5"].shift(2)
    # 严格上拐: MA5[t] > MA5[t-1] AND MA5[t-1] <= MA5[t-2]
    big["ma5_uptick"] = (big["ma5"] > big["ma5_prev"]) & \
                          (big["ma5_prev"] <= big["ma5_prev2"])

    # 启动子: 未来 5 日涨 ≥10% AND 回撤 ≤5%
    print(f"[2] 计算 5 日 forward 启动子 label ...", flush=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    N = 5
    big["max_high_next"] = (big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
    big["min_low_next"] = (big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))

    valid = big.dropna(subset=["next_open", "max_high_next", "min_low_next",
                                  "ma5", "ma5_prev", "ma5_prev2"]).copy()
    valid["upside"] = valid["max_high_next"] / valid["next_open"] - 1
    valid["downside"] = valid["min_low_next"] / valid["next_open"] - 1
    valid["is_strict"] = (valid["upside"] >= 0.10) & (valid["downside"] >= -0.05)

    n_total = len(valid)
    n_uptick = int(valid["ma5_uptick"].sum())
    n_strict = int(valid["is_strict"].sum())
    n_uptick_and_strict = int((valid["ma5_uptick"] & valid["is_strict"]).sum())
    n_uptick_not_strict = int((valid["ma5_uptick"] & ~valid["is_strict"]).sum())
    n_not_uptick_strict = int((~valid["ma5_uptick"] & valid["is_strict"]).sum())

    print(f"\n=== 核心指标 ===\n")
    print(f"  有效 bar (含 MA5 + 5 日 forward): {n_total:,}", flush=True)
    print(f"  MA5 上拐 bar: {n_uptick:,} ({n_uptick/n_total*100:.2f}%)", flush=True)
    print(f"  启动子 bar (5日涨10%回撤5%): {n_strict:,} ({n_strict/n_total*100:.2f}%)", flush=True)
    print(f"\n--- 联合分布 (核心!) ---", flush=True)
    print(f"  MA5 上拐 AND 启动子:    {n_uptick_and_strict:,} "
           f"({n_uptick_and_strict/n_total*100:.3f}%)", flush=True)
    print(f"  MA5 上拐 但 NOT 启动子: {n_uptick_not_strict:,} "
           f"({n_uptick_not_strict/n_total*100:.3f}%)", flush=True)
    print(f"  NOT MA5 上拐 但启动子:  {n_not_uptick_strict:,} "
           f"({n_not_uptick_strict/n_total*100:.3f}%)", flush=True)

    print(f"\n--- 信噪比 (TCN 训练价值) ---", flush=True)
    if n_uptick > 0:
        precision_uptick = n_uptick_and_strict / n_uptick
        print(f"  MA5 上拐时点的启动子比例: {precision_uptick*100:.2f}% "
               f"(无过滤基线 {n_strict/n_total*100:.2f}%)", flush=True)
        print(f"  信噪比放大: {precision_uptick / (n_strict/n_total):.2f}x", flush=True)

    print(f"\n--- 启动子覆盖率 (启动子有多少被 MA5 上拐捕捉) ---", flush=True)
    if n_strict > 0:
        recall_uptick = n_uptick_and_strict / n_strict
        print(f"  启动子在 MA5 上拐时点触发的比例 (recall): "
               f"{recall_uptick*100:.2f}%", flush=True)
        print(f"  漏掉的启动子: {n_not_uptick_strict:,} "
               f"({(1-recall_uptick)*100:.2f}%, 这些没 MA5 上拐信号触发)", flush=True)

    # ±1 bar 放宽
    print(f"\n--- 放宽到 ±1 bar (MA5 上拐前后 1 天) ---", flush=True)
    valid["ma5_uptick_next1"] = valid.groupby("ts_code")["ma5_uptick"].shift(-1)
    valid["ma5_uptick_prev1"] = valid.groupby("ts_code")["ma5_uptick"].shift(1)
    valid["ma5_uptick_window"] = (valid["ma5_uptick"] |
                                     valid["ma5_uptick_next1"].fillna(False) |
                                     valid["ma5_uptick_prev1"].fillna(False))
    n_window = int(valid["ma5_uptick_window"].sum())
    n_window_strict = int((valid["ma5_uptick_window"] & valid["is_strict"]).sum())
    print(f"  ±1 bar 内 MA5 上拐: {n_window:,} ({n_window/n_total*100:.2f}%)", flush=True)
    if n_window > 0:
        prec_window = n_window_strict / n_window
        print(f"  窗口内启动子比例: {prec_window*100:.2f}% "
               f"(放大 {prec_window / (n_strict/n_total):.2f}x)", flush=True)
    if n_strict > 0:
        recall_window = n_window_strict / n_strict
        print(f"  窗口覆盖启动子: {recall_window*100:.2f}%", flush=True)

    # 跌启动子 (空头模式) - 用户提的反向学习
    print(f"\n\n=== 反向: MA5 下拐 + 跌启动子 ===\n", flush=True)
    big["ma5_downtick"] = (big["ma5"] < big["ma5_prev"]) & \
                            (big["ma5_prev"] >= big["ma5_prev2"])
    valid["ma5_downtick"] = (valid["ma5"] < valid["ma5_prev"]) & \
                              (valid["ma5_prev"] >= valid["ma5_prev2"])
    valid["is_strict_down"] = (valid["downside"] <= -0.10) & (valid["upside"] <= 0.05)
    n_downtick = int(valid["ma5_downtick"].sum())
    n_strict_down = int(valid["is_strict_down"].sum())
    n_downtick_strict = int((valid["ma5_downtick"] & valid["is_strict_down"]).sum())
    print(f"  MA5 下拐 bar: {n_downtick:,} ({n_downtick/n_total*100:.2f}%)", flush=True)
    print(f"  跌启动子 bar (5日跌10%反弹5%): {n_strict_down:,} ({n_strict_down/n_total*100:.2f}%)",
           flush=True)
    print(f"  MA5 下拐 AND 跌启动子: {n_downtick_strict:,} "
           f"({n_downtick_strict/n_total*100:.3f}%)", flush=True)
    if n_downtick > 0:
        prec_down = n_downtick_strict / n_downtick
        print(f"  MA5 下拐时点跌启动子比例: {prec_down*100:.2f}% "
               f"(基线 {n_strict_down/n_total*100:.2f}%, 放大 {prec_down / (n_strict_down/n_total):.2f}x)",
               flush=True)
    if n_strict_down > 0:
        rec_down = n_downtick_strict / n_strict_down
        print(f"  跌启动子在 MA5 下拐时点 recall: {rec_down*100:.2f}%", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
