"""寻找"真正的启动子触发条件" - 批量验证 12+ 候选触发.

目标: 在已知启动子样本 (5 日涨 ≥10% & 回撤 ≤5%) 中, 找哪些前置 K 线条件
能显著放大信噪比 (≥ 2x), 同时保持合理 recall (≥ 30%).

候选包括:
A. 单点条件 (今日 K 线特征):
   - 放量 (vol > MA20 × N)
   - 跳空高开
   - 突破前高
   - 长阳实体
   - 锤子线 / 长下影
   - 接近涨停

B. 多 K 序列模式 (前 5-10 根 + 今日):
   - 缩量整理 + 放量突破
   - 多日下跌 + 长阳反转
   - 横盘整理 + 突破
   - 阴包阳后大阳
   - 双底 + 突破颈线

C. 组合 (上述 AND/OR):
   - 放量 + 突破前高
   - 长阳 + 跳空
   - 缩量整理 + 长阳

输出每个触发的: 频率 / 启动子比例 / 放大倍数 / recall
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent


def define_triggers(g: pd.DataFrame) -> dict:
    """对单个 ts_code 的 DataFrame g 计算所有触发条件.

    g 必须按 trade_date 升序, 含列: open, high, low, close, vol
    返回 dict[触发名 -> bool Series]
    """
    out = {}
    # 基础统计
    g_close = g["close"]
    g_open = g["open"]
    g_high = g["high"]
    g_low = g["low"]
    g_vol = g["vol"]

    vol_ma20 = g_vol.rolling(20, min_periods=20).mean()
    vol_ma5 = g_vol.rolling(5, min_periods=5).mean()
    high_max_20 = g_high.rolling(20, min_periods=20).max().shift(1)
    high_max_10 = g_high.rolling(10, min_periods=10).max().shift(1)
    low_min_10 = g_low.rolling(10, min_periods=10).min().shift(1)
    close_prev = g_close.shift(1)
    high_max_60 = g_high.rolling(60, min_periods=60).max().shift(1)

    # === A 单点 ===
    out["A1_vol_2x_ma20"] = g_vol > vol_ma20 * 2.0
    out["A2_vol_15x_ma20"] = g_vol > vol_ma20 * 1.5
    out["A3_gap_up_1pct"] = g_open > close_prev * 1.01
    out["A4_gap_up_2pct"] = g_open > close_prev * 1.02
    out["A5_break_20d_high"] = g_close > high_max_20
    out["A6_break_10d_high"] = g_close > high_max_10
    out["A7_break_60d_high"] = g_close > high_max_60
    out["A8_big_red_3pct"] = (g_close / g_open - 1) > 0.03
    out["A9_big_red_5pct"] = (g_close / g_open - 1) > 0.05
    out["A10_hammer"] = ((g_low.combine(g_open, min).combine(g_close, min) -
                            g_low).abs() / (g_high - g_low + 1e-9) > 0.4) & (g_close > g_open)
    out["A11_near_limit_up"] = (g_high / close_prev - 1) > 0.095  # 接近涨停 (主板 10%)

    # === B 序列模式 ===
    # B1: 缩量整理 (前 5 日 vol 都 < ma20) + 今日放量 1.5x
    cond_5d_low_vol = (g_vol.shift(1) < vol_ma20.shift(1)) & \
                       (g_vol.shift(2) < vol_ma20.shift(2)) & \
                       (g_vol.shift(3) < vol_ma20.shift(3)) & \
                       (g_vol.shift(4) < vol_ma20.shift(4)) & \
                       (g_vol.shift(5) < vol_ma20.shift(5))
    out["B1_squeeze_release"] = cond_5d_low_vol & (g_vol > vol_ma20 * 1.5)

    # B2: 多日下跌 (前 3 日 close 递减) + 长阳反转
    out["B2_down_3d_then_red"] = (g_close.shift(3) > g_close.shift(2)) & \
                                    (g_close.shift(2) > g_close.shift(1)) & \
                                    (g_close > g_open) & \
                                    ((g_close / g_open - 1) > 0.03)

    # B3: 横盘整理 (10 日 振幅 < 8%) + 突破前高
    range_10 = (g_high.rolling(10, min_periods=10).max().shift(1) /
                g_low.rolling(10, min_periods=10).min().shift(1) - 1)
    out["B3_consolidation_break"] = (range_10 < 0.08) & (g_close > high_max_10)

    # B4: 阴包阳后大阳 (前 2 日: 大阴 + 小阳 / 持平) + 今日大阳
    out["B4_dark_cloud_recover"] = (g_close.shift(2) < g_open.shift(2)) & \
                                      ((g_open.shift(2) - g_close.shift(2)) /
                                       g_open.shift(2) > 0.03) & \
                                      (g_close > g_open) & \
                                      ((g_close / g_open - 1) > 0.04)

    # B5: 双底 + 突破颈线 (10 日内两次低点接近 + 突破中间高点)
    # 简化: 前 5-10 日内 low_min_5 ≈ low_min_10 (二底接近) + 突破前 5 日 high max
    low_min_5 = g_low.rolling(5, min_periods=5).min().shift(1)
    high_max_5 = g_high.rolling(5, min_periods=5).max().shift(1)
    out["B5_double_bottom_break"] = (abs(low_min_5 - low_min_10) / low_min_10 < 0.02) & \
                                       (g_close > high_max_5)

    # B6: 长下影锤子 + 次日跟进
    yesterday_hammer = ((g_low.shift(1).combine(g_open.shift(1), min).combine(
        g_close.shift(1), min) - g_low.shift(1)).abs() /
        (g_high.shift(1) - g_low.shift(1) + 1e-9) > 0.4)
    out["B6_hammer_follow"] = yesterday_hammer & (g_close > g_open) & \
                                ((g_close / g_open - 1) > 0.02)

    # === C 组合 ===
    out["C1_vol2x_break20"] = out["A1_vol_2x_ma20"] & out["A5_break_20d_high"]
    out["C2_big_red_gap"] = out["A8_big_red_3pct"] & out["A3_gap_up_1pct"]
    out["C3_squeeze_break"] = out["B1_squeeze_release"] & out["A5_break_20d_high"]
    out["C4_break60_vol15x"] = out["A7_break_60d_high"] & out["A2_vol_15x_ma20"]

    return out


def main():
    t0 = time.time()
    print(f"=== 真正启动子触发分析 (15+ 候选 × 启动子 label) ===\n", flush=True)

    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    print(f"daily 文件: {len(files)} 个", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date",
                                            "open", "high", "low", "close", "vol"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name"]].drop_duplicates("ts_code")
    st_set = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    big = big[~big["ts_code"].isin(st_set)].reset_index(drop=True)
    print(f"排 ST 后: {len(big):,} bar", flush=True)

    # 计算所有触发条件 (groupby ts_code)
    print(f"\n[1] 计算 15+ 触发条件 (groupby ts_code apply) ...", flush=True)
    t1 = time.time()
    all_triggers = {}
    for ts, g in big.groupby("ts_code", sort=False):
        triggers = define_triggers(g)
        for name, ser in triggers.items():
            if name not in all_triggers:
                all_triggers[name] = []
            all_triggers[name].append(ser)
    # concat 回到 big
    for name, series_list in all_triggers.items():
        big[name] = pd.concat(series_list).sort_index().reindex(big.index)
    print(f"  触发条件计算完成 {time.time()-t1:.0f}s", flush=True)

    # 启动子 label (5 日)
    print(f"\n[2] 计算 5 日启动子 label ...", flush=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    N = 5
    big["max_high_next"] = (big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
    big["min_low_next"] = (big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))

    valid = big.dropna(subset=["next_open", "max_high_next", "min_low_next"]).copy()
    valid["upside"] = valid["max_high_next"] / valid["next_open"] - 1
    valid["downside"] = valid["min_low_next"] / valid["next_open"] - 1
    valid["is_pump"] = (valid["upside"] >= 0.10) & (valid["downside"] >= -0.05)

    n_total = len(valid)
    n_pump = int(valid["is_pump"].sum())
    baseline = n_pump / n_total
    print(f"  有效 bar: {n_total:,}, 启动子 bar: {n_pump:,} ({baseline*100:.2f}%, 基线)",
           flush=True)

    # 每个触发的 precision (= 触发时是启动子 比例) + recall + 放大
    print(f"\n[3] 评估每个触发的信噪比放大 + recall:\n", flush=True)
    print(f"  {'触发':28s} {'频率%':7s} {'启动子比例%':12s} {'放大x':8s} {'recall%':8s} {'F1':6s}",
           flush=True)
    results = []
    for name in [c for c in valid.columns if c.startswith(("A", "B", "C"))]:
        if not valid[name].dtype == bool: continue
        n_trig = int(valid[name].sum())
        n_trig_pump = int((valid[name] & valid["is_pump"]).sum())
        if n_trig == 0: continue
        freq = n_trig / n_total
        precision = n_trig_pump / n_trig
        recall = n_trig_pump / n_pump
        amplify = precision / baseline
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        results.append({
            "trigger": name, "freq": freq, "precision": precision,
            "amplify": amplify, "recall": recall, "f1": f1,
            "n_trig": n_trig, "n_trig_pump": n_trig_pump,
        })
        print(f"  {name:28s} {freq*100:6.2f} {precision*100:11.2f} "
               f"{amplify:7.2f} {recall*100:7.2f} {f1*100:5.2f}", flush=True)

    df = pd.DataFrame(results).sort_values("amplify", ascending=False)
    out_p = ROOT / "output" / "trigger_analysis"
    out_p.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p / "trigger_compare.csv", index=False)

    print(f"\n--- Top 5 by 放大倍数 ---", flush=True)
    for _, r in df.head(5).iterrows():
        print(f"  {r['trigger']:28s} 放大 {r['amplify']:.2f}x  "
               f"precision={r['precision']*100:.1f}%  recall={r['recall']*100:.1f}%", flush=True)

    print(f"\n--- Top 5 by F1 (precision+recall 平衡) ---", flush=True)
    df_f1 = df.sort_values("f1", ascending=False)
    for _, r in df_f1.head(5).iterrows():
        print(f"  {r['trigger']:28s} F1 {r['f1']*100:.1f}  "
               f"precision={r['precision']*100:.1f}%  recall={r['recall']*100:.1f}%", flush=True)

    print(f"\n  输出: {out_p / 'trigger_compare.csv'}")
    print(f"  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
