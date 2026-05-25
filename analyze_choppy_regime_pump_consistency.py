"""震荡 regime 下 pump 信号一致性验证.

用户假设:
  - 双均线 MA5/MA60 (或 MA20/MA60) 反复缠绕 = 震荡行情
  - 震荡时模型方向不确定, 应该 pump↑ 和 pump↓ "同高同低", 而非一高一低
  - 趋势行情时 pump 信号清晰 (一高一低)

验证:
  1. 定义震荡指标 (3 种):
     - ma5_ma60_cross_60d: 过去 60 日 MA5 上下穿 MA60 次数 (越多越震荡)
     - ma5_ma60_dist_std: MA5/MA60 距离 std (低 = 缠绕)
     - ma20_ma60_cross_60d: MA20 上下穿 MA60 次数

  2. 把样本分两组: 震荡 (top 20% 缠绕) vs 趋势 (bot 20%)
  3. 比较两组:
     - pump↑ / pump↓ 均值 + std
     - 信号一致性指标 |pump↑ - pump↓|
     - "极端组合" 占比 (pump↑ 高 & pump↓ 低 = 清晰多 ; pump↑ 低 & pump↓ 高 = 清晰空)
     - "矛盾组合" 占比 (pump↑ 与 pump↓ 同高 或 同低)
     - 实际未来 5 日收益分布

  4. 如果成立: 震荡组矛盾比例显著高 + 实战收益差
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

OOS_START = "20251001"
OOS_END = "20260331"


def main():
    print(f"\n=== 震荡 regime 下 pump 一致性验证 ===\n", flush=True)

    # 1. 加载 daily, 算 MA5/MA20/MA60 + 缠绕指标
    print("[1] 加载 daily 算 MA 缠绕 ...", flush=True)
    daily_dir = ROOT / "output/tushare_cache/daily"
    files = sorted(daily_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["ma5"] = big.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    big["ma20"] = big.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(20, min_periods=20).mean())
    big["ma60"] = big.groupby("ts_code")["close"].transform(
        lambda x: x.rolling(60, min_periods=60).mean())
    big["ma5_above_ma60"] = (big["ma5"] > big["ma60"]).astype(int)
    big["ma20_above_ma60"] = (big["ma20"] > big["ma60"]).astype(int)
    # 过去 60 日上下穿次数 = abs(diff).sum()
    big["ma5_60_cross_60d"] = big.groupby("ts_code")["ma5_above_ma60"].transform(
        lambda x: x.diff().abs().rolling(60, min_periods=30).sum())
    big["ma20_60_cross_60d"] = big.groupby("ts_code")["ma20_above_ma60"].transform(
        lambda x: x.diff().abs().rolling(60, min_periods=30).sum())
    # 距离归一化 std (低 = 缠绕)
    big["ma5_60_dist"] = (big["ma5"] - big["ma60"]) / big["close"] * 100
    big["ma5_60_dist_abs_avg"] = big.groupby("ts_code")["ma5_60_dist"].transform(
        lambda x: x.rolling(20, min_periods=10).apply(lambda y: y.abs().mean()))
    # 防 leak: 用 shift(1)
    for c in ["ma5_60_cross_60d", "ma20_60_cross_60d", "ma5_60_dist_abs_avg"]:
        big[c] = big.groupby("ts_code")[c].shift(1)

    # 加 forward 5 日 r5 label
    big["next_open"] = big.groupby("ts_code")["close"].shift(-1)
    # 简化 r5 = close[t+5] / close[t] - 1 (代表 5 日实际表现)
    big["close_5d"] = big.groupby("ts_code")["close"].shift(-5)
    big["r5_actual"] = (big["close_5d"] / big["close"] - 1) * 100

    # 2. 推理 v3b pump_3way (5/15 截面之前的 OOS)
    print("\n[2] 加载 v3b 模型推理 OOS 期 pump↑/↓ ...", flush=True)
    df = load_window(OOS_START, OOS_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    lf = pd.read_parquet(ROOT / "output/long_return_features/features.parquet")
    lf["trade_date"] = lf["trade_date"].astype(str)
    df = df.merge(lf, on=["ts_code", "trade_date"], how="left")

    prod = ROOT / "output" / "production" / "r5_pump_3way_lgbm_v3b"
    b = lgb.Booster(model_str=(prod / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((prod / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})
    if "industry" in df.columns:
        df["industry_id"] = df["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in fc:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    proba = b.predict(df[fc].astype("float32"))
    df["P_neutral"] = proba[:, 0]
    df["P_down"] = proba[:, 1]
    df["P_up"] = proba[:, 2]

    # 3. merge 缠绕指标 + r5_actual
    print("\n[3] merge MA 缠绕 + r5 实际收益 ...", flush=True)
    df = df.merge(
        big[["ts_code", "trade_date", "ma5_60_cross_60d", "ma20_60_cross_60d",
              "ma5_60_dist_abs_avg", "r5_actual"]],
        on=["ts_code", "trade_date"], how="left")

    # 只看 OOS + 数据齐全
    df = df.dropna(subset=["ma5_60_cross_60d", "ma5_60_dist_abs_avg", "r5_actual", "P_up", "P_down"])
    df = df[(df["trade_date"] >= OOS_START) & (df["trade_date"] <= OOS_END)]
    print(f"  样本数: {len(df):,}", flush=True)

    # 4. 分组 (按 ma5_60_cross_60d): 趋势 (cross 少) vs 震荡 (cross 多)
    cross_p20 = df["ma5_60_cross_60d"].quantile(0.20)
    cross_p80 = df["ma5_60_cross_60d"].quantile(0.80)
    print(f"\n  缠绕分位: p20 = {cross_p20:.1f} 次, p80 = {cross_p80:.1f} 次", flush=True)
    df["regime"] = "中性"
    df.loc[df["ma5_60_cross_60d"] <= cross_p20, "regime"] = "趋势 (cross 少)"
    df.loc[df["ma5_60_cross_60d"] >= cross_p80, "regime"] = "震荡 (cross 多)"

    # 衍生信号
    df["consistency"] = (df["P_up"] - df["P_down"]).abs()  # 高 = 一致 (一高一低)
    df["clear_up"] = (df["P_up"] >= df["P_up"].quantile(0.8)) & \
                       (df["P_down"] <= df["P_down"].quantile(0.2))
    df["clear_down"] = (df["P_down"] >= df["P_down"].quantile(0.8)) & \
                         (df["P_up"] <= df["P_up"].quantile(0.2))
    df["contradiction_both_high"] = (df["P_up"] >= df["P_up"].quantile(0.7)) & \
                                       (df["P_down"] >= df["P_down"].quantile(0.7))
    df["contradiction_both_low"] = (df["P_up"] <= df["P_up"].quantile(0.3)) & \
                                      (df["P_down"] <= df["P_down"].quantile(0.3))

    # 5. 三组对比
    print(f"\n=== Regime × pump 一致性 + 实际 r5 ===\n", flush=True)
    print(f"  {'regime':20s} {'n':>8s} {'P↑均':>7s} {'P↓均':>7s} {'|↑-↓|':>7s} "
           f"{'清晰多%':>8s} {'清晰空%':>8s} {'同高%':>7s} {'同低%':>7s} {'r5 实%':>8s}")
    for reg in ["趋势 (cross 少)", "中性", "震荡 (cross 多)"]:
        sub = df[df["regime"] == reg]
        if sub.empty: continue
        print(f"  {reg:20s} {len(sub):>8,d} "
               f"{sub['P_up'].mean():>7.3f} {sub['P_down'].mean():>7.3f} "
               f"{sub['consistency'].mean():>7.3f} "
               f"{sub['clear_up'].mean()*100:>7.2f}% {sub['clear_down'].mean()*100:>7.2f}% "
               f"{sub['contradiction_both_high'].mean()*100:>6.2f}% "
               f"{sub['contradiction_both_low'].mean()*100:>6.2f}% "
               f"{sub['r5_actual'].mean():>+7.2f}%", flush=True)

    # 6. 关键测试: P↑ 高 + P↓ 低 在不同 regime 下的 r5 表现 (是不是震荡组失效?)
    print(f"\n=== 高确定性多头 (P↑ 高 & P↓ 低) 在各 regime 下的 r5 ===\n", flush=True)
    print(f"  {'regime':20s} {'清晰多 n':>10s} {'清晰多 r5':>10s} | "
           f"{'清晰空 n':>10s} {'清晰空 r5':>10s} | {'同高 n':>8s} {'同高 r5':>10s}")
    for reg in ["趋势 (cross 少)", "中性", "震荡 (cross 多)"]:
        sub = df[df["regime"] == reg]
        if sub.empty: continue
        cu = sub[sub["clear_up"]]
        cd = sub[sub["clear_down"]]
        ch = sub[sub["contradiction_both_high"]]
        cu_r5 = cu["r5_actual"].mean() if len(cu) > 0 else 0
        cd_r5 = cd["r5_actual"].mean() if len(cd) > 0 else 0
        ch_r5 = ch["r5_actual"].mean() if len(ch) > 0 else 0
        print(f"  {reg:20s} {len(cu):>10,d} {cu_r5:>+9.3f}% | "
               f"{len(cd):>10,d} {cd_r5:>+9.3f}% | "
               f"{len(ch):>8,d} {ch_r5:>+9.3f}%", flush=True)

    # 7. 假设验证总结
    print(f"\n=== 假设验证 ===\n", flush=True)
    trend = df[df["regime"] == "趋势 (cross 少)"]
    choppy = df[df["regime"] == "震荡 (cross 多)"]
    print(f"  H1: 震荡组 |P↑-P↓| 小于趋势组?", flush=True)
    print(f"      趋势 mean = {trend['consistency'].mean():.3f}, 震荡 mean = {choppy['consistency'].mean():.3f}",
           flush=True)
    print(f"      差异 = {trend['consistency'].mean() - choppy['consistency'].mean():+.3f}",
           flush=True)
    print(f"\n  H2: 震荡组矛盾 (同高/同低) 占比大于趋势组?", flush=True)
    print(f"      趋势 矛盾 = {(trend['contradiction_both_high']|trend['contradiction_both_low']).mean()*100:.1f}%",
           flush=True)
    print(f"      震荡 矛盾 = {(choppy['contradiction_both_high']|choppy['contradiction_both_low']).mean()*100:.1f}%",
           flush=True)
    print(f"\n  H3: 震荡组的清晰多 r5 实战收益是否显著低于趋势组?", flush=True)
    trend_cu = trend[trend["clear_up"]]["r5_actual"].mean()
    choppy_cu = choppy[choppy["clear_up"]]["r5_actual"].mean()
    print(f"      趋势组清晰多 r5 = {trend_cu:+.3f}%", flush=True)
    print(f"      震荡组清晰多 r5 = {choppy_cu:+.3f}%", flush=True)
    print(f"      差异 = {trend_cu - choppy_cu:+.3f}pp", flush=True)


if __name__ == "__main__":
    main()
