"""验证 OOS 期 (20260301-20260515) 是否普涨阶段."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"

df = pd.read_parquet(F3, columns=["ts_code", "trade_time", "trade_date", "r1_next_open"])
df["trade_time"] = pd.to_datetime(df["trade_time"])
df["trade_date"] = df["trade_date"].astype(str)

# 只看 EOD bar (避免单日重复)
eod = df[df["trade_time"].dt.hour == 15].copy()
eod = eod.dropna(subset=["r1_next_open"])
eod = eod[eod["r1_next_open"].abs() <= 20]

# OOS 切片
oos = eod[eod["trade_date"] >= "20260301"]
in_sample = eod[(eod["trade_date"] >= "20250101") & (eod["trade_date"] < "20260301")]

print(f"=== OOS (≥ 20260301) ===")
print(f"日数: {oos['trade_date'].nunique()}")
print(f"r1_next_open 全市场均值: {oos['r1_next_open'].mean():+.3f}%")
print(f"r1_next_open 全市场中位: {oos['r1_next_open'].median():+.3f}%")
print(f"上涨比例 (r1 > 0): {(oos['r1_next_open'] > 0).mean()*100:.1f}%")
print(f"大涨比例 (r1 > 2%): {(oos['r1_next_open'] > 2).mean()*100:.1f}%")
print(f"涨停接力比例 (r1 > 8%): {(oos['r1_next_open'] > 8).mean()*100:.1f}%")
print()

# 按日统计
print(f"=== 每日 r1 全市场均值分布 ===")
day_stats = oos.groupby("trade_date")["r1_next_open"].agg(["mean", "median"])
print(f"日均值 均值: {day_stats['mean'].mean():+.3f}%")
print(f"日均值 中位: {day_stats['mean'].median():+.3f}%")
print(f"日均值 std: {day_stats['mean'].std():.3f}%")
print(f"普涨日数 (日均 > 0.5%): {(day_stats['mean'] > 0.5).sum()}/{len(day_stats)}")
print(f"大涨日数 (日均 > 1%): {(day_stats['mean'] > 1.0).sum()}/{len(day_stats)}")
print(f"下跌日数 (日均 < 0): {(day_stats['mean'] < 0).sum()}/{len(day_stats)}")
print()

# 对比 in-sample
print(f"=== 对比训练期 (2025-2026/02) ===")
print(f"训练期 r1 均值: {in_sample['r1_next_open'].mean():+.3f}%")
print(f"训练期 上涨比例: {(in_sample['r1_next_open'] > 0).mean()*100:.1f}%")
print()

# 最强 10 日
print(f"=== OOS 内 10 个最大日 ===")
top10 = day_stats.sort_values("mean", ascending=False).head(10)
for d, r in top10.iterrows():
    n_up = (oos[oos["trade_date"]==d]["r1_next_open"] > 2).sum()
    n_tot = (oos["trade_date"]==d).sum()
    print(f"  {d}: 均值 {r['mean']:+.3f}%, 中位 {r['median']:+.3f}%, "
          f"大涨股 {n_up}/{n_tot} ({n_up/n_tot*100:.0f}%)")
