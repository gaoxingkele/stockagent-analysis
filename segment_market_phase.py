"""市场风格分段: 用上证指数 (000001.SH) 60 日 MA + 60 日涨跌幅划分牛/熊/震荡。

输出: output/backtest_3y_2023_2026/market_phases.json
  [{trade_date, close, ma60, mom60, phase: bull/bear/sideways}]
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import tushare as ts

OUT = ROOT / "output" / "backtest_3y_2023_2026" / "market_phases.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

# 拉上证指数
df = pro.index_daily(ts_code="000001.SH", start_date="20221001", end_date="20260331")
df = df.sort_values("trade_date").reset_index(drop=True)

closes = df["close"].astype(float).values

# MA60
ma60 = np.full(len(closes), np.nan)
for i in range(59, len(closes)):
    ma60[i] = closes[i-59:i+1].mean()

# 60日动量
mom60 = np.full(len(closes), np.nan)
for i in range(60, len(closes)):
    mom60[i] = (closes[i] - closes[i-60]) / closes[i-60] * 100

phases = []
for i, row in df.iterrows():
    td = row["trade_date"]
    if td < "20230101": continue
    c = closes[i]; m = ma60[i]; mo = mom60[i]
    if np.isnan(m) or np.isnan(mo):
        ph = "unknown"
    elif c > m and mo > 5:
        ph = "bull"
    elif c < m and mo < -5:
        ph = "bear"
    else:
        ph = "sideways"
    phases.append({
        "trade_date": td, "close": float(c),
        "ma60": float(m) if not np.isnan(m) else None,
        "mom60": float(mo) if not np.isnan(mo) else None,
        "phase": ph,
    })

OUT.write_text(json.dumps(phases, ensure_ascii=False), encoding="utf-8")

# 各 phase 占比
from collections import Counter
cnt = Counter(p["phase"] for p in phases)
print(f"上证指数 2023-01 ~ 2026-03 阶段分布:")
for k, v in cnt.most_common():
    print(f"  {k}: {v} 天 ({v/len(phases)*100:.1f}%)")

# 各 phase 区间
print(f"\n阶段切换点:")
last_phase = None
for p in phases:
    if p["phase"] != last_phase:
        print(f"  {p['trade_date']}  →  {p['phase']}  (close={p['close']:.0f}, mom60={p['mom60']:+.1f}%)" if p['mom60'] else f"  {p['trade_date']}  →  {p['phase']}")
        last_phase = p["phase"]

print(f"\n写入: {OUT}")
