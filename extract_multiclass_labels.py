#!/usr/bin/env python3
"""新版多类标签提取 — 5 类样本.

正样本 (3 类, 都要求 max_dd >= -5% AND MA5 不连续 5 天向下):
  mild       : max_gain > 15%, < 20%
  strong     : max_gain > 20%, < 25%
  aggressive : max_gain > 25%

负样本:
  fake_pump  : 前 5 天涨过 +3%, 后续跌破 -8%, 跌破后 20 日内没再回到买点

中性:
  sideways   : max_gain < 12% AND max_dd > -8%

不属于任何类:
  other      : 其他混合情况

输出: output/labels_v2/multiclass_labels.parquet
列: ts_code, trade_date, entry_open, max_gain_20, max_dd_20,
    ma5_max_consec_down, early_peak_pct, later_low_pct, label
"""
from __future__ import annotations
import os, struct, time, json, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR = ROOT / "output" / "labels_v2"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

START = "20230101"
END   = "20260126"
LOOKAHEAD = 20

# 阈值
MAX_DD_THRESH = -5.0     # 正样本回撤不能 < -5%
MA5_CONSEC_DOWN = 5      # 正样本 MA5 不连续 5 天向下 (>=5 reject)
MA5_FLAT = 0.003         # MA5 走平 / 向下 阈值

# 假上涨参数
FAKE_EARLY_DAYS = 5      # 前 N 天看最高
FAKE_EARLY_PEAK = 3.0    # 前 N 天最高 >= +3%
FAKE_DUMP = -8.0         # 之后跌破 -8%

# 中性参数
SIDEWAYS_MAX = 12.0      # max_gain < 12%
SIDEWAYS_DD = -8.0       # max_dd > -8%


def read_tdx(market, code):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    try: data = p.read_bytes()
    except: return None
    n = len(data) // 32
    if n == 0: return None
    out = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        out.append((d.strftime("%Y%m%d"),
                     f[1]/100.0, f[2]/100.0, f[3]/100.0, f[4]/100.0))
    return out


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def label_one_sample(daily, t0_idx) -> dict | None:
    """对单股单日打标签."""
    n = len(daily)
    if t0_idx + 1 + LOOKAHEAD > n: return None
    entry_open = daily[t0_idx + 1][1]
    if entry_open <= 0: return None

    future = daily[t0_idx + 1: t0_idx + 1 + LOOKAHEAD]
    if len(future) < LOOKAHEAD: return None

    highs = np.array([r[2] for r in future])
    lows  = np.array([r[3] for r in future])
    closes = np.array([r[4] for r in future])

    max_gain = (highs.max() / entry_open - 1) * 100
    max_dd   = (lows.min() / entry_open - 1) * 100

    # MA5 (用历史 close 计算 + 未来 close)
    if t0_idx + 1 < 4: return None  # 至少 4 天历史以算 MA5
    hist_closes = np.array([daily[t0_idx + 1 - 4 + j][4] for j in range(4)])
    ma5_seq = []
    for i in range(LOOKAHEAD):
        if i < 1:  # 第一天 MA5
            window = np.append(hist_closes, closes[0])
        else:
            # 滚动: 最近 5 天 close
            if i < 5:
                window = np.append(hist_closes[i:], closes[:i+1])
            else:
                window = closes[i-4: i+1]
        ma5_seq.append(window.mean())
    ma5_seq = np.array(ma5_seq)

    # MA5 连续向下天数
    consec_down = 0; max_consec = 0
    for i in range(1, len(ma5_seq)):
        change = (ma5_seq[i] - ma5_seq[i-1]) / max(ma5_seq[i-1], 1e-9)
        if change < -MA5_FLAT:
            consec_down += 1
            max_consec = max(max_consec, consec_down)
        else:
            consec_down = 0

    # 前 5 天最高 + 后续是否跌破
    early_peak = (highs[:FAKE_EARLY_DAYS].max() / entry_open - 1) * 100
    early_peak_idx = int(np.argmax(highs[:FAKE_EARLY_DAYS]))
    after_peak_lows = lows[early_peak_idx + 1:] if early_peak_idx + 1 < LOOKAHEAD else np.array([])
    after_peak_low = (after_peak_lows.min() / entry_open - 1) * 100 if len(after_peak_lows) > 0 else 0

    # ── 标签判定 ──
    label = "other"

    # 1. 假上涨
    if early_peak >= FAKE_EARLY_PEAK and after_peak_low <= FAKE_DUMP:
        # 找到跌破点
        dump_idx = None
        for i in range(early_peak_idx + 1, LOOKAHEAD):
            if (lows[i] / entry_open - 1) * 100 <= FAKE_DUMP:
                dump_idx = i; break
        if dump_idx is not None and dump_idx < LOOKAHEAD - 1:
            # 跌破后是否回到买点
            after_dump_max = (highs[dump_idx + 1:].max() / entry_open - 1) * 100
            if after_dump_max < 0:
                label = "fake_pump"

    # 2. 正样本: dd >= -5%, MA5 不连续 5 天向下
    if label == "other" and max_dd >= MAX_DD_THRESH and max_consec < MA5_CONSEC_DOWN:
        if max_gain >= 25:
            label = "aggressive"
        elif max_gain >= 20:
            label = "strong"
        elif max_gain >= 15:
            label = "mild"

    # 3. 震荡
    if label == "other" and max_gain < SIDEWAYS_MAX and max_dd > SIDEWAYS_DD:
        label = "sideways"

    return {
        "trade_date": daily[t0_idx][0],
        "entry_open": round(entry_open, 3),
        "max_gain_20": round(max_gain, 2),
        "max_dd_20": round(max_dd, 2),
        "ma5_max_consec_down": int(max_consec),
        "early_peak_pct": round(early_peak, 2),
        "after_peak_low_pct": round(after_peak_low, 2),
        "label": label,
    }


def main():
    t0 = time.time()
    print("加载股票列表...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code","trade_date"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        parts.append(df)
    keys = pd.concat(parts).drop_duplicates()
    ts_codes = keys["ts_code"].unique()
    print(f"股票数: {len(ts_codes)}")

    rows = []
    for i, ts in enumerate(ts_codes):
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(ts_codes)}] {int(time.time()-t0)}s, 已 {len(rows)} 行")
        c, m = code_market(ts)
        daily = read_tdx(m, c)
        if not daily or len(daily) < 30: continue
        # 索引 trade_date → idx
        # 只看在 (START, END) 内的 t0
        for j in range(4, len(daily) - LOOKAHEAD - 1):
            td = daily[j][0]
            if td < START or td > END: continue
            res = label_one_sample(daily, j)
            if res is None: continue
            res["ts_code"] = ts
            rows.append(res)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "multiclass_labels.parquet", index=False)
    print(f"\n总样本: {len(df):,}")

    # 分布
    print("\n=== 标签分布 ===")
    dist = df["label"].value_counts()
    for k, v in dist.items():
        print(f"  {k:<12}: {v:>10,} ({v/len(df)*100:>5.2f}%)")

    # 各正样本 max_gain 分布
    print("\n=== 各类样本特征统计 ===")
    for lab in ["mild", "strong", "aggressive", "fake_pump", "sideways", "other"]:
        sub = df[df["label"] == lab]
        if len(sub) == 0: continue
        print(f"\n[{lab}] n={len(sub):,}")
        print(f"  max_gain: 均值 {sub['max_gain_20'].mean():.2f}% 中位 {sub['max_gain_20'].median():.2f}% "
              f"min {sub['max_gain_20'].min():.2f}% max {sub['max_gain_20'].max():.2f}%")
        print(f"  max_dd:   均值 {sub['max_dd_20'].mean():.2f}% 中位 {sub['max_dd_20'].median():.2f}%")
        print(f"  MA5 连续向下: 均值 {sub['ma5_max_consec_down'].mean():.2f} 天")

    # 按年分布
    print("\n=== 按年标签分布 ===")
    df["_year"] = df["trade_date"].str[:4]
    yp = df.pivot_table(index="_year", columns="label",
                          values="ts_code", aggfunc="count", fill_value=0)
    print(yp.to_string())

    # 元数据
    summary = {
        "total": int(len(df)),
        "n_stocks": int(df["ts_code"].nunique()),
        "n_dates": int(df["trade_date"].nunique()),
        "label_dist": dist.to_dict(),
        "thresholds": {
            "MAX_DD_THRESH": MAX_DD_THRESH, "MA5_CONSEC_DOWN": MA5_CONSEC_DOWN,
            "FAKE_EARLY_PEAK": FAKE_EARLY_PEAK, "FAKE_DUMP": FAKE_DUMP,
            "SIDEWAYS_MAX": SIDEWAYS_MAX, "SIDEWAYS_DD": SIDEWAYS_DD,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n输出: {OUT_DIR}, 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
