# -*- coding: utf-8 -*-
"""MS-001 — 市场状态指标 GOOD_TIMES (paper 复刻, 事前无前瞻) + 可行性诊断.

paper "In Good and in Bad Times?" 的 GOOD TIMES 复刻 (规避前瞻):
  MA200_state: 上月末指数收盘 > 其 200日均线 → 1 else 0
  UP_state:    上月末指数 > 3年前价格 → 1 else 0
  GOOD_TIMES = mean(MA200_state, UP_state) ∈ {0(熊), 0.5(过渡), 1(牛)}
  多指数 (沪深300 000300.SH + 中证500 000905.SH) 各算后再平均。
  每月初完全事前可用 (只用 <= 上月末信息)。

诊断: 打印 GOOD_TIMES 在 2022-2026 + de-lookahead OOS 窗(2024-10~2026-06) 的逐月序列,
判断 timing 测试是否有足够 bull/bear 变化 (若 OOS 窗几乎恒定 → 诚实数据测不动)。

输出: research/cache/ms_goodtimes.parquet (month, gt_hs300, gt_csi500, GOOD_TIMES)
数据: pro.index_daily (000300.SH/000905.SH); ST 无关 (指数). 不碰生产 (R05)。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache"
IDX_CACHE = CACHE / "ms_index_daily.parquet"
OUT_P = CACHE / "ms_goodtimes.parquet"
INDICES = {"000300.SH": "沪深300", "000905.SH": "中证500"}
FETCH_START, FETCH_END = "20180101", "20260612"


def _pro():
    if not os.environ.get("TUSHARE_TOKEN"):
        for line in open(ROOT / ".env", encoding="utf-8"):
            if line.startswith("TUSHARE_TOKEN="):
                os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
    import tushare as ts
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    return ts.pro_api()


def fetch_index() -> pd.DataFrame:
    if IDX_CACHE.exists():
        return pd.read_parquet(IDX_CACHE)
    pro = _pro()
    frames = []
    for code in INDICES:
        # index_daily 单次上限, 分年拉
        parts = []
        for y in range(2018, 2027):
            d = pro.index_daily(ts_code=code, start_date=f"{y}0101", end_date=f"{y}1231")
            if len(d):
                parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df = df[["ts_code", "trade_date", "close"]]
        frames.append(df)
        print(f"  {code} ({INDICES[code]}): {len(df)} 日", flush=True)
    allidx = pd.concat(frames, ignore_index=True)
    allidx["trade_date"] = allidx["trade_date"].astype(str)
    allidx = allidx.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    allidx.to_parquet(IDX_CACHE, index=False)
    return allidx


def build_goodtimes(idx: pd.DataFrame) -> pd.DataFrame:
    """每指数: 日级 MA200 + 3年前价格 → 月末状态 → GOOD_TIMES (赋下月, 事前)。"""
    out_per = {}
    for code in INDICES:
        d = idx[idx["ts_code"] == code].sort_values("trade_date").reset_index(drop=True)
        d["ma200"] = d["close"].rolling(200, min_periods=200).mean()
        d["px_3y_ago"] = d["close"].shift(750)   # ~3年(250交易日/年)
        d["ma200_state"] = (d["close"] > d["ma200"]).astype(float)
        d["up_state"] = (d["close"] > d["px_3y_ago"]).astype(float)
        d.loc[d["ma200"].isna(), "ma200_state"] = np.nan
        d.loc[d["px_3y_ago"].isna(), "up_state"] = np.nan
        d["month"] = d["trade_date"].str[:6]
        # 每月取该月最后一个交易日的状态 = 月末状态 (用于赋"下月")
        me = d.groupby("month").tail(1)[["month", "ma200_state", "up_state"]].copy()
        me["gt"] = (me["ma200_state"] + me["up_state"]) / 2.0
        # 月末状态赋下一月 (事前: 进入下月时已知上月末)
        months = sorted(me["month"])
        nxt = {months[i]: months[i + 1] for i in range(len(months) - 1)}
        me["apply_month"] = me["month"].map(nxt)
        out_per[code] = me.dropna(subset=["apply_month"]).set_index("apply_month")["gt"]
    g = pd.DataFrame(out_per)
    g.columns = [f"gt_{c.split('.')[0]}" for c in INDICES]
    g["GOOD_TIMES"] = g.mean(axis=1)
    g = g.reset_index().rename(columns={"apply_month": "month", "index": "month"})
    return g


def main():
    print("\n=== MS-001 GOOD_TIMES 市场状态指标 + 可行性诊断 ===\n", flush=True)
    idx = fetch_index()
    g = build_goodtimes(idx)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)} ({len(g)} 月)\n", flush=True)

    g2 = g[(g["month"] >= "202201")].copy()
    print("逐月 GOOD_TIMES (2022-01 起):", flush=True)
    for _, r in g2.iterrows():
        bar = {0.0: "熊", 0.5: "过渡", 1.0: "牛"}.get(round(r["GOOD_TIMES"], 1), f"{r['GOOD_TIMES']:.2f}")
        mark = " <OOS" if r["month"] >= "202410" else ""
        print(f"  {r['month']}  GT={r['GOOD_TIMES']:.2f} ({bar})  "
              f"[300={r['gt_000300']:.1f} 500={r['gt_000905']:.1f}]{mark}", flush=True)

    oos = g2[g2["month"] >= "202410"]
    full = g2
    print(f"\n[诊断] 全样本 2022-2026 ({len(full)}月): GT 分布 "
          f"熊={(full['GOOD_TIMES']==0).sum()} 过渡={((full['GOOD_TIMES']>0)&(full['GOOD_TIMES']<1)).sum()} "
          f"牛={(full['GOOD_TIMES']==1).sum()} | std={full['GOOD_TIMES'].std():.3f}", flush=True)
    print(f"[诊断] de-lookahead OOS 窗 2024-10~2026-06 ({len(oos)}月): GT 分布 "
          f"熊={(oos['GOOD_TIMES']==0).sum()} 过渡={((oos['GOOD_TIMES']>0)&(oos['GOOD_TIMES']<1)).sum()} "
          f"牛={(oos['GOOD_TIMES']==1).sum()} | std={oos['GOOD_TIMES'].std():.3f}", flush=True)
    feasible = oos["GOOD_TIMES"].std() > 0.15
    print(f"\n[结论] OOS窗 GOOD_TIMES {'有足够变化, timing测试可行' if feasible else '变化不足, 诚实OOS窗测不动timing (需更长序列+lookahead警告)'}"
          f" (std阈值0.15)", flush=True)
    return feasible


if __name__ == "__main__":
    main()
