#!/usr/bin/env python3
"""实盘推荐审计: 把 0420 起所有留档推荐按满期统一回评。

核心指标 (区分 beta 拖累 vs 选股 alpha):
  - ret_H:    次日开盘买入, 持有 H 交易日的收益
  - mkt_ew:   同窗口全市场 (排 ST, open/close 有效) 等权平均收益 = "平均个股"
  - alpha:    ret_H - mkt_ew
  - pctile:   每只票 ret_H 在全市场横截面的分位 (50=平均, <50=选差, >50=有skill)
入场=次日开盘; 出场=H 交易日后收盘。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "output/tushare_cache/daily"

# ── 交易日索引 + 价格面板 ──
TDS = sorted(p.stem for p in CACHE.glob("*.parquet"))
TD_IDX = {d: i for i, d in enumerate(TDS)}
_panel: dict[str, pd.DataFrame] = {}

def panel(d: str) -> pd.DataFrame:
    if d not in _panel:
        df = pd.read_parquet(CACHE / f"{d}.parquet").set_index("ts_code")[["open", "close"]]
        _panel[d] = df
    return _panel[d]

basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[["ts_code", "name"]].drop_duplicates("ts_code").set_index("ts_code")
ST = set(basic[basic["name"].fillna("").str.contains("ST")].index)

def td_at(d: str, off: int) -> str | None:
    i = TD_IDX.get(d)
    if i is None or i + off >= len(TDS) or i + off < 0:
        return None
    return TDS[i + off]

def fwd_returns(entry_open_date: str, exit_close_date: str) -> pd.Series:
    """全市场 (排ST, open>0,close>0) 从 entry 开盘到 exit 收盘的收益 Series."""
    e = panel(entry_open_date); x = panel(exit_close_date)
    j = e.join(x["close"], rsuffix="_exit")
    j = j[(j["open"] > 0) & (j["close_exit"] > 0)]
    j = j[~j.index.isin(ST)]
    return (j["close_exit"] / j["open"] - 1.0) * 100.0

def eval_rec(date: str, tickers: list[str], label: str, horizons=(5, 10, 20)) -> list[dict]:
    rows = []
    entry = td_at(date, 1)
    if entry is None:
        return rows
    picks = [t for t in tickers if t not in ST]
    for H in horizons:
        exit_d = td_at(entry, H - 1)
        if exit_d is None:
            continue  # 未满期
        mkt = fwd_returns(entry, exit_d)
        if mkt.empty:
            continue
        pr = mkt.reindex(picks).dropna()
        if pr.empty:
            continue
        # 横截面分位 (每只票在全市场的百分位)
        rank_pct = mkt.rank(pct=True) * 100
        pctiles = rank_pct.reindex(pr.index).dropna()
        rows.append({
            "label": label, "date": date, "entry": entry, "exit": exit_d, "H": H,
            "n": len(pr), "pool": len(picks),
            "ret": round(pr.mean(), 2), "med": round(pr.median(), 2),
            "win%": round((pr > 0).mean() * 100, 0),
            "mkt_ew": round(mkt.mean(), 2),
            "alpha": round(pr.mean() - mkt.mean(), 2),
            "pctile_med": round(pctiles.median(), 1),
        })
    return rows

# ── 推荐集抽取 ──
def load_picks():
    out = []  # (date, tickers, label)
    # V7c 主推池
    for f in sorted((ROOT / "output/v7c_full_inference").glob("v7c_inference_2026*.csv")):
        d = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        rec = df[df["v7c_recommend"] == True]
        if len(rec):
            out.append((d, rec["ts_code"].tolist(), f"V7c主推@{d}"))
    # V12 final 排序 top20
    for f in sorted((ROOT / "output/v12_inference").glob("v12_inference_final_2026*.csv")):
        d = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        if "rank" in df.columns:
            df = df.sort_values("rank")
        out.append((d, df["ts_code"].head(20).tolist(), f"V12final20@{d}"))
    # daily_pick top20 (ratio 排序)
    for f in sorted((ROOT / "output/daily_pick").glob("top20_2026*.csv")):
        d = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        out.append((d, df["ts_code"].tolist(), f"Top20ratio@{d}"))
    return out

def main():
    recs = load_picks()
    print(f"审计 {len(recs)} 份推荐, 行情至 {TDS[-1]}\n")
    all_rows = []
    for d, tk, lab in recs:
        all_rows += eval_rec(d, tk, lab)
    res = pd.DataFrame(all_rows)
    res.to_csv(ROOT / "research/audit_live_recs_results.csv", index=False, encoding="utf-8-sig")

    # 按满期 H 分组汇总
    for H in (5, 10, 20):
        sub = res[res["H"] == H]
        if sub.empty:
            continue
        print(f"==================== 满 {H} 交易日 ({len(sub)} 份满期) ====================")
        print(sub[["label", "n", "ret", "win%", "mkt_ew", "alpha", "pctile_med"]].to_string(index=False))
        print(f"  >>> 合计: 平均收益 {sub['ret'].mean():+.2f}%  平均α {sub['alpha'].mean():+.2f}pp  "
              f"中位分位 {sub['pctile_med'].median():.1f}  正α份数 {(sub['alpha']>0).sum()}/{len(sub)}\n")

if __name__ == "__main__":
    main()
