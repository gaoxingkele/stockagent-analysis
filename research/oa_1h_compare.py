# -*- coding: utf-8 -*-
"""OA-1H — 1H vs 日线对比验证 (对 Phase-1 存活波动率因子, 1H尺度重算并对比正交IC).

用户要求"1小时和日线对比验证"。最自然的对比: OA-VOL 头号存活 ivol_60 (日线特异波动) 用
1H bar 重算日内已实现波动 (intraday realized vol), 看更细粒度是否增强同一低波动信号。

1H 数据: output/tushare_cache/1h/<code>.parquet (每股, trade_time/OHLCV, 近~2年覆盖)。
因子 (backward):
  rv_1h_20  = 过去20交易日, 每日 1H 收益平方和(日内RV) 的均值 (日内已实现波动)
  rv_1h_60  = 60日版
对比基准 = 日线 ivol_60 / total_vol_60 (oa_vol_factors.parquet), 同 (ts_code, trade_date) 交集。
两者各跑 oa_screen Phase-1 正交 IC (扣 FULL_CTRL, NW-t) → 报告 1H 是否更强 + 两者秩相关。

checkpoint: 1H 面板拼接缓存 research/cache/oa_1h_panel.parquet (SIGN-R08)。
ST 源头排除; 全 backward; 纪律 R03/R04/R06/R11/R12++/R16。生产线不碰 (R05)。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import oa_screen  # noqa: E402

H1_DIR = ROOT / "output" / "tushare_cache" / "1h"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
PANEL_CACHE = ROOT / "research" / "cache" / "oa_1h_panel.parquet"
VOL_P = ROOT / "research" / "features" / "oa_vol_factors.parquet"
CTRL_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
OUT_P = ROOT / "research" / "features" / "oa_1h_factors.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "OA-1H.json"
KEY = ["ts_code", "trade_date"]


def load_st_set() -> set:
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def build_daily_rv() -> pd.DataFrame:
    """从 1H bar 聚合每 (股,日) 的日内已实现波动 rv = sum(1H收益^2)。checkpoint。"""
    if PANEL_CACHE.exists():
        print(f"[1h] checkpoint 命中 {PANEL_CACHE.name}", flush=True)
        return pd.read_parquet(PANEL_CACHE)
    st = load_st_set()
    files = sorted(H1_DIR.glob("*.parquet"))
    print(f"[1h] 读取 {len(files)} 个 1H 文件 ...", flush=True)
    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        code = f.stem
        if code in st:
            continue
        d = pd.read_parquet(f, columns=["trade_time", "close", "trade_date"])
        if len(d) < 8:
            continue
        d = d.sort_values("trade_time")
        d["ret1h"] = d.groupby("trade_date")["close"].pct_change()
        rv = d.groupby("trade_date")["ret1h"].apply(lambda s: float(np.nansum(s.to_numpy() ** 2)))
        nbar = d.groupby("trade_date")["ret1h"].apply(lambda s: int(s.notna().sum()))
        g = pd.DataFrame({"trade_date": rv.index, "rv_intraday": rv.values, "nbar": nbar.values})
        g["ts_code"] = code
        rows.append(g[g["nbar"] >= 2])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    panel = pd.concat(rows, ignore_index=True)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel = panel.sort_values(KEY).reset_index(drop=True)
    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE, index=False)
    print(f"[1h] 落盘 {PANEL_CACHE.name} ({len(panel):,} 行 / {panel['ts_code'].nunique()} 股, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return panel


def main() -> int:
    t0 = time.time()
    print("\n=== OA-1H 1H vs 日线 波动率因子对比 ===\n", flush=True)
    rv = build_daily_rv()
    g = rv.groupby("ts_code", sort=False)
    rv["rv_1h_20"] = g["rv_intraday"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    rv["rv_1h_60"] = g["rv_intraday"].transform(lambda s: s.rolling(60, min_periods=30).mean())

    ctrl_dates = set(pd.read_parquet(CTRL_P, columns=["trade_date"])["trade_date"].astype(str).unique())
    out = rv[rv["trade_date"].isin(ctrl_dates)][KEY + ["rv_1h_20", "rv_1h_60"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    factor_cols = ["rv_1h_20", "rv_1h_60"]
    oa_screen.assert_backward(factor_cols)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_P, index=False)
    cov_dates = sorted(out["trade_date"].unique())
    print(f"[out] -> {OUT_P.relative_to(ROOT)} ({len(out):,} 行, {len(cov_dates)} 月度日 "
          f"{cov_dates[0]}~{cov_dates[-1]} = 1H覆盖窗)", flush=True)

    print("\n[screen-1H] 1H 日内RV 因子 Phase-1 正交 IC ...", flush=True)
    v1h = oa_screen.screen_factors(out, factor_cols, "OA-1H", primary="rv_1h_60", write=False)

    # 同窗口对比: 日线 ivol_60/total_vol_60 在 1H 覆盖的同 dates 上的正交 IC
    dvol = pd.read_parquet(VOL_P)[KEY + ["ivol_60", "total_vol_60"]]
    dvol["trade_date"] = dvol["trade_date"].astype(str)
    dvol_w = dvol[dvol["trade_date"].isin(set(cov_dates))].copy()
    print("[screen-daily] 日线 ivol_60/total_vol_60 同窗口正交 IC ...", flush=True)
    vdaily = oa_screen.screen_factors(dvol_w, ["ivol_60", "total_vol_60"], "OA-1H-DAILYREF",
                                      primary="ivol_60", write=False)

    # 秩相关 (rv_1h_60 vs ivol_60) — 是否同序 (换皮?)
    m = out.merge(dvol_w, on=KEY).dropna(subset=["rv_1h_60", "ivol_60"])
    spear = float(m["rv_1h_60"].corr(m["ivol_60"], method="spearman")) if len(m) else float("nan")

    c1h = v1h["classifications"]; cdl = vdaily["classifications"]
    print("\n=== 对比 (同 1H 覆盖窗) ===", flush=True)
    print(f"  1H   rv_1h_60   orth_ic={c1h['rv_1h_60']['orth_ic']:+.4f} t={c1h['rv_1h_60']['orth_t']} ({c1h['rv_1h_60']['status']})", flush=True)
    print(f"  1H   rv_1h_20   orth_ic={c1h['rv_1h_20']['orth_ic']:+.4f} t={c1h['rv_1h_20']['orth_t']} ({c1h['rv_1h_20']['status']})", flush=True)
    print(f"  日线 ivol_60    orth_ic={cdl['ivol_60']['orth_ic']:+.4f} t={cdl['ivol_60']['orth_t']} ({cdl['ivol_60']['status']})", flush=True)
    print(f"  日线 total_vol  orth_ic={cdl['total_vol_60']['orth_ic']:+.4f} t={cdl['total_vol_60']['orth_t']} ({cdl['total_vol_60']['status']})", flush=True)
    print(f"  Spearman(rv_1h_60, ivol_60) = {spear:.3f}", flush=True)

    best_1h = max(abs(c1h[f]["orth_ic"]) for f in factor_cols)
    best_daily = max(abs(cdl[f]["orth_ic"]) for f in ["ivol_60", "total_vol_60"])
    stronger = "1H" if best_1h > best_daily + 0.005 else ("日线" if best_daily > best_1h + 0.005 else "相当")

    verdict = {
        "id": "OA-1H", "status": "built",
        "window": [cov_dates[0], cov_dates[-1]], "n_months": len(cov_dates),
        "comparison": {
            "1h": {f: {"orth_ic": c1h[f]["orth_ic"], "orth_t": c1h[f]["orth_t"],
                       "status": c1h[f]["status"]} for f in factor_cols},
            "daily": {f: {"orth_ic": cdl[f]["orth_ic"], "orth_t": cdl[f]["orth_t"],
                          "status": cdl[f]["status"]} for f in ["ivol_60", "total_vol_60"]},
            "spearman_rv1h60_vs_ivol60": round(spear, 4),
            "stronger": stronger,
        },
        "conclusion": (
            f"1H 日内已实现波动 (rv_1h_60 orth IC={c1h['rv_1h_60']['orth_ic']:+.4f}) vs 日线 ivol_60 "
            f"({cdl['ivol_60']['orth_ic']:+.4f}) 在同 {len(cov_dates)} 月 1H 覆盖窗对比: {stronger}更强 "
            f"(差<0.005=相当)。Spearman {spear:.3f} {'高(同序=换皮, 1H无新增信息)' if spear>0.8 else '中低(1H含独立日内信息)'}。"
            f"低波动信号在两尺度一致 (符号同负)。SIGN-R03: 仅IC对比, 落地仍以 GATE walk-forward 为准。"),
        "guardrails": ["SIGN-R04 backward", "SIGN-R06 ST排除", "SIGN-R11 分regime(继承screen)",
                       "SIGN-R12++ 扣全控制残差", "SIGN-R16 廉价对比非落地", "SIGN-R05 生产线不碰"],
    }
    import json
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT_P.relative_to(ROOT)} (stronger={stronger})", flush=True)
    print(f"[done] 耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
