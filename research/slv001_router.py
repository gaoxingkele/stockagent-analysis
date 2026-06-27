# -*- coding: utf-8 -*-
"""SLV-001 — 三原型路由器 + 三桶覆盖统计 (杠铃 + 独漏中间横盘 的总体确认).

Phase 0 (零训练廉价基础) 第一步: 不建任何模型, 只做"按预注册边界把全市场分桶 +
量化各桶机会"。设计见 research/sleeves/DESIGN.md, 边界冻结于 plans/prd.json 的
preRegisteredGate.router_boundaries (SIGN-R01, 绝不在结果上调)。

路由器边界 (FROZEN, 逐字落实):
  SQUAT    : past_r5 < -0.05
  BASE     : abs(ma20_slope_norm) < 0.0015/日  且  boll_bw_pctile(250d) < 0.30
             且  整理时长 >= 15日  且  past_r5 ∈ [-0.05, +0.05]
  MOMENTUM : past_r5 > +0.05  且  close>ma20>ma60 (多头排列)
             且  pyr_velocity_20_60 当日横截面分位 > 0.65
  NONE     : 不满足任一
  (三桶在 past_r5 上天然互斥: <-0.05 / [-0.05,+0.05] / >+0.05 → 无优先级冲突)

true_launch 标签 (同 v3c up, 全因果, 复用 pl001/t004 口径):
  次日开盘建仓, 前向5日 max_gain>=10% & max_dd>=-5%。

因子定义 (全 backward, 因果; 价格因子自算, pyr 取自 v7_extras):
  past_r5         = close/close.shift(5) - 1
  ma20/ma60       = close 的 20/60 日均线
  ma20_slope_norm = (ma20/ma20.shift(20) - 1)/20        (每日归一斜率)
  boll_bw         = 2*std20/ma20
  boll_bw_pctile  = boll_bw 在本股过去 250 日的分位 (rolling rank, min 100)
  整理时长        = abs(ma20_slope_norm)<0.0015 且 boll_bw_pctile<0.30 的尾部连续天数
  pyr_velocity_20_60 = v7_extras 现成列 (pyr_20d - pyr_60d), 当日横截面 pct rank

红线纪律:
  - 边界冻结 (SIGN-R01); ST 源头排除 (SIGN-R06); 全因果零泄漏 (SIGN-R04, 标签留 cache)。
  - 大缓存写 research/cache/slv001/ (gitignored), 不写 research/features/。
  - 分 regime 复核 (SIGN-R11, 用 RG-001 regime_timeline)。
  - 中间描述统计非 ship 决策 (SIGN-R03); 生产线只读 (SIGN-R05)。

产出:
  research/cache/slv001/factor_panel.parquet  (checkpoint, 含中间因子)
  research/cache/slv001/router_panel.parquet  (ts_code,trade_date,archetype,is_launch,regime,past_r5)
  research/cache/slv001_results.json
  research/verdicts/SLV-001.json              (status=built + 三表)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "slv001"
CACHE.mkdir(parents=True, exist_ok=True)
DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
V7X = ROOT / "output" / "v7_extras"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
FACTOR_CKPT = CACHE / "factor_panel.parquet"
ROUTER_OUT = CACHE / "router_panel.parquet"
RESULTS = ROOT / "research" / "cache" / "slv001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "SLV-001.json"

# ── 数据窗口 ──
LOAD_START = "20220601"   # 价格回看缓冲 (250d boll + 60d MA)
STUDY_START = "20230103"  # 统计起点 (pyr 与 regime 均自此可用)

# ── 冻结边界常量 (SIGN-R01) ──
SQUAT_R5 = -0.05
BASE_R5_LO, BASE_R5_HI = -0.05, 0.05
MOM_R5 = 0.05
MA20_SLOPE_THR = 0.0015          # 每日归一斜率阈值
BW_PCTILE_THR = 0.30             # 250d 布林带宽分位
CONSOL_MIN_DAYS = 15             # 整理时长
PYR_CSRANK_THR = 0.65            # 动量速度横截面分位

# ── true_launch 标签 (同 v3c up) ──
PUMP_UP, PUMP_DD, FORWARD = 0.10, 0.05, 5

KEY = ["ts_code", "trade_date"]


def load_st_set() -> set:
    if not BASIC.exists():
        return set()
    b = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(b[b["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_daily(st: set) -> pd.DataFrame:
    print(f"[data] 加载 daily OHLC {LOAD_START}+ (ST 源头排除)...", flush=True)
    cols = ["ts_code", "trade_date", "open", "high", "low", "close"]
    parts = []
    for f in sorted(DAILY.glob("*.parquet")):
        if f.stem < LOAD_START:
            continue
        parts.append(pd.read_parquet(f, columns=cols))
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    big = big[big["close"].notna() & (big["close"] > 0)]
    if st:
        n0 = len(big)
        big = big[~big["ts_code"].isin(st)]
        print(f"  ST 排除 {n0 - len(big):,} 行 ({len(st)} 只)", flush=True)
    return big.sort_values(KEY).reset_index(drop=True)


def load_pyr() -> pd.DataFrame:
    """pyr_velocity_20_60 取自 v7_extras (主 + ext, 去重保最新)。"""
    print("[data] 加载 v7_extras pyr_velocity_20_60 (主 + ext)...", flush=True)
    parts = []
    for f in sorted(V7X.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["ts_code", "trade_date", "pyr_velocity_20_60"])
        parts.append(df)
    pyr = pd.concat(parts, ignore_index=True)
    pyr["trade_date"] = pyr["trade_date"].astype(str)
    pyr = pyr.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    print(f"  pyr {len(pyr):,} 行 ({pyr['trade_date'].min()}~{pyr['trade_date'].max()})", flush=True)
    return pyr


def build_factors(df: pd.DataFrame, pyr: pd.DataFrame) -> pd.DataFrame:
    """逐股因果因子 + true_launch 标签 + 路由分桶。df 已按 (ts_code,trade_date) 排序。"""
    g = df.groupby("ts_code", sort=False)
    close = df["close"]

    # 价格均线 / 斜率 / 带宽
    ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    ma60 = g["close"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    df["past_r5"] = close / g["close"].shift(5) - 1.0
    ma20_lag = g["close"].transform(lambda s: s.rolling(20, min_periods=15).mean()).groupby(
        df["ts_code"], sort=False).shift(20)
    df["ma20_slope_norm"] = (ma20 / ma20_lag - 1.0) / 20.0
    std20 = g["close"].transform(lambda s: s.rolling(20, min_periods=15).std())
    boll_bw = 2.0 * std20 / ma20
    df["boll_bw_pctile"] = boll_bw.groupby(df["ts_code"], sort=False).transform(
        lambda s: s.rolling(250, min_periods=100).rank(pct=True))
    df["ma_bull"] = (close > ma20) & (ma20 > ma60)

    # 整理时长: abs(slope)<thr 且 带宽分位<thr 的尾部连续天数
    in_consol = ((df["ma20_slope_norm"].abs() < MA20_SLOPE_THR)
                 & (df["boll_bw_pctile"] < BW_PCTILE_THR))
    s = in_consol.fillna(False).astype(int)
    grp_reset = s.eq(0).groupby(df["ts_code"], sort=False).cumsum()
    df["consol_duration"] = s.groupby([df["ts_code"].values, grp_reset.values]).cumsum().values

    # true_launch 标签 (次日开盘建仓, 前向5日 max_gain>=10% & max_dd>=-5%)
    next_open = g["open"].shift(-1)
    max_high = g["high"].transform(
        lambda s: s.rolling(FORWARD, min_periods=FORWARD).max().shift(-FORWARD))
    min_low = g["low"].transform(
        lambda s: s.rolling(FORWARD, min_periods=FORWARD).min().shift(-FORWARD))
    upside = max_high / next_open - 1.0
    downside = min_low / next_open - 1.0
    df["is_launch"] = ((upside >= PUMP_UP) & (downside >= -PUMP_DD)).astype("float")
    df.loc[max_high.isna() | next_open.isna(), "is_launch"] = np.nan

    # pyr 横截面分位 (当日全市场 pct rank)
    df = df.merge(pyr, on=KEY, how="left")
    df["pyr_csrank"] = df.groupby("trade_date")["pyr_velocity_20_60"].rank(pct=True)

    # ── 路由分桶 (互斥, 逐字边界) ──
    r5 = df["past_r5"]
    archetype = pd.Series("NONE", index=df.index, dtype=object)
    squat = r5 < SQUAT_R5
    base = ((df["ma20_slope_norm"].abs() < MA20_SLOPE_THR)
            & (df["boll_bw_pctile"] < BW_PCTILE_THR)
            & (df["consol_duration"] >= CONSOL_MIN_DAYS)
            & (r5 >= BASE_R5_LO) & (r5 <= BASE_R5_HI))
    mom = (r5 > MOM_R5) & df["ma_bull"] & (df["pyr_csrank"] > PYR_CSRANK_THR)
    archetype[squat] = "SQUAT"
    archetype[base] = "BASE"
    archetype[mom] = "MOMENTUM"
    archetype[r5.isna()] = "NA"
    df["archetype"] = archetype
    return df


def main():
    t0 = time.time()
    print("\n=== SLV-001 三原型路由器 + 三桶覆盖统计 ===\n", flush=True)

    if FACTOR_CKPT.exists():
        print(f"[ckpt] 命中 {FACTOR_CKPT.name}, 跳过因子构建", flush=True)
        df = pd.read_parquet(FACTOR_CKPT)
    else:
        st = load_st_set()
        daily = load_daily(st)
        print(f"[data] daily {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / "
              f"{daily['trade_date'].nunique()} 日", flush=True)
        pyr = load_pyr()
        print("[calc] 构建因子 + 标签 + 分桶...", flush=True)
        df = build_factors(daily, pyr)
        df.to_parquet(FACTOR_CKPT, index=False)
        print(f"[ckpt] 因子面板落盘 {FACTOR_CKPT.name} ({len(df):,} 行)", flush=True)

    # regime (SIGN-R11)
    if REGIME_P.exists():
        rg = pd.read_parquet(REGIME_P, columns=["trade_date", "regime"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        df = df.merge(rg, on="trade_date", how="left")
    else:
        df["regime"] = "unknown"

    # 研究面板: 统计窗口内 + 标签可算
    panel = df[(df["trade_date"] >= STUDY_START) & df["is_launch"].notna()].copy()
    panel["is_launch"] = panel["is_launch"].astype(int)
    print(f"[panel] 研究面板 {len(panel):,} 行 ({panel['trade_date'].min()}~"
          f"{panel['trade_date'].max()}), 启动率 {panel['is_launch'].mean():.4f}", flush=True)

    # ── 落 router_panel (无 forward 因子列名; is_launch 是标签但不命中黑名单) ──
    out_cols = KEY + ["archetype", "is_launch", "regime", "past_r5"]
    panel[out_cols].to_parquet(ROUTER_OUT, index=False)
    print(f"[out] router_panel -> {ROUTER_OUT.relative_to(ROOT)} ({len(panel):,} 行)", flush=True)

    # ── 泄漏自查: 输出列名不撞 forward 黑名单 ──
    fwd_exact = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                {f"dd{n}" for n in (5, 10, 20, 30, 40)} | {"max_gain", "fwd_ret", "label", "target"}
    bad = [c for c in out_cols if c.lower() in fwd_exact
           or c.lower().startswith(("fwd_", "future_", "next_")) or c.lower().endswith("_forward")]
    assert not bad, f"router_panel 列名撞 forward 黑名单: {bad}"

    ARCH = ["SQUAT", "BASE", "MOMENTUM", "NONE", "NA"]
    overall_rate = float(panel["is_launch"].mean())
    n_total = len(panel)
    n_launch = int(panel["is_launch"].sum())

    # 表 a: 各 archetype 占 stock-days %
    share_days = (panel["archetype"].value_counts(normalize=True)).reindex(ARCH).fillna(0).round(5)
    # 表 b: 各 archetype 占 true_launch %
    lp = panel[panel["is_launch"] == 1]
    share_launch = (lp["archetype"].value_counts(normalize=True)).reindex(ARCH).fillna(0).round(5)
    # 表 c: 各 archetype 桶内启动率
    rate_in_bucket = panel.groupby("archetype")["is_launch"].mean().reindex(ARCH).round(5)
    n_in_bucket = panel.groupby("archetype")["is_launch"].size().reindex(ARCH).fillna(0).astype(int)

    table = {}
    for a in ARCH:
        table[a] = {
            "share_of_days": float(share_days[a]),
            "share_of_launches": float(share_launch[a]),
            "launch_rate": float(rate_in_bucket[a]) if pd.notna(rate_in_bucket[a]) else None,
            "n_days": int(n_in_bucket[a]),
            "lift_vs_overall": round(float(rate_in_bucket[a]) / overall_rate, 3)
            if pd.notna(rate_in_bucket[a]) and overall_rate > 0 else None,
        }

    # 分 regime 复核 (SIGN-R11): 桶内启动率 by regime
    by_regime = {}
    for rgm in ["momentum", "reversal", "mixed"]:
        sub = panel[panel["regime"] == rgm]
        if len(sub) < 5000:
            continue
        rr = sub.groupby("archetype")["is_launch"].mean().reindex(ARCH).round(5)
        by_regime[rgm] = {a: (float(rr[a]) if pd.notna(rr[a]) else None) for a in ARCH}

    # 杠铃确认: 两端 (SQUAT/MOMENTUM) launch_rate 高于整体, 中间 BASE 桶机会量化
    squat_lift = table["SQUAT"]["lift_vs_overall"]
    mom_lift = table["MOMENTUM"]["lift_vs_overall"]
    base_lift = table["BASE"]["lift_vs_overall"]
    barbell = (squat_lift is not None and mom_lift is not None
               and squat_lift > 1.0 and mom_lift > 1.0)

    results = {
        "task": "SLV-001", "elapsed_s": round(time.time() - t0, 1),
        "window": [STUDY_START, panel["trade_date"].max()],
        "n_stock_days": n_total, "n_launches": n_launch,
        "overall_launch_rate": round(overall_rate, 5),
        "router_boundaries": {
            "SQUAT": "past_r5 < -0.05",
            "BASE": "abs(ma20_slope_norm)<0.0015/d & boll_bw_pctile(250d)<0.30 & consol_dur>=15 & past_r5∈[-0.05,0.05]",
            "MOMENTUM": "past_r5>0.05 & close>ma20>ma60 & pyr_velocity_20_60 csrank>0.65",
            "NONE": "其余",
        },
        "true_launch_label": "次日开盘建仓, 前向5日 max_gain>=0.10 & max_dd>=-0.05 (同 v3c up)",
        "table_abc": table,
        "by_regime_launch_rate": by_regime,
        "barbell_confirmed": bool(barbell),
        "guardrails": ["SIGN-R01 边界冻结逐字落实", "SIGN-R03 描述统计非ship",
                       "SIGN-R04 全因果标签留cache", "SIGN-R06 ST源头排除", "SIGN-R11 分regime"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print(f"\n--- 三表 (整体启动率 {overall_rate:.4f}, {n_launch:,}/{n_total:,}) ---", flush=True)
    print(f"  {'archetype':10s} {'占天数%':>9s} {'占启动%':>9s} {'桶内启动率':>11s} {'lift':>7s} {'n_days':>11s}",
          flush=True)
    for a in ARCH:
        t = table[a]
        lr = f"{t['launch_rate']:.4f}" if t["launch_rate"] is not None else "—"
        lf = f"{t['lift_vs_overall']:.2f}" if t["lift_vs_overall"] is not None else "—"
        print(f"  {a:10s} {t['share_of_days']*100:>8.2f}% {t['share_of_launches']*100:>8.2f}% "
              f"{lr:>11s} {lf:>7s} {t['n_days']:>11,d}", flush=True)
    print(f"\n  杠铃确认(SQUAT&MOMENTUM lift>1): {barbell} "
          f"(squat {squat_lift} / mom {mom_lift} / base {base_lift})", flush=True)
    print(f"  分regime桶内启动率: {json.dumps(by_regime, ensure_ascii=False)}", flush=True)

    # ── verdict ──
    base_share = table["BASE"]["share_of_days"]
    conclusion = (
        f"路由器建成并按冻结边界给 {n_total:,} 个 stock-day 分桶 (窗口 {STUDY_START}~"
        f"{panel['trade_date'].max()}, 整体启动率 {overall_rate:.4f})。"
        f"三表关键数: 占天数 SQUAT {table['SQUAT']['share_of_days']:.1%} / "
        f"BASE {base_share:.1%} / MOMENTUM {table['MOMENTUM']['share_of_days']:.1%} / "
        f"NONE {table['NONE']['share_of_days']:.1%}; 桶内启动率 SQUAT "
        f"{table['SQUAT']['launch_rate']:.4f}(lift {squat_lift}) / BASE "
        f"{table['BASE']['launch_rate']:.4f}(lift {base_lift}) / MOMENTUM "
        f"{table['MOMENTUM']['launch_rate']:.4f}(lift {mom_lift}) / NONE "
        f"{table['NONE']['launch_rate']:.4f}。"
        f"杠铃{'复现' if barbell else '未复现'} (PL-001: 两端启动率高): SQUAT/MOMENTUM lift "
        f"{'均>1' if barbell else '非均>1'}; 中间 BASE 横盘桶 lift {base_lift} "
        f"(占天数仅 {base_share:.1%} 但桶内启动率 {table['BASE']['launch_rate']:.4f}) "
        f"= '独漏中间横盘'机会量化。分 regime 复核见 metrics。"
        f" 描述统计非 ship, MOMENTUM 是否建由 SLV-003 分散性闸定 (SIGN-R03)。")

    VERDICT.write_text(json.dumps({
        "id": "SLV-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "window": [STUDY_START, panel["trade_date"].max()],
            "n_stock_days": n_total, "n_launches": n_launch,
            "overall_launch_rate": round(overall_rate, 5),
            "table_a_share_of_days": {a: table[a]["share_of_days"] for a in ARCH},
            "table_b_share_of_launches": {a: table[a]["share_of_launches"] for a in ARCH},
            "table_c_launch_rate": {a: table[a]["launch_rate"] for a in ARCH},
            "lift_vs_overall": {a: table[a]["lift_vs_overall"] for a in ARCH},
            "barbell_confirmed": bool(barbell),
            "by_regime_launch_rate": by_regime,
        },
        "artifacts": ["research/cache/slv001/router_panel.parquet",
                      "research/cache/slv001/factor_panel.parquet",
                      "research/cache/slv001_results.json"],
        "note": "边界冻结逐字落实 (SIGN-R01); 全因果 ST 源头排除 (R04/R06); 大缓存写 cache 非 features; "
                "分 regime (R11); 中间统计非 ship 决策 (R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] SLV-001 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
