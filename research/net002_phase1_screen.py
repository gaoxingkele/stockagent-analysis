# -*- coding: utf-8 -*-
"""NET-002 — 概念 lead-lag Phase-1 廉价筛查 (扣[自身动量+市场]正交 IC).

核心问 (北极星): 滞后票扣掉**自身动量**(+市场 beta)后, 还跟不跟同概念领先票?
若 lead-lag 信号只是"本股自己过去也在涨"(动量) 或"大盘 beta"的换皮 → no_residual,
该轨廉价 REJECT (SEAT 轨还在)。若扣完仍有显著正交 IC → residual_signal, 进 GATE walk-forward。

输入 (全部因果, 复用):
  research/features/concept_leadlag.parquet  (NET-001, cl_pr1/5/20/lead_ratio, 邻居 ≤t-1 收益)
  research/cache/fu002_momentum.parquet      (自身动量 mom_5/20/60, 月末 rebalance, 因果)
  research/cache/rt004_r20_label.parquet     (前向 r20 = close[t+20]/close[t]-1, 因果, label)
  research/features/regime_timeline.parquet  (RG-001 regime: momentum/mixed/reversal, 分层)
  output/tushare_cache/daily/*.parquet       (算市场 beta 控制: 等权市场日收益 + trailing 60d beta)

方法 (逐月横截面, 复刻 FU-003 框架):
  - 月末 rebalance (mom 缓存的 54 个月末日, 47 个有前向 r20 → ≥36月 功率达标)。
  - 原始 rank-IC: 逐月 Spearman(信号, r20) 再跨月平均; t = mean/(std/sqrt(n))。
  - 正交 IC (SIGN-R12++): 单日横截面把信号对控制集 OLS 残差化 (控制标准化), 残差再 rank-IC。
      变体: raw / minus_mom(扣自身动量) / minus_market(扣 beta) / orth_full(扣动量+市场)。
  - 分 regime (SIGN-R11): 全期 + momentum/mixed/reversal 分别报, 决策不只看全期平均。

防泄漏 (SIGN-R04):
  - 信号侧 NET-001 已保证邻居 ≤t-1 + leave-one-out (双重防泄漏)。
  - r20 是 label (因果 close.shift(-20)), 落 research/cache, 不写任何 features parquet。
  - 市场 beta 用 trailing 60d (≤当日) 等权市场收益, 因果。
  - 横截面 rank-IC 天然差掉"当日全市场常数项"; 市场 beta 控制捕捉 per-stock 市场暴露。

纪律: SIGN-R03 IC 仅 go/no-go 非落地 (只认 GATE walk-forward α); SIGN-R06 ST 源头排除 (NET-001
  继承 + mom 继承); SIGN-R01 阈值冻结。

裁决: research/verdicts/NET-002.json
  status ∈ {residual_signal, no_residual} (prd 注册口径)
  residual_signal 当且仅当 最强信号 orth_full(扣动量+市场) 全期 |IC|>=0.02 且 |t|>=3。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CL_P = ROOT / "research" / "features" / "concept_leadlag.parquet"
MOM_P = ROOT / "research" / "cache" / "fu002_momentum.parquet"
LAB_P = ROOT / "research" / "cache" / "rt004_r20_label.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"

BETA_CACHE = ROOT / "research" / "cache" / "net002_beta.parquet"   # 月末 beta (forward-free)
PANEL_CACHE = ROOT / "research" / "cache" / "net002_panel.parquet"  # 含 r20 → 留 cache 不进 features
IC_TABLE_P = ROOT / "research" / "cache" / "net002_ic_table.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "NET-002.json"

KEY = ["ts_code", "trade_date"]
SIGNALS = ["cl_pr1", "cl_pr5", "cl_pr20", "cl_lead_ratio"]  # 待筛信号 (邻居滞后收益 + 先动比例)
PRIMARY = "cl_pr5"   # 北极星头号信号 (邻居过去5d收益), cl_lead_ratio 一并报

MOM = ["mom_5", "mom_20", "mom_60"]
MARKET = ["beta"]
VARIANTS = {
    "raw": [],
    "minus_mom": MOM,
    "minus_market": MARKET,
    "orth_full": MOM + MARKET,
}

BETA_WIN = 60        # 市场 beta trailing 窗口 (交易日)
BETA_MINP = 40
MIN_XS = 30          # 单月最少横截面股票数
MIN_MONTHS = 36      # 功率下限 (prd: ≥36月)

# gate 阈值 (冻结 R01, 同 prd.preRegisteredGate.phase1_screen)
IC_SIG = 0.02
T_SIG = 3.0
IC_COLLAPSE = 0.01


def std_col(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    m, s = np.nanmean(a), np.nanstd(a)
    return (a - m) / s if s > 0 else a * 0.0


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def tstat(ic: np.ndarray) -> float:
    ic = ic[~np.isnan(ic)]
    if len(ic) < 2:
        return np.nan
    se = ic.std(ddof=1) / np.sqrt(len(ic))
    return float(ic.mean() / se) if se > 0 else np.nan


def load_st_set() -> set:
    if not BASIC_P.exists():
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def build_beta(month_ends: list[str], st: set) -> pd.DataFrame:
    """市场 beta 控制: 等权市场日收益 + per-stock trailing 60d beta, 仅在月末取值 (因果)."""
    if BETA_CACHE.exists():
        print(f"[beta] checkpoint 命中 {BETA_CACHE.name}", flush=True)
        return pd.read_parquet(BETA_CACHE)

    print("[beta] 加载 daily close 算等权市场收益 + trailing 60d beta ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(DAILY_DIR.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px[(px["close"] > 0)].drop_duplicates(KEY, keep="last")
    if st:
        px = px[~px["ts_code"].isin(st)]
    px = px.sort_values(KEY).reset_index(drop=True)

    px["r1"] = px.groupby("ts_code")["close"].pct_change()
    mkt = px.groupby("trade_date")["r1"].mean().rename("mkt")           # 等权市场日收益
    px = px.merge(mkt, on="trade_date", how="left")
    px["r1m"] = px["r1"] * px["mkt"]

    # per-stock trailing 60d: E[r1], E[r1*mkt], E[mkt]  → cov = E[r1m]-E[r1]E[mkt]
    roll = (px.groupby("ts_code")[["r1", "r1m", "mkt"]]
            .rolling(BETA_WIN, min_periods=BETA_MINP).mean()
            .reset_index(level=0, drop=True))
    cov = roll["r1m"] - roll["r1"] * roll["mkt"]

    # 市场 var (trailing 60d), 按日同值, 在市场序列上算一次
    mser = mkt.reset_index().sort_values("trade_date")
    mser["m2"] = mser["mkt"] ** 2
    em = mser["mkt"].rolling(BETA_WIN, min_periods=BETA_MINP).mean()
    em2 = mser["m2"].rolling(BETA_WIN, min_periods=BETA_MINP).mean()
    mser["var_mkt"] = (em2 - em ** 2).values
    px = px.merge(mser[["trade_date", "var_mkt"]], on="trade_date", how="left")

    px["beta"] = cov.values / px["var_mkt"].replace(0, np.nan)
    out = px[px["trade_date"].isin(set(month_ends))][["ts_code", "trade_date", "beta"]]
    out = out.dropna(subset=["beta"]).reset_index(drop=True)
    BETA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(BETA_CACHE, index=False)
    print(f"[beta] 落盘 {BETA_CACHE.name} ({len(out):,} 行, {out['trade_date'].nunique()} 月末)", flush=True)
    return out


def build_panel(st: set) -> pd.DataFrame:
    if PANEL_CACHE.exists():
        print(f"[panel] checkpoint 命中 {PANEL_CACHE.name}", flush=True)
        return pd.read_parquet(PANEL_CACHE)

    mom = pd.read_parquet(MOM_P)
    mom["trade_date"] = mom["trade_date"].astype(str)
    month_ends = sorted(mom["trade_date"].unique())
    print(f"[panel] mom 月末 rebalance 日 {len(month_ends)} 个 "
          f"({month_ends[0]}~{month_ends[-1]})", flush=True)

    cl = pd.read_parquet(CL_P)
    cl["trade_date"] = cl["trade_date"].astype(str)
    cl = cl[cl["trade_date"].isin(set(month_ends))][KEY + SIGNALS]

    lab = pd.read_parquet(LAB_P)[["ts_code", "trade_date", "r20"]]
    lab["trade_date"] = lab["trade_date"].astype(str)

    beta = build_beta(month_ends, st)

    df = (cl.merge(mom, on=KEY, how="inner")          # inner: ST 已源头排除 → 自然继承
            .merge(beta, on=KEY, how="left")
            .merge(lab, on=KEY, how="inner"))         # 只保留有前向 r20 的月
    if st:
        df = df[~df["ts_code"].isin(st)]
    df = df[df["r20"].notna()].reset_index(drop=True)

    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]]
    rg["trade_date"] = rg["trade_date"].astype(str)
    df = df.merge(rg, on="trade_date", how="left")
    df["regime"] = df["regime"].fillna("unknown")

    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_CACHE, index=False)
    print(f"[panel] 落盘 {PANEL_CACHE.name} ({len(df):,} 行, "
          f"{df['trade_date'].nunique()} 月, {df['ts_code'].nunique()} 股)", flush=True)
    return df


def monthly_ic_series(df: pd.DataFrame, sig: str, ctrl: list[str]) -> pd.DataFrame:
    """逐月: (可选)对 ctrl OLS 残差化 sig, 再 rank-IC vs r20。返回 [date, ic, regime]。"""
    need = [sig, "r20"] + ctrl
    rows = []
    for d, sub in df.groupby("trade_date", sort=True):
        s = sub.dropna(subset=need)
        if len(s) < MIN_XS:
            continue
        y = s[sig].to_numpy(float)
        if ctrl:
            X = np.column_stack([np.ones(len(s))] + [std_col(s[c].to_numpy()) for c in ctrl])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            v = y - X @ beta
        else:
            v = y
        ic = rank_ic(v, s["r20"].to_numpy(float))
        rows.append((d, ic, s["regime"].iloc[0] if "regime" in s else "unknown"))
    return pd.DataFrame(rows, columns=["date", "ic", "regime"])


def main() -> int:
    t0 = time.time()
    print("\n=== NET-002 概念 lead-lag Phase-1 廉价筛查 (扣[动量+市场]正交IC) ===\n", flush=True)
    st = load_st_set()
    print(f"[st] ST 源头排除集合 {len(st)} 只", flush=True)
    df = build_panel(st)
    n_months = df["trade_date"].nunique()
    print(f"[data] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / {n_months} 月 "
          f"({df['trade_date'].min()}~{df['trade_date'].max()}); "
          f"beta 非空率={df['beta'].notna().mean():.3f}; "
          f"regime 分布={df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    table_rows = []
    summary = {}
    for sig in SIGNALS:
        summary[sig] = {}
        print(f"\n--- {sig} ---", flush=True)
        for vname, ctrl in VARIANTS.items():
            ser = monthly_ic_series(df, sig, ctrl)
            ic = ser["ic"].to_numpy(float)
            n = int(np.sum(~np.isnan(ic)))
            mean_ic = float(np.nanmean(ic)) if n else np.nan
            t = tstat(ic)
            reg_block = {}
            for rg, g in ser.groupby(ser["regime"].fillna("unknown")):
                gic = g["ic"].to_numpy(float)
                gt = tstat(gic)
                reg_block[str(rg)] = {
                    "ic": round(float(np.nanmean(gic)), 4) if len(gic) else None,
                    "t": round(gt, 2) if len(gic) > 1 and not np.isnan(gt) else None,
                    "n": int(np.sum(~np.isnan(gic)))}
            summary[sig][vname] = {
                "ic": round(mean_ic, 4),
                "t": round(t, 2) if not np.isnan(t) else None,
                "n_months": n, "by_regime": reg_block}
            print(f"  {vname:13s} IC={mean_ic:+.4f} t={t:+.2f} n={n}", flush=True)
            for _, r in ser.iterrows():
                table_rows.append({"signal": sig, "variant": vname,
                                   "date": r["date"], "ic": r["ic"], "regime": r["regime"]})

    ic_table = pd.DataFrame(table_rows)
    IC_TABLE_P.parent.mkdir(parents=True, exist_ok=True)
    ic_table.to_parquet(IC_TABLE_P, index=False)
    print(f"\n[ic-table] -> {IC_TABLE_P.relative_to(ROOT)} ({len(ic_table):,} 行)", flush=True)

    # ── 决策: 最强信号的 orth_full(扣动量+市场) 是否过线 ──
    def orth(sig):
        return summary[sig]["orth_full"]
    best_sig = max(SIGNALS, key=lambda s: abs(orth(s)["ic"]) if orth(s)["ic"] is not None else -1)
    bo = orth(best_sig)
    bo_ic, bo_t = bo["ic"], (bo["t"] or 0.0)
    power_ok = bo["n_months"] >= MIN_MONTHS

    passed = abs(bo_ic) >= IC_SIG and abs(bo_t) >= T_SIG
    status = "residual_signal" if passed else "no_residual"

    # 信号来源定位 (R12): raw → 单扣动量 → 单扣市场 → 全扣, 看在哪一步塌
    prim = summary[PRIMARY]
    print(f"\n[决策] 最强信号 {best_sig}: orth_full IC={bo_ic:+.4f} t={bo_t:+.2f} "
          f"n={bo['n_months']} → status={status}", flush=True)
    print(f"[定位] {PRIMARY}: raw IC={prim['raw']['ic']:+.4f} → "
          f"minus_mom={prim['minus_mom']['ic']:+.4f} → "
          f"minus_market={prim['minus_market']['ic']:+.4f} → "
          f"orth_full={prim['orth_full']['ic']:+.4f}", flush=True)

    if status == "residual_signal":
        conclusion = (
            f"概念 lead-lag Phase-1: 最强信号 {best_sig} 扣[自身动量+市场]残差后 "
            f"全期 IC={bo_ic:+.4f} |t|={abs(bo_t):.2f} (n={bo['n_months']}月), "
            f"过线 (需 |IC|>={IC_SIG} & |t|>={T_SIG}) → residual_signal: 滞后票扣自身动量后"
            f"仍跟同概念领先票, 进 GATE-001 walk-forward (最终只认 walk-forward α, SIGN-R03)。")
    else:
        conclusion = (
            f"概念 lead-lag Phase-1: 最强信号 {best_sig} 扣[自身动量+市场]残差后 "
            f"全期 IC={bo_ic:+.4f} |t|={abs(bo_t):.2f} (n={bo['n_months']}月) 未过显著线 "
            f"(需 |IC|>={IC_SIG} & |t|>={T_SIG}); {PRIMARY} raw IC={prim['raw']['ic']:+.4f} → "
            f"扣动量 {prim['minus_mom']['ic']:+.4f} → 全扣 {prim['orth_full']['ic']:+.4f}。"
            f"→ no_residual: lead-lag 信号 ≈ 自身动量/市场 beta 换皮, 扣完无独立残差。"
            f"该轨廉价 REJECT 记录 (SEAT 轨仍在, GATE 由 SEAT-002 决定)。")

    verdict = {
        "id": "NET-002",
        "status": status,
        "conclusion": conclusion,
        "screen_thresholds": {"ic_sig": IC_SIG, "t_sig": T_SIG,
                              "ic_collapse": IC_COLLAPSE, "min_months": MIN_MONTHS,
                              "applies_to": "best signal orth_full(扣动量+市场) full-sample residual IC"},
        "label": "fwd r20 = close[t+20]/close[t]-1 (因果, 复用 rt004_r20_label.parquet)",
        "controls": {"mom": MOM, "market": MARKET,
                     "market_def": f"per-stock trailing {BETA_WIN}d beta vs 等权市场日收益 (因果)"},
        "power_ok_ge36months": power_ok,
        "best_signal": best_sig,
        "primary_signal": PRIMARY,
        "metrics": {
            "n_rebalance_months": int(bo["n_months"]),
            "best_orth_full_ic": bo_ic,
            "best_orth_full_t": bo["t"],
            "by_signal": summary,
            "primary_ablation": {
                "raw_ic": prim["raw"]["ic"], "raw_t": prim["raw"]["t"],
                "minus_mom_ic": prim["minus_mom"]["ic"],
                "minus_market_ic": prim["minus_market"]["ic"],
                "orth_full_ic": prim["orth_full"]["ic"], "orth_full_t": prim["orth_full"]["t"],
            },
        },
        "ablation_note": (
            "逐步消融 (raw → 单扣动量 → 单扣市场 beta → 全扣) 定位 lead-lag 信号来源: "
            "若 raw 有 IC 但扣自身动量即塌 = 滞后票收益本质是自身动量 (邻居/本股同概念共涨), "
            "非独立 lead-lag alpha。横截面 rank-IC 天然差掉当日全市场常数项, beta 控制额外捕捉 "
            "per-stock 市场暴露。"),
        "artifacts": ["research/cache/net002_ic_table.parquet",
                      "research/cache/net002_panel.parquet",
                      "research/cache/net002_beta.parquet"],
        "guardrails": ["SIGN-R01 阈值冻结未改", "SIGN-R02 负结果=合法完成",
                       "SIGN-R03 IC 仅 go/no-go 非落地", "SIGN-R04 r20 因果落 cache 不进 features",
                       "SIGN-R06 ST 源头排除 (NET-001 + mom 继承)", "SIGN-R11 分 regime 评估",
                       "SIGN-R12++ 扣[自身动量+市场]残差化消融"],
        "downstream": ("no_residual → 该轨 REJECT 记录, GATE-001 是否跑由 SEAT-002 是否 residual 决定; "
                       "若两轨皆 no_residual → GATE-001 status=REJECT 不跑 walk-forward。"
                       if status == "no_residual"
                       else "residual_signal → 进 GATE-001 walk-forward (NET 臂)。"),
    }
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
