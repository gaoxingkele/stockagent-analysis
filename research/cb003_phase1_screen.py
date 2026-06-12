# -*- coding: utf-8 -*-
"""CB-003 — Phase-1 廉价筛查: 蓄势信号扣[动量+市场+现有形态因子] 正交残差 IC.

北极星第三问的前置防自欺闸: CB-002 已证盲点真实 (past_r5≥0 横盘/微涨启动被系统性漏掉),
但"系统漏掉这族启动" ≠ "蓄势信号本身有独立 alpha"。本任务问的是另一件事:

  **蓄势信号 (MA20走平 + 收敛 + 缩量) 对前向 r20 的横截面预测力, 扣掉
    [动量 + 市场 + 现有标准形态因子 (MA斜率/ADX/布林带宽/past_r5/RSI)] 之后,
    还剩不剩独立残差?**

若蓄势信号只是现有 MA/ADX/布林/past_r5 因子的非线性换皮 → orth_full 残差塌缩 →
no_residual → 廉价 REJECT, CB-004 skip (SIGN-R12++ 消融闸; 整轮第 N 次防"价格再编码"自欺)。
若扣完仍 |IC|>=0.02 & |t|>=3 → residual_signal → 进 CB-004 walk-forward (最终只认 α, R03)。

口径 (复用 SEAT-002 残差 rank-IC 框架, 但密集面板改**月末 rebalance**):
  - 截面: 每个日历月最后一个交易日取全市场横截面 (非重叠 r20 标签, 避免日采样 t 膨胀)。
  - 信号侧 (蓄势, 来自 CB-001 consolidation.parquet, 全 backward):
      s_flat       = -cs_ma20_absslope        (越大=MA20越走平)
      s_bw_squeeze = -cs_bw_squeeze            (越大=布林带宽越收敛, 三角/楔)
      s_amp_squeeze= -cs_amp_squeeze           (越大=振幅越收窄)
      s_vol_shrink = -cs_vol_ratio             (越大=越缩量蓄势)
      s_coil       = 上四者横截面 z 之和        (合成"蓄势强度", 北极星头号信号)
      s_consol     = cs_is_consolidation(0/1)   (CB-001 候选合成布尔)
  - 标签: fr20 = close[t+20]/close[t]-1 (前向, 因果, **仅 outcome 不入 features**)。
  - 控制 (扣除, 现有标准 TA, 全 backward ≤t):
      MOM     = mom_5, mom_20, mom_60               (价格动量)
      MARKET  = beta                                (trailing 60d vs 等权市场)
      PATTERN = cs_ma20_slope(MA斜率,有符号), adx_14, rsi_14, cs_boll_bw(布林原始带宽), cs_past_r5
  - 逐步消融: raw → 单扣 MOM → 单扣 MARKET → 单扣 PATTERN → 全扣(orth_full), 定位 IC 来源。
  - 分 regime (SIGN-R11): 每月 IC 按 regime_timeline 主导 regime 聚合, 全期+分 regime 各报。

纪律: SIGN-R01 阈值冻结 (= prd.preRegisteredGate.phase1_screen, |IC|>=0.02 & |t|>=3);
  SIGN-R03 IC 仅 go/no-go 非落地; SIGN-R04 fr20 仅 outcome 不进 features/不命中黑名单列名;
  SIGN-R06 ST 源头排除; SIGN-R08 控制面板 checkpoint; SIGN-R12++ 扣现有形态因子消融。

裁决: research/verdicts/CB-003.json — status ∈ {residual_signal, no_residual} + 残差 IC 表。
  no_residual → 本脚本顺手把 prd.json CB-004.skip=true (precondition skip, playbook CB-003 分叉)。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
CONSOL_P = ROOT / "research" / "features" / "consolidation.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"

CACHE_DIR = ROOT / "research" / "cache" / "cb003"
CTRL_CACHE = CACHE_DIR / "controls_monthend.parquet"   # 月末控制+标签 (fr20 → cache 不进 features)
IC_TABLE_P = CACHE_DIR / "ic_table.parquet"
PRD_P = ROOT / "plans" / "prd.json"
VERDICT_P = ROOT / "research" / "verdicts" / "CB-003.json"

KEY = ["ts_code", "trade_date"]

# 信号 (蓄势, 全部 backward; 方向已对齐为"越大越蓄势")
SIGNALS = ["s_coil", "s_consol", "s_flat", "s_bw_squeeze", "s_amp_squeeze", "s_vol_shrink"]
PRIMARY = "s_coil"   # 北极星头号: 合成蓄势强度

# 控制组 (逐步消融)
MOM = ["mom_5", "mom_20", "mom_60"]
MARKET = ["beta"]
PATTERN = ["cs_ma20_slope", "adx_14", "rsi_14", "cs_boll_bw", "cs_past_r5"]
VARIANTS = {
    "raw": [],
    "minus_mom": MOM,
    "minus_market": MARKET,
    "minus_pattern": PATTERN,
    "orth_full": MOM + MARKET + PATTERN,
}

FWD_H = 20           # 前向 r20
BETA_WIN, BETA_MINP = 60, 40
ADX_N = RSI_N = 14
MIN_XS = 30          # 单月末最少横截面股票数 (密集面板远超此)
MIN_MONTHS = 36      # 功率下限 (prd: ≥36月)

# 冻结阈值 (SIGN-R01, = prd.preRegisteredGate.phase1_screen)
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


def wilder(s: pd.Series) -> pd.Series:
    """Wilder 平滑 = ewm(alpha=1/N, adjust=False)."""
    return s.ewm(alpha=1.0 / ADX_N, min_periods=ADX_N, adjust=False).mean()


def build_controls(st: set) -> pd.DataFrame:
    """全历史 daily 算 动量 + 市场 beta + ADX(14) + RSI(14), 再 subsample 到月末 (因果, ≤t).

    PATTERN 里的 cs_ma20_slope / cs_boll_bw / cs_past_r5 复用 consolidation.parquet, 此处只补
    ADX/RSI/动量/beta。标签 fr20 = close[t+20]/close[t]-1 (前向, 仅 outcome)。
    """
    if CTRL_CACHE.exists():
        print(f"[ctrl] checkpoint 命中 {CTRL_CACHE.name}", flush=True)
        return pd.read_parquet(CTRL_CACHE)

    print("[ctrl] 加载全历史 OHLCV ...", flush=True)
    cols = ["ts_code", "trade_date", "high", "low", "close"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(DAILY_DIR.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px[px["close"] > 0].drop_duplicates(KEY, keep="last")
    if st:
        before = len(px)
        px = px[~px["ts_code"].isin(st)]
        print(f"  ST 源头排除 {before - len(px):,} 行 ({len(st)} 只)", flush=True)
    px = px.sort_values(KEY).reset_index(drop=True)
    g = px.groupby("ts_code", sort=False)

    # 动量 (trailing)
    for h in (5, 20, 60):
        px[f"mom_{h}"] = g["close"].transform(lambda s, h=h: s / s.shift(h) - 1.0)

    # 前向标签 fr20 (= outcome, 仅 cache)
    px["fr20"] = g["close"].transform(lambda s: s.shift(-FWD_H) / s - 1.0)

    # 市场 beta: 等权市场日收益 + per-stock trailing 60d beta = cov/var
    px["r1"] = g["close"].pct_change()
    mkt = px.groupby("trade_date")["r1"].mean().rename("mkt")
    px = px.merge(mkt, on="trade_date", how="left")
    g = px.groupby("ts_code", sort=False)
    px["r1m"] = px["r1"] * px["mkt"]
    roll = (px.groupby("ts_code")[["r1", "r1m", "mkt"]]
            .rolling(BETA_WIN, min_periods=BETA_MINP).mean().reset_index(level=0, drop=True))
    cov = roll["r1m"] - roll["r1"] * roll["mkt"]
    mser = mkt.reset_index().sort_values("trade_date")
    mser["m2"] = mser["mkt"] ** 2
    em = mser["mkt"].rolling(BETA_WIN, min_periods=BETA_MINP).mean()
    em2 = mser["m2"].rolling(BETA_WIN, min_periods=BETA_MINP).mean()
    mser["var_mkt"] = (em2 - em ** 2).values
    px = px.merge(mser[["trade_date", "var_mkt"]], on="trade_date", how="left")
    px["beta"] = cov.values / px["var_mkt"].replace(0, np.nan)

    # RSI(14) Wilder
    delta = g["close"].transform(lambda s: s.diff())
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.groupby(px["ts_code"], sort=False).transform(wilder)
    avg_dn = down.groupby(px["ts_code"], sort=False).transform(wilder)
    rs = avg_up / avg_dn.replace(0, np.nan)
    px["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    px.loc[avg_dn == 0, "rsi_14"] = 100.0

    # ADX(14) Wilder
    prev_close = g["close"].transform(lambda s: s.shift(1))
    up_move = g["high"].transform(lambda s: s.diff())
    down_move = -g["low"].transform(lambda s: s.diff())
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum.reduce([
        (px["high"] - px["low"]).to_numpy(),
        (px["high"] - prev_close).abs().to_numpy(),
        (px["low"] - prev_close).abs().to_numpy(),
    ])
    px["_pdm"], px["_mdm"], px["_tr"] = plus_dm, minus_dm, tr
    atr = px.groupby("ts_code", sort=False)["_tr"].transform(wilder)
    s_pdm = px.groupby("ts_code", sort=False)["_pdm"].transform(wilder)
    s_mdm = px.groupby("ts_code", sort=False)["_mdm"].transform(wilder)
    pdi = 100.0 * s_pdm / atr.replace(0, np.nan)
    mdi = 100.0 * s_mdm / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    px["adx_14"] = dx.groupby(px["ts_code"], sort=False).transform(wilder)

    # subsample 月末 (每月最后一个交易日)
    px["month"] = px["trade_date"].str[:6]
    month_end = px.groupby("month")["trade_date"].transform("max")
    me = px[px["trade_date"] == month_end].copy()
    keep = KEY + ["month", "fr20"] + MOM + MARKET + ["adx_14", "rsi_14"]
    me = me[keep].reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    me.to_parquet(CTRL_CACHE, index=False)
    print(f"[ctrl] 月末控制+标签落盘 {CTRL_CACHE.name} "
          f"({len(me):,} 行, {me['trade_date'].nunique()} 月末)", flush=True)
    return me


def build_panel(st: set) -> pd.DataFrame:
    """月末横截面: 蓄势信号 (consolidation.parquet) + 控制 (含 PATTERN 复用 cs_*) + 标签 + regime."""
    ctrl = build_controls(st)
    me_dates = set(ctrl["trade_date"].unique())

    consol = pd.read_parquet(CONSOL_P)
    consol["trade_date"] = consol["trade_date"].astype(str)
    consol = consol[consol["trade_date"].isin(me_dates)]
    if st:
        consol = consol[~consol["ts_code"].isin(st)]

    # 蓄势信号 (方向: 越大越蓄势)
    consol["s_flat"] = -consol["cs_ma20_absslope"]
    consol["s_bw_squeeze"] = -consol["cs_bw_squeeze"]
    consol["s_amp_squeeze"] = -consol["cs_amp_squeeze"]
    consol["s_vol_shrink"] = -consol["cs_vol_ratio"]
    consol["s_consol"] = consol["cs_is_consolidation"].astype("float")

    sig_keep = KEY + SIGNALS[2:] + ["s_consol", "cs_ma20_slope", "cs_boll_bw", "cs_past_r5"]
    df = consol[sig_keep].merge(ctrl, on=KEY, how="inner")

    # s_coil = 四蓄势分量横截面 z 之和 (按月末截面标准化, 因果)
    comps = ["s_flat", "s_bw_squeeze", "s_amp_squeeze", "s_vol_shrink"]
    z = df.groupby("trade_date")[comps].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0)
    df["s_coil"] = z.sum(axis=1)

    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]]
    rg["trade_date"] = rg["trade_date"].astype(str)
    df = df.merge(rg, on="trade_date", how="left")
    df["regime"] = df["regime"].fillna("unknown")
    return df


def monthly_ic(df: pd.DataFrame, sig: str, ctrl: list[str]) -> pd.DataFrame:
    """逐月末: 残差化 sig 对 ctrl (含截距), 再 rank-IC vs fr20。返回 [month, ic, regime]."""
    need = [sig, "fr20"] + ctrl
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
        ic = rank_ic(v, s["fr20"].to_numpy(float))
        rows.append((d[:6], ic, s["regime"].mode().iloc[0] if len(s) else "unknown"))
    return pd.DataFrame(rows, columns=["month", "ic", "regime"])


def summarize(ser: pd.DataFrame) -> dict:
    ic = ser["ic"].to_numpy(float)
    n = int(np.sum(~np.isnan(ic)))
    t = tstat(ic)
    reg = {}
    for rg, gg in ser.groupby(ser["regime"].fillna("unknown")):
        gic = gg["ic"].to_numpy(float)
        gt = tstat(gic)
        reg[str(rg)] = {"ic": round(float(np.nanmean(gic)), 4) if len(gic) else None,
                        "t": round(gt, 2) if len(gic) > 1 and not np.isnan(gt) else None,
                        "n": int(np.sum(~np.isnan(gic)))}
    return {"ic": round(float(np.nanmean(ic)), 4) if n else None,
            "t": round(t, 2) if not np.isnan(t) else None,
            "n_months": n, "by_regime": reg}


def set_cb004_skip(reason: str) -> None:
    """no_residual → CB-004 precondition skip (playbook CB-003 分叉)。"""
    d = json.loads(PRD_P.read_text(encoding="utf-8"))
    for t in d.get("features", []):
        if t["id"] == "CB-004":
            t["skip"] = True
            t["skipReason"] = reason
    PRD_P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prd] CB-004.skip=true ({reason})", flush=True)


def main() -> int:
    t0 = time.time()
    print("\n=== CB-003 蓄势信号 Phase-1 廉价筛查 (扣[动量+市场+现有形态因子]正交IC) ===\n", flush=True)
    st = load_st_set()
    print(f"[st] ST 源头排除集合 {len(st)} 只", flush=True)
    df = build_panel(st)
    n_months = df["month"].nunique()
    print(f"[data] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / {df['trade_date'].nunique()} 月末 "
          f"/ {n_months} 月 ({df['trade_date'].min()}~{df['trade_date'].max()}); "
          f"fr20 非空率={df['fr20'].notna().mean():.3f} adx 非空率={df['adx_14'].notna().mean():.3f} "
          f"rsi={df['rsi_14'].notna().mean():.3f} beta={df['beta'].notna().mean():.3f}; "
          f"regime={df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    table_rows, summary = [], {}
    for sig in SIGNALS:
        summary[sig] = {}
        print(f"\n--- {sig} ---", flush=True)
        for vname, ctrl in VARIANTS.items():
            ser = monthly_ic(df, sig, ctrl)
            summary[sig][vname] = summarize(ser)
            s = summary[sig][vname]
            print(f"  {vname:14s} IC={s['ic']:+.4f} t={s['t']} n={s['n_months']}", flush=True)
            for _, r in ser.iterrows():
                table_rows.append({"signal": sig, "variant": vname,
                                   "month": r["month"], "ic": r["ic"], "regime": r["regime"]})

    ic_table = pd.DataFrame(table_rows)
    IC_TABLE_P.parent.mkdir(parents=True, exist_ok=True)
    ic_table.to_parquet(IC_TABLE_P, index=False)
    print(f"\n[ic-table] -> {IC_TABLE_P.relative_to(ROOT)} ({len(ic_table):,} 行)", flush=True)

    # ── 决策: 最强蓄势信号的 orth_full(扣动量+市场+现有形态因子) 是否过线 ──
    def orth_ic(s):
        v = summary[s]["orth_full"]["ic"]
        return abs(v) if v is not None else -1
    best = max(SIGNALS, key=orth_ic)
    bo = summary[best]["orth_full"]
    bo_ic, bo_t = bo["ic"], (bo["t"] or 0.0)
    power_ok = bo["n_months"] >= MIN_MONTHS
    passed = (bo_ic is not None) and abs(bo_ic) >= IC_SIG and abs(bo_t) >= T_SIG
    status = "residual_signal" if passed else "no_residual"

    pr = summary[PRIMARY]
    print(f"\n[决策] 最强 {best}: orth_full IC={bo_ic:+.4f} t={bo_t:+.2f} n={bo['n_months']} "
          f"→ status={status}", flush=True)
    print(f"[定位 {PRIMARY}] raw={pr['raw']['ic']:+.4f} → -mom={pr['minus_mom']['ic']:+.4f} → "
          f"-market={pr['minus_market']['ic']:+.4f} → -pattern={pr['minus_pattern']['ic']:+.4f} → "
          f"orth_full={pr['orth_full']['ic']:+.4f}", flush=True)

    if status == "residual_signal":
        conclusion = (
            f"蓄势信号 Phase-1 (月末 rebalance {n_months}月): 最强 {best} 扣[动量+市场+现有形态因子"
            f"(MA斜率/ADX/布林/past_r5/RSI)]残差后 全期 IC={bo_ic:+.4f} |t|={abs(bo_t):.2f} "
            f"(n={bo['n_months']}月), 过线 (需 |IC|>={IC_SIG} & |t|>={T_SIG}) → residual_signal: "
            f"蓄势信号非现有 TA 因子换皮, 有独立预测残差 → 进 CB-004 walk-forward "
            f"(最终只认 walk-forward α, SIGN-R03)。")
    else:
        conclusion = (
            f"蓄势信号 Phase-1 (月末 rebalance {n_months}月): 最强 {best} 扣[动量+市场+现有形态因子"
            f"(MA斜率/ADX/布林/past_r5/RSI)]残差后 全期 IC={bo_ic:+.4f} |t|={abs(bo_t):.2f} "
            f"(n={bo['n_months']}月) 未过显著线 (需 |IC|>={IC_SIG} & |t|>={T_SIG}); "
            f"{PRIMARY}: raw IC={pr['raw']['ic']:+.4f} → 扣动量 {pr['minus_mom']['ic']:+.4f} → "
            f"扣形态因子 {pr['minus_pattern']['ic']:+.4f} → 全扣 {pr['orth_full']['ic']:+.4f}。"
            f"→ no_residual: 蓄势信号(MA20走平+收敛+缩量)对 r20 的预测力被现有 MA斜率/ADX/布林/"
            f"past_r5 标准 TA 因子吸收 (≈换皮), 无独立残差 → 廉价 REJECT, 跳过 CB-004 walk-forward "
            f"(SIGN-R12++ 消融闸)。注: 与 CB-002 盲点真实性不矛盾 — 盲点='系统漏掉这族启动', "
            f"本任务='蓄势特征是否提供超越现有因子的增量预测', 二者独立。")

    verdict = {
        "id": "CB-003",
        "status": status,
        "conclusion": conclusion,
        "alignment": "月末 rebalance (每月最后交易日全市场横截面, r20 标签近非重叠; vs SEAT-002 事件对齐)",
        "screen_thresholds": {"ic_sig": IC_SIG, "t_sig": T_SIG, "ic_collapse": IC_COLLAPSE,
                              "min_months": MIN_MONTHS,
                              "applies_to": "最强蓄势信号 orth_full(扣动量+市场+现有形态因子) 全期残差 IC"},
        "label": f"fr20 = close[t+{FWD_H}]/close[t]-1 (前向因果 outcome, 仅 cache 不进 features)",
        "signals": {"tested": SIGNALS, "primary": PRIMARY,
                    "def": "蓄势信号方向已对齐为'越大越蓄势'(s_flat/s_*squeeze/s_vol_shrink=负号), "
                           "s_coil=四分量月末截面 z 之和, s_consol=CB-001 候选布尔"},
        "controls": {"mom": MOM, "market": MARKET, "pattern": PATTERN,
                     "pattern_note": "现有标准 TA: cs_ma20_slope(MA斜率有符号), adx_14, rsi_14, "
                                     "cs_boll_bw(布林原始带宽), cs_past_r5; 全 backward ≤t",
                     "market_def": f"per-stock trailing {BETA_WIN}d beta vs 等权市场"},
        "power_ok_ge36months": power_ok,
        "best_signal": best,
        "metrics": {
            "n_months": int(bo["n_months"]),
            "best_orth_full_ic": bo_ic, "best_orth_full_t": bo["t"],
            "by_signal": summary,
            "primary_ablation": {
                "signal": PRIMARY,
                "raw_ic": pr["raw"]["ic"], "raw_t": pr["raw"]["t"],
                "minus_mom_ic": pr["minus_mom"]["ic"],
                "minus_market_ic": pr["minus_market"]["ic"],
                "minus_pattern_ic": pr["minus_pattern"]["ic"],
                "orth_full_ic": pr["orth_full"]["ic"], "orth_full_t": pr["orth_full"]["t"],
            },
        },
        "ablation_note": (
            "逐步消融 (raw → 单扣动量 → 单扣市场 → 单扣现有形态因子 → 全扣) 定位蓄势信号来源: "
            "若 raw 有 IC 但单扣 PATTERN(MA斜率/ADX/布林/past_r5/RSI) 即塌 = 蓄势信号本质是现有"
            "标准 TA 形态因子的非线性换皮, 非独立增量。横截面 rank-IC 天然差掉当日全市场常数项。"),
        "artifacts": [str(IC_TABLE_P.relative_to(ROOT)), str(CTRL_CACHE.relative_to(ROOT))],
        "guardrails": ["SIGN-R01 阈值冻结未改", "SIGN-R02 负结果=合法完成",
                       "SIGN-R03 IC 仅 go/no-go 非落地", "SIGN-R04 fr20 仅 outcome 不进 features",
                       "SIGN-R06 ST 源头排除", "SIGN-R08 控制面板 checkpoint",
                       "SIGN-R11 分 regime 评估", "SIGN-R12++ 扣[动量+市场+现有形态因子]消融"],
        "downstream": ("residual_signal → 进 CB-004 walk-forward (蓄势候选臂)。"
                       if status == "residual_signal"
                       else "no_residual → 蓄势信号≈现有形态因子换皮, 廉价 REJECT; CB-004 skip:true "
                            "(playbook CB-003 分叉, SIGN-R03 中间指标不落地)。"),
    }
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} -> {VERDICT_P.relative_to(ROOT)}", flush=True)

    if status == "no_residual":
        set_cb004_skip("CB-003=no_residual: 蓄势信号扣[动量+市场+现有形态因子]无残差(换皮), "
                       "precondition skip (playbook CB-003 分叉)。")

    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
