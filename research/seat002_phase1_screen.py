# -*- coding: utf-8 -*-
"""SEAT-002 — 龙虎榜席位印记 Phase-1 廉价筛查 (扣[自身动量+市场]正交 IC).

核心问 (北极星): 当日上榜股按"买方席位历史战绩"加权后, 扣掉**本股自身动量**(+市场 beta)
后, 高战绩席位跟随的股票还跑不跑赢? 若 sf_edge 信号只是"被聪明席位买的股本来就在涨"
(动量) 或市场 beta 换皮 → no_residual, 该轨廉价 REJECT。扣完仍显著 → residual_signal,
进 GATE walk-forward (与 NET 轨同闸)。

事件对齐 (vs NET-002 月末对齐): 龙虎榜信号天然稀疏 (仅上榜股 D+1 有值, 46K 行)。
月末 rebalance 会把横截面砍到几乎为零。故改**事件对齐**: 每个 signal_date t (=D+1) 是一个
横截面截面 (当日所有上榜股), 算当日残差 rank-IC vs 该股从 t 起的 fwd_r5/r20, 再按月聚合,
跨 ~41 月做 t 检验 (≥30月 功率达标, SIGN-R11 分 regime)。

输入 (全部因果, 复用):
  research/features/seat_footprint.parquet     (SEAT-001 信号 sf_edge_r5/r20/winrate, 已 D+1 + ST 排除)
  research/cache/seat001/fwd_returns.parquet   (fr5/fr20 = close[t+H]/close[t]-1, 因果 label)
  research/features/regime_timeline.parquet    (RG-001 regime: momentum/mixed/reversal, 分层)
  output/tushare_cache/daily/*.parquet         (算自身动量 mom_5/20/60 + 市场 beta 控制)

防泄漏 (SIGN-R04):
  - 信号侧 SEAT-001 已保证 D+1 生效 + 席位战绩仅用 avail<=d 已兑现 (时间维 loo)。
  - 标签 fr5/fr20 是从 signal_date t 起的严格未来收益, 与信号不重叠。
  - 自身动量 mom_h=close[t]/close[t-h]-1 (≤t, backward); 市场 beta trailing 60d (≤t)。
  - 横截面 rank-IC 天然差掉当日全市场常数项; beta 控制额外捕捉 per-stock 市场暴露。

纪律: SIGN-R03 IC 仅 go/no-go 非落地 (只认 GATE walk-forward α); SIGN-R06 ST 源头排除
  (SEAT-001 继承 + daily 控制侧再剔); SIGN-R01 阈值冻结; SIGN-R12++ 扣[动量+市场]消融。

裁决: research/verdicts/SEAT-002.json
  status ∈ {residual_signal, no_residual} (prd 注册口径)
  residual_signal 当且仅当 最强 (信号×label) orth_full 全期 |IC|>=0.02 且 |t|>=3。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SF_P = ROOT / "research" / "features" / "seat_footprint.parquet"
FWD_P = ROOT / "research" / "cache" / "seat001" / "fwd_returns.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"

CTRL_CACHE = ROOT / "research" / "cache" / "seat002_controls.parquet"   # mom+beta @ 信号日
PANEL_CACHE = ROOT / "research" / "cache" / "seat002_panel.parquet"     # 含 fr → cache 不进 features
IC_TABLE_P = ROOT / "research" / "cache" / "seat002_ic_table.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "SEAT-002.json"

KEY = ["ts_code", "trade_date"]
# (信号, 自然 label): 席位偏短线主看 r5, sf_edge_r20 看 r20
SIG_LABELS = [
    ("sf_edge_r5", "fr5"),
    ("sf_winrate_r5", "fr5"),
    ("sf_edge_r20", "fr20"),
    ("sf_edge_r5", "fr20"),     # 交叉报: r5 战绩对 r20 收益
]
PRIMARY = ("sf_edge_r5", "fr5")   # 北极星头号: 席位 r5 战绩 → r5 收益

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
MIN_XS = 15          # 单日最少横截面股票数 (数据显示每日 >=17)
MIN_MONTHS = 30      # 功率下限 (prd: ≥30月)

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


def build_controls(signal_dates: set, st: set) -> pd.DataFrame:
    """自身动量 mom_5/20/60 (trailing) + 市场 beta (trailing 60d), 仅在信号日取值 (因果)."""
    if CTRL_CACHE.exists():
        print(f"[ctrl] checkpoint 命中 {CTRL_CACHE.name}", flush=True)
        return pd.read_parquet(CTRL_CACHE)

    print("[ctrl] 加载 daily close 算自身动量 + 等权市场 beta ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(DAILY_DIR.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px[(px["close"] > 0)].drop_duplicates(KEY, keep="last")
    if st:
        px = px[~px["ts_code"].isin(st)]
    px = px.sort_values(KEY).reset_index(drop=True)

    g = px.groupby("ts_code")["close"]
    for h in (5, 20, 60):
        px[f"mom_{h}"] = g.transform(lambda s, h=h: s / s.shift(h) - 1.0)

    # 市场 beta: 等权市场日收益 + per-stock trailing 60d beta = cov/var
    px["r1"] = g.pct_change()
    mkt = px.groupby("trade_date")["r1"].mean().rename("mkt")
    px = px.merge(mkt, on="trade_date", how="left")
    px["r1m"] = px["r1"] * px["mkt"]
    roll = (px.groupby("ts_code")[["r1", "r1m", "mkt"]]
            .rolling(BETA_WIN, min_periods=BETA_MINP).mean()
            .reset_index(level=0, drop=True))
    cov = roll["r1m"] - roll["r1"] * roll["mkt"]
    mser = mkt.reset_index().sort_values("trade_date")
    mser["m2"] = mser["mkt"] ** 2
    em = mser["mkt"].rolling(BETA_WIN, min_periods=BETA_MINP).mean()
    em2 = mser["m2"].rolling(BETA_WIN, min_periods=BETA_MINP).mean()
    mser["var_mkt"] = (em2 - em ** 2).values
    px = px.merge(mser[["trade_date", "var_mkt"]], on="trade_date", how="left")
    px["beta"] = cov.values / px["var_mkt"].replace(0, np.nan)

    out = px[px["trade_date"].isin(signal_dates)][KEY + MOM + MARKET].reset_index(drop=True)
    CTRL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CTRL_CACHE, index=False)
    print(f"[ctrl] 落盘 {CTRL_CACHE.name} ({len(out):,} 行, {out['trade_date'].nunique()} 信号日)", flush=True)
    return out


def build_panel(st: set) -> pd.DataFrame:
    if PANEL_CACHE.exists():
        print(f"[panel] checkpoint 命中 {PANEL_CACHE.name}", flush=True)
        return pd.read_parquet(PANEL_CACHE)

    sf = pd.read_parquet(SF_P)
    sf["trade_date"] = sf["trade_date"].astype(str)
    if st:
        sf = sf[~sf["ts_code"].isin(st)]   # SEAT-001 已排, 双保险
    signal_dates = set(sf["trade_date"].unique())

    fwd = pd.read_parquet(FWD_P)
    fwd["trade_date"] = fwd["trade_date"].astype(str)

    ctrl = build_controls(signal_dates, st)

    df = (sf.merge(fwd, on=KEY, how="left")        # label: 从信号日 t 起的 fr5/fr20
            .merge(ctrl, on=KEY, how="left"))
    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]]
    rg["trade_date"] = rg["trade_date"].astype(str)
    df = df.merge(rg, on="trade_date", how="left")
    df["regime"] = df["regime"].fillna("unknown")

    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_CACHE, index=False)
    print(f"[panel] 落盘 {PANEL_CACHE.name} ({len(df):,} 行, "
          f"{df['trade_date'].nunique()} 信号日, {df['ts_code'].nunique()} 股)", flush=True)
    return df


def monthly_ic_series(df: pd.DataFrame, sig: str, label: str, ctrl: list[str]) -> pd.DataFrame:
    """事件对齐: 逐**信号日**残差化 sig 对 ctrl, 再 rank-IC vs label; 按月聚合 → 月 IC 序列。

    返回 [month, ic, regime]: ic=该月各日 IC 均值, regime=该月主导 regime。
    """
    need = [sig, label] + ctrl
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
        ic = rank_ic(v, s[label].to_numpy(float))
        rows.append((d, ic, s["regime"].iloc[0] if "regime" in s else "unknown"))
    daily = pd.DataFrame(rows, columns=["date", "ic", "regime"])
    if daily.empty:
        return pd.DataFrame(columns=["month", "ic", "regime"])
    daily["month"] = daily["date"].str[:6]
    # 月聚合: 日 IC 均值; regime 取该月众数
    agg = daily.groupby("month").agg(
        ic=("ic", "mean"),
        regime=("regime", lambda x: x.value_counts().index[0])).reset_index()
    return agg


def main() -> int:
    t0 = time.time()
    print("\n=== SEAT-002 席位印记 Phase-1 廉价筛查 (扣[动量+市场]正交IC, 事件对齐) ===\n", flush=True)
    st = load_st_set()
    print(f"[st] ST 源头排除集合 {len(st)} 只", flush=True)
    df = build_panel(st)
    n_months = df["trade_date"].str[:6].nunique()
    print(f"[data] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / "
          f"{df['trade_date'].nunique()} 信号日 / {n_months} 月 "
          f"({df['trade_date'].min()}~{df['trade_date'].max()}); "
          f"fr5 非空率={df['fr5'].notna().mean():.3f} fr20={df['fr20'].notna().mean():.3f} "
          f"beta 非空率={df['beta'].notna().mean():.3f}; "
          f"regime={df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    table_rows = []
    summary = {}
    for sig, label in SIG_LABELS:
        pair = f"{sig}->{label}"
        summary[pair] = {}
        print(f"\n--- {pair} ---", flush=True)
        for vname, ctrl in VARIANTS.items():
            ser = monthly_ic_series(df, sig, label, ctrl)
            ic = ser["ic"].to_numpy(float)
            n = int(np.sum(~np.isnan(ic)))
            mean_ic = float(np.nanmean(ic)) if n else np.nan
            t = tstat(ic)
            reg_block = {}
            for rg, gg in ser.groupby(ser["regime"].fillna("unknown")):
                gic = gg["ic"].to_numpy(float)
                gt = tstat(gic)
                reg_block[str(rg)] = {
                    "ic": round(float(np.nanmean(gic)), 4) if len(gic) else None,
                    "t": round(gt, 2) if len(gic) > 1 and not np.isnan(gt) else None,
                    "n": int(np.sum(~np.isnan(gic)))}
            summary[pair][vname] = {
                "ic": round(mean_ic, 4),
                "t": round(t, 2) if not np.isnan(t) else None,
                "n_months": n, "by_regime": reg_block}
            print(f"  {vname:13s} IC={mean_ic:+.4f} t={t:+.2f} n={n}", flush=True)
            for _, r in ser.iterrows():
                table_rows.append({"pair": pair, "signal": sig, "label": label,
                                   "variant": vname, "month": r["month"],
                                   "ic": r["ic"], "regime": r["regime"]})

    ic_table = pd.DataFrame(table_rows)
    IC_TABLE_P.parent.mkdir(parents=True, exist_ok=True)
    ic_table.to_parquet(IC_TABLE_P, index=False)
    print(f"\n[ic-table] -> {IC_TABLE_P.relative_to(ROOT)} ({len(ic_table):,} 行)", flush=True)

    # ── 决策: 最强 (信号×label) 的 orth_full(扣动量+市场) 是否过线 ──
    pairs = [f"{s}->{l}" for s, l in SIG_LABELS]

    def orth(p):
        return summary[p]["orth_full"]
    best_pair = max(pairs, key=lambda p: abs(orth(p)["ic"]) if orth(p)["ic"] is not None else -1)
    bo = orth(best_pair)
    bo_ic, bo_t = bo["ic"], (bo["t"] or 0.0)
    power_ok = bo["n_months"] >= MIN_MONTHS

    passed = abs(bo_ic) >= IC_SIG and abs(bo_t) >= T_SIG
    status = "residual_signal" if passed else "no_residual"

    prim_key = f"{PRIMARY[0]}->{PRIMARY[1]}"
    prim = summary[prim_key]
    print(f"\n[决策] 最强 {best_pair}: orth_full IC={bo_ic:+.4f} t={bo_t:+.2f} "
          f"n={bo['n_months']} → status={status}", flush=True)
    print(f"[定位] {prim_key}: raw IC={prim['raw']['ic']:+.4f} → "
          f"minus_mom={prim['minus_mom']['ic']:+.4f} → "
          f"minus_market={prim['minus_market']['ic']:+.4f} → "
          f"orth_full={prim['orth_full']['ic']:+.4f}", flush=True)

    if status == "residual_signal":
        conclusion = (
            f"席位印记 Phase-1 (事件对齐 {n_months}月): 最强 {best_pair} 扣[自身动量+市场]残差后 "
            f"全期 IC={bo_ic:+.4f} |t|={abs(bo_t):.2f} (n={bo['n_months']}月), "
            f"过线 (需 |IC|>={IC_SIG} & |t|>={T_SIG}) → residual_signal: 高战绩席位跟随扣自身动量后"
            f"仍有独立预测, 进 GATE-001 walk-forward (最终只认 walk-forward α, SIGN-R03)。")
    else:
        conclusion = (
            f"席位印记 Phase-1 (事件对齐 {n_months}月): 最强 {best_pair} 扣[自身动量+市场]残差后 "
            f"全期 IC={bo_ic:+.4f} |t|={abs(bo_t):.2f} (n={bo['n_months']}月) 未过显著线 "
            f"(需 |IC|>={IC_SIG} & |t|>={T_SIG}); {prim_key} raw IC={prim['raw']['ic']:+.4f} → "
            f"扣动量 {prim['minus_mom']['ic']:+.4f} → 全扣 {prim['orth_full']['ic']:+.4f}。"
            f"→ no_residual: 席位战绩信号扣完[动量+市场]无独立残差 (聪明席位跟随≈被买的股本就在动)。"
            f"该轨廉价 REJECT 记录。")

    verdict = {
        "id": "SEAT-002",
        "status": status,
        "conclusion": conclusion,
        "alignment": "event-aligned (每信号日 D+1 一横截面, 按月聚合跨月 t 检验; vs NET-002 月末对齐)",
        "screen_thresholds": {"ic_sig": IC_SIG, "t_sig": T_SIG,
                              "ic_collapse": IC_COLLAPSE, "min_months": MIN_MONTHS,
                              "applies_to": "best (signal×label) orth_full(扣动量+市场) full-sample residual IC"},
        "labels": "fr5/fr20 = close[t+H]/close[t]-1 (因果, 从信号日 t 起, 复用 seat001 fwd_returns)",
        "controls": {"mom": MOM, "market": MARKET,
                     "mom_def": "mom_h=close[t]/close[t-h]-1 (trailing, ≤t)",
                     "market_def": f"per-stock trailing {BETA_WIN}d beta vs 等权市场日收益 (因果)"},
        "power_ok_ge30months": power_ok,
        "best_pair": best_pair,
        "primary_pair": prim_key,
        "metrics": {
            "n_months": int(bo["n_months"]),
            "best_orth_full_ic": bo_ic,
            "best_orth_full_t": bo["t"],
            "by_pair": summary,
            "primary_ablation": {
                "raw_ic": prim["raw"]["ic"], "raw_t": prim["raw"]["t"],
                "minus_mom_ic": prim["minus_mom"]["ic"],
                "minus_market_ic": prim["minus_market"]["ic"],
                "orth_full_ic": prim["orth_full"]["ic"], "orth_full_t": prim["orth_full"]["t"],
            },
        },
        "ablation_note": (
            "逐步消融 (raw → 单扣动量 → 单扣市场 beta → 全扣) 定位席位信号来源: "
            "若 raw 有 IC 但扣自身动量即塌 = 聪明席位跟随本质是被买股的自身动量 (席位追涨/接力), "
            "非独立行为 alpha。横截面 rank-IC 天然差掉当日全市场常数项, beta 控制额外捕捉市场暴露。"),
        "artifacts": ["research/cache/seat002_ic_table.parquet",
                      "research/cache/seat002_panel.parquet",
                      "research/cache/seat002_controls.parquet"],
        "guardrails": ["SIGN-R01 阈值冻结未改", "SIGN-R02 负结果=合法完成",
                       "SIGN-R03 IC 仅 go/no-go 非落地", "SIGN-R04 fr 因果 label 不进 features",
                       "SIGN-R06 ST 源头排除 (SEAT-001 + daily 控制侧)", "SIGN-R11 分 regime 评估",
                       "SIGN-R12++ 扣[自身动量+市场]残差化消融"],
        "downstream": (
            "no_residual → SEAT 轨 REJECT; NET-002 亦 no_residual → 双轨皆否 → "
            "GATE-001 status=REJECT 不跑 walk-forward (responsePlaybook GATE 分叉)。"
            if status == "no_residual"
            else "residual_signal → 进 GATE-001 walk-forward (SEAT 臂)。"),
    }
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
