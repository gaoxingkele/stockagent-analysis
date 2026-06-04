# -*- coding: utf-8 -*-
"""FU-002 — 成长因子构建 + 控制集 (R12++ 消融核心).

输入: research/features/fundamental_pit.parquet (FU-001, point-in-time, ann_date 对齐)。
产出: research/features/fundamental_factors.parquet, 键 (ts_code, trade_date), 含
  成长因子 (winsor + 横截面 z + 行业中性 + 组合) + 控制集 (动量/size/value/其他基本面),
  全部 point-in-time 因果, 不含任何 forward 字段 (label 由 FU-003 另接, 不落本 parquet)。

成长因子 (待验证的正交候选):
  growth_or   = 行业中性 z(winsor(or_yoy))          营收增速
  growth_np   = 行业中性 z(winsor(netprofit_yoy))    利润增速
  growth_composite = 0.5*(growth_or + growth_np)     成长组合

控制集 (R12++: 决定成长是否只是 size/value/动量 reskin):
  动量    mom_5 / mom_20 / mom_60   (从 daily close, 月末采样, backward)
  size    ln_total_mv               (ln total_mv)
  value   inv_pe / inv_pb           (1/pe, 1/pb; 仅正值)
  其他基本面 roe / margin           (roe, grossprofit_margin)

纪律:
  - SIGN-R04 泄漏前置闸: 成长用 FU-001 的 ann_date 对齐值; 动量从 daily close 月末采样
    (close/close.shift(n)-1, 全 backward); 本 parquet **不含 forward 字段** (label 在 FU-003)。
  - SIGN-R06 ST 已在 FU-001 源头排除 (面板继承)。
  - SIGN-R08 长任务 checkpoint: 动量面板缓存, 已完成跳过。
  - SIGN-R11 regime: merge regime_timeline 便于 FU-003 分层 (缺则 NaN, 不强行回填)。

verdict: research/verdicts/FU-002.json status=built + 因子/控制清单 + 相关性矩阵 (成长 vs 控制,
  逐日横截面 Spearman 均值)。中间相关性非落地 (SIGN-R03), 仅描述结构。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
PIT_P = ROOT / "research" / "features" / "fundamental_pit.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
CACHE = ROOT / "research" / "cache"
MOM_CACHE = CACHE / "fu002_momentum.parquet"
OUT_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "FU-002.json"

KEY = ["ts_code", "trade_date"]
GROWTH_RAW = ["or_yoy", "netprofit_yoy"]
MOM_LAGS = (5, 20, 60)
CONTROLS = ["mom_5", "mom_20", "mom_60", "ln_total_mv", "inv_pe", "inv_pb", "roe", "margin"]
GROWTH_FACTORS = ["growth_or", "growth_np", "growth_composite"]

# forward 黑名单 (落盘前自查, 与 verify.py 一致)
FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret", "label", "target"}


def winsor_z_neutral(s: pd.Series, industry: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    """单日横截面: winsorize -> z-score -> 行业中性 (减行业均值)。"""
    x = s.astype(float).copy()
    valid = x.notna()
    if valid.sum() < 10:
        return pd.Series(np.nan, index=s.index)
    ql, qh = x[valid].quantile([lo, hi])
    x = x.clip(ql, qh)
    mu, sd = x[valid].mean(), x[valid].std(ddof=0)
    z = (x - mu) / sd if sd > 0 else x * 0.0
    # 行业中性: 减各行业 z 均值 (NaN 行业归一组)
    ind = industry.fillna("__NA__")
    z = z - z.groupby(ind).transform("mean")
    return z


def build_factors_for_date(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    out["growth_or"] = winsor_z_neutral(g["or_yoy"], g["industry"])
    out["growth_np"] = winsor_z_neutral(g["netprofit_yoy"], g["industry"])
    out["growth_composite"] = out[["growth_or", "growth_np"]].mean(axis=1, skipna=True)
    # 若两者皆缺则组合也缺
    both_na = g["or_yoy"].isna() & g["netprofit_yoy"].isna()
    out.loc[both_na, "growth_composite"] = np.nan
    return out


def build_momentum(dates: list[str]) -> pd.DataFrame:
    """从 daily close 计算 mom_5/20/60 (backward), 月末 rebalance 日采样。"""
    if MOM_CACHE.exists():
        print(f"[mom] checkpoint 命中 {MOM_CACHE.name}", flush=True)
        return pd.read_parquet(MOM_CACHE)
    print("[mom] 加载 daily close 计算 mom_5/20/60 (因果, 月末采样) ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(DAILY.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = (px[px["close"] > 0].drop_duplicates(KEY, keep="last")
          .sort_values(KEY).reset_index(drop=True))
    g = px.groupby("ts_code", sort=False)["close"]
    for n in MOM_LAGS:
        px[f"mom_{n}"] = g.transform(lambda s, n=n: s / s.shift(n) - 1.0)
    out = px[px["trade_date"].isin(dates)][KEY + [f"mom_{n}" for n in MOM_LAGS]].copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MOM_CACHE, index=False)
    print(f"[mom] -> {MOM_CACHE.relative_to(ROOT)} ({len(out):,} 行)", flush=True)
    return out


def xs_spearman(df: pd.DataFrame, a: str, b: str) -> float:
    """逐日横截面 Spearman 相关均值 (rank-Pearson per date, 再跨日平均)。"""
    sub = df[["trade_date", a, b]].dropna()
    if sub.empty:
        return np.nan
    ra = sub.groupby("trade_date")[a].rank()
    rb = sub.groupby("trade_date")[b].rank()
    tmp = pd.DataFrame({"d": sub["trade_date"].values, "a": ra.values, "b": rb.values})
    tmp["ab"] = tmp["a"] * tmp["b"]
    agg = tmp.groupby("d").agg(n=("a", "size"), sa=("a", "sum"), sb=("b", "sum"),
                               saa=("a", lambda x: float(np.dot(x, x))),
                               sbb=("b", lambda x: float(np.dot(x, x))),
                               sab=("ab", "sum"))
    n = agg["n"]
    cov = agg["sab"] - agg["sa"] * agg["sb"] / n
    va = agg["saa"] - agg["sa"] ** 2 / n
    vb = agg["sbb"] - agg["sb"] ** 2 / n
    denom = np.sqrt(va * vb)
    corr = (cov / denom).where((denom > 0) & (n >= 5)).dropna()
    return float(corr.mean()) if len(corr) else np.nan


def main() -> int:
    t0 = time.time()
    print("\n=== FU-002 成长因子 + 控制集 ===\n", flush=True)

    pit = pd.read_parquet(PIT_P)
    pit["trade_date"] = pit["trade_date"].astype(str)
    dates = sorted(pit["trade_date"].unique())
    print(f"[pit] {len(pit):,} 行 / {pit['ts_code'].nunique()} 股 / {len(dates)} 月度网格 "
          f"({dates[0]}~{dates[-1]})", flush=True)

    # 行业 (stock_basic, 用于行业中性)
    basic = pd.read_parquet(BASIC)[["ts_code", "industry"]].drop_duplicates("ts_code")
    df = pit.merge(basic, on="ts_code", how="left")
    n_no_ind = int(df["industry"].isna().sum())
    print(f"[industry] merge stock_basic; 无行业 {n_no_ind} 行 (归 __NA__ 组)", flush=True)

    # ── 控制集: size / value / 其他基本面 (PIT) ──
    df["ln_total_mv"] = np.log(df["total_mv"].where(df["total_mv"] > 0))
    df["inv_pe"] = np.where(df["pe"] > 0, 1.0 / df["pe"], np.nan)
    df["inv_pb"] = np.where(df["pb"] > 0, 1.0 / df["pb"], np.nan)
    df["roe"] = df["roe"]
    df["margin"] = df["grossprofit_margin"]

    # ── 控制集: 动量 (daily close 月末采样) ──
    mom = build_momentum(dates)
    df = df.merge(mom, on=KEY, how="left")
    for n in MOM_LAGS:
        cov = float(df[f"mom_{n}"].notna().mean())
        print(f"[ctrl] mom_{n} 覆盖 {cov:.1%}", flush=True)

    # ── 成长因子: 逐日 winsor + z + 行业中性 + 组合 ──
    print("\n[factor] 逐日 winsorize(1/99) + 横截面 z + 行业中性 + 组合 ...", flush=True)
    fac = (df.groupby("trade_date", group_keys=False)[["or_yoy", "netprofit_yoy", "industry"]]
             .apply(build_factors_for_date))
    df = df.join(fac)
    for f in GROWTH_FACTORS:
        cov = float(df[f].notna().mean())
        print(f"  {f:18s} 覆盖 {cov:.1%}  mean={np.nanmean(df[f]):+.4f} std={np.nanstd(df[f]):.4f}",
              flush=True)

    # ── regime merge (便于 FU-003 分层; 缺则 NaN, 不回填) ──
    if REGIME_P.exists():
        reg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]].copy()
        reg["trade_date"] = reg["trade_date"].astype(str)
        df = df.merge(reg, on="trade_date", how="left")
        cov_reg = float(df["regime"].notna().mean())
        print(f"[regime] merge regime_timeline; 覆盖 {cov_reg:.1%} "
              f"({df['regime'].dropna().unique().tolist()})", flush=True)
    else:
        df["regime"] = np.nan

    # ── 落盘 (仅因子 + 控制 + 结构列, 无 forward 字段) ──
    keep = KEY + ["industry", "regime"] + GROWTH_RAW + GROWTH_FACTORS + CONTROLS
    out = df[keep].sort_values(KEY).reset_index(drop=True)

    bad = [c for c in out.columns if c.lower() in FORWARD_TOKENS
           or c.lower().startswith(("fwd_", "future_", "next_", "r1_next"))
           or c.lower().endswith("_forward")]
    assert not bad, f"输出含 forward 字段: {bad}"
    out.to_parquet(OUT_P, index=False)
    print(f"\n[out] -> {OUT_P.relative_to(ROOT)} ({len(out):,} 行 × {len(out.columns)} 列)", flush=True)

    # ── 相关性矩阵: 成长因子 vs 控制 (逐日横截面 Spearman 均值) ──
    print("\n[corr] 成长 vs 控制 (逐日横截面 Spearman 均值) ...", flush=True)
    corr_mat = {}
    for gf in GROWTH_FACTORS:
        row = {}
        for c in CONTROLS:
            row[c] = round(xs_spearman(out, gf, c), 4)
        # 成长因子之间
        for other in GROWTH_FACTORS:
            if other != gf:
                row[other] = round(xs_spearman(out, gf, other), 4)
        corr_mat[gf] = row
        top = sorted(((abs(v), c) for c, v in row.items() if c in CONTROLS and not np.isnan(v)),
                     reverse=True)[:3]
        print(f"  {gf:18s} 最强控制相关: " +
              ", ".join(f"{c}={row[c]:+.3f}" for _, c in top), flush=True)

    # 覆盖率分市值档 (size = ln_total_mv 三分位)
    cov_by_bucket = {}
    valid_mv = out["ln_total_mv"].notna()
    if valid_mv.any():
        q = out.loc[valid_mv, "ln_total_mv"].rank(pct=True)
        bucket = pd.cut(q, [0, 1/3, 2/3, 1.0], labels=["small", "mid", "large"],
                        include_lowest=True)
        tmp = out.loc[valid_mv].assign(_b=bucket.values)
        for b, gg in tmp.groupby("_b", observed=True):
            cov_by_bucket[str(b)] = round(float(gg["growth_composite"].notna().mean()), 4)

    verdict = {
        "task": "FU-002",
        "status": "built",
        "conclusion": (
            f"成长因子 (growth_or/growth_np/growth_composite, winsor+横截面z+行业中性) "
            f"+ 控制集 (动量mom_5/20/60 + size ln_mv + value inv_pe/pb + 其他基本面 roe/margin) "
            f"落盘 {len(out):,} 行 × {out['trade_date'].nunique()} 月度网格; growth_composite 覆盖 "
            f"{out['growth_composite'].notna().mean():.1%}; growth_composite vs size 相关 "
            f"{corr_mat['growth_composite']['ln_total_mv']:+.3f} / vs value(inv_pe) "
            f"{corr_mat['growth_composite']['inv_pe']:+.3f} / vs mom_20 "
            f"{corr_mat['growth_composite']['mom_20']:+.3f} (逐日横截面 Spearman 均值; "
            f"FU-003 据此残差化消融)。无 forward 字段 (label 在 FU-003 另接)。"),
        "artifact": str(OUT_P.relative_to(ROOT)),
        "responsePlaybook_branch": "n/a (构建任务, 无分叉; 相关性结构供 FU-003 R12++ 消融)",
        "metrics": {
            "rows": int(len(out)),
            "n_stocks": int(out["ts_code"].nunique()),
            "n_rebalance_dates": int(out["trade_date"].nunique()),
            "date_range": [out["trade_date"].min(), out["trade_date"].max()],
            "growth_factors": GROWTH_FACTORS,
            "growth_raw": GROWTH_RAW,
            "controls": CONTROLS,
            "control_groups": {
                "momentum": ["mom_5", "mom_20", "mom_60"],
                "size": ["ln_total_mv"],
                "value": ["inv_pe", "inv_pb"],
                "other_fundamental": ["roe", "margin"],
            },
            "factor_construction": "per-date winsorize(1/99) -> cross-sectional z -> industry-neutral (minus industry mean)",
            "coverage_growth_composite": round(float(out["growth_composite"].notna().mean()), 4),
            "coverage_by_mv_bucket": cov_by_bucket,
            "coverage_controls": {c: round(float(out[c].notna().mean()), 4) for c in CONTROLS},
            "n_no_industry_rows": n_no_ind,
            "corr_growth_vs_controls_xs_spearman": corr_mat,
            "regime_coverage": round(float(out["regime"].notna().mean()), 4),
            "leakage_free": True,
            "no_forward_fields": True,
        },
        "guardrails": ["SIGN-R04 PIT 因果 + 无 forward 字段", "SIGN-R06 ST 继承 FU-001 源头排除",
                       "SIGN-R08 动量面板 checkpoint", "SIGN-R11 regime 已 merge 供 FU-003 分层",
                       "SIGN-R12++ 控制集=动量+size+value+其他基本面"],
    }
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
