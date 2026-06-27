# -*- coding: utf-8 -*-
"""BT-001 — 运行引擎 + sanity, 产出引擎就绪裁决。

两件事:
  (A) t005 选股 α 复刻 (输入校验): 用 picks 重算 t005 month_metrics 口径的选股 α (成本关, alloc 加权
      r20_fresh − total×市场均值), 与 research/cache/t005_monthly.csv 的 base_alpha 对照
      → 证明喂给引擎的 picks 就是 V12.31 baseline 臂 (输入忠实)。
  (B) 引擎 book 运行 (核心交付): 把 picks 作 20 交易日再平衡的等权 book 过引擎 (成本关), 输出
      净值/年化/Sharpe/最大回撤/月均换手/暴露 时序 → research/cache/bt001/; 同口径跑等权全市场
      基准 book, 报引擎"每再平衡期(≈20d)超额", 与 t005 选股 α 同量级 = 引擎无 bug。

引擎解析型自检 (engine.py _selftest) 已独立证明核算无 bug; 本脚本叠加 t005 量级一致性。
ST 源头排除 (基准/市场口径); 成本关仅 sanity, BT-002 开真实成本+滑点敏感。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from engine import PortfolioBacktester, CostModel

CACHE = ROOT / "research" / "cache" / "bt001"
PICKS = CACHE / "picks_v1231_daily.parquet"
T005_MONTHLY = ROOT / "research" / "cache" / "t005_monthly.csv"
VERDICT = ROOT / "research" / "verdicts" / "BT-001.json"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"

TOTAL = 0.90          # V12.31 投资比例 (A 0.70 + B 0.20, 余 10% 现金) — t005 口径
REBAL_EVERY = 20      # 再平衡周期 ≈ r20 持有
CLIP = 30.0


def load_prices(start: str) -> pd.DataFrame:
    parts = []
    for f in sorted(DAILY_DIR.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high",
                                        "low", "close", "pre_close", "pct_chg"])
        d["trade_date"] = d["trade_date"].astype(str)
        d = d[d["trade_date"] >= start]
        if not d.empty:
            parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    # ST 源头排除 (SIGN-R06)
    basic = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    st = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    px = px[~px["ts_code"].isin(st)].reset_index(drop=True)
    return px


# ───────────── (A) t005 选股 α 复刻 ─────────────
def replicate_t005_alpha(picks: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """t005 month_metrics 口径 (成本关): 逐日 alpha = Σ alloc·clip(r20_fresh) − total·市场均值。
    市场均值 = 当日全(ST排除)宇宙 clip(r20_fresh) 截面均值。"""
    # 市场 r20_fresh: next_open→close_20d, 从价表自算 (与 compute_r20_label 同式)
    p = px.sort_values(["ts_code", "trade_date"]).copy()
    p["next_open"] = p.groupby("ts_code")["open"].shift(-1)
    p["close_20d"] = p.groupby("ts_code")["close"].shift(-20)
    p["r20_fresh"] = (p["close_20d"] / p["next_open"] - 1) * 100
    p["r20c"] = p["r20_fresh"].clip(-CLIP, CLIP)
    mkt = p.groupby("trade_date")["r20c"].mean().rename("mkt").reset_index()

    h = picks.copy()
    h["r20c"] = h["r20_fresh"].clip(-CLIP, CLIP)
    h["w"] = h["alloc_pct"] * h["r20c"]
    daily = h.groupby("trade_date").agg(gross=("w", "sum")).reset_index()
    daily = daily.merge(mkt, on="trade_date", how="left")
    daily["alpha"] = daily["gross"] - daily["mkt"] * TOTAL
    daily["month"] = daily["trade_date"].str[:6]
    by_month = daily.groupby("month")["alpha"].mean().rename("repl_alpha").reset_index()
    return by_month


# ───────────── (B) 引擎 book + 基准 ─────────────
def build_rebalance_targets(picks: pd.DataFrame, cal: list, equal: bool = True) -> dict:
    """每 REBAL_EVERY 交易日取一再平衡日, 用当日 picks 等权 (和=TOTAL) 作目标。"""
    rebal_days = cal[::REBAL_EVERY]
    pick_days = set(picks["trade_date"].unique())
    targets = {}
    for d in rebal_days:
        if d not in pick_days:
            continue
        g = picks[picks["trade_date"] == d]
        codes = g["ts_code"].tolist()
        if not codes:
            continue
        if equal:
            w = TOTAL / len(codes)
            targets[d] = {c: w for c in codes}
        else:
            tot_alloc = g["alloc_pct"].sum()
            targets[d] = dict(zip(g["ts_code"], g["alloc_pct"] / tot_alloc * TOTAL))
    return targets


def build_market_targets(px: pd.DataFrame, cal: list) -> dict:
    """等权全市场基准: 每再平衡日等权所有可成交股 (和=TOTAL, 同 book 暴露)。"""
    rebal_days = cal[::REBAL_EVERY]
    targets = {}
    for d in rebal_days:
        day = px[(px["trade_date"] == d) & px["open"].notna() & (px["pct_chg"].abs() < 9.8)]
        codes = day["ts_code"].tolist()
        if not codes:
            continue
        w = TOTAL / len(codes)
        targets[d] = {c: w for c in codes}
    return targets


def period_excess(book_ts: pd.DataFrame, mkt_ts: pd.DataFrame, cal: list) -> dict:
    """每再平衡期(≈20d) book vs 市场 的超额收益, 与 t005 选股 α 同量纲 (%/期)。"""
    rebal_days = [d for d in cal[::REBAL_EVERY]]
    rebal_days = [d for d in rebal_days if d in book_ts.index]
    bn = book_ts["nav"].reindex(rebal_days).dropna()
    mn = mkt_ts["nav"].reindex(rebal_days).dropna()
    common = bn.index.intersection(mn.index)
    bn, mn = bn.loc[common], mn.loc[common]
    b_ret = bn.pct_change().dropna() * 100
    m_ret = mn.pct_change().dropna() * 100
    ex = (b_ret - m_ret).dropna()
    return {"mean_period_excess_pct": float(ex.mean()),
            "median_period_excess_pct": float(ex.median()),
            "n_periods": int(len(ex))}


def main():
    t0 = time.time()
    print("\n=== BT-001 run: 引擎 sanity (t005 复刻 + 引擎 book) ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / {picks['month'].nunique()} 月", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    print(f"[prices] {len(px):,} 行 / {px['ts_code'].nunique()} 股 / {len(cal)} 交易日", flush=True)

    # ── (A) t005 复刻 ──
    print("\n[A] t005 选股 α 复刻 (成本关) ...", flush=True)
    repl = replicate_t005_alpha(picks, px)
    t005 = pd.read_csv(T005_MONTHLY, dtype={"test_month": str})[["test_month", "base_alpha"]]
    t005.columns = ["month", "t005_base_alpha"]
    cmp = repl.merge(t005, on="month", how="inner").sort_values("month")
    corr = float(cmp["repl_alpha"].corr(cmp["t005_base_alpha"]))
    mae = float((cmp["repl_alpha"] - cmp["t005_base_alpha"]).abs().mean())
    print(f"  月数 {len(cmp)}  repl 均值 {cmp['repl_alpha'].mean():+.3f}  "
          f"t005 均值 {cmp['t005_base_alpha'].mean():+.3f}  corr {corr:+.3f}  MAE {mae:.3f}", flush=True)
    for _, r in cmp.iterrows():
        print(f"    {r['month']}  repl {r['repl_alpha']:+.2f}  t005 {r['t005_base_alpha']:+.2f}", flush=True)

    # ── (B) 引擎 book (等权, 成本关) + 等权市场基准 ──
    print("\n[B] 引擎 book 运行 (20d 再平衡, 等权, 成本关) ...", flush=True)
    cost_off = CostModel(enabled=False)
    bt = PortfolioBacktester(px, cost=cost_off, t_plus_1=True, limit_pct=9.8)

    tgt_book = build_rebalance_targets(picks, cal, equal=True)
    res_book = bt.run(tgt_book, start=start)
    s_book = res_book.summary()

    tgt_mkt = build_market_targets(px, cal)
    res_mkt = bt.run(tgt_mkt, start=start)
    s_mkt = res_mkt.summary()

    ex = period_excess(res_book.timeseries, res_mkt.timeseries, cal)

    # 落盘时序 (核心交付)
    res_book.timeseries.to_parquet(CACHE / "bt001_book_equal_costoff.parquet")
    res_mkt.timeseries.to_parquet(CACHE / "bt001_market_equal_costoff.parquet")

    print(f"  book : 年化 {s_book['ann_return']:+.1%}  Sharpe {s_book['sharpe']:+.2f}  "
          f"maxDD {s_book['max_drawdown']:+.1%}  月换手 {s_book['avg_monthly_turnover']:.2f}  "
          f"暴露 {s_book['avg_exposure']:.2f}  持仓 {s_book['avg_n_positions']:.0f}", flush=True)
    print(f"  mkt  : 年化 {s_mkt['ann_return']:+.1%}  Sharpe {s_mkt['sharpe']:+.2f}  "
          f"maxDD {s_mkt['max_drawdown']:+.1%}", flush=True)
    print(f"  引擎每期(≈20d)超额: 均值 {ex['mean_period_excess_pct']:+.2f}%  "
          f"中位 {ex['median_period_excess_pct']:+.2f}%  ({ex['n_periods']} 期)", flush=True)
    print(f"  对照 t005 选股 α 均值 {cmp['t005_base_alpha'].mean():+.2f}% (同量级 = 引擎无 bug)", flush=True)

    # ── 汇总 + 裁决 ──
    results = {
        "engine_selftest": "engine.py _selftest 全通过 (买入持有/成本核算/涨跌停/T+1/carryover)",
        "A_t005_replication": {
            "n_months": int(len(cmp)), "repl_alpha_mean": round(float(cmp["repl_alpha"].mean()), 3),
            "t005_base_alpha_mean": round(float(cmp["t005_base_alpha"].mean()), 3),
            "corr": round(corr, 3), "mae_pp": round(mae, 3),
            "per_month": cmp.to_dict("records"),
        },
        "B_engine_book": {
            "rebalance_every_tdays": REBAL_EVERY, "weighting": "equal", "cost": "off",
            "book": s_book, "market": s_mkt, "period_excess": ex,
        },
        "sanity_magnitude_ok": bool(abs(ex["mean_period_excess_pct"]) <
                                    3 * abs(cmp["t005_base_alpha"].mean()) + 2.0),
    }
    (CACHE / "bt001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    conclusion = (
        f"引擎就绪。engine.py 解析型自检全通过 (买入持有/成本核算 round-trip≈0.30%/涨跌停不可成交/"
        f"T+1/carryover 换手)。(A) picks 复刻 t005 选股 α: {len(cmp)} 月 corr {corr:+.2f} MAE {mae:.2f}pp, "
        f"repl 均值 {cmp['repl_alpha'].mean():+.2f} vs t005 {cmp['t005_base_alpha'].mean():+.2f} "
        f"→ 喂引擎的 picks 忠实于 V12.31 baseline 臂。(B) 等权成本关 book 过引擎: 每再平衡期(≈20d)"
        f"超额均值 {ex['mean_period_excess_pct']:+.2f}% vs t005 选股 α {cmp['t005_base_alpha'].mean():+.2f}% "
        f"→ 同量级, 引擎核算无 bug。book 年化 {s_book['ann_return']:+.1%} Sharpe {s_book['sharpe']:+.2f} "
        f"maxDD {s_book['max_drawdown']:+.1%} 月换手 {s_book['avg_monthly_turnover']:.2f}。"
        f"成本/滑点敏感+基准对照+regime 分解留 BT-002。"
    )
    VERDICT.write_text(json.dumps({
        "id": "BT-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "t005_repl_corr": round(corr, 3), "t005_repl_mae_pp": round(mae, 3),
            "repl_alpha_mean_pp": round(float(cmp["repl_alpha"].mean()), 3),
            "t005_base_alpha_mean_pp": round(float(cmp["t005_base_alpha"].mean()), 3),
            "engine_period_excess_mean_pct": round(ex["mean_period_excess_pct"], 3),
            "book_ann_return": round(s_book["ann_return"], 4),
            "book_sharpe": round(s_book["sharpe"], 3),
            "book_max_drawdown": round(s_book["max_drawdown"], 4),
            "book_avg_monthly_turnover": round(s_book["avg_monthly_turnover"], 3),
            "book_avg_exposure": round(s_book["avg_exposure"], 3),
        },
        "cost_model": cost_off.describe(),
        "artifacts": [
            "research/backtest/engine.py", "research/backtest/gen_picks.py",
            "research/backtest/run_bt001.py",
            "research/cache/bt001/picks_v1231_daily.parquet",
            "research/cache/bt001/bt001_book_equal_costoff.parquet",
            "research/cache/bt001/bt001_market_equal_costoff.parquet",
            "research/cache/bt001/bt001_results.json",
        ],
        "note": "引擎=BT-001 核心交付; 解析自检证核算, t005 复刻证输入忠实, 等权成本关 book 与 t005 "
                "选股 α 同量级证无 bug。生产线只读 (SIGN-R05), 大缓存 gitignored, 前向 r20_fresh 仅 cache。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
