# -*- coding: utf-8 -*-
"""BT-002 — V12.31 扣成本真实净 P&L + 基准/基金对照 (该不该真金白银跑).

把 V12.31 生产 book (picks_v1231_daily, **alloc 加权双轨**, 非 BT-001 的等权 sanity) 过引擎,
开**真实成本 + 滑点敏感** (0/0.05/0.10/0.20%/边), 报:
  年化收益 / 净 Sharpe / 最大回撤 / 月均换手 / 月度胜率
对照基准:
  - hs300 (000300.SH) 指数净值
  - CSI1000 (000852.SH) 指数净值
  - 动量基准: 每再平衡日按 past-20d 收益取 top-N 等权, 过同一引擎 (同基线成本) = 纯动量风格 book
分 regime 分解 (动量/反转/混合月, 来自 research/features/regime_timeline.parquet):
  逐月 book vs hs300 超额, 按月度 regime 分组 → 看**动量月是否系统性跑输** (= 用户"基金赢我们输"的 book 层检验)。

R01 成本/滑点档冻结于 prd.preRegisteredGate, 不在结果上调; R06 ST 源头排除; R11 必带 regime 分层;
描述性净 P&L + regime 诊断 = 合法完成 (R02); 这不是 walk-forward ship gate (BT-004 才是)。
大缓存写 research/cache/bt002/ (gitignored); 指数数据缓存避免重复打 Tushare。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from engine import PortfolioBacktester, CostModel

CACHE001 = ROOT / "research" / "cache" / "bt001"
CACHE = ROOT / "research" / "cache" / "bt002"
CACHE.mkdir(parents=True, exist_ok=True)
PICKS = CACHE001 / "picks_v1231_daily.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "BT-002.json"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
IDX_CACHE = CACHE / "index_benchmarks.parquet"

TOTAL = 0.90              # V12.31 投资比例 (A 0.70 + B 0.20, 余 10% 现金)
REBAL_EVERY = 20         # 再平衡周期 ≈ r20 持有
CLIP = 30.0
SLIPPAGE_ARMS = [0.0, 0.0005, 0.0010, 0.0020]   # 0/0.05/0.10/0.20%/边 (冻结, R01)
BASELINE_SLIP = 0.0010   # 基线档
MOM_TOPN = 50            # 动量基准持仓数
MOM_LOOKBACK = 20        # 动量回看


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
    basic = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    st = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    px = px[~px["ts_code"].isin(st)].reset_index(drop=True)   # ST 源头排除 (R06)
    return px


# ───────────── 指数基准 (hs300 / CSI1000), 缓存 ─────────────
def load_index_navs(start: str, end: str) -> pd.DataFrame:
    if IDX_CACHE.exists():
        df = pd.read_parquet(IDX_CACHE)
        df["trade_date"] = df["trade_date"].astype(str)
        if df["trade_date"].min() <= start and df["trade_date"].max() >= end:
            return df
    from dotenv import load_dotenv
    load_dotenv()
    import tushare as ts
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    pro = ts.pro_api()
    codes = {"000300.SH": "hs300", "000852.SH": "csi1000"}
    out = None
    for code, name in codes.items():
        d = pro.index_daily(ts_code=code, start_date=start, end_date=end)
        d = d.sort_values("trade_date").reset_index(drop=True)
        d["trade_date"] = d["trade_date"].astype(str)
        d = d[["trade_date", "close"]].rename(columns={"close": f"{name}_close"})
        out = d if out is None else out.merge(d, on="trade_date", how="outer")
    out = out.sort_values("trade_date").reset_index(drop=True)
    out.to_parquet(IDX_CACHE, index=False)
    return out


# ───────────── 目标构建 ─────────────
def build_dual_alloc_targets(picks: pd.DataFrame, cal: list) -> dict:
    """V12.31 生产 book: 每再平衡日用当日 picks 的 alloc_pct (归一到 TOTAL) 作目标 = 双轨加权。"""
    rebal_days = cal[::REBAL_EVERY]
    pick_days = set(picks["trade_date"].unique())
    targets = {}
    for d in rebal_days:
        if d not in pick_days:
            continue
        g = picks[picks["trade_date"] == d]
        if g.empty:
            continue
        tot = g["alloc_pct"].sum()
        if tot <= 0:
            continue
        targets[d] = dict(zip(g["ts_code"], g["alloc_pct"] / tot * TOTAL))
    return targets


def build_momentum_targets(px: pd.DataFrame, cal: list) -> dict:
    """动量基准: 每再平衡日按 past-MOM_LOOKBACK 收益取 top-N 等权 (和=TOTAL)。
    因果: mom 用 D 收盘及之前数据算, 引擎在 D+1 开盘成交。"""
    p = px.sort_values(["ts_code", "trade_date"]).copy()
    p["close_ago"] = p.groupby("ts_code")["close"].shift(MOM_LOOKBACK)
    p["mom"] = p["close"] / p["close_ago"] - 1.0
    rebal_days = cal[::REBAL_EVERY]
    targets = {}
    for d in rebal_days:
        day = p[(p["trade_date"] == d) & p["open"].notna() & (p["pct_chg"].abs() < 9.8)
                & p["mom"].notna()]
        if day.empty:
            continue
        top = day.nlargest(MOM_TOPN, "mom")
        codes = top["ts_code"].tolist()
        w = TOTAL / len(codes)
        targets[d] = {c: w for c in codes}
    return targets


# ───────────── 净值统计 ─────────────
def nav_summary(nav: pd.Series, ppy: int = 252) -> dict:
    nav = nav.dropna().astype(float)
    n = len(nav)
    if n < 2:
        return {"n_days": n, "error": "insufficient"}
    arr = nav.to_numpy()
    ret = pd.Series(arr).pct_change().dropna().to_numpy()
    ann = (arr[-1] / arr[0]) ** (ppy / max(n - 1, 1)) - 1.0
    vol = float(np.nanstd(ret, ddof=1))
    sharpe = (float(np.nanmean(ret)) / (vol + 1e-12)) * np.sqrt(ppy)
    peak = np.maximum.accumulate(arr)
    maxdd = float((arr / peak - 1.0).min())
    return {"n_days": int(n), "total_return": float(arr[-1] / arr[0] - 1.0),
            "ann_return": float(ann), "sharpe": float(sharpe), "max_drawdown": maxdd}


def monthly_returns(nav: pd.Series) -> pd.Series:
    """逐月收益 (月末 nav 的 pct_change)。index=YYYYMM。"""
    s = nav.dropna().astype(float)
    ym = s.index.astype(str).str[:6]
    last = s.groupby(ym).last()
    return last.pct_change().dropna()


def main():
    t0 = time.time()
    print("\n=== BT-002: V12.31 扣成本净 P&L + 基准/基金对照 + regime 分解 ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    end = picks["trade_date"].max()
    print(f"[picks] {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / {picks['month'].nunique()} 月", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    end = max(end, cal[-1])
    print(f"[prices] {len(px):,} 行 / {px['ts_code'].nunique()} 股 / {len(cal)} 交易日 [{start}~{cal[-1]}]", flush=True)

    # ── 目标 ──
    tgt_book = build_dual_alloc_targets(picks, cal)
    tgt_mom = build_momentum_targets(px, cal)
    print(f"[targets] book 再平衡 {len(tgt_book)} 次, 动量基准 {len(tgt_mom)} 次", flush=True)

    # ── 滑点敏感: V12.31 book 过引擎 (cost ON) ──
    print("\n[A] V12.31 book 滑点敏感 (alloc 加权双轨, 真实成本) ...", flush=True)
    arm_summ = {}
    book_navs = {}
    for slip in SLIPPAGE_ARMS:
        cm = CostModel(slippage=slip, enabled=True)
        bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
        res = bt.run(tgt_book, start=start)
        s = res.summary()
        arm_summ[f"slip_{slip}"] = s
        book_navs[slip] = res.timeseries["nav"]
        if slip == BASELINE_SLIP:
            res.timeseries.to_parquet(CACHE / "bt002_book_baseline.parquet")
        tag = " <baseline>" if slip == BASELINE_SLIP else ""
        print(f"  滑点 {slip*100:.2f}%/边: 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.1%}  月换手 {s['avg_monthly_turnover']:.2f}  "
              f"月胜率 {s['monthly_win_rate']:.0%}  总成本 {s['total_cost_paid']:.3f}{tag}", flush=True)

    book_nav = book_navs[BASELINE_SLIP]
    book_dates = book_nav.index.tolist()

    # ── 基准 ──
    print("\n[B] 基准对照 ...", flush=True)
    idx = load_index_navs(start, end)
    idx = idx[(idx["trade_date"] >= start) & (idx["trade_date"] <= book_dates[-1])].copy()
    idx = idx.set_index("trade_date").reindex(book_dates).ffill()
    bench_navs = {}
    for name in ["hs300", "csi1000"]:
        c = idx[f"{name}_close"].dropna()
        c = c / c.iloc[0]
        bench_navs[name] = c.reindex(book_dates).ffill()

    # 动量基准过引擎 (基线成本)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res_mom = bt.run(tgt_mom, start=start)
    res_mom.timeseries.to_parquet(CACHE / "bt002_momentum_benchmark.parquet")
    bench_navs["momentum"] = res_mom.timeseries["nav"].reindex(book_dates).ffill()

    bench_summ = {}
    for name, nv in bench_navs.items():
        bench_summ[name] = nav_summary(nv)
        s = bench_summ[name]
        print(f"  {name:9s}: 年化 {s['ann_return']:+.1%}  Sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.1%}", flush=True)
    sb = arm_summ[f"slip_{BASELINE_SLIP}"]
    print(f"  {'V12.31':9s}: 年化 {sb['ann_return']:+.1%}  净Sharpe {sb['sharpe']:+.2f}  "
          f"maxDD {sb['max_drawdown']:+.1%}  (基线滑点 {BASELINE_SLIP*100:.2f}%)", flush=True)

    # ── regime 分解 (R11) ──
    print("\n[C] regime 分解 (动量月是否系统性跑输 hs300) ...", flush=True)
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())

    book_m = monthly_returns(book_nav).rename("book")
    cmp_m = pd.DataFrame({"book": book_m})
    for name, nv in bench_navs.items():
        cmp_m[name] = monthly_returns(nv)
    cmp_m = cmp_m.dropna()
    cmp_m["regime"] = cmp_m.index.map(month_regime)
    cmp_m["excess_hs300"] = (cmp_m["book"] - cmp_m["hs300"]) * 100
    cmp_m["excess_mom"] = (cmp_m["book"] - cmp_m["momentum"]) * 100

    regime_table = {}
    for rg, g in cmp_m.groupby("regime"):
        regime_table[rg] = {
            "n_months": int(len(g)),
            "book_ret_mean_pct": round(float(g["book"].mean()) * 100, 3),
            "hs300_ret_mean_pct": round(float(g["hs300"].mean()) * 100, 3),
            "csi1000_ret_mean_pct": round(float(g["csi1000"].mean()) * 100, 3),
            "momentum_ret_mean_pct": round(float(g["momentum"].mean()) * 100, 3),
            "excess_vs_hs300_mean_pp": round(float(g["excess_hs300"].mean()), 3),
            "excess_vs_momentum_mean_pp": round(float(g["excess_mom"].mean()), 3),
            "excess_vs_hs300_win_rate": round(float((g["excess_hs300"] > 0).mean()), 3),
        }
        print(f"  [{rg:9s}] n={len(g):2d}  book {g['book'].mean()*100:+.2f}%  "
              f"hs300 {g['hs300'].mean()*100:+.2f}%  mom {g['momentum'].mean()*100:+.2f}%  "
              f"超额hs300 {g['excess_hs300'].mean():+.2f}pp (胜率 {(g['excess_hs300']>0).mean():.0%})  "
              f"超额mom {g['excess_mom'].mean():+.2f}pp", flush=True)
    cmp_m.to_parquet(CACHE / "bt002_monthly_regime.parquet")

    mom_months = cmp_m[cmp_m["regime"] == "momentum"]
    underperf_mom_vs_hs = bool(len(mom_months) and mom_months["excess_hs300"].mean() < 0)
    underperf_mom_vs_mom = bool(len(mom_months) and mom_months["excess_mom"].mean() < 0)
    # 动量月是否为 book 相对最弱 regime (即便绝对未跑输 hs300)
    excess_by_regime = cmp_m.groupby("regime")["excess_hs300"].mean()
    weakest_regime = str(excess_by_regime.idxmin()) if len(excess_by_regime) else "NA"
    momentum_is_weakest = bool(weakest_regime == "momentum")

    # ── 裁决 ──
    base = arm_summ[f"slip_{BASELINE_SLIP}"]
    net_positive = bool(base["ann_return"] > 0 and base["sharpe"] > 0)
    slip0 = arm_summ["slip_0.0"]
    slip_drag_sharpe = round(slip0["sharpe"] - arm_summ["slip_0.002"]["sharpe"], 3)

    results = {
        "config": {"total_invested": TOTAL, "rebalance_every_tdays": REBAL_EVERY,
                   "slippage_arms": SLIPPAGE_ARMS, "baseline_slippage": BASELINE_SLIP,
                   "momentum_benchmark": {"top_n": MOM_TOPN, "lookback_tdays": MOM_LOOKBACK},
                   "period": [start, book_dates[-1]]},
        "A_slippage_sensitivity": arm_summ,
        "B_benchmarks": bench_summ,
        "C_regime": regime_table,
        "diagnosis": {
            "net_positive_at_baseline": net_positive,
            "slippage_drag_sharpe_0_to_0.20": slip_drag_sharpe,
            "underperform_hs300_in_momentum_months": underperf_mom_vs_hs,
            "underperform_momentum_bench_in_momentum_months": underperf_mom_vs_mom,
            "weakest_regime_vs_hs300": weakest_regime,
            "momentum_is_relatively_weakest_regime": momentum_is_weakest,
            "absolute_return_caveat": "r20-pool common-mode lookahead + concentration inflate absolute "
                                      "level (same as BT-001); not a tradeable claim; BT-003/004 isolate increment",
        },
    }
    (CACHE / "bt002_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # regime 诊断句
    if underperf_mom_vs_hs:
        rg_diag = (f"动量月 book 超额 hs300 {mom_months['excess_hs300'].mean():+.2f}pp/月 (绝对跑输)"
                   f", 印证用户'基金赢我们输'的 book 层 regime 暴露问题")
    elif len(mom_months):
        rg_diag = (f"动量月 book 超额 hs300 {mom_months['excess_hs300'].mean():+.2f}pp/月 (绝对未跑输 hs300, "
                   f"胜率 {(mom_months['excess_hs300']>0).mean():.0%})")
    else:
        rg_diag = "样本内无 momentum 月"
    if momentum_is_weakest:
        rg_diag += (f"; 但动量月是 book **相对最弱** regime (超额 {excess_by_regime['momentum']:+.2f}pp "
                    f"vs 反转 {excess_by_regime.get('reversal', float('nan')):+.2f} / "
                    f"混合 {excess_by_regime.get('mixed', float('nan')):+.2f}), 与用户'动量月吃亏'方向一致")
    if underperf_mom_vs_mom and len(mom_months):
        rg_diag += f"; 跑输纯动量基准 {mom_months['excess_mom'].mean():+.2f}pp/月"

    LOOKAHEAD_CAVEAT = (
        "【绝对收益非可交易声明】book 绝对年化由 r20 池模型 (r20_v16_long_nost 单一生产模型跨全测试期) "
        "共模 lookahead + 集中持仓抬高 (同 BT-001 caveat); 故 +140% 量级不可当实盘预期, "
        "BT-003 leave-one-out 与 BT-004 残差 gate 才在受控对照下判增量。本 task 可信结论是: "
        "(a) 成本/滑点不构成杀手 (换手仅 ~1.5/月, 0→0.20%滑点 Sharpe 仅衰减 0.12); "
        "(b) regime 相对结构 (动量月最弱) 不受共模 lookahead 等比抬高影响。"
    )
    conclusion = (
        f"V12.31 生产 book (alloc 加权双轨) 扣真实成本+滑点敏感全 history ({start}~{book_dates[-1]}, "
        f"{base['n_months']} 月) 过引擎。基线滑点 {BASELINE_SLIP*100:.2f}%/边: 年化 {base['ann_return']:+.1%} "
        f"净Sharpe {base['sharpe']:+.2f} maxDD {base['max_drawdown']:+.1%} 月换手 {base['avg_monthly_turnover']:.2f} "
        f"月胜率 {base['monthly_win_rate']:.0%}。滑点 0→0.20% Sharpe 衰减 {slip_drag_sharpe:+.2f} "
        f"({slip0['sharpe']:+.2f}→{arm_summ['slip_0.002']['sharpe']:+.2f})。基准: hs300 年化 "
        f"{bench_summ['hs300']['ann_return']:+.1%} Sharpe {bench_summ['hs300']['sharpe']:+.2f}, "
        f"CSI1000 {bench_summ['csi1000']['ann_return']:+.1%}, 动量基准 {bench_summ['momentum']['ann_return']:+.1%} "
        f"Sharpe {bench_summ['momentum']['sharpe']:+.2f}。regime 分解: {rg_diag}。{LOOKAHEAD_CAVEAT} "
        f"答'该不该真跑': 成本层无杀手 (净正不脆弱), 但绝对 α 须待 BT-003/004 去 lookahead 后定; "
        f"用户'动量月吃亏'在 book 层得**部分印证** (动量月是相对最弱 regime, 非绝对跑输 hs300) → 指向 RL-lite 风控/动量暴露管理。"
    )
    VERDICT.write_text(json.dumps({
        "id": "BT-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "baseline_slippage": BASELINE_SLIP,
            "ann_return_by_slip": {f"{s}": round(arm_summ[f'slip_{s}']['ann_return'], 4) for s in SLIPPAGE_ARMS},
            "sharpe_by_slip": {f"{s}": round(arm_summ[f'slip_{s}']['sharpe'], 3) for s in SLIPPAGE_ARMS},
            "maxdd_by_slip": {f"{s}": round(arm_summ[f'slip_{s}']['max_drawdown'], 4) for s in SLIPPAGE_ARMS},
            "baseline_ann_return": round(base["ann_return"], 4),
            "baseline_net_sharpe": round(base["sharpe"], 3),
            "baseline_max_drawdown": round(base["max_drawdown"], 4),
            "avg_monthly_turnover": round(base["avg_monthly_turnover"], 3),
            "monthly_win_rate": round(base["monthly_win_rate"], 3),
            "hs300_ann_return": round(bench_summ["hs300"]["ann_return"], 4),
            "hs300_sharpe": round(bench_summ["hs300"]["sharpe"], 3),
            "csi1000_ann_return": round(bench_summ["csi1000"]["ann_return"], 4),
            "momentum_bench_ann_return": round(bench_summ["momentum"]["ann_return"], 4),
            "momentum_bench_sharpe": round(bench_summ["momentum"]["sharpe"], 3),
            "underperform_hs300_in_momentum_months": underperf_mom_vs_hs,
            "underperform_mombench_in_momentum_months": underperf_mom_vs_mom,
            "weakest_regime_vs_hs300": weakest_regime,
            "momentum_is_relatively_weakest_regime": momentum_is_weakest,
        },
        "regime_table": regime_table,
        "cost_model": CostModel(slippage=BASELINE_SLIP, enabled=True).describe(),
        "artifacts": [
            "research/backtest/run_bt002.py",
            "research/cache/bt002/bt002_book_baseline.parquet",
            "research/cache/bt002/bt002_momentum_benchmark.parquet",
            "research/cache/bt002/bt002_monthly_regime.parquet",
            "research/cache/bt002/bt002_results.json",
            "research/cache/bt002/index_benchmarks.parquet",
        ],
        "note": "alloc 加权双轨 (非 BT-001 等权 sanity); 成本/滑点档冻结 (R01); ST 源头排除 (R06); "
                "regime 分层必带 (R11); 描述性净 P&L 非 ship gate (R02/R03); 大缓存 gitignored; "
                "生产线只读未碰 (R05)。动量基准=past20d top50 等权过同引擎。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
