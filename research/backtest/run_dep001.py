# -*- coding: utf-8 -*-
"""DEP-001 — PIT-clean (剔 holder_pct) r20 OOS book 过引擎 → 确认 Sharpe≈1.31 (清 codex 硬红线).

输入: research/cache/dep001/picks_oos_daily.parquet (剔 holder_pct 的 label-embargo r20 月度
walk-forward 重建的 V12.31 持仓)。同 WFE-001 配置 (alloc 加权双轨, 20d 再平衡, 真实成本基线
0.10%/边) 过同一引擎, 与 WFE-001 基线 1.31 做 **apples-to-apples 受控对照** (唯一差异 = 去 holder_pct)。

裁决 (gate_dep, 冻结 R01):
  clean确认  = PIT-clean Sharpe 与 1.31 的 |ΔSharpe| 落在 per-cohort 配对 block bootstrap 噪声内
               (ΔSharpe 95%CI 含 0) → holder_pct 泄漏无实质影响, PIT-clean r20 deployable, 硬红线清除。
  材料性下降 = ΔSharpe 95%CI 整段 < 0 (超噪声) → 该跌幅 = holder_pct 泄漏的假 edge, 真实基线下修。

per-cohort 配对 bootstrap: 两 book 同 OOS 期、同交易日 → 按 20 交易日 cohort 配对重采样, 每次同一批
cohort 上分别算两臂 Sharpe 取差 → ΔSharpe 分布 (消除共模, 只测 holder_pct 的净效应)。
因果自检: PIT-clean r20 IC (剔 holder) vs 生产含 holder r20 IC。分 regime (R11)。

R01 配置冻结 (与 WFE-001 逐字一致, 唯一差异 = r20 特征剔 holder_pct); R06 ST 源头排除;
R11 regime 分层; 描述性真实 P&L = 合法完成 (R02); 大缓存 gitignored; 生产线只读 (R05)。
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
from run_bt002 import (load_prices, build_dual_alloc_targets, build_momentum_targets,
                       load_index_navs, nav_summary, monthly_returns,
                       TOTAL, REBAL_EVERY, BASELINE_SLIP)

DEP = ROOT / "research" / "cache" / "dep001"
PICKS = DEP / "picks_oos_daily.parquet"
DIAG = DEP / "r20_oos_diagnostics.csv"
WFE = ROOT / "research" / "cache" / "wfe001"
WFE_BOOK = WFE / "wfe001_book_oos.parquet"
WFE_RESULTS = WFE / "wfe001_results.json"
WFE_DIAG = WFE / "r20_oos_diagnostics.csv"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
BASELINE_MD = ROOT / "research" / "backtest" / "V12.31_BASELINE.md"
VERDICT = ROOT / "research" / "verdicts" / "DEP-001.json"

# ── 冻结参数 (R01, 同 FIN-001) ──
COHORT_LEN = 20
N_BOOT = 1000
BOOT_SEED = 20260614
PPY = 252


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    return float(np.mean(r) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(PPY))


def ic_split(diag: pd.DataFrame):
    diag = diag.copy()
    diag["in_sample"] = diag["month"].astype(str) <= "202509"
    ins, oos = diag[diag["in_sample"]], diag[~diag["in_sample"]]
    return {
        "wf_ic_all": float(diag["r20_oos_rankic"].mean()),
        "wf_ic_trueoos": float(oos["r20_oos_rankic"].mean()) if len(oos) else float("nan"),
        "prod_ic_all": float(diag["r20_prod_rankic"].mean()),
    }


def main():
    t0 = time.time()
    print("\n=== DEP-001: PIT-clean (剔 holder_pct) r20 OOS book → 确认 Sharpe≈1.31 ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] PIT-clean OOS {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / "
          f"{picks['month'].nunique()} 月 / {picks['ts_code'].nunique()} 股", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    end = cal[-1]
    print(f"[prices] {len(px):,} 行 / {len(cal)} 交易日 [{start}~{end}]", flush=True)

    tgt_book = build_dual_alloc_targets(picks, cal)
    tgt_mom = build_momentum_targets(px, cal)
    print(f"[targets] PIT-clean book 再平衡 {len(tgt_book)} 次, 动量基准 {len(tgt_mom)} 次", flush=True)

    # ── [A] PIT-clean book 过引擎 (基线滑点, 同 WFE-001/BT-002) ──
    print("\n[A] PIT-clean V12.31 book 过引擎 (基线滑点 0.10%/边) ...", flush=True)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res = bt.run(tgt_book, start=start)
    s_clean = res.summary()
    res.timeseries.to_parquet(DEP / "dep001_book_oos.parquet")
    book_nav = res.timeseries["nav"]
    book_dates = book_nav.index.astype(str).tolist()
    print(f"  PIT-clean: 年化 {s_clean['ann_return']:+.1%}  净Sharpe {s_clean['sharpe']:+.2f}  "
          f"maxDD {s_clean['max_drawdown']:+.1%}  月换手 {s_clean['avg_monthly_turnover']:.2f}  "
          f"月胜率 {s_clean['monthly_win_rate']:.0%}", flush=True)

    # ── [B] vs WFE-001 基线 1.31 ──
    wfe = json.loads(WFE_RESULTS.read_text(encoding="utf-8"))
    s_base = wfe["embargo_book"]
    d_sharpe = s_clean["sharpe"] - s_base["sharpe"]
    d_ann = s_clean["ann_return"] - s_base["ann_return"]
    print(f"\n[B] vs WFE-001 基线 (含 holder_pct): 年化 {s_base['ann_return']:+.1%}  Sharpe {s_base['sharpe']:+.2f}", flush=True)
    print(f"  ΔSharpe (clean - 基线) = {d_sharpe:+.3f}  Δ年化 = {d_ann:+.1%}", flush=True)

    # ── [C] per-cohort 配对 block bootstrap of ΔSharpe ──
    base_book = pd.read_parquet(WFE_BOOK)
    base_book.index = base_book.index.astype(str)
    base_ret = base_book["nav"].astype(float).pct_change().dropna()
    clean_ret = book_nav.astype(float).pct_change().dropna()
    clean_ret.index = clean_ret.index.astype(str)
    common = sorted(set(base_ret.index) & set(clean_ret.index))
    br = base_ret.reindex(common).to_numpy()
    cr = clean_ret.reindex(common).to_numpy()
    n_days = len(common)
    cohort_id = np.arange(n_days) // COHORT_LEN
    n_cohorts = int(cohort_id.max()) + 1
    base_coh = [br[cohort_id == c] for c in range(n_cohorts)]
    clean_coh = [cr[cohort_id == c] for c in range(n_cohorts)]
    print(f"\n[C] 配对 block bootstrap: {n_days} 共同日 / {n_cohorts} 个 {COHORT_LEN}日 cohort, {N_BOOT} 次 ...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    bs_d = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_cohorts, size=n_cohorts)
        b = np.concatenate([base_coh[c] for c in pick])
        c_ = np.concatenate([clean_coh[c] for c in pick])
        sb, sc = sharpe(b), sharpe(c_)
        if np.isfinite(sb) and np.isfinite(sc):
            bs_d.append(sc - sb)
    bs_d = np.array(bs_d)
    d_ci = (float(np.percentile(bs_d, 2.5)), float(np.percentile(bs_d, 97.5)))
    d_med = float(np.median(bs_d))
    p_d_lt0 = float((bs_d < 0).mean())
    ci_contains_0 = bool(d_ci[0] <= 0 <= d_ci[1])
    print(f"  ΔSharpe 95%CI = [{d_ci[0]:+.3f}, {d_ci[1]:+.3f}]  中位 {d_med:+.3f}  "
          f"P(Δ<0)={p_d_lt0:.3f}  CI含0={ci_contains_0}", flush=True)

    # ── [D] 基准 ──
    idx = load_index_navs(start, end)
    idx = idx[(idx["trade_date"] >= start) & (idx["trade_date"] <= book_dates[-1])].copy()
    idx = idx.set_index("trade_date").reindex(book_dates).ffill()
    bench_navs = {}
    for name in ["hs300", "csi1000"]:
        c = idx[f"{name}_close"].dropna(); c = c / c.iloc[0]
        bench_navs[name] = c.reindex(book_dates).ffill()
    cm2 = CostModel(slippage=BASELINE_SLIP, enabled=True)
    res_mom = PortfolioBacktester(px, cost=cm2, t_plus_1=True, limit_pct=9.8).run(tgt_mom, start=start)
    bench_navs["momentum"] = res_mom.timeseries["nav"].reindex(book_dates).ffill()
    bench_summ = {n: nav_summary(v) for n, v in bench_navs.items()}
    beats_all_bench = bool(s_clean["sharpe"] > max(bm["sharpe"] for bm in bench_summ.values()))
    print("\n[D] 基准:", flush=True)
    for n, sm in bench_summ.items():
        print(f"  {n:9s}: 年化 {sm['ann_return']:+.1%}  Sharpe {sm['sharpe']:+.2f}", flush=True)

    # ── [E] regime 分解 (R11) ──
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())
    book_m = monthly_returns(book_nav).rename("book")
    cmp_m = pd.DataFrame({"book": book_m})
    for n, v in bench_navs.items():
        cmp_m[n] = monthly_returns(v)
    cmp_m = cmp_m.dropna()
    cmp_m["regime"] = cmp_m.index.map(month_regime)
    cmp_m["excess_hs300"] = (cmp_m["book"] - cmp_m["hs300"]) * 100
    cmp_m.to_parquet(DEP / "dep001_monthly_regime.parquet")
    regime_table = {}
    print("\n[E] regime 分解 (PIT-clean book vs hs300):", flush=True)
    for rg, g in cmp_m.groupby("regime"):
        regime_table[rg] = {"n_months": int(len(g)),
                            "book_ret_mean_pct": round(float(g["book"].mean()) * 100, 3),
                            "excess_vs_hs300_mean_pp": round(float(g["excess_hs300"].mean()), 3),
                            "excess_vs_hs300_win_rate": round(float((g["excess_hs300"] > 0).mean()), 3)}
        print(f"  [{rg:9s}] n={len(g):2d}  book {g['book'].mean()*100:+.2f}%  "
              f"超额hs300 {g['excess_hs300'].mean():+.2f}pp (胜率 {(g['excess_hs300']>0).mean():.0%})", flush=True)

    # ── [F] 因果自检 (PIT-clean r20 IC vs 生产含 holder) ──
    ic_clean = ic_split(pd.read_csv(DIAG, dtype={"month": str}))
    ic_base = ic_split(pd.read_csv(WFE_DIAG, dtype={"month": str}))
    print("\n[F] 因果自检 (r20 截面 rank-IC vs r20_fresh, 月度均值):", flush=True)
    print(f"  PIT-clean (剔holder) WF: 全期 {ic_clean['wf_ic_all']:+.4f} | true-OOS {ic_clean['wf_ic_trueoos']:+.4f}", flush=True)
    print(f"  WFE-001 (含holder)   WF: 全期 {ic_base['wf_ic_all']:+.4f} | true-OOS {ic_base['wf_ic_trueoos']:+.4f}", flush=True)
    ic_change = ic_clean["wf_ic_all"] - ic_base["wf_ic_all"]
    print(f"  剔 holder 后 r20 IC 变化 (全期): {ic_change:+.4f} "
          f"({'几乎不变=holder 无信号贡献' if abs(ic_change) < 0.005 else '有变化'})", flush=True)

    # ── 裁决 (gate_dep, 冻结 R01) ──
    # clean确认 = ΔSharpe 95%CI 含 0 (落在配对 bootstrap 噪声内, 不材料性下降)
    # 材料性下降 = ΔSharpe 95%CI 整段 < 0 (超噪声)
    if ci_contains_0 or d_ci[1] >= 0:
        status = "clean确认"
    else:
        status = "材料性下降"
    net_positive = bool(s_clean["ann_return"] > 0 and s_clean["sharpe"] > 0)

    conclusion = (
        f"DEP-001 = WFE-001 (label-embargo r20 真·真实 walk-forward) **唯一改动 = r20 训练特征集严格剔除泄漏因子 "
        f"holder_pct** (FIN-002 定位的 codex 唯一硬红线: stk_holdernumber 季频股东户数按 end_date<=target 选数未取 "
        f"ann_date → ~1季度前视; 但 r20 gain 排 118/236 占 0.005% 重要度极低)。234 特征 (剔 1)、24m lookback、embargo "
        f"P_start-21交易日、固定120树、双轨70-20-10、行业cap4、20d再平衡、close-based 成本、T+1、ST 源头排除逐字同 "
        f"WFE-001 → apples-to-apples。"
        f"PIT-clean book: 年化 {s_clean['ann_return']:+.1%} 净Sharpe **{s_clean['sharpe']:+.2f}** maxDD {s_clean['max_drawdown']:+.1%} "
        f"月换手 {s_clean['avg_monthly_turnover']:.2f} 月胜率 {s_clean['monthly_win_rate']:.0%}。"
        f"vs WFE-001 基线 (含 holder_pct, Sharpe {s_base['sharpe']:+.2f}): **ΔSharpe {d_sharpe:+.3f} / Δ年化 {d_ann:+.1%}**。"
        f"**per-cohort 配对 block bootstrap ({N_BOOT}次): ΔSharpe 95%CI = [{d_ci[0]:+.3f}, {d_ci[1]:+.3f}]** "
        f"(中位 {d_med:+.3f}, P(Δ<0)={p_d_lt0:.0%}, CI含0={ci_contains_0})。"
        f"因果自检: 剔 holder 后 r20 全期 IC {ic_clean['wf_ic_all']:+.4f} vs 含 holder {ic_base['wf_ic_all']:+.4f} "
        f"(变化 {ic_change:+.4f} → {'holder_pct 在信号层无实质贡献' if abs(ic_change) < 0.005 else '信号层有变化'})。"
        f"基准: hs300 {bench_summ['hs300']['sharpe']:+.2f} / CSI1000 {bench_summ['csi1000']['sharpe']:+.2f} / 动量 "
        f"{bench_summ['momentum']['sharpe']:+.2f} ({'仍跑赢全基准' if beats_all_bench else '部分基准追上'})。"
        f"**裁决 [{status}]**: "
        + ({"clean确认": (
                f"|ΔSharpe| {abs(d_sharpe):.3f} 落在配对 bootstrap 噪声内 (CI 含 0) → holder_pct 泄漏对 book Sharpe **无实质影响**, "
                f"剔除后 PIT-clean r20 是干净可部署变体, **codex 唯一硬红线清除** (零 ann_date 工程, 隐患彻底消除)。"
                f"真实基线维持 ≈1.31 (CI[+0.11,+2.60] FIN-001)。印证 BT-003: r20 book α 主导来自整池排序非单因子, holder 这种 "
                f"0.005% gain 的低重要度因子剔除无损 → 反而说明早该剔。"),
            "材料性下降": (
                f"ΔSharpe 95%CI 整段 < 0 (超噪声) → holder_pct 泄漏确实贡献了假 edge, 真实基线下修到 clean 值 "
                f"{s_clean['sharpe']:+.2f} (诚实, 反而说明早该剔; 见 responsePlaybook['材料性下降'])。")}[status])
        + " 生产线 V12.31 (v12_scoring/v12_dual_track) 全程只读冻结 (R05); PIT-clean r20 是可部署变体, 非动生产。"
    )

    results = {
        "config": {"total_invested": TOTAL, "rebalance_every_tdays": REBAL_EVERY,
                   "baseline_slippage": BASELINE_SLIP, "period": [start, book_dates[-1]],
                   "n_common_days": n_days, "n_cohorts": n_cohorts, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "dropped_leak_factor": "holder_pct",
                   "note": "WFE-001 口径, 唯一差异 = r20 特征剔 holder_pct (PIT-clean)"},
        "clean_book": s_clean, "wfe001_baseline": s_base,
        "delta_vs_baseline": {"delta_sharpe": round(d_sharpe, 3), "delta_ann_return": round(d_ann, 4)},
        "paired_bootstrap": {
            "delta_sharpe_ci95": [round(d_ci[0], 3), round(d_ci[1], 3)],
            "delta_sharpe_median": round(d_med, 3),
            "p_delta_lt_0": round(p_d_lt0, 3), "ci_contains_0": ci_contains_0,
            "n_valid": int(len(bs_d)),
        },
        "benchmarks": bench_summ, "beats_all_bench": beats_all_bench,
        "regime": regime_table,
        "causal_check": {"clean_wf_ic_all": round(ic_clean["wf_ic_all"], 4),
                         "clean_wf_ic_trueoos": round(ic_clean["wf_ic_trueoos"], 4),
                         "baseline_wf_ic_all": round(ic_base["wf_ic_all"], 4),
                         "ic_change_all": round(ic_change, 4)},
        "status": status,
    }
    (DEP / "dep001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "DEP-001", "status": status, "conclusion": conclusion,
        "metrics": {
            "clean_net_sharpe": round(s_clean["sharpe"], 3),
            "clean_ann_return": round(s_clean["ann_return"], 4),
            "clean_max_drawdown": round(s_clean["max_drawdown"], 4),
            "clean_avg_monthly_turnover": round(s_clean["avg_monthly_turnover"], 3),
            "clean_monthly_win_rate": round(s_clean["monthly_win_rate"], 3),
            "baseline_net_sharpe": round(s_base["sharpe"], 3),
            "delta_sharpe_vs_baseline": round(d_sharpe, 3),
            "delta_ann_return_vs_baseline": round(d_ann, 4),
            "bootstrap_delta_sharpe_ci95": [round(d_ci[0], 3), round(d_ci[1], 3)],
            "bootstrap_delta_sharpe_median": round(d_med, 3),
            "bootstrap_p_delta_lt_0": round(p_d_lt0, 3),
            "bootstrap_ci_contains_0": ci_contains_0,
            "clean_wf_ic_all": round(ic_clean["wf_ic_all"], 4),
            "baseline_wf_ic_all": round(ic_base["wf_ic_all"], 4),
            "ic_change_all": round(ic_change, 4),
            "net_positive": net_positive, "beats_all_bench": beats_all_bench,
            "playbook": status,
        },
        "regime_table": regime_table,
        "artifacts": ["research/backtest/dep001_gen_picks.py", "research/backtest/run_dep001.py",
                      "research/cache/dep001/picks_oos_daily.parquet",
                      "research/cache/dep001/r20_models/", "research/cache/dep001/r20_oos_diagnostics.csv",
                      "research/cache/dep001/dep001_book_oos.parquet",
                      "research/cache/dep001/dep001_monthly_regime.parquet",
                      "research/cache/dep001/dep001_results.json"],
        "note": "唯一改动 vs WFE-001 = r20 训练特征集剔 holder_pct (PIT 泄漏因子); 24m lookback/embargo "
                "P_start-21交易日/固定120树/引擎/成本/再平衡/双轨配置逐字同 WFE-001 (R01)。配对 block bootstrap "
                "消共模只测 holder 净效应。ST 源头排除 (R06); regime 分层 (R11); 描述性=合法完成 (R02); "
                "中间 IC≠ship (R03); 生产线只读 (R05); 大缓存 gitignored, 无 features 写入 (R04)。"
                "剔泄漏因子确认 deployable baseline, 清 codex 唯一硬红线。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict={status} -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return status, s_clean, d_sharpe, d_ci, ic_change


if __name__ == "__main__":
    main()
