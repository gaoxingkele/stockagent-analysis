# -*- coding: utf-8 -*-
"""WFE-001 — embargo OOS picks 过 book 引擎 → 真·真实可交易 P&L + vs WF-001(无embargo 1.84) 额外缩水.

输入: research/cache/wfe001/picks_oos_daily.parquet (label-embargo r20 月度 walk-forward 重建的 V12.31 持仓)。
同 BT-002/WF-001 配置 (alloc 加权双轨, 20d 再平衡, 真实成本基线 0.10%/边) 过同一引擎, 报:
  真·真实 年化/净Sharpe/maxDD/月换手/月胜率  +  vs WF-001 (无 embargo, Sharpe 1.84) 的**额外缩水**
  +  vs BT-002 注水版 (单一生产 r20, Sharpe 2.78) 的累计缩水。
因果自检: embargo 后 WF 月度重训 r20 IC (true-OOS 段) 应 <= WF-001 无 embargo 版 (近截止 label 泄漏被切掉)。
分 regime (R11)。verdict 按 prd.responsePlaybook.WFE-001 (embargo后仍稳 / embargo后大缩 / embargo后塌) 命中。

R01 配置冻结 (与 WF-001/BT-002 逐字一致, 唯一差异 = picks 由 embargo OOS r20 生成); R06 ST 源头排除;
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

WFE = ROOT / "research" / "cache" / "wfe001"
PICKS = WFE / "picks_oos_daily.parquet"
DIAG = WFE / "r20_oos_diagnostics.csv"
WF001_RESULTS = ROOT / "research" / "cache" / "wf001" / "wf001_results.json"
WF001_DIAG = ROOT / "research" / "cache" / "wf001" / "r20_oos_diagnostics.csv"
BT002 = ROOT / "research" / "cache" / "bt002" / "bt002_results.json"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "WFE-001.json"


def ic_split(diag: pd.DataFrame):
    """月度 r20 IC 拆 in-sample(≤202509) / true-OOS / 全期。"""
    diag = diag.copy()
    diag["prod_in_train_window"] = diag["month"].astype(str) <= "202509"
    ins = diag[diag["prod_in_train_window"]]
    oos = diag[~diag["prod_in_train_window"]]
    return {
        "wf_ic_all": float(diag["r20_oos_rankic"].mean()),
        "wf_ic_insample": float(ins["r20_oos_rankic"].mean()) if len(ins) else float("nan"),
        "wf_ic_trueoos": float(oos["r20_oos_rankic"].mean()) if len(oos) else float("nan"),
        "prod_ic_insample": float(ins["r20_prod_rankic"].mean()) if len(ins) else float("nan"),
        "prod_ic_trueoos": float(oos["r20_prod_rankic"].mean()) if len(oos) else float("nan"),
    }


def main():
    t0 = time.time()
    print("\n=== WFE-001: label-embargo r20 OOS book → 真·真实可交易 P&L ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] embargo OOS {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / "
          f"{picks['month'].nunique()} 月 / {picks['ts_code'].nunique()} 股", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    end = cal[-1]
    print(f"[prices] {len(px):,} 行 / {len(cal)} 交易日 [{start}~{end}]", flush=True)

    tgt_book = build_dual_alloc_targets(picks, cal)
    tgt_mom = build_momentum_targets(px, cal)
    print(f"[targets] embargo book 再平衡 {len(tgt_book)} 次, 动量基准 {len(tgt_mom)} 次", flush=True)

    # ── embargo book 过引擎 (基线滑点, 同 WF-001/BT-002) ──
    print("\n[A] embargo V12.31 book 过引擎 (基线滑点 0.10%/边) ...", flush=True)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res = bt.run(tgt_book, start=start)
    s_emb = res.summary()
    res.timeseries.to_parquet(WFE / "wfe001_book_oos.parquet")
    book_nav = res.timeseries["nav"]
    book_dates = book_nav.index.tolist()
    print(f"  embargo: 年化 {s_emb['ann_return']:+.1%}  净Sharpe {s_emb['sharpe']:+.2f}  "
          f"maxDD {s_emb['max_drawdown']:+.1%}  月换手 {s_emb['avg_monthly_turnover']:.2f}  "
          f"月胜率 {s_emb['monthly_win_rate']:.0%}", flush=True)

    # ── vs WF-001 (无 embargo, 1.84) + BT-002 注水版 (2.78) ──
    wf001 = json.loads(WF001_RESULTS.read_text(encoding="utf-8"))
    s_wf001 = wf001["oos_book"]
    bt002 = json.loads(BT002.read_text(encoding="utf-8"))
    s_la = bt002["A_slippage_sensitivity"][f"slip_{BASELINE_SLIP}"]
    extra_sharpe = s_emb["sharpe"] - s_wf001["sharpe"]
    extra_ann = s_emb["ann_return"] - s_wf001["ann_return"]
    cum_sharpe = s_emb["sharpe"] - s_la["sharpe"]
    cum_ann = s_emb["ann_return"] - s_la["ann_return"]
    print(f"\n[B] vs WF-001 (无 embargo): 年化 {s_wf001['ann_return']:+.1%} Sharpe {s_wf001['sharpe']:+.2f}", flush=True)
    print(f"  额外缩水 (embargo - WF001): ΔSharpe {extra_sharpe:+.2f}  Δ年化 {extra_ann:+.1%}", flush=True)
    print(f"  累计缩水 (embargo - BT002注水 {s_la['sharpe']:+.2f}): ΔSharpe {cum_sharpe:+.2f}  Δ年化 {cum_ann:+.1%}", flush=True)

    # ── 基准 ──
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
    print("\n[C] 基准:", flush=True)
    for n, sm in bench_summ.items():
        print(f"  {n:9s}: 年化 {sm['ann_return']:+.1%}  Sharpe {sm['sharpe']:+.2f}", flush=True)

    # ── regime 分解 (R11) ──
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
    cmp_m.to_parquet(WFE / "wfe001_monthly_regime.parquet")
    regime_table = {}
    print("\n[D] regime 分解 (embargo book vs hs300):", flush=True)
    for rg, g in cmp_m.groupby("regime"):
        regime_table[rg] = {"n_months": int(len(g)),
                            "book_ret_mean_pct": round(float(g["book"].mean()) * 100, 3),
                            "excess_vs_hs300_mean_pp": round(float(g["excess_hs300"].mean()), 3),
                            "excess_vs_hs300_win_rate": round(float((g["excess_hs300"] > 0).mean()), 3)}
        print(f"  [{rg:9s}] n={len(g):2d}  book {g['book'].mean()*100:+.2f}%  "
              f"超额hs300 {g['excess_hs300'].mean():+.2f}pp (胜率 {(g['excess_hs300']>0).mean():.0%})", flush=True)

    # ── 因果自检: embargo vs WF-001-无embargo 的 r20 IC ──
    diag_emb = pd.read_csv(DIAG, dtype={"month": str})
    ic_emb = ic_split(diag_emb)
    ic_wf001 = ic_split(pd.read_csv(WF001_DIAG, dtype={"month": str}))
    print("\n[E] 因果自检 (r20 截面 rank-IC vs r20_fresh, 月度均值):", flush=True)
    print(f"  WF-001 无embargo WF: 全期 {ic_wf001['wf_ic_all']:+.4f} | true-OOS {ic_wf001['wf_ic_trueoos']:+.4f}", flush=True)
    print(f"  WFE-001 embargo  WF: 全期 {ic_emb['wf_ic_all']:+.4f} | true-OOS {ic_emb['wf_ic_trueoos']:+.4f}", flush=True)
    ic_drop_all = ic_emb["wf_ic_all"] - ic_wf001["wf_ic_all"]
    print(f"  embargo 后 r20 IC 变化 (全期): {ic_drop_all:+.4f} "
          f"({'再降=近截止label泄漏被切' if ic_drop_all < 0 else '基本持平=泄漏量小'})", flush=True)

    # ── 裁决 (playbook: embargo后仍稳 / embargo后大缩 / embargo后塌) ──
    net_positive = bool(s_emb["ann_return"] > 0 and s_emb["sharpe"] > 0)
    beats_all_bench = bool(s_emb["sharpe"] > max(bm["sharpe"] for bm in bench_summ.values()))
    if s_emb["sharpe"] <= 0.3:
        playbook = "embargo后塌"
    elif extra_sharpe < -0.5 or s_emb["sharpe"] < 0.6 * s_wf001["sharpe"]:
        playbook = "embargo后大缩"
    else:
        playbook = "embargo后仍稳"

    conclusion = (
        f"WFE-001 = WF-001 + label-availability embargo (训练截止收紧到 P_start - {21} 交易日, 保证训练样本 r20 "
        f"前向20日标签在预测前已全部可知; codex 第二次 review 抓到 WF-001 仍按特征日<train_end 切, 近截止样本标签延伸进预测月)。"
        f"embargo 后**真·真实可交易**: 年化 {s_emb['ann_return']:+.1%} 净Sharpe {s_emb['sharpe']:+.2f} maxDD {s_emb['max_drawdown']:+.1%} "
        f"月换手 {s_emb['avg_monthly_turnover']:.2f} 月胜率 {s_emb['monthly_win_rate']:.0%}。"
        f"vs WF-001 无embargo版 (年化 {s_wf001['ann_return']:+.1%} Sharpe {s_wf001['sharpe']:+.2f}): "
        f"**额外缩水 ΔSharpe {extra_sharpe:+.2f} / Δ年化 {extra_ann:+.1%}** (label embargo 再削掉 WF-001 残留的近截止泄漏)。"
        f"vs BT-002 注水版 (Sharpe {s_la['sharpe']:+.2f}): 累计缩水 ΔSharpe {cum_sharpe:+.2f} / Δ年化 {cum_ann:+.1%}。"
        f"基准: hs300 Sharpe {bench_summ['hs300']['sharpe']:+.2f} / CSI1000 {bench_summ['csi1000']['sharpe']:+.2f} / 动量 {bench_summ['momentum']['sharpe']:+.2f} "
        f"({'仍跑赢全基准' if beats_all_bench else '已被部分基准追上'})。"
        f"因果自检: embargo 后 WF r20 IC 全期 {ic_emb['wf_ic_all']:+.4f} vs WF-001 无embargo {ic_wf001['wf_ic_all']:+.4f} (变化 {ic_drop_all:+.4f})。"
        f"playbook 命中 [{playbook}]: "
        + ({"embargo后仍稳": "加 embargo 后真实 Sharpe 仍显著正 (小幅再缩) → V12.31 真·可交易数定案, 进部署/holdout 讨论 (WFE-002 止盈复测须基于这批 embargo picks)。",
            "embargo后大缩": "embargo 再削大块 → 真实优势比 WF-001 想象更小, 实盘预期再下修 (WFE-002 须基于这批 embargo picks)。",
            "embargo后塌": "近零/负 → r20 优势主要是 label 泄漏+记忆, 重大发现, 整个评估口径重估。"}[playbook])
        + " 注: 这是 label embargo 后的真·真实可交易地基, 替代 WF-001 的 1.84 (相对 Δ/受控对照结论不受影响)。"
    )

    results = {
        "config": {"total_invested": TOTAL, "rebalance_every_tdays": REBAL_EVERY,
                   "baseline_slippage": BASELINE_SLIP, "period": [start, book_dates[-1]],
                   "n_months": int(len(cmp_m)), "embargo_tdays": 21,
                   "delookahead": "r20 月度 walk-forward + label-availability embargo (训练截止 <= P_start-21交易日)"},
        "embargo_book": s_emb, "wf001_no_embargo": s_wf001, "lookahead_bt002": s_la,
        "extra_shrinkage_vs_wf001": {"delta_sharpe": round(extra_sharpe, 3), "delta_ann_return": round(extra_ann, 4)},
        "cumulative_shrinkage_vs_bt002": {"delta_sharpe": round(cum_sharpe, 3), "delta_ann_return": round(cum_ann, 4)},
        "benchmarks": bench_summ, "regime": regime_table,
        "causal_check": {"embargo_wf_ic_all": round(ic_emb["wf_ic_all"], 4),
                         "embargo_wf_ic_trueoos": round(ic_emb["wf_ic_trueoos"], 4),
                         "wf001_no_embargo_ic_all": round(ic_wf001["wf_ic_all"], 4),
                         "wf001_no_embargo_ic_trueoos": round(ic_wf001["wf_ic_trueoos"], 4),
                         "ic_change_all": round(ic_drop_all, 4)},
        "playbook": playbook, "beats_all_bench": beats_all_bench,
    }
    (WFE / "wfe001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "WFE-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "embargo_ann_return": round(s_emb["ann_return"], 4),
            "embargo_net_sharpe": round(s_emb["sharpe"], 3),
            "embargo_max_drawdown": round(s_emb["max_drawdown"], 4),
            "embargo_avg_monthly_turnover": round(s_emb["avg_monthly_turnover"], 3),
            "embargo_monthly_win_rate": round(s_emb["monthly_win_rate"], 3),
            "wf001_no_embargo_sharpe": round(s_wf001["sharpe"], 3),
            "wf001_no_embargo_ann_return": round(s_wf001["ann_return"], 4),
            "extra_shrink_delta_sharpe": round(extra_sharpe, 3),
            "extra_shrink_delta_ann_return": round(extra_ann, 4),
            "cum_shrink_delta_sharpe_vs_bt002": round(cum_sharpe, 3),
            "embargo_wf_ic_all": round(ic_emb["wf_ic_all"], 4),
            "embargo_wf_ic_trueoos": round(ic_emb["wf_ic_trueoos"], 4),
            "net_positive": net_positive,
            "beats_all_bench": beats_all_bench,
            "playbook": playbook,
        },
        "regime_table": regime_table,
        "artifacts": ["research/backtest/wfe001_gen_picks.py", "research/backtest/run_wfe001.py",
                      "research/cache/wfe001/picks_oos_daily.parquet",
                      "research/cache/wfe001/r20_models/", "research/cache/wfe001/r20_oos_diagnostics.csv",
                      "research/cache/wfe001/wfe001_book_oos.parquet",
                      "research/cache/wfe001/wfe001_monthly_regime.parquet",
                      "research/cache/wfe001/wfe001_results.json"],
        "note": "唯一改动 vs WF-001 = r20 训练截止从 '特征日 < train_end' 收紧到 'label 可知 (trade_date <= P_start-21交易日)'; "
                "24m lookback 起点/固定120树/引擎/成本/再平衡/双轨配置逐字同 WF-001 (R01)。ST 源头排除 (R06); "
                "regime 分层 (R11); 描述性真实 P&L=合法完成 (R02); 生产线只读 (R05); 大缓存 gitignored (R04)。"
                "吃进 codex 第二次 review 第1条 (label embargo)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built playbook={playbook} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
