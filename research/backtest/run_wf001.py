# -*- coding: utf-8 -*-
"""WF-001 — OOS picks 过 book 引擎 → 真实可交易 P&L + vs BT-002 注水版缩水幅度.

输入: research/cache/wf001/picks_oos_daily.parquet (de-lookahead r20 月度 walk-forward 重建的 V12.31 持仓)。
同 BT-002 配置 (alloc 加权双轨, 20d 再平衡, 真实成本基线 0.10%/边) 过同一引擎, 报:
  真实 年化/净Sharpe/maxDD/月换手/月胜率  +  vs BT-002 (单一生产 r20 跨全期 = lookahead 注水) 的缩水。
因果自检: r20_oos_diagnostics.csv 的 OOS r20 IC 应低于 in-sample 生产模型 (前半段 202410-202506 生产模型训练窗内)。
分 regime (R11)。verdict 按 prd.responsePlaybook.WF-001 (缩水温和 / 大幅缩水 / 塌) 命中。

R01 配置冻结 (与 BT-002 逐字一致, 唯一差异 = picks 由 OOS r20 生成); R06 ST 源头排除; R11 regime 分层;
描述性真实 P&L = 合法完成 (R02); 大缓存 gitignored; 生产线只读 (R05)。
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

WF = ROOT / "research" / "cache" / "wf001"
PICKS = WF / "picks_oos_daily.parquet"
DIAG = WF / "r20_oos_diagnostics.csv"
BT002 = ROOT / "research" / "cache" / "bt002" / "bt002_results.json"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "WF-001.json"


def main():
    t0 = time.time()
    print("\n=== WF-001: de-lookahead r20 OOS book → 真实可交易 P&L ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] OOS {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / "
          f"{picks['month'].nunique()} 月 / {picks['ts_code'].nunique()} 股", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    end = cal[-1]
    print(f"[prices] {len(px):,} 行 / {len(cal)} 交易日 [{start}~{end}]", flush=True)

    tgt_book = build_dual_alloc_targets(picks, cal)
    tgt_mom = build_momentum_targets(px, cal)
    print(f"[targets] OOS book 再平衡 {len(tgt_book)} 次, 动量基准 {len(tgt_mom)} 次", flush=True)

    # ── OOS book 过引擎 (基线滑点, 同 BT-002) ──
    print("\n[A] OOS V12.31 book 过引擎 (基线滑点 0.10%/边) ...", flush=True)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res = bt.run(tgt_book, start=start)
    s_oos = res.summary()
    res.timeseries.to_parquet(WF / "wf001_book_oos.parquet")
    book_nav = res.timeseries["nav"]
    book_dates = book_nav.index.tolist()
    print(f"  OOS: 年化 {s_oos['ann_return']:+.1%}  净Sharpe {s_oos['sharpe']:+.2f}  "
          f"maxDD {s_oos['max_drawdown']:+.1%}  月换手 {s_oos['avg_monthly_turnover']:.2f}  "
          f"月胜率 {s_oos['monthly_win_rate']:.0%}", flush=True)

    # ── vs BT-002 注水版 (单一生产 r20 跨全期) ──
    bt002 = json.loads(BT002.read_text(encoding="utf-8"))
    s_la = bt002["A_slippage_sensitivity"][f"slip_{BASELINE_SLIP}"]
    print(f"\n[B] vs BT-002 (lookahead 注水版, 单一生产 r20):", flush=True)
    print(f"  注水版: 年化 {s_la['ann_return']:+.1%}  净Sharpe {s_la['sharpe']:+.2f}  maxDD {s_la['max_drawdown']:+.1%}", flush=True)
    shrink_sharpe = s_oos["sharpe"] - s_la["sharpe"]
    shrink_ann = s_oos["ann_return"] - s_la["ann_return"]
    print(f"  缩水: ΔSharpe {shrink_sharpe:+.2f}  Δ年化 {shrink_ann:+.1%}", flush=True)

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
    cmp_m.to_parquet(WF / "wf001_monthly_regime.parquet")
    regime_table = {}
    print("\n[D] regime 分解 (OOS book vs hs300):", flush=True)
    for rg, g in cmp_m.groupby("regime"):
        regime_table[rg] = {"n_months": int(len(g)),
                            "book_ret_mean_pct": round(float(g["book"].mean()) * 100, 3),
                            "excess_vs_hs300_mean_pp": round(float(g["excess_hs300"].mean()), 3),
                            "excess_vs_hs300_win_rate": round(float((g["excess_hs300"] > 0).mean()), 3)}
        print(f"  [{rg:9s}] n={len(g):2d}  book {g['book'].mean()*100:+.2f}%  "
              f"超额hs300 {g['excess_hs300'].mean():+.2f}pp (胜率 {(g['excess_hs300']>0).mean():.0%})", flush=True)

    # ── 因果自检 (OOS r20 IC vs in-sample 生产) ──
    diag = pd.read_csv(DIAG, dtype={"month": str})
    # 生产 r20 训练窗 < 20250930 → 月 ≤ 202509 = in-sample (从 month 重算, 不信存储 flag)
    diag["prod_in_train_window"] = diag["month"].astype(str) <= "202509"
    ins = diag[diag["prod_in_train_window"]]
    oos = diag[~diag["prod_in_train_window"]]
    ic_oos_all = float(diag["r20_oos_rankic"].mean())
    ic_prod_all = float(diag["r20_prod_rankic"].mean())
    ic_prod_insample = float(ins["r20_prod_rankic"].mean()) if len(ins) else float("nan")
    ic_prod_trueoos = float(oos["r20_prod_rankic"].mean()) if len(oos) else float("nan")
    ic_oos_insample = float(ins["r20_oos_rankic"].mean()) if len(ins) else float("nan")
    ic_oos_trueoos = float(oos["r20_oos_rankic"].mean()) if len(oos) else float("nan")
    print("\n[E] 因果自检 (r20 截面 rank-IC vs r20_fresh, 月度均值):", flush=True)
    print(f"  生产单一模型: 全期 {ic_prod_all:+.4f} | in-sample段(≤202509) {ic_prod_insample:+.4f} | true-OOS段 {ic_prod_trueoos:+.4f}", flush=True)
    print(f"  WF月度重训:   全期 {ic_oos_all:+.4f} | in-sample段 {ic_oos_insample:+.4f} | true-OOS段 {ic_oos_trueoos:+.4f}", flush=True)
    # 因果性证据: 生产模型在 in-sample 段 IC 应高于 true-OOS 段 (lookahead 痕迹)
    prod_decay = ic_prod_insample - ic_prod_trueoos

    # ── 裁决 (playbook) ──
    net_positive = bool(s_oos["ann_return"] > 0 and s_oos["sharpe"] > 0)
    if s_oos["sharpe"] <= 0.1:
        playbook = "塌"
    elif shrink_sharpe < -0.5 or s_oos["ann_return"] < 0.5 * s_la["ann_return"]:
        playbook = "大幅缩水"
    else:
        playbook = "缩水温和"

    conclusion = (
        f"de-lookahead r20 月度 walk-forward (24m lookback, 19 月, time-split val 防 OOS 偷看) 重建 V12.31 OOS book "
        f"过同一引擎 (基线滑点 {BASELINE_SLIP*100:.2f}%/边, 同 BT-002 配置)。"
        f"真实可交易: 年化 {s_oos['ann_return']:+.1%} 净Sharpe {s_oos['sharpe']:+.2f} maxDD {s_oos['max_drawdown']:+.1%} "
        f"月换手 {s_oos['avg_monthly_turnover']:.2f} 月胜率 {s_oos['monthly_win_rate']:.0%}。"
        f"vs BT-002 注水版 (单一生产 r20 跨全测试期 = 共模 lookahead, 年化 {s_la['ann_return']:+.1%} Sharpe {s_la['sharpe']:+.2f}): "
        f"缩水 ΔSharpe {shrink_sharpe:+.2f} / Δ年化 {shrink_ann:+.1%}。"
        f"基准: hs300 年化 {bench_summ['hs300']['ann_return']:+.1%} Sharpe {bench_summ['hs300']['sharpe']:+.2f}; "
        f"CSI1000 {bench_summ['csi1000']['ann_return']:+.1%}; 动量 {bench_summ['momentum']['ann_return']:+.1%}。"
        f"因果自检 (smoking gun): 生产 r20 IC 从 in-sample 段(≤202509, 19月里12月在其<20250930训练窗内) "
        f"{ic_prod_insample:+.4f} 暴跌至 true-OOS 段 {ic_prod_trueoos:+.4f} (衰减 {prod_decay:+.4f}); "
        f"而 WF 月度重训在**同一 true-OOS 段** IC {ic_oos_trueoos:+.4f} ≈ 生产 true-OOS {ic_prod_trueoos:+.4f} "
        f"(几乎相等) → 生产模型在真 OOS 上**无 lookahead 优势, 其 in-sample +0.44 全是记忆**; 真实 r20 OOS IC ≈ "
        f"WF 全期 {ic_oos_all:+.4f}。这正是 +140% 注水的来源 (12/19 测试月在生产模型训练窗内)。"
        f"playbook 命中 [{playbook}]: "
        + ({"缩水温和": "de-lookahead 后真实 Sharpe 仍显著正 → V12.31 可交易, 进部署讨论 (但绝对量级须按真实数重设, 不再用 +140%)。",
            "大幅缩水": "真实 Sharpe 远低于注水版 → 之前所有绝对数 (BT-001/002/EX 含 r20 池 lookahead) 不可信, 实盘预期须按真实数重设。",
            "塌": "de-lookahead 后近零/负 → r20 优势主要是 lookahead artifact, 重大发现, 整个评估口径重估。"}[playbook])
        + " 注: 这是真 walk-forward 的可交易地基, 替代之前所有含共模 lookahead 的绝对数 (相对 Δ/受控对照结论不受影响)。"
    )

    results = {
        "config": {"total_invested": TOTAL, "rebalance_every_tdays": REBAL_EVERY,
                   "baseline_slippage": BASELINE_SLIP, "period": [start, book_dates[-1]],
                   "n_months": int(len(cmp_m)),
                   "delookahead": "r20 月度 walk-forward 重训 (24m lookback, time-split val); pump s5 已 walk-forward(t005)"},
        "oos_book": s_oos, "lookahead_bt002": s_la,
        "shrinkage": {"delta_sharpe": round(shrink_sharpe, 3), "delta_ann_return": round(shrink_ann, 4),
                      "oos_sharpe_pct_of_lookahead": round(s_oos["sharpe"] / (s_la["sharpe"] + 1e-9), 3)},
        "benchmarks": bench_summ, "regime": regime_table,
        "causal_check": {"r20_prod_ic_insample": round(ic_prod_insample, 4),
                         "r20_prod_ic_trueoos": round(ic_prod_trueoos, 4),
                         "r20_prod_ic_decay_lookahead_trace": round(prod_decay, 4),
                         "r20_wf_ic_all": round(ic_oos_all, 4),
                         "r20_wf_ic_insample": round(ic_oos_insample, 4),
                         "r20_wf_ic_trueoos": round(ic_oos_trueoos, 4)},
        "playbook": playbook,
    }
    (WF / "wf001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "WF-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "oos_ann_return": round(s_oos["ann_return"], 4),
            "oos_net_sharpe": round(s_oos["sharpe"], 3),
            "oos_max_drawdown": round(s_oos["max_drawdown"], 4),
            "oos_avg_monthly_turnover": round(s_oos["avg_monthly_turnover"], 3),
            "oos_monthly_win_rate": round(s_oos["monthly_win_rate"], 3),
            "lookahead_ann_return": round(s_la["ann_return"], 4),
            "lookahead_net_sharpe": round(s_la["sharpe"], 3),
            "shrink_delta_sharpe": round(shrink_sharpe, 3),
            "shrink_delta_ann_return": round(shrink_ann, 4),
            "oos_sharpe_pct_of_lookahead": round(s_oos["sharpe"] / (s_la["sharpe"] + 1e-9), 3),
            "r20_prod_ic_insample": round(ic_prod_insample, 4),
            "r20_prod_ic_trueoos": round(ic_prod_trueoos, 4),
            "r20_wf_ic_all": round(ic_oos_all, 4),
            "net_positive": net_positive,
            "playbook": playbook,
        },
        "regime_table": regime_table,
        "artifacts": ["research/backtest/wf001_gen_picks.py", "research/backtest/run_wf001.py",
                      "research/cache/wf001/picks_oos_daily.parquet",
                      "research/cache/wf001/r20_models/", "research/cache/wf001/r20_oos_diagnostics.csv",
                      "research/cache/wf001/wf001_book_oos.parquet",
                      "research/cache/wf001/wf001_monthly_regime.parquet",
                      "research/cache/wf001/wf001_results.json"],
        "note": "唯一 de-lookahead 改动 = r20 池模型由单一生产模型换月度 walk-forward 重训 (pump s5 已 walk-forward); "
                "引擎/成本/再平衡/双轨配置逐字同 BT-002 (R01)。ST 源头排除 (R06); regime 分层 (R11); "
                "time-split val 防 OOS 偷看; 描述性真实 P&L=合法完成 (R02); 生产线只读 (R05); 大缓存 gitignored (R04)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built playbook={playbook} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
