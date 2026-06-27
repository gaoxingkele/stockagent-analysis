# -*- coding: utf-8 -*-
"""FIN-003 — triple-barrier V12.32 挑战者 book 过引擎 + gate_tb 断言 (apples-to-apples vs 1.31 基线).

输入: research/cache/fin003/picks_oos_daily.parquet (TB-score 替代 r20 重建的 V12.31 dual-track 持仓)。
基线: research/cache/wfe001/wfe001_book_oos.parquet (WFE-001 embargo 真·真实 book, FIN-001 定 净Sharpe 1.31)。

两 book 过**同一引擎/成本/再平衡/双轨配置** (唯一差异 = 池 filter 信号: TB-score vs r20), 在同一组交易日上
对齐 → 受控对照 (共模成本/集中相消, 只看相对 TB vs 基线; SIGN-R03/R13)。

gate_tb (prd.preRegisteredGate.gate_tb, 冻结 R01, 不移门柱):
  PASS = TB 净Sharpe > 1.31 基线  AND  maxDD 不升  AND  ΔSharpe block bootstrap CI 不含0
         AND  单月 outlier 剔后仍优  AND  分 regime 不伤 (动量改善 + 反转不伤, SIGN-R11)
  真小 = ΔSharpe>0 但 CI 含0 / 增益小    REJECT = <=1.31 或伤 maxDD/regime
  (responsePlaybook.FIN-003: PASS→V12.32候选进独立holdout; 真小→记候选不强上; REJECT→均值回归选股近最优文档化)

ST 已源头排除 (R06); regime 分层 (R11); 描述性受控对照=合法完成 (R02); 生产线只读 (R05);
大缓存 research/cache/fin003/ gitignored (R04)。
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
from run_bt002 import (load_prices, build_dual_alloc_targets,
                       nav_summary, monthly_returns,
                       TOTAL, REBAL_EVERY, BASELINE_SLIP)

FIN3 = ROOT / "research" / "cache" / "fin003"
PICKS = FIN3 / "picks_oos_daily.parquet"
DIAG = FIN3 / "tb_oos_diagnostics.csv"
WFE_BOOK = ROOT / "research" / "cache" / "wfe001" / "wfe001_book_oos.parquet"
WFE_RESULTS = ROOT / "research" / "cache" / "wfe001" / "wfe001_results.json"
FIN1 = ROOT / "research" / "cache" / "fin001" / "fin001_results.json"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "FIN-003.json"

PPY = 252
COHORT_LEN = 20          # = 再平衡/持有期, 自然依赖块 (与 FIN-001 同)
N_BOOT = 1000
BOOT_SEED = 20260614


def sharpe(rets: np.ndarray) -> float:
    r = np.asarray(rets, float); r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    return float(np.mean(r) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(PPY))


def max_dd(nav: np.ndarray) -> float:
    nav = np.asarray(nav, float)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def main():
    t0 = time.time()
    print("\n=== FIN-003: triple-barrier V12.32 book → gate_tb vs 1.31 基线 ===\n", flush=True)

    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] TB embargo OOS {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / "
          f"{picks['month'].nunique()} 月 / {picks['ts_code'].nunique()} 股", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    print(f"[prices] {len(px):,} 行 / {len(cal)} 交易日 [{start}~{cal[-1]}]", flush=True)

    # ── TB book 过引擎 (与 WFE-001 逐字同配置) ──
    print("\n[A] TB V12.32 book 过引擎 (基线滑点 0.10%/边, 同 WFE-001) ...", flush=True)
    tgt_book = build_dual_alloc_targets(picks, cal)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res = bt.run(tgt_book, start=start)
    s_tb = res.summary()
    res.timeseries.to_parquet(FIN3 / "fin003_book_oos.parquet")
    tb_nav = res.timeseries["nav"].astype(float)
    tb_nav.index = tb_nav.index.astype(str)
    print(f"  TB: 年化 {s_tb['ann_return']:+.1%}  净Sharpe {s_tb['sharpe']:+.2f}  "
          f"maxDD {s_tb['max_drawdown']:+.1%}  月换手 {s_tb['avg_monthly_turnover']:.2f}  "
          f"月胜率 {s_tb['monthly_win_rate']:.0%}", flush=True)

    # ── 基线 book (WFE-001, FIN-001 定 1.31) ──
    base_book = pd.read_parquet(WFE_BOOK)
    base_book.index = base_book.index.astype(str)
    base_nav = base_book["nav"].astype(float)
    fin1 = json.loads(FIN1.read_text(encoding="utf-8"))
    baseline_point_sharpe = fin1["point_estimate"]["net_sharpe"]   # = 1.31 (FIN-001 口径一致)

    # ── 对齐两 book 到公共交易日 (受控对照) ──
    common = [d for d in tb_nav.index if d in set(base_nav.index)]
    tb_n = tb_nav.reindex(common).astype(float)
    ba_n = base_nav.reindex(common).astype(float)
    tb_r = tb_n.pct_change()
    ba_r = ba_n.pct_change()
    aligned = pd.DataFrame({"tb": tb_r, "ba": ba_r}).dropna()
    tb_ret = aligned["tb"].to_numpy()
    ba_ret = aligned["ba"].to_numpy()
    dates = aligned.index.tolist()
    n_days = len(dates)

    s_tb_aln = sharpe(tb_ret)
    s_ba_aln = sharpe(ba_ret)
    dd_tb = max_dd(np.cumprod(1.0 + tb_ret))
    dd_ba = max_dd(np.cumprod(1.0 + ba_ret))
    d_sharpe = s_tb_aln - s_ba_aln
    print(f"\n[B] 对齐 {n_days} 日 [{dates[0]}~{dates[-1]}] (受控对照, 同引擎/成本/再平衡):", flush=True)
    print(f"  基线(r20) Sharpe {s_ba_aln:+.3f} maxDD {dd_ba:+.1%}  (FIN-001 报 {baseline_point_sharpe:+.2f})", flush=True)
    print(f"  TB(V12.32) Sharpe {s_tb_aln:+.3f} maxDD {dd_tb:+.1%}", flush=True)
    print(f"  ΔSharpe (TB - 基线) = {d_sharpe:+.3f}", flush=True)

    # ── ① block bootstrap ΔSharpe CI (paired cohort 重采样) ──
    cohort_id = np.arange(n_days) // COHORT_LEN
    n_cohorts = int(cohort_id.max()) + 1
    tb_co = [tb_ret[cohort_id == c] for c in range(n_cohorts)]
    ba_co = [ba_ret[cohort_id == c] for c in range(n_cohorts)]
    rng = np.random.default_rng(BOOT_SEED)
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_cohorts, size=n_cohorts)
        s_t = sharpe(np.concatenate([tb_co[c] for c in pick]))
        s_b = sharpe(np.concatenate([ba_co[c] for c in pick]))
        if np.isfinite(s_t) and np.isfinite(s_b):
            bs.append(s_t - s_b)
    bs = np.array(bs)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    p_gt0 = float((bs > 0).mean())
    ci_excludes_0 = bool(ci[0] > 0 or ci[1] < 0)
    print(f"\n[①bootstrap] ΔSharpe 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]  P(Δ>0)={p_gt0:.3f}  "
          f"(不含0? {ci_excludes_0})", flush=True)

    # ── ② 单月 outlier 剔后仍优 ──
    ym = pd.Index(dates).str[:6]
    tb_mret = pd.Series(tb_ret, index=dates).groupby(ym).apply(lambda g: float(np.prod(1+g.to_numpy())-1))
    ba_mret = pd.Series(ba_ret, index=dates).groupby(ym).apply(lambda g: float(np.prod(1+g.to_numpy())-1))
    # 剔掉对 TB 贡献最大的单月 (最有利 TB 的月), 看是否仍优
    tb_excess_m = (tb_mret - ba_mret)
    best_m = str(tb_excess_m.idxmax())
    keep = np.asarray(ym) != best_m
    s_tb_ex = sharpe(tb_ret[keep]); s_ba_ex = sharpe(ba_ret[keep])
    d_sharpe_ex = s_tb_ex - s_ba_ex
    outlier_robust = bool(s_tb_ex > s_ba_ex)
    print(f"[②单月outlier] 剔最利TB月 {best_m} (Δ收益 {tb_excess_m.max()*100:+.2f}pp): "
          f"TB Sharpe {s_tb_ex:+.3f} vs 基线 {s_ba_ex:+.3f} (ΔSharpe {d_sharpe_ex:+.3f}, TB仍优? {outlier_robust})", flush=True)

    # ── ③ 分 regime 不伤 (R11) ──
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())
    rg_tbl = {}
    for rg in ["momentum", "reversal", "mixed"]:
        months_rg = [m for m in tb_mret.index if month_regime.get(m) == rg]
        if not months_rg:
            continue
        tb_mean = float(tb_mret[months_rg].mean()) * 100
        ba_mean = float(ba_mret[months_rg].mean()) * 100
        rg_tbl[rg] = {"n_months": len(months_rg),
                      "tb_monthly_ret_pct": round(tb_mean, 3),
                      "base_monthly_ret_pct": round(ba_mean, 3),
                      "delta_pp": round(tb_mean - ba_mean, 3)}
    print("\n[③regime 不伤 (TB vs 基线 月收益)]:", flush=True)
    for rg, d in rg_tbl.items():
        print(f"  [{rg:9s}] n={d['n_months']:2d}  TB {d['tb_monthly_ret_pct']:+.2f}%  "
              f"基线 {d['base_monthly_ret_pct']:+.2f}%  Δ {d['delta_pp']:+.2f}pp", flush=True)
    # 不伤: 动量改善 (Δ>0) 且 反转不伤 (Δ>=-tol). mixed n小, 不作硬条件.
    REG_TOL = 0.05
    mom_d = rg_tbl.get("momentum", {}).get("delta_pp", -99)
    rev_d = rg_tbl.get("reversal", {}).get("delta_pp", -99)
    regime_ok = bool(mom_d > 0 and rev_d >= -REG_TOL)

    # ── gate_tb 五条件 (冻结 R01) ──
    c_sharpe = bool(s_tb_aln > s_ba_aln)
    c_maxdd = bool(dd_tb >= dd_ba - 1e-9)        # 不升 = 不更负
    c_ci = ci_excludes_0
    c_outlier = outlier_robust
    c_regime = regime_ok
    conds = {"net_sharpe>baseline": c_sharpe, "maxDD_not_up": c_maxdd,
             "bootstrap_CI_excludes_0": c_ci, "single_month_outlier_robust": c_outlier,
             "regime_not_hurt(mom_up+rev_not_hurt)": c_regime}
    passed = all(conds.values())
    if passed:
        status = "PASS"; pb = "PASS"
    elif d_sharpe > 0:
        status = "真小"; pb = "真小"
    else:
        status = "REJECT"; pb = "REJECT"
    fail = [k for k, v in conds.items() if not v]

    # ── TB-score IC 诊断 (描述性) ──
    tb_ic = np.nan
    if DIAG.exists():
        dd = pd.read_csv(DIAG)
        if "tb_oos_rankic_vs_r20fresh" in dd.columns:
            tb_ic = float(dd["tb_oos_rankic_vs_r20fresh"].mean())

    pb_text = {
        "PASS": ("triple-barrier 干净跑赢 1.31 基线 → V12.32 候选, 进独立 holdout (不拿来修饰 V12.31, "
                 "生产线 V12.31 仍冻结 R05)。用户 '限幅度+回撤' 直觉在选股 label 层兑现。"),
        "真小": ("ΔSharpe>0 但 bootstrap CI 含0 / 增益小 → 记 V12.32 候选不强上 (样本短功率低, 同 FIN-001 "
                 "CI 宽诊断); 不替换 V12.31 基线。"),
        "REJECT": ("triple-barrier 在选股 label 层 <=1.31 基线 或 伤 maxDD/regime → 用户 'triple-barrier' "
                   "直觉在**选股层**不兑现 (出场层 WFE-002 也已否止盈); 均值回归选股已近最优, 文档化 "
                   "(第 20 个被反过拟合脚手架诚实判的假设)。生产线 V12.31 冻结不动。"),
    }[pb]

    conclusion = (
        f"FIN-003 triple-barrier V12.32 挑战者 (双屏障 +15%/-8% + 40d backstop, 参数预注册冻结 R01) = "
        f"WFE-001 embargo r20 walk-forward 的独立挑战者, 唯一改动 = r20 回归 label 换三屏障 label "
        f"(embargo 收紧到 P_start-41交易日, 因屏障 horizon 40d), TB-score 替代 r20 作池 filter, 池内仍按 "
        f"ratio_s5 排序, 过同一引擎 (close-based 成本 0.10%/边 + 双轨 + 行业cap + 20d 再平衡)。"
        f"对齐 {n_days} 日受控对照 (vs WFE-001 基线 1.31, 共模成本/集中相消): "
        f"**TB 净Sharpe {s_tb_aln:+.2f} vs 基线 {s_ba_aln:+.2f} → ΔSharpe {d_sharpe:+.2f}**; "
        f"maxDD TB {dd_tb:+.1%} vs 基线 {dd_ba:+.1%}。"
        f"① block bootstrap ΔSharpe 95% CI=[{ci[0]:+.2f},{ci[1]:+.2f}] P(Δ>0)={p_gt0:.0%} (不含0? {ci_excludes_0}); "
        f"② 剔最利TB单月 {best_m} 后 ΔSharpe {d_sharpe_ex:+.2f} (TB仍优? {outlier_robust}); "
        f"③ regime: 动量 Δ{mom_d:+.2f}pp / 反转 Δ{rev_d:+.2f}pp (不伤? {regime_ok})。"
        f"TB-score 对 r20_fresh 截面 IC≈{tb_ic:+.4f} (描述性)。"
        f"gate_tb 五条件 {conds} → **status={status}**。{pb_text}"
    )

    results = {
        "config": {"tb_X": 0.15, "tb_Y": 0.08, "tb_horizon": 40, "embargo_tdays": 41,
                   "total_invested": TOTAL, "rebalance_every_tdays": REBAL_EVERY,
                   "baseline_slippage": BASELINE_SLIP, "aligned_days": n_days,
                   "period": [dates[0], dates[-1]], "n_cohorts": n_cohorts,
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "note": "FIN-003 是独立 V12.32 挑战者; apples-to-apples vs WFE-001 基线 1.31 "
                           "(同池/cap/双轨/成本/embargo, 唯一差异=池filter信号 TB-score vs r20)"},
        "tb_book_summary": s_tb,
        "aligned": {"tb_sharpe": round(s_tb_aln, 3), "base_sharpe": round(s_ba_aln, 3),
                    "delta_sharpe": round(d_sharpe, 3), "tb_maxdd": round(dd_tb, 4),
                    "base_maxdd": round(dd_ba, 4), "baseline_fin001_sharpe": baseline_point_sharpe},
        "bootstrap": {"delta_sharpe_ci95": [round(ci[0], 3), round(ci[1], 3)],
                      "p_delta_gt_0": round(p_gt0, 3), "ci_excludes_0": ci_excludes_0,
                      "n_valid": int(len(bs))},
        "single_month_outlier": {"dropped_month": best_m,
                                 "tb_sharpe_ex": round(s_tb_ex, 3), "base_sharpe_ex": round(s_ba_ex, 3),
                                 "delta_sharpe_ex": round(d_sharpe_ex, 3), "tb_still_better": outlier_robust},
        "regime": rg_tbl, "tb_score_ic_vs_r20fresh": round(tb_ic, 4) if np.isfinite(tb_ic) else None,
        "gate_conditions": conds, "gate_status": status, "playbook": pb,
    }
    (FIN3 / "fin003_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "FIN-003", "status": status, "conclusion": conclusion,
        "metrics": {
            "sharpe": round(s_tb_aln, 3),
            "delta_sharpe_vs_baseline": round(d_sharpe, 3),
            "baseline_sharpe": round(s_ba_aln, 3),
            "tb_max_drawdown": round(dd_tb, 4),
            "base_max_drawdown": round(dd_ba, 4),
            "worst_month": round(float(tb_mret.min()) * 100, 3),
            "bootstrap_delta_sharpe_ci95": [round(ci[0], 3), round(ci[1], 3)],
            "bootstrap_p_delta_gt_0": round(p_gt0, 3),
            "ci_excludes_0": ci_excludes_0,
            "single_month_outlier_dropped": best_m,
            "delta_sharpe_ex_outlier": round(d_sharpe_ex, 3),
            "outlier_robust": outlier_robust,
            "tb_ann_return": round(s_tb["ann_return"], 4),
            "tb_score_ic_vs_r20fresh": round(tb_ic, 4) if np.isfinite(tb_ic) else None,
            "playbook": pb,
        },
        "by_regime": rg_tbl,
        "gate_conditions": conds,
        "gate_failed": fail,
        "artifacts": ["research/backtest/fin003_gen_picks.py", "research/backtest/run_fin003.py",
                      "research/cache/fin003/picks_oos_daily.parquet",
                      "research/cache/fin003/tb_label.parquet",
                      "research/cache/fin003/r20_models/",
                      "research/cache/fin003/tb_oos_diagnostics.csv",
                      "research/cache/fin003/fin003_book_oos.parquet",
                      "research/cache/fin003/fin003_results.json"],
        "note": "triple-barrier 是独立 V12.32 挑战者 (非 V12.31 基线补丁); 双屏障+15%/-8%/40d backstop 参数预注册"
                "冻结不搜 (R01); embargo 41交易日 (屏障 horizon 40d, label 可知; R04); TB-score 替代 r20 作池"
                "filter, 池内仍 ratio_s5 排序; 同引擎/成本/双轨/cap/20d再平衡 apples-to-apples vs WFE-001 基线"
                "1.31 (FIN-001 口径); ΔSharpe block bootstrap CI + 单月outlier + 分regime (R11) 共判 gate_tb; "
                "ST 源头排除 (R06); 描述性受控对照=合法完成 (R02); 中间IC≠ship (R03); 生产线 V12.31 只读冻结 "
                "(R05); 大缓存 gitignored (R04)。吃进 codex 第三次 review③ (triple-barrier 独立分支全验证)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict={status} playbook={pb} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
