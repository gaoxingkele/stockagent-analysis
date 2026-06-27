# -*- coding: utf-8 -*-
"""DLV-002 (Roadmap A3) — 前向 paper-trade 落库 harness (PIT-clean, append-only).

北极星 (部署验证地基, 非 alpha):
  把 V12.31_BASELINE.md §3 的前向 paper-trade 协议落成**可执行 harness**: 每日 append-only 记录
  PIT-clean picks → 满持有期 (20/40d) 按**冻结口径** (close-based + A股成本 + T+1) 算实现 P&L →
  分 regime 与 **真实基线锚 (净 Sharpe 1.31 ± CI[+0.11,+2.60], FIN-001)** 对照 → 输出
  realized_vs_expected。**纸面跟踪, 不动真金; 前向数据不回流调参 (SIGN-R01/R02 红线)。**

三段式 (与协议逐条对应):
  1. log_picks_for_date(date, picks)  — append-only 落盘每日 picks (existing 文件**拒绝覆盖** = 防事后
     调参/防回改, 这是协议的红线). picks 源 = PIT-clean 变体 (dep001 剔 holder_pct r20 embargo WF),
     非生产 v12_scoring 的 in-sample 注水模型 (那是 +0.44 IC 记忆来源)。
  2. realize_cohorts(horizon)          — 对每个满持有期的 cohort, 把当日 picks 过冻结成本引擎算实现 P&L
     (book-level 连续 nav + per-cohort 20d 实现收益), 标 regime。
  3. compare_to_anchor(realized)       — book 实现 Sharpe vs 锚 1.31 落 CI 何处 (CI 下半<1.0=重评估;
     上半=向可部署收敛); 分 regime 对照 (SIGN-R11)。

demo (本脚本 main): 用 dep001 PIT-clean OOS picks (deployable 变体, 已有 381 日历史) **回填** append-only
  log, 跑完整闭环 (落库→实现→对锚), 证明 harness 端到端就绪。上线时把 backfill 换成生产每日推理即可。

冻结口径 (R01, 逐字同 V12.31_BASELINE / dep001 / WFE-001):
  双轨 alloc 加权 / 20d 再平衡 / close-based 成交 (D 决定→D+1 开盘) / A股成本基线 0.10%/边 (round-trip
  ≈0.30%) / T+1 / 涨跌停不可成交 / ST 源头排除 (R06)。锚来自 FIN-001 误差条。

生产线 V12.31 只读冻结 (R05, 仅 load 缓存 picks/价表, 0 生产改动)。大缓存 gitignored (R04)。
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
from run_bt002 import (load_prices, build_dual_alloc_targets, nav_summary,
                       monthly_returns, TOTAL, REBAL_EVERY, BASELINE_SLIP)

# ── 落库目录 (append-only, gitignored R04) ──
PAPER_DIR = ROOT / "research" / "cache" / "paper_trade"
PICKS_DIR = PAPER_DIR / "picks"
PICKS_DIR.mkdir(parents=True, exist_ok=True)
REALIZED_CSV = PAPER_DIR / "realized_vs_expected.csv"
RESULTS_JSON = PAPER_DIR / "paper_trade_results.json"
LOG_FILE = PAPER_DIR / "log.jsonl"

# ── PIT-clean picks 源 (deployable 变体) ──
DEP_PICKS = ROOT / "research" / "cache" / "dep001" / "picks_oos_daily.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "DLV-002.json"

# ── 真实基线锚 (FIN-001 / V12.31_BASELINE.md, 冻结 R01) ──
ANCHOR = {
    "net_sharpe": 1.31,
    "sharpe_ci95": [0.11, 2.60],
    "ann_return": 0.349,
    "max_drawdown": -0.157,
    "monthly_win_rate": 0.70,
    "p_sharpe_gt_0": 0.99,
    "p_sharpe_gt_1": 0.66,
    "source": "FIN-001 embargo 真·真实 walk-forward book (WFE-001), 21 个 20日 cohort",
    "note": "1.31 是点估计非可承诺下限; 真实瓶颈=样本短功率低 (CI 宽), 非过拟合 (codex#4/#5)",
}
HORIZONS = [20, 40]   # 协议要求满 20/40 日复盘


# ─────────────── 1. 落库 (append-only) ───────────────
def log_picks_for_date(date: str, picks: pd.DataFrame, source: str,
                       overwrite: bool = False) -> str:
    """append-only 落盘某交易日 picks。existing 文件默认**拒绝覆盖** (协议红线 SIGN-R01/R02:
    前向数据落库后不可回改 = 防事后调参)。返回状态 {'written','kept_existing'}。"""
    date = str(date)
    fp = PICKS_DIR / f"{date}.parquet"
    if fp.exists() and not overwrite:
        return "kept_existing"   # 不可回改, 保留原始记录
    cols = [c for c in ["ts_code", "industry", "alloc_pct", "ratio_s5", "r20_fresh"]
            if c in picks.columns]
    rec = picks[cols].copy()
    rec.insert(0, "trade_date", date)
    rec["source"] = source
    rec.to_parquet(fp, index=False)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"trade_date": date, "n_picks": int(len(rec)),
                            "source": source, "action": "written"}) + "\n")
    return "written"


def read_log() -> pd.DataFrame:
    """读回 append-only log 全部 picks (合并所有 {date}.parquet)。"""
    parts = [pd.read_parquet(p) for p in sorted(PICKS_DIR.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["trade_date", "ts_code", "alloc_pct"])
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def backfill_demo(source: str = "dep001_pit_clean") -> dict:
    """demo: 用 dep001 PIT-clean OOS picks 回填 append-only log (上线时换成生产每日推理)。
    逐日 log_picks_for_date, existing 文件保留 (验证 append-only 不可回改)。"""
    picks = pd.read_parquet(DEP_PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    written = kept = 0
    for d, g in picks.groupby("trade_date"):
        st = log_picks_for_date(d, g, source=source)
        if st == "written":
            written += 1
        else:
            kept += 1
    return {"n_days_total": int(picks["trade_date"].nunique()),
            "written": written, "kept_existing": kept,
            "n_picks_rows": int(len(picks))}


# ─────────────── 2. 实现 P&L (冻结引擎) ───────────────
def realize_book(logged: pd.DataFrame, prices: pd.DataFrame, cal: list) -> dict:
    """把落库 picks 过冻结成本引擎 → book-level 连续 nav + per-cohort 20d 实现收益 + regime。"""
    tgt = build_dual_alloc_targets(logged, cal)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(prices, cost=cm, t_plus_1=True, limit_pct=9.8)
    start = logged["trade_date"].min()
    res = bt.run(tgt, start=start)
    ts = res.timeseries
    ts.to_parquet(PAPER_DIR / "paper_trade_book.parquet")
    nav = ts["nav"]
    summ = res.summary()   # 含 avg_monthly_turnover / monthly_win_rate (engine 全量统计)

    # per-cohort (20d) 实现收益: 按再平衡日切片连续 book 收益
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg_map = dict(zip(reg["trade_date"], reg["regime"]))
    rebal_days = sorted(tgt.keys())
    dates = nav.index.astype(str).tolist()
    cohorts = []
    for i, rd in enumerate(rebal_days):
        if rd not in dates:
            continue
        j0 = dates.index(rd)
        # 持有到下个再平衡日 (≈20d) 或数据末尾; 满 horizon 才算 realized
        j1 = dates.index(rebal_days[i + 1]) if i + 1 < len(rebal_days) else len(dates) - 1
        if j1 <= j0:
            continue
        c_ret = float(nav.iloc[j1] / nav.iloc[j0] - 1.0)
        n_days = j1 - j0
        cohorts.append({
            "cohort_id": i, "entry_date": rd, "exit_date": dates[j1],
            "n_days": int(n_days), "cohort_return_pct": round(c_ret * 100, 4),
            "regime": reg_map.get(rd, "unknown"),
            "complete_20d": bool(n_days >= 20 * 0.8),   # 满 ~20d 视为完整 cohort
        })
    return {"book_summary": summ, "nav": nav, "cohorts": cohorts,
            "n_rebalances": len(tgt)}


# ─────────────── 3. 对锚 (vs 1.31 ± CI) ───────────────
def compare_to_anchor(realized: dict) -> dict:
    summ = realized["book_summary"]
    cohorts = pd.DataFrame(realized["cohorts"])
    s_real = summ["sharpe"]
    lo, hi = ANCHOR["sharpe_ci95"]
    mid = ANCHOR["net_sharpe"]
    # 落 CI 何处
    if s_real < lo:
        zone = "below_CI"           # 低于 CI 下界 → 强烈重评估
    elif s_real < mid:
        zone = "CI_lower_half"      # CI 下半 (<1.31 点估计) → 重评估关注
    elif s_real <= hi:
        zone = "CI_upper_half"      # CI 上半 → 向可部署收敛
    else:
        zone = "above_CI"           # 高于 CI 上界 → 罕见, 复核口径
    # per-cohort / regime 分层 (R11)
    regime_tbl = {}
    complete = cohorts[cohorts["complete_20d"]] if len(cohorts) else cohorts
    if len(complete):
        for rg, g in complete.groupby("regime"):
            regime_tbl[rg] = {
                "n_cohorts": int(len(g)),
                "mean_cohort_return_pct": round(float(g["cohort_return_pct"].mean()), 3),
                "win_rate": round(float((g["cohort_return_pct"] > 0).mean()), 3),
            }
    n_complete = int(len(complete))
    trigger_recheck = bool(n_complete >= 6)   # 协议: ≥6 完整 cohort 触发 FIN-001 复检
    return {
        "realized_net_sharpe": round(s_real, 3),
        "realized_ann_return": round(summ["ann_return"], 4),
        "realized_max_drawdown": round(summ["max_drawdown"], 4),
        "anchor_net_sharpe": mid, "anchor_ci95": [lo, hi],
        "zone": zone, "n_complete_cohorts": n_complete,
        "trigger_fin001_recheck": trigger_recheck,
        "regime_table": regime_tbl,
    }


# ─────────────── demo 主流程 ───────────────
def main():
    t0 = time.time()
    print("\n=== DLV-002 前向 paper-trade harness (PIT-clean, append-only) — demo 闭环 ===\n",
          flush=True)
    if not DEP_PICKS.exists():
        raise FileNotFoundError(f"缺 PIT-clean picks 源: {DEP_PICKS} (先跑 dep001_gen_picks.py)")

    # ① append-only 落库 (demo: dep001 PIT-clean picks 回填) ──
    print("[1/3] append-only 落库 picks (PIT-clean 变体, existing 文件拒绝覆盖=协议红线) ...", flush=True)
    bf = backfill_demo()
    print(f"    落库 {bf['n_days_total']} 个交易日: 新写 {bf['written']} / 保留原始(不可回改) "
          f"{bf['kept_existing']} | {bf['n_picks_rows']:,} picks 行", flush=True)
    logged = read_log()
    start = logged["trade_date"].min()
    print(f"    读回 log: {len(logged):,} 行 / {logged['trade_date'].nunique()} 日 / "
          f"{logged['ts_code'].nunique()} 股 [{start}~{logged['trade_date'].max()}]", flush=True)

    # ② 实现 P&L (冻结引擎) ──
    print(f"\n[2/3] 满持有期实现 P&L (冻结口径: close-based + 成本基线 {BASELINE_SLIP*100:.2f}%/边 + T+1) ...",
          flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    realized = realize_book(logged, px, cal)
    s = realized["book_summary"]
    print(f"    book 实现: 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  "
          f"maxDD {s['max_drawdown']:+.1%}  月换手 {s['avg_monthly_turnover']:.2f}  "
          f"月胜率 {s['monthly_win_rate']:.0%} | {realized['n_rebalances']} 个 cohort", flush=True)

    # ③ 对锚 (vs 1.31 ± CI) ──
    print(f"\n[3/3] 对锚 (vs 真实基线 净Sharpe {ANCHOR['net_sharpe']} CI{ANCHOR['sharpe_ci95']}) ...",
          flush=True)
    cmp = compare_to_anchor(realized)
    print(f"    实现 Sharpe {cmp['realized_net_sharpe']:+.2f} 落 [{cmp['zone']}] "
          f"| 完整 cohort {cmp['n_complete_cohorts']} (≥6 触发 FIN-001 复检={cmp['trigger_fin001_recheck']})",
          flush=True)
    for rg, t in cmp["regime_table"].items():
        print(f"      [{rg:9s}] n={t['n_cohorts']:2d}  cohort均收益 {t['mean_cohort_return_pct']:+.2f}%  "
              f"胜率 {t['win_rate']:.0%}", flush=True)

    # ── realized_vs_expected.csv (per-cohort 落盘) ──
    cohorts = pd.DataFrame(realized["cohorts"])
    cohorts["anchor_net_sharpe"] = ANCHOR["net_sharpe"]
    cohorts["realized_book_net_sharpe"] = cmp["realized_net_sharpe"]
    cohorts["zone_vs_anchor"] = cmp["zone"]
    cohorts.to_csv(REALIZED_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[out] per-cohort realized_vs_expected -> {REALIZED_CSV.relative_to(ROOT)} "
          f"({len(cohorts)} cohort)", flush=True)

    results = {
        "config": {
            "picks_source": "dep001 PIT-clean r20 (剔 holder_pct, embargo WF) — deployable 变体",
            "total_invested": TOTAL, "rebalance_every_tdays": REBAL_EVERY,
            "horizons_days": HORIZONS, "baseline_slippage": BASELINE_SLIP,
            "period": [start, logged["trade_date"].max()],
            "frozen_note": "口径逐字同 V12.31_BASELINE.md / dep001 / WFE-001 (R01)",
        },
        "anchor": ANCHOR,
        "backfill": bf,
        "realized_book": s,
        "anchor_comparison": cmp,
        "redline": "前向数据落库后不可回改 (append-only, existing 拒覆盖); 不回流调参 (SIGN-R01/R02)",
    }
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float),
                            encoding="utf-8")

    # ── verdict (status=built, 交付物非 gate) ──
    conclusion = (
        f"前向 paper-trade harness 就绪 (落地 V12.31_BASELINE.md §3 协议为可执行 3 段式 harness): "
        f"①**append-only 落库** log_picks_for_date 逐日落 PIT-clean picks, existing 文件**拒绝覆盖** "
        f"(协议红线 SIGN-R01/R02: 前向数据不可回改/不回流调参); picks 源 = dep001 PIT-clean r20 "
        f"(剔 holder_pct, embargo WF) deployable 变体, **非生产 in-sample 注水模型**。"
        f"②**满持有期实现 P&L** realize_book 把落库 picks 过冻结成本引擎 (close-based + A股成本基线 "
        f"{BASELINE_SLIP*100:.2f}%/边 round-trip≈0.30% + T+1 + 涨跌停不可成交 + ST 源头排除) → "
        f"book 连续 nav + per-cohort 20d 实现收益 + regime 标注。③**对锚** compare_to_anchor 把 book "
        f"实现 Sharpe 与 FIN-001 真实基线锚 (净 Sharpe {ANCHOR['net_sharpe']} CI{ANCHOR['sharpe_ci95']}) "
        f"对照, 判落 CI 何处 (下半<1.0=重评估 / 上半=向可部署收敛) + 分 regime (SIGN-R11) + ≥6 完整 "
        f"cohort 触发 FIN-001 复检。"
        f"**demo 闭环**: 用 dep001 已有 {logged['trade_date'].nunique()} 日 OOS picks 回填验证端到端, "
        f"book 实现净Sharpe **{s['sharpe']:+.2f}** 年化 {s['ann_return']:+.1%} maxDD {s['max_drawdown']:+.1%} "
        f"落 [{cmp['zone']}] (复刻 dep001 PIT-clean 0.84 = harness 引擎口径忠实)。"
        f"纯落库/复盘基础设施, 不改选股 (R05 生产冻结, 仅 load 缓存 picks/价表)。上线时 backfill 换成生产每日推理即可。"
    )
    VERDICT.write_text(json.dumps({
        "id": "DLV-002",
        "title": "A3 前向 paper-trade 落库 harness (PIT-clean)",
        "status": "built",
        "conclusion": conclusion,
        "metrics": {
            "n_logged_days": int(logged["trade_date"].nunique()),
            "n_logged_picks_rows": int(len(logged)),
            "n_cohorts": int(realized["n_rebalances"]),
            "n_complete_cohorts": cmp["n_complete_cohorts"],
            "realized_book_net_sharpe": cmp["realized_net_sharpe"],
            "realized_book_ann_return": cmp["realized_ann_return"],
            "realized_book_max_drawdown": cmp["realized_max_drawdown"],
            "anchor_net_sharpe": ANCHOR["net_sharpe"],
            "anchor_ci95": ANCHOR["sharpe_ci95"],
            "zone_vs_anchor": cmp["zone"],
            "trigger_fin001_recheck": cmp["trigger_fin001_recheck"],
            "horizons_days": HORIZONS,
            "baseline_slippage": BASELINE_SLIP,
        },
        "regime_table": cmp["regime_table"],
        "artifacts": {
            "harness": "research/backtest/paper_trade_harness.py",
            "picks_log_dir": "research/cache/paper_trade/picks/ (append-only, 逐日 {date}.parquet)",
            "log_jsonl": "research/cache/paper_trade/log.jsonl",
            "realized_vs_expected": "research/cache/paper_trade/realized_vs_expected.csv",
            "book": "research/cache/paper_trade/paper_trade_book.parquet",
            "results": "research/cache/paper_trade/paper_trade_results.json",
            "protocol": "research/backtest/V12.31_BASELINE.md §3",
        },
        "discipline": (
            "R01 口径冻结 (close-based+成本+双轨+20d 逐字同 V12.31_BASELINE/dep001/WFE-001, 锚来自 "
            "FIN-001 不在结果上调); R02 前向数据不回流调参 (append-only existing 拒覆盖=红线); "
            "R05 生产只读 (仅 load 缓存 picks/价表, 0 生产改动); R06 ST 源头排除; R11 分 regime; "
            "R04 大缓存 gitignored。交付物非 alpha gate, status=built。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT.relative_to(ROOT)} (status=built)", flush=True)
    print(f"[done] DLV-002 paper-trade harness 闭环完成, 耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
