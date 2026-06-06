# -*- coding: utf-8 -*-
"""SL-002 — 成本 × 持有期 × 容量 敏感性 (Arm_A 席位战绩排序).

承接 SL-001: r5 龙虎榜席位事件 sleeve 引擎已建, 主测 (H=5/N=10/30bps往返) 净/笔 Arm_A=-1.73%,
年化换手≈50轮成本拖累≈15%/yr。**短线 sleeve 成败常在成本/容量** — 本任务在 gate (SL-003) 之前
先把网格看清:

  1. 成本 × 持有期网格: cost ∈ {0, 15, 30}bps 往返 × H ∈ {3, 5, 10} 交易日,
     报告 Arm_A 各组合 net 月度收益 / Sharpe / 年化换手。
  2. 容量估算: 基于上榜股 D+1(event_date) 成交额, 在"不冲击≤当日成交额 1%"假设下,
     估 top-N 等权组合可容纳资金 (min-binding=等权下最小成交额股约束, sum=松上界)。

纪律:
  - SIGN-R03 中间指标 ≠ 落地: 本 task 仅 status=built (敏感性描述), 净 α walk-forward 裁决在 SL-003。
  - SIGN-R01 冻结: 复用 SL-001 引擎/常数 (SEAT_SORT/TOP_N/MOM_WIN/RET_CLIP), 仅扫 cost/H。
  - SIGN-R06 ST 源头排除 (引擎内已做); SIGN-R08 picks 按 H checkpoint。
  - 月度收益口径 = "事件等权": 当月所有 entry-day Arm_A 篮子 net 收益均值 (split capital across 当月事件);
    Sharpe = 月度序列 mean/std × sqrt(12) (年化), 明确标注口径, 不是可复利净值曲线。

输出:
  research/cache/sl001/picks_H{H}.parquet     (各 H 的 picks, checkpoint)
  research/cache/sl002_grid.parquet           (cost×H×指标 长表)
  research/verdicts/SL-002.json               (status=built + 敏感性表 + 容量估算)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import sl001_sleeve_engine as eng  # noqa: E402  复用引擎 (载体/常数/因果一致)

CACHE = ROOT / "research" / "cache"
SL_CACHE = CACHE / "sl001"
GRID_OUT = CACHE / "sl002_grid.parquet"
VERDICT = ROOT / "research" / "verdicts" / "SL-002.json"

# ── 事前注册网格 (冻结 SIGN-R01) ──
COST_GRID_BPS = [0, 15, 30]      # round-trip bps
H_GRID = [3, 5, 10]              # 持有交易日
TOP_N = eng.TOP_N                # 10, 复用
CAP_FRAC = 0.01                  # 容量假设: 单股 ≤ 当日成交额 1% (不冲击)
AMOUNT_UNIT_YUAN = 1000.0        # Tushare daily amount 单位 = 千元 → ×1000 = 元


def picks_for_H(px, cand, sf, cal, H) -> pd.DataFrame:
    """各 H 的 stock 级 picks, 带 checkpoint (SIGN-R08)。"""
    cp = SL_CACHE / f"picks_H{H}.parquet"
    if cp.exists():
        print(f"[picks H={H}] checkpoint 命中 {cp.name}", flush=True)
        return pd.read_parquet(cp)
    pk = eng.build_picks(px, cand, sf, cal, H)
    pk.to_parquet(cp, index=False)
    print(f"[picks H={H}] 落盘 {cp.name} ({len(pk):,} 行, 席位可信 {int(pk['has_seat'].sum()):,})",
          flush=True)
    return pk


def arm_A_gross(picks: pd.DataFrame) -> pd.DataFrame:
    """逐 entry_date Arm_A 篮子毛收益 (cost=0; net 在分析层按 cost 扣)。"""
    pnl = eng.run_arms(picks, TOP_N, cost_rt=0.0)
    a = pnl[pnl["arm"] == "A"][["entry_date", "month", "gross", "n_stocks"]].copy()
    return a.reset_index(drop=True)


def grid_metrics(a: pd.DataFrame, H: int, cost_bps: float) -> dict:
    """给定 Arm_A 毛收益序列 + cost + H → net 月度收益/Sharpe/换手。"""
    cost_rt = cost_bps / 10000.0
    net = a["gross"].to_numpy() - cost_rt
    # 月度: 当月所有 entry-day 篮子 net 收益均值 (事件等权)
    tmp = a[["month"]].copy()
    tmp["net"] = net
    monthly = tmp.groupby("month")["net"].mean()
    m = monthly.to_numpy()
    mean_m = float(m.mean())
    std_m = float(m.std(ddof=1)) if len(m) > 1 else float("nan")
    sharpe_ann = (mean_m / std_m * np.sqrt(12.0)) if std_m and std_m > 0 else float("nan")
    roundtrips = 252.0 / H
    return {
        "H": H,
        "cost_bps_roundtrip": int(cost_bps),
        "n_event_days": int(len(a)),
        "n_months": int(len(monthly)),
        "net_mean_per_trade_pct": round(float(net.mean()) * 100, 4),
        "net_monthly_mean_pct": round(mean_m * 100, 4),
        "net_monthly_std_pct": round(std_m * 100, 4) if std_m == std_m else None,
        "sharpe_ann": round(sharpe_ann, 3) if sharpe_ann == sharpe_ann else None,
        "pos_month_ratio": round(float((m > 0).mean()), 4),
        "ann_roundtrips_full_invest": round(roundtrips, 1),
        "ann_cost_drag_pct": round(roundtrips * cost_rt * 100, 2),
    }


def capacity_estimate(picks_H5: pd.DataFrame) -> dict:
    """容量: 基于上榜股 event_date 成交额, ≤当日成交额 1% 不冲击假设。

    等权 top-N: 每股配 capital/N, 约束 capital/N ≤ 1%·amount_i ∀i → capital ≤ N·min_i(1%·amount_i)。
    报告 min-binding(等权硬约束) 与 sum(松上界=各股各吃 1%) 两口径, 按 entry_date 取分位。
    """
    a = eng.run_arms  # noqa  (仅复用排序逻辑下面手算 Arm_A 选股)
    seat = picks_H5[picks_H5["has_seat"]].copy()
    seat = seat.dropna(subset=["entry_amount"])
    per_day = []
    for ed, g in seat.groupby("entry_date"):
        sel = g.sort_values([eng.SEAT_SORT, eng.SEAT_TIE], ascending=False).head(TOP_N)
        amt_yuan = sel["entry_amount"].to_numpy() * AMOUNT_UNIT_YUAN  # 千元→元
        n = len(sel)
        if n == 0:
            continue
        cap_per_stock = CAP_FRAC * amt_yuan
        per_day.append({
            "entry_date": ed,
            "n": n,
            "cap_sum_yuan": float(cap_per_stock.sum()),          # 松: 各股各吃 1%
            "cap_minbind_yuan": float(n * cap_per_stock.min()),  # 紧: 等权受最小成交额股约束
        })
    cap = pd.DataFrame(per_day)
    q = lambda col, p: float(np.quantile(cap[col], p))  # noqa: E731
    yi = 1e8  # 亿元
    return {
        "assumption": f"单股建仓 ≤ event_date 成交额 {CAP_FRAC:.0%}; 等权 top-{TOP_N}; 无冲击",
        "amount_unit": "Tushare daily amount=千元, ×1000=元",
        "n_event_days_with_amount": int(len(cap)),
        "cap_minbind_yiyuan": {
            "median": round(q("cap_minbind_yuan", 0.5) / yi, 3),
            "p25": round(q("cap_minbind_yuan", 0.25) / yi, 3),
            "p75": round(q("cap_minbind_yuan", 0.75) / yi, 3),
        },
        "cap_sum_yiyuan": {
            "median": round(q("cap_sum_yuan", 0.5) / yi, 3),
            "p25": round(q("cap_sum_yuan", 0.25) / yi, 3),
            "p75": round(q("cap_sum_yuan", 0.75) / yi, 3),
        },
        "note": ("min-binding=等权下可部署资金(受当日最小成交额成分股约束); sum=各成分股各吃 1% 的松上界。"
                 "短线高换手日内反复进出会进一步压缩有效容量(此为单次建仓静态估算)。"),
    }


def main() -> int:
    t0 = time.time()
    SL_CACHE.mkdir(parents=True, exist_ok=True)
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== SL-002 成本 × 持有期 × 容量 敏感性 ===\n", flush=True)

    st = eng.load_st_set()
    print(f"[st] ST 源头排除 {len(st)} 只", flush=True)
    px = eng.load_prices(st)
    cal = sorted(px["trade_date"].unique())
    print(f"[cal] {len(cal)} 交易日 ({cal[0]}~{cal[-1]})", flush=True)
    cand = eng.load_candidates(st, set(px["ts_code"].unique()))
    sf = pd.read_parquet(eng.SF_P)
    sf["trade_date"] = sf["trade_date"].astype(str)

    # ── 成本 × 持有期网格 ──
    rows = []
    picks_by_H = {}
    for H in H_GRID:
        pk = picks_for_H(px, cand, sf, cal, H)
        picks_by_H[H] = pk
        a = arm_A_gross(pk)
        print(f"[H={H}] Arm_A entry 日 {len(a)} / 月 {a['month'].nunique()} / "
              f"avg持仓 {a['n_stocks'].mean():.1f}", flush=True)
        for cost_bps in COST_GRID_BPS:
            rows.append(grid_metrics(a, H, cost_bps))

    grid = pd.DataFrame(rows)
    grid.to_parquet(GRID_OUT, index=False)
    print(f"[grid] -> {GRID_OUT.relative_to(ROOT)} ({len(grid)} cells)", flush=True)

    # ── 打印网格 ──
    print("\n=== Arm_A 净敏感性 (cost round-trip × 持有期 H) ===", flush=True)
    print(f"  {'H':>3s} {'cost_bps':>9s} {'net/笔%':>9s} {'净月度%':>9s} "
          f"{'Sharpe(年)':>10s} {'正月%':>7s} {'年换手':>7s} {'成本拖累%/yr':>12s}", flush=True)
    for _, r in grid.iterrows():
        print(f"  {int(r['H']):>3d} {int(r['cost_bps_roundtrip']):>9d} "
              f"{r['net_mean_per_trade_pct']:>+9.4f} {r['net_monthly_mean_pct']:>+9.4f} "
              f"{(r['sharpe_ann'] if r['sharpe_ann'] is not None else float('nan')):>10.3f} "
              f"{r['pos_month_ratio']:>7.1%} {r['ann_roundtrips_full_invest']:>7.1f} "
              f"{r['ann_cost_drag_pct']:>12.2f}", flush=True)

    # ── 容量 (基于主测 H=5 picks) ──
    cap = capacity_estimate(picks_by_H[5])
    print("\n=== 容量估算 (≤当日成交额 1%, 等权 top-N) ===", flush=True)
    print(f"  min-binding 中位 {cap['cap_minbind_yiyuan']['median']} 亿元 "
          f"(p25={cap['cap_minbind_yiyuan']['p25']} / p75={cap['cap_minbind_yiyuan']['p75']})", flush=True)
    print(f"  sum(松上界)  中位 {cap['cap_sum_yiyuan']['median']} 亿元 "
          f"(p25={cap['cap_sum_yiyuan']['p25']} / p75={cap['cap_sum_yiyuan']['p75']})", flush=True)

    # 主测 cell (H=5, 30bps) 摘要供 verdict
    main_cell = grid[(grid["H"] == 5) & (grid["cost_bps_roundtrip"] == 30)].iloc[0].to_dict()

    verdict = {
        "id": "SL-002", "status": "built",
        "conclusion": (
            f"成本×持有期×容量敏感性已建。Arm_A 各档 net 月度收益/Sharpe/换手见网格 "
            f"({len(grid)} cells: cost{COST_GRID_BPS}bps × H{H_GRID})。"
            f"主测 (H=5/30bps往返): net/笔 {main_cell['net_mean_per_trade_pct']:+.3f}%, "
            f"净月度 {main_cell['net_monthly_mean_pct']:+.3f}%, Sharpe(年) {main_cell['sharpe_ann']}, "
            f"年换手≈{main_cell['ann_roundtrips_full_invest']}轮→成本拖累≈{main_cell['ann_cost_drag_pct']}%/yr。"
            f"持有期越短换手/成本拖累越高 (H=3→{252/3:.0f}轮); 成本从 0→30bps 单调侵蚀。"
            f"容量(≤成交额1%/等权): min-binding 中位 {cap['cap_minbind_yiyuan']['median']} 亿元。"
            f"龙虎榜净买入股 r5 原始收益本就为负 (SL-001), 净月度多为负属预期 — "
            f"净 α(Arm_A−等权市场) 与席位技能净增量(Arm_A−Arm_B) walk-forward 裁决在 SL-003 (SIGN-R03)。"),
        "artifact": str(GRID_OUT.relative_to(ROOT)),
        "design": {
            "grid": {"cost_bps_roundtrip": COST_GRID_BPS, "H_hold_days": H_GRID, "top_n": TOP_N},
            "monthly_convention": "事件等权: 当月所有 entry-day Arm_A 篮子 net 收益均值; Sharpe=月序列 mean/std×√12",
            "turnover": "年化换手=252/H 满仓连续轮次; 成本拖累=换手×cost_rt",
            "capacity_assumption": cap["assumption"],
            "frozen": "复用 SL-001 引擎/常数 (SEAT_SORT/TOP_N/MOM_WIN/RET_CLIP), 仅扫 cost/H (SIGN-R01)",
        },
        "metrics": {
            "grid": grid.to_dict(orient="records"),
            "main_cell_H5_30bps": main_cell,
            "capacity": cap,
        },
        "artifacts": [str(GRID_OUT.relative_to(ROOT))] +
                     [f"research/cache/sl001/picks_H{H}.parquet" for H in H_GRID],
        "guardrails": ["SIGN-R03 敏感性描述≠落地, 净α裁决在 SL-003",
                       "SIGN-R01 冻结网格复用引擎常数", "SIGN-R06 ST 源头排除(引擎内)",
                       "SIGN-R08 picks 按 H checkpoint"],
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status=built -> {VERDICT.relative_to(ROOT)}", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
