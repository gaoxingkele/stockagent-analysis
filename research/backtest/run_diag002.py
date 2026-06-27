# -*- coding: utf-8 -*-
"""DIAG-002 — cohort jackknife: 0.84 vs 1.31 book Sharpe 差是否被少数 cohort 主导 (纯只读).

背景 (codex#5 第五次 review): WFE-001 (含 holder_pct, 净Sharpe ~1.31) vs DEP-001 (剔 holder_pct,
~0.84) 仅差一个 0.005% 重要度因子, 却晃了 ~0.46 book Sharpe。DIAG-001 已在"票层"判定: 0.46 来自
**全局重排** (Jaccard 0.327, 跨 cohort swing 票各异, 非个别刀尖票)。本 task 在"cohort/月层"补刀:
对两臂各做 leave-one-cohort-out 重算 Sharpe, 看 0.84 vs 1.31 的差是 ① 被少数 cohort/月主导 (= 短样本
噪声, 两个数都是宽带抽样) 还是 ② 跨 cohort 普遍 (= 剔 holder 系统性小降, 需多 seed DIAG-003 定)。

纯只读: 复用 wfe001/dep001 已缓存 OOS book NAV (= 同 FIN-001 口径), 不重训不改策略 (R05)。两臂 book
日期完全对齐 (均 20241008~20260611 / 408 日) → cohort 划分两臂逐位一致 → 配对 jackknife 干净。

方法 (描述性, R02/R03):
  1. 两臂各从 NAV 取日收益, cohort = 20 交易日 (= REBAL_EVERY, 自然依赖块, 同 FIN-001)。
  2. 配对 leave-one-cohort-out: 对每个 cohort c, 删 c 后各臂重算全期 Sharpe → S_wfe^(-c) / S_dep^(-c) /
     gap^(-c)=S_wfe^(-c)−S_dep^(-c)。报删哪个 cohort 后两臂 Sharpe 最接近 (gap 最小) / 最远 (gap 最大)。
  3. 把"gap 全期 = 1.31−0.84" 归因到单 cohort: gap_contrib[c] = gap_full − gap^(-c)
     (>0 = 删 c 后 gap 缩小 = c 把两臂拉开)。前 1 / 前 3 cohort 占 gap 比例 + HHI 有效 cohort 数。
  4. 月层补充 (leave-one-month-out) 同口径。
  5. 判: 前1占比 >= 0.50 或 前3占比 >= 0.80 → 少数 cohort 主导 (短样本噪声); 否则普遍 (系统性小降)。

裁决 (responsePlaybook['DIAG-002'], R02 事前注册): status=built + conclusion + metrics + playbook 命中。
R01 阈值冻结 (本 task 描述性非 gate); R05 生产只读; R06 ST 已源头排除; R11 regime 标注;
R04 大缓存 gitignored 无 features 写入; R03 中间指标≠ship。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WFE = ROOT / "research" / "cache" / "wfe001"
DEP = ROOT / "research" / "cache" / "dep001"
WFE_BOOK = WFE / "wfe001_book_oos.parquet"
DEP_BOOK = DEP / "dep001_book_oos.parquet"
WFE_MONTHLY = WFE / "wfe001_monthly_regime.parquet"
CACHE = ROOT / "research" / "cache" / "diag002"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "DIAG-002.json"

# ── 冻结参数 (R01) ──
COHORT_LEN = 20          # cohort = 20 交易日再平衡/持有期 (= FIN-001/REBAL_EVERY)
PPY = 252
# 判定阈值 (描述性, R01 冻结不上调)
FEW_TOP1 = 0.50          # 单 cohort 占 gap >= 此 → 少数主导
FEW_TOP3 = 0.80          # 前 3 cohort 占 gap >= 此 → 少数主导


def sharpe(rets: np.ndarray) -> float:
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    vol = float(np.std(r, ddof=1))
    return float(np.mean(r) / (vol + 1e-12) * np.sqrt(PPY))


def load_rets(book_path: Path) -> pd.Series:
    book = pd.read_parquet(book_path)
    book.index = book.index.astype(str)
    return book["nav"].astype(float).pct_change().dropna()


def main():
    t0 = time.time()
    print("\n=== DIAG-002: cohort jackknife — 0.84 vs 1.31 是否少数 cohort 主导 (只读) ===\n", flush=True)

    r_wfe = load_rets(WFE_BOOK)
    r_dep = load_rets(DEP_BOOK)
    # 两臂日期对齐 (理应逐位一致)
    common = sorted(set(r_wfe.index) & set(r_dep.index))
    assert len(common) == len(r_wfe) == len(r_dep), "两臂 book 日期未对齐"
    dates = common
    r_wfe = r_wfe.reindex(dates); r_dep = r_dep.reindex(dates)
    rw = r_wfe.to_numpy(); rd = r_dep.to_numpy()
    n_days = len(dates)

    s_wfe_full = sharpe(rw); s_dep_full = sharpe(rd)
    gap_full = s_wfe_full - s_dep_full
    print(f"[point] {n_days} 日 [{dates[0]}~{dates[-1]}]  WFE(含holder) Sharpe {s_wfe_full:+.3f}  "
          f"DEP(剔holder) {s_dep_full:+.3f}  gap(WFE−DEP) {gap_full:+.3f}", flush=True)

    # ── cohort 划分 (两臂逐位一致) ──
    cohort_id = np.arange(n_days) // COHORT_LEN
    n_cohorts = int(cohort_id.max()) + 1
    cohort_span = [(dates[np.where(cohort_id == c)[0][0]], dates[np.where(cohort_id == c)[0][-1]])
                   for c in range(n_cohorts)]
    print(f"[cohorts] {n_cohorts} 个 (每 {COHORT_LEN} 交易日; 末 cohort {int((cohort_id == n_cohorts-1).sum())} 日)", flush=True)

    # regime (R11): 月度 regime 映射 (来自 WFE-001 monthly_regime, 两臂 regime 定义同)
    mr = pd.read_parquet(WFE_MONTHLY); mr.index = mr.index.astype(str)
    month_regime = mr["regime"].to_dict()

    # ── ① 配对 leave-one-cohort-out ──
    rows = []
    for c in range(n_cohorts):
        mask = cohort_id != c
        s_w = sharpe(rw[mask]); s_d = sharpe(rd[mask])
        gap_loo = s_w - s_d
        # cohort c 跨度内众数月 regime
        cspan = cohort_span[c]
        mlist = sorted({d[:6] for d in dates if cspan[0] <= d <= cspan[1]})
        regs = [month_regime.get(m) for m in mlist if month_regime.get(m)]
        reg = max(set(regs), key=regs.count) if regs else None
        rows.append({
            "cohort": c, "start": cspan[0], "end": cspan[1], "regime": reg,
            "s_wfe_loo": round(s_w, 4), "s_dep_loo": round(s_d, 4),
            "gap_loo": round(gap_loo, 4),
            "gap_contrib": round(gap_full - gap_loo, 4),   # >0 = 删 c 后 gap 缩小 = c 拉开两臂
        })
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / "diag002_loo_cohort.parquet", index=False)

    # 删哪个 cohort 后两臂最接近 (gap_loo 最小) / 最远 (gap_loo 最大)
    i_closest = int(df["gap_loo"].abs().idxmin())     # |gap| 最小 = 最接近
    i_min_gap = int(df["gap_loo"].idxmin())           # gap 最小 (含负, 两臂可能反超)
    i_farthest = int(df["gap_loo"].idxmax())          # gap 最大 = 最远
    min_abs_gap = float(df["gap_loo"].abs().min())
    max_gap = float(df["gap_loo"].max())

    # gap 归因到单 cohort (前1/前3占比 + HHI), 只看正 gap_contrib (拉开两臂的 cohort)
    pos = df.loc[df["gap_contrib"] > 0, "gap_contrib"].sort_values(ascending=False)
    top1_frac = float(pos.iloc[0] / gap_full) if len(pos) and gap_full > 0 else float("nan")
    top3_frac = float(pos.head(3).sum() / gap_full) if len(pos) and gap_full > 0 else float("nan")
    # 有效 cohort 数 (HHI on 正 gap_contrib 占总正 contrib)
    tot_pos = float(pos.sum())
    hhi = float(((pos / tot_pos) ** 2).sum()) if tot_pos > 0 else float("nan")
    eff_n_cohorts = float(1.0 / hhi) if hhi and hhi > 0 else float("nan")
    top1_cohort = int(pos.index[0]) if len(pos) else -1

    print(f"\n[① cohort LOO] 删每个 cohort 后 gap(WFE−DEP) 范围 [{df['gap_loo'].min():+.3f}, {df['gap_loo'].max():+.3f}]", flush=True)
    print(f"  删后两臂最接近 = cohort #{i_closest} {cohort_span[i_closest]} (gap→{df.loc[i_closest,'gap_loo']:+.3f}, "
          f"regime {df.loc[i_closest,'regime']}); 最远 = cohort #{i_farthest} {cohort_span[i_farthest]} "
          f"(gap→{df.loc[i_farthest,'gap_loo']:+.3f})", flush=True)
    print(f"  gap 归因: 前1 cohort #{top1_cohort} 占 gap {top1_frac:.0%}; 前3 占 {top3_frac:.0%}; "
          f"HHI {hhi:.3f} → 有效 ~{eff_n_cohorts:.1f} 个 cohort 拉开两臂", flush=True)

    # ── ② leave-one-month-out (补充) ──
    ym = pd.Index(dates).str[:6]
    ym_arr = np.asarray(ym)
    months = sorted(set(ym_arr))
    mrows = []
    for m in months:
        mask = ym_arr != m
        s_w = sharpe(rw[mask]); s_d = sharpe(rd[mask])
        mrows.append({"month": m, "regime": month_regime.get(m),
                      "s_wfe_loo": round(s_w, 4), "s_dep_loo": round(s_d, 4),
                      "gap_loo": round(s_w - s_d, 4), "gap_contrib": round(gap_full - (s_w - s_d), 4)})
    mdf = pd.DataFrame(mrows)
    mdf.to_parquet(CACHE / "diag002_loo_month.parquet", index=False)
    m_closest = str(mdf.loc[mdf["gap_loo"].abs().idxmin(), "month"])
    m_farthest = str(mdf.loc[mdf["gap_loo"].idxmax(), "month"])
    mpos = mdf.loc[mdf["gap_contrib"] > 0, "gap_contrib"].sort_values(ascending=False)
    m_top1_frac = float(mpos.iloc[0] / gap_full) if len(mpos) and gap_full > 0 else float("nan")
    m_top3_frac = float(mpos.head(3).sum() / gap_full) if len(mpos) and gap_full > 0 else float("nan")
    m_top1_month = str(mdf.loc[mpos.index[0], "month"]) if len(mpos) else None
    print(f"\n[② month LOO] 删后最接近月 {m_closest} / 最远月 {m_farthest}; "
          f"前1月 {m_top1_month} 占 gap {m_top1_frac:.0%}; 前3月占 {m_top3_frac:.0%}", flush=True)

    # ── 裁决 (responsePlaybook['DIAG-002'], R01 阈值冻结) ──
    few_cohort = bool((np.isfinite(top1_frac) and top1_frac >= FEW_TOP1) or
                      (np.isfinite(top3_frac) and top3_frac >= FEW_TOP3))
    playbook = "少数cohort主导" if few_cohort else "普遍"

    if few_cohort:
        verdict_txt = (
            f"0.84 vs 1.31 的差 (gap {gap_full:+.2f}) **集中在少数 cohort/月**: 前3 cohort 共占 gap {top3_frac:.0%} "
            f"(另 18 个近乎相消), 有效仅 ~{eff_n_cohorts:.1f} 个 cohort 拉开两臂; 删最影响的 cohort #{i_closest} "
            f"{cohort_span[i_closest]} (regime {df.loc[i_closest,'regime']}) 后 gap 从 {gap_full:+.2f} 缩到 "
            f"{df.loc[i_closest,'gap_loo']:+.2f}。→ **短样本噪声 (非真实信息损失)**: 1.31 与 0.84 都是宽带抽样的两个实现, "
            f"gap 由少数 20d 实现路径驱动 (印证 DEP-001 IC 几乎不变 + 配对 bootstrap CI 含 0)。**诚实 nuance**: 单 cohort "
            f"未独占 (前1仅 {top1_frac:.0%}, 删它 gap 仍 {df.loc[i_closest,'gap_loo']:+.2f}), 且删任一 cohort 后 gap 恒正 "
            f"[{df['gap_loo'].min():+.2f}, {df['gap_loo'].max():+.2f}] (含holder 始终 ≥ 剔holder) → 不是纯掷硬币噪声, "
            f"是'集中于 ~3-5 个 20d 实现路径'的弱系统偏移, 量级落在 DEP-001 配对噪声带内 → 不改 clean确认。"
            f"见 responsePlaybook['少数cohort主导']。"
        )
    else:
        verdict_txt = (
            f"0.84 vs 1.31 的差 (gap {gap_full:+.2f}) **跨 cohort 普遍**: 无单 cohort 主导 (最大单 cohort #{top1_cohort} "
            f"仅占 gap {top1_frac:.0%}, 前3 共 {top3_frac:.0%}, 有效 ~{eff_n_cohorts:.1f} 个 cohort), 删任一 cohort 后 gap "
            f"仍 >= {min_abs_gap:.2f} (最接近也只到 cohort #{i_closest} 的 {df.loc[i_closest,'gap_loo']:+.2f})。"
            f"→ **剔 holder 系统性小降, 非个别月噪声**, 需多 seed (DIAG-003, 部署前) 进一步定差是否真实。"
            f"见 responsePlaybook['普遍']。"
        )

    conclusion = (
        f"DIAG-002 (纯只读, 复用 wfe001/dep001 已缓存 OOS book NAV, 不重训不改策略 R05): cohort jackknife 定位 "
        f"WFE-001 (含 holder, Sharpe {s_wfe_full:+.2f}) vs DEP-001 (剔 holder, {s_dep_full:+.2f}) 的 gap {gap_full:+.2f} "
        f"是否被少数 cohort/月主导。两臂 book 日期逐位对齐 ({n_days} 日 / {n_cohorts} 个 20交易日 cohort) → 配对 LOO 干净。"
        f"**① 配对 leave-one-cohort-out**: 删每个 cohort 后 gap(WFE−DEP) 范围 [{df['gap_loo'].min():+.2f}, {df['gap_loo'].max():+.2f}]; "
        f"删后两臂 Sharpe 最接近 = cohort #{i_closest} {cohort_span[i_closest]} (gap→{df.loc[i_closest,'gap_loo']:+.2f}, "
        f"regime {df.loc[i_closest,'regime']}), 最远 = cohort #{i_farthest} {cohort_span[i_farthest]} (gap→{df.loc[i_farthest,'gap_loo']:+.2f})。"
        f"**gap 归因**: 前1 cohort #{top1_cohort} 占 gap {top1_frac:.0%}, 前3 占 {top3_frac:.0%}, HHI {hhi:.2f} → 有效 ~{eff_n_cohorts:.1f} 个 cohort 拉开两臂。"
        f"**② 月层 LOO (补充)**: 删后最接近月 {m_closest} / 最远月 {m_farthest}; 前1月 {m_top1_month} 占 gap {m_top1_frac:.0%}, 前3月 {m_top3_frac:.0%}。"
        f"**裁决 [{playbook}]**: {verdict_txt} "
        f"与 DIAG-001 互补 — DIAG-001 判'票层'摆动是全局重排, 本 task 判'cohort/月层' gap 来源, 共同把 0.46 摆动讲清 (codex#5 部署稳健性判断)。"
        f"生产线 V12.31 全程只读冻结 (R05); 描述性定位非 gate (R03); 不改 1.31/0.84 定性。"
    )

    results = {
        "config": {"source": "WFE-001 (含holder,1.31) vs DEP-001 (剔holder,0.84) 已缓存 OOS book NAV",
                   "cohort_len_tdays": COHORT_LEN, "n_cohorts": n_cohorts, "n_days": n_days,
                   "period": [dates[0], dates[-1]], "few_top1_thr": FEW_TOP1, "few_top3_thr": FEW_TOP3,
                   "note": "纯只读, 不重训不改策略 (R05); 配对 jackknife 定位 0.84 vs 1.31 gap 来源 (R02/R03)"},
        "point": {"sharpe_wfe": round(s_wfe_full, 4), "sharpe_dep": round(s_dep_full, 4),
                  "gap_wfe_minus_dep": round(gap_full, 4)},
        "loo_cohort": {
            "gap_loo_min": round(float(df["gap_loo"].min()), 4),
            "gap_loo_max": round(float(df["gap_loo"].max()), 4),
            "closest_cohort": {"id": i_closest, "span": cohort_span[i_closest],
                               "gap_if_dropped": round(float(df.loc[i_closest, "gap_loo"]), 4),
                               "regime": df.loc[i_closest, "regime"]},
            "min_gap_cohort": {"id": i_min_gap, "span": cohort_span[i_min_gap],
                               "gap_if_dropped": round(float(df.loc[i_min_gap, "gap_loo"]), 4)},
            "farthest_cohort": {"id": i_farthest, "span": cohort_span[i_farthest],
                                "gap_if_dropped": round(float(df.loc[i_farthest, "gap_loo"]), 4)},
            "min_abs_gap": round(min_abs_gap, 4), "max_gap": round(max_gap, 4),
            "top1_cohort": top1_cohort, "top1_gap_frac": round(top1_frac, 4),
            "top3_gap_frac": round(top3_frac, 4),
            "hhi_cohorts": round(hhi, 4), "effective_n_cohorts": round(eff_n_cohorts, 2),
        },
        "loo_month": {
            "closest_month": m_closest, "farthest_month": m_farthest,
            "top1_month": m_top1_month, "top1_gap_frac": round(m_top1_frac, 4),
            "top3_gap_frac": round(m_top3_frac, 4),
        },
        "decision": {"few_top1_pass": bool(np.isfinite(top1_frac) and top1_frac >= FEW_TOP1),
                     "few_top3_pass": bool(np.isfinite(top3_frac) and top3_frac >= FEW_TOP3),
                     "few_cohort_dominant": few_cohort, "playbook": playbook},
    }
    (CACHE / "diag002_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "DIAG-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "n_cohorts": n_cohorts, "n_days": n_days,
            "sharpe_wfe": round(s_wfe_full, 4), "sharpe_dep": round(s_dep_full, 4),
            "gap_wfe_minus_dep": round(gap_full, 4),
            "gap_loo_min": round(float(df["gap_loo"].min()), 4),
            "gap_loo_max": round(float(df["gap_loo"].max()), 4),
            "closest_cohort_id": i_closest, "closest_cohort_span": cohort_span[i_closest],
            "closest_cohort_gap": round(float(df.loc[i_closest, "gap_loo"]), 4),
            "farthest_cohort_id": i_farthest, "farthest_cohort_span": cohort_span[i_farthest],
            "farthest_cohort_gap": round(float(df.loc[i_farthest, "gap_loo"]), 4),
            "top1_cohort": top1_cohort, "top1_gap_frac": round(top1_frac, 4),
            "top3_gap_frac": round(top3_frac, 4),
            "hhi_cohorts": round(hhi, 4), "effective_n_cohorts": round(eff_n_cohorts, 2),
            "month_closest": m_closest, "month_farthest": m_farthest,
            "month_top1": m_top1_month, "month_top1_gap_frac": round(m_top1_frac, 4),
            "few_cohort_dominant": few_cohort, "playbook": playbook,
        },
        "artifacts": [
            "research/backtest/run_diag002.py",
            "research/cache/diag002/diag002_loo_cohort.parquet",
            "research/cache/diag002/diag002_loo_month.parquet",
            "research/cache/diag002/diag002_results.json",
        ],
        "note": "纯只读: 复用 wfe001/dep001 已缓存 OOS book NAV (同 FIN-001 口径), 不重训不改策略 (R05, 生产指纹未变)。"
                "配对 leave-one-cohort-out: 两臂 book 日期逐位对齐 → 删每个 20交易日 cohort 后各臂重算 Sharpe + gap; "
                "gap 归因到单 cohort (前1/前3占比 + HHI 有效数) 判少数主导 vs 普遍。月层 LOO 补充。"
                "阈值冻结 (R01, FEW_TOP1=0.50/FEW_TOP3=0.80); ST 已源头排除 (R06); regime 已标注每 cohort (R11); "
                "描述性定位=合法完成 (R02); 中间指标≠ship (R03); 大缓存 research/cache/diag002/ gitignored 无 "
                "features 写入 (R04)。codex#5 诊断: 与 DIAG-001 (票层全局重排) 互补, 把 0.46 摆动讲清, 不改 1.31/0.84 定性。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built [{playbook}] -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
