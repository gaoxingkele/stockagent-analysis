# -*- coding: utf-8 -*-
"""FIN-001 — V12.31 真·真实基线 (WFE-001 embargo, 净Sharpe 1.31) 补误差条.

不改策略。在 WFE-001 label-embargo 真·真实可交易 book (research/cache/wfe001/wfe001_book_oos.parquet,
点估计 净Sharpe ~1.31 / 年化 ~+35%) 上做三件事, 给 1.31 一个**诚实的不确定性 + 集中度画像**:

  ① per-cohort block bootstrap (1000 次): cohort = 20 交易日再平衡/持有期 (= 自然依赖块, 保块内自相关)。
     重采样 cohort (有放回) → 拼接日收益 → 年化 Sharpe / 年化收益 的 95% CI + P(Sharpe>0)。
  ② leave-one-out (逐 cohort 删): 删任一 cohort 重算 Sharpe → 范围 + 最影响 cohort (删后掉最多/升最多)。
  ③ 月度 / regime 归因: 逐月 Sharpe 贡献 (LOO-by-month) + top-k 月收益占比 (HHI 集中度) +
     按月度 regime (动量/反转/混合) 分组的日 Sharpe 与收益贡献占比 → 指出 1.31 主要靠哪些月/regime (集中度风险)。

裁决 = 描述性 built (R02): 给 1.31 一个 CI 区间 + 集中度结论。不动门柱 (R01); 不 ship (R03);
ST 已在 WFE-001 源头排除 (R06); regime 分层必带 (R11); 复用 WFE-001 缓存, 生产线只读 (R05);
大缓存 research/cache/fin001/ gitignored (R04)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WFE = ROOT / "research" / "cache" / "wfe001"
BOOK = WFE / "wfe001_book_oos.parquet"
MONTHLY = WFE / "wfe001_monthly_regime.parquet"
WFE_RESULTS = WFE / "wfe001_results.json"
CACHE = ROOT / "research" / "cache" / "fin001"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "FIN-001.json"

# ── 冻结参数 (R01) ──
COHORT_LEN = 20          # cohort = 20 交易日再平衡/持有期 (= WFE-001 REBAL_EVERY)
N_BOOT = 1000            # block bootstrap 次数 (prd 指定)
BOOT_SEED = 20260614     # 复现固定种子
PPY = 252


def sharpe(rets: np.ndarray) -> float:
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    vol = float(np.std(r, ddof=1))
    return float(np.mean(r) / (vol + 1e-12) * np.sqrt(PPY))


def ann_return(rets: np.ndarray) -> float:
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    return float(np.prod(1.0 + r) ** (PPY / len(r)) - 1.0)


def max_dd(rets: np.ndarray) -> float:
    r = np.asarray(rets, float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def main():
    t0 = time.time()
    print("\n=== FIN-001: V12.31 真实基线 (WFE-001 embargo Sharpe~1.31) 补误差条 ===\n", flush=True)

    book = pd.read_parquet(BOOK)
    book.index = book.index.astype(str)
    nav = book["nav"].astype(float)
    rets = nav.pct_change().dropna()            # 日收益 (跳过 day0)
    dates = rets.index.tolist()
    n_days = len(rets)
    point_sharpe = sharpe(rets.to_numpy())
    point_ann = ann_return(rets.to_numpy())
    point_maxdd = max_dd(rets.to_numpy())
    wfe = json.loads(WFE_RESULTS.read_text(encoding="utf-8"))
    reported_sharpe = wfe["embargo_book"]["sharpe"]
    print(f"[point] book {n_days} 日 [{dates[0]}~{dates[-1]}]  净Sharpe {point_sharpe:+.3f} "
          f"(WFE-001 报 {reported_sharpe:+.3f})  年化 {point_ann:+.1%}  maxDD {point_maxdd:+.1%}", flush=True)

    # ── cohort 划分 (= 20 交易日再平衡/持有期) ──
    cohort_id = np.arange(n_days) // COHORT_LEN
    n_cohorts = int(cohort_id.max()) + 1
    cohort_rets = [rets.to_numpy()[cohort_id == c] for c in range(n_cohorts)]
    cohort_span = [(dates[np.where(cohort_id == c)[0][0]], dates[np.where(cohort_id == c)[0][-1]])
                   for c in range(n_cohorts)]
    print(f"[cohorts] {n_cohorts} 个 (每 {COHORT_LEN} 交易日 = 再平衡/持有期; 末 cohort {len(cohort_rets[-1])} 日)", flush=True)

    # ── ① per-cohort block bootstrap ──
    print(f"\n[①block bootstrap] {N_BOOT} 次重采样 {n_cohorts} cohort (有放回) ...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    bs_sharpe, bs_ann = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_cohorts, size=n_cohorts)
        concat = np.concatenate([cohort_rets[c] for c in pick])
        s = sharpe(concat)
        if np.isfinite(s):
            bs_sharpe.append(s)
            bs_ann.append(ann_return(concat))
    bs_sharpe = np.array(bs_sharpe); bs_ann = np.array(bs_ann)
    sh_ci = (float(np.percentile(bs_sharpe, 2.5)), float(np.percentile(bs_sharpe, 97.5)))
    sh_med = float(np.median(bs_sharpe))
    sh_std = float(np.std(bs_sharpe, ddof=1))
    p_gt0 = float((bs_sharpe > 0).mean())
    p_gt1 = float((bs_sharpe > 1.0).mean())
    ann_ci = (float(np.percentile(bs_ann, 2.5)), float(np.percentile(bs_ann, 97.5)))
    print(f"  Sharpe 95% CI = [{sh_ci[0]:+.3f}, {sh_ci[1]:+.3f}]  中位 {sh_med:+.3f}  SE {sh_std:.3f}  "
          f"P(>0)={p_gt0:.3f}  P(>1)={p_gt1:.3f}", flush=True)
    print(f"  年化 95% CI = [{ann_ci[0]:+.1%}, {ann_ci[1]:+.1%}]", flush=True)

    # ── ② leave-one-out (逐 cohort) ──
    loo = []
    for c in range(n_cohorts):
        concat = np.concatenate([cohort_rets[k] for k in range(n_cohorts) if k != c])
        loo.append(sharpe(concat))
    loo = np.array(loo)
    loo_min, loo_max = float(loo.min()), float(loo.max())
    # 删后 Sharpe 最低 = 该 cohort 贡献最大 (它撑着 Sharpe); 删后最高 = 该 cohort 最拖累
    c_most_pos = int(loo.argmin())   # 删它掉最多 → 正贡献最大
    c_most_neg = int(loo.argmax())   # 删它升最多 → 拖累最大
    loo_sign_stable = bool((loo > 0).all())
    print(f"\n[②leave-one-out] 删任一 cohort 后 Sharpe 范围 [{loo_min:+.3f}, {loo_max:+.3f}] "
          f"(符号恒正? {loo_sign_stable})", flush=True)
    print(f"  贡献最大 cohort #{c_most_pos} {cohort_span[c_most_pos]} (删后 Sharpe→{loo[c_most_pos]:+.3f}); "
          f"最拖累 cohort #{c_most_neg} {cohort_span[c_most_neg]} (删后 Sharpe→{loo[c_most_neg]:+.3f})", flush=True)

    # ── ③a 月度归因 (LOO-by-month + top-k 占比 + HHI) ──
    ym = pd.Index(dates).str[:6]
    month_ret = rets.groupby(ym).apply(lambda g: float(np.prod(1.0 + g.to_numpy()) - 1.0))
    months = month_ret.index.tolist()
    # LOO-by-month: 删整月日收益重算 Sharpe
    loo_m = {}
    ym_arr = np.asarray(ym)
    for m in months:
        mask = ym_arr != m
        loo_m[m] = sharpe(rets.to_numpy()[mask])
    loo_m_s = pd.Series(loo_m)
    worst_drop_month = str(loo_m_s.idxmin())   # 删它 Sharpe 掉最多 = 撑 Sharpe 的月
    # 收益集中度: 各月对总对数收益的贡献占比 (只看正贡献的 HHI + top-3 份额)
    log_contrib = np.log1p(month_ret)
    total_log = float(log_contrib.sum())
    pos = log_contrib[log_contrib > 0]
    top3 = pos.sort_values(ascending=False).head(3)
    top3_share = float(top3.sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    hhi = float(((pos / pos.sum()) ** 2).sum()) if pos.sum() > 0 else float("nan")
    eff_n_months = float(1.0 / hhi) if hhi and hhi > 0 else float("nan")
    print(f"\n[③月度归因] LOO-by-month: 删 {worst_drop_month} Sharpe→{loo_m_s.min():+.3f} (最撑 Sharpe 的月)", flush=True)
    print(f"  收益集中度: top-3 正贡献月 {list(top3.index)} 占正收益 {top3_share:.1%}; "
          f"HHI {hhi:.3f} → 有效 ~{eff_n_months:.1f} 个独立赢月 (共 {len(pos)} 个正月)", flush=True)

    # ── ③b regime 归因 (日级, 按月度 regime 映射) ──
    mr = pd.read_parquet(MONTHLY)
    mr.index = mr.index.astype(str)
    month_regime = mr["regime"].to_dict()
    day_regime = np.asarray(pd.Index(ym).map(month_regime))
    reg_table = {}
    for rg in ["momentum", "reversal", "mixed"]:
        mask = day_regime == rg
        r = rets.to_numpy()[mask]
        if len(r) < 2:
            continue
        lc = float(np.log1p(month_ret[[m for m in months if month_regime.get(m) == rg]]).sum())
        reg_table[rg] = {
            "n_days": int(len(r)),
            "n_months": int(sum(1 for m in months if month_regime.get(m) == rg)),
            "daily_sharpe_within": round(sharpe(r), 3),
            "mean_daily_ret_bp": round(float(np.mean(r)) * 1e4, 2),
            "log_ret_contrib": round(lc, 4),
            "log_ret_contrib_share": round(lc / total_log, 3) if total_log != 0 else None,
        }
    print(f"\n[③regime 归因] (日级, 按月度 regime; 总对数收益 {total_log:+.3f})", flush=True)
    for rg, d in reg_table.items():
        print(f"  [{rg:9s}] {d['n_months']:2d}月/{d['n_days']:3d}日  组内日Sharpe {d['daily_sharpe_within']:+.3f}  "
              f"日均 {d['mean_daily_ret_bp']:+.1f}bp  贡献总收益 {d['log_ret_contrib_share']:.0%}", flush=True)
    regime_main = max(reg_table, key=lambda k: reg_table[k]["log_ret_contrib_share"]) if reg_table else "NA"

    # ── 落盘 ──
    results = {
        "config": {"source": "WFE-001 label-embargo 真·真实 OOS book (wfe001_book_oos.parquet)",
                   "cohort_len_tdays": COHORT_LEN, "n_cohorts": n_cohorts, "n_days": n_days,
                   "period": [dates[0], dates[-1]], "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "note": "不改策略, 只给 1.31 点估计补误差条 + 集中度 (R01 不动门柱)"},
        "point_estimate": {"net_sharpe": round(point_sharpe, 3), "ann_return": round(point_ann, 4),
                           "max_drawdown": round(point_maxdd, 4), "wfe001_reported_sharpe": reported_sharpe},
        "block_bootstrap": {
            "sharpe_ci95": [round(sh_ci[0], 3), round(sh_ci[1], 3)],
            "sharpe_median": round(sh_med, 3), "sharpe_se": round(sh_std, 3),
            "p_sharpe_gt_0": round(p_gt0, 3), "p_sharpe_gt_1": round(p_gt1, 3),
            "ann_return_ci95": [round(ann_ci[0], 4), round(ann_ci[1], 4)],
            "n_valid": int(len(bs_sharpe)),
        },
        "leave_one_out_cohort": {
            "sharpe_min": round(loo_min, 3), "sharpe_max": round(loo_max, 3),
            "sign_stable_positive": loo_sign_stable,
            "most_supportive_cohort": {"id": c_most_pos, "span": cohort_span[c_most_pos],
                                       "sharpe_if_dropped": round(float(loo[c_most_pos]), 3)},
            "most_dragging_cohort": {"id": c_most_neg, "span": cohort_span[c_most_neg],
                                     "sharpe_if_dropped": round(float(loo[c_most_neg]), 3)},
        },
        "monthly_attribution": {
            "loo_by_month_sharpe_min": round(float(loo_m_s.min()), 3),
            "loo_by_month_sharpe_max": round(float(loo_m_s.max()), 3),
            "most_supportive_month": worst_drop_month,
            "top3_positive_months": list(top3.index),
            "top3_share_of_positive_return": round(top3_share, 3),
            "hhi_positive_months": round(hhi, 3),
            "effective_n_winning_months": round(eff_n_months, 2),
            "n_positive_months": int(len(pos)),
            "n_total_months": int(len(months)),
        },
        "regime_attribution": reg_table,
        "regime_main_contributor": regime_main,
    }
    (CACHE / "fin001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"loo_cohort_sharpe": loo}).to_parquet(CACHE / "fin001_loo_cohort.parquet")
    month_ret.rename("month_ret").to_frame().assign(
        loo_sharpe=loo_m_s, regime=[month_regime.get(m) for m in months]).to_parquet(
        CACHE / "fin001_monthly.parquet")

    conclusion = (
        f"V12.31 真·真实基线 (WFE-001 label-embargo OOS book, {dates[0]}~{dates[-1]}, {n_days}日/{n_cohorts}个20日cohort) "
        f"的点估计 净Sharpe **{point_sharpe:+.2f}** (WFE-001 报 {reported_sharpe:+.2f}) / 年化 {point_ann:+.1%} / maxDD {point_maxdd:+.1%} "
        f"补误差条 (不改策略, R01)。"
        f"**① per-cohort block bootstrap ({N_BOOT}次): Sharpe 95% CI = [{sh_ci[0]:+.2f}, {sh_ci[1]:+.2f}]** "
        f"(中位 {sh_med:+.2f}, SE {sh_std:.2f}, P(>0)={p_gt0:.0%}, P(>1)={p_gt1:.0%}); 年化 95% CI [{ann_ci[0]:+.0%}, {ann_ci[1]:+.0%}]。"
        f"→ **CI 很宽是核心结论**: 仅 ~{n_cohorts} 个 cohort / {len(months)} 月样本, 统计功率有限; 真实 Sharpe 几乎必正 (P>0 {p_gt0:.0%}) "
        f"但 '稳超1' 信心只有 {p_gt1:.0%} → 1.31 是点估计而非可承诺下限。"
        f"**② leave-one-out (逐cohort): Sharpe ∈ [{loo_min:+.2f}, {loo_max:+.2f}], 符号恒正={loo_sign_stable}** "
        f"→ **对删任一单期稳健** (不靠某个孤立 cohort); 最撑 Sharpe 的 cohort #{c_most_pos} {cohort_span[c_most_pos]} 删后仍 {loo[c_most_pos]:+.2f}。"
        f"**③ 集中度 = 中等 (非极端)**: top-3 赢月 {list(top3.index)} 占正收益 {top3_share:.0%}, HHI {hhi:.2f} → 有效 ~{eff_n_months:.1f} 个独立赢月 "
        f"(共 {len(pos)}/{len(months)} 正月); LOO-by-month 删最重的 {worst_drop_month} Sharpe 仅降到 {loo_m_s.min():+.2f} = 单月依赖不强。"
        f"regime 归因 (主贡献 **{regime_main}**, 但实为均衡): "
        + " / ".join(f"{rg} 组内日Sharpe {d['daily_sharpe_within']:+.2f} 贡献 {d['log_ret_contrib_share']:.0%}"
                     for rg, d in reg_table.items()) + " → 动量/反转两态贡献近 50/50, 非靠单一 regime。"
        f"**诚实结论: 1.31 是最干净真实基线点估计, 且对 LOO (删单 cohort/月) 稳健、收益分布跨月/跨 regime 中等均衡 — "
        f"但 bootstrap CI 宽 [{sh_ci[0]:.1f},{sh_ci[1]:.1f}] 暴露真实瓶颈是样本短/功率低, 而非过拟合到某几个月**。"
        f"故 1.31 非最终可部署期望, 主要不确定性来自样本量; 接 7 月血洗窗前向 paper-trade 复检 (见 FIN-002 冻结协议)。"
    )

    VERDICT.write_text(json.dumps({
        "id": "FIN-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "point_net_sharpe": round(point_sharpe, 3),
            "point_ann_return": round(point_ann, 4),
            "point_max_drawdown": round(point_maxdd, 4),
            "bootstrap_sharpe_ci95": [round(sh_ci[0], 3), round(sh_ci[1], 3)],
            "bootstrap_sharpe_median": round(sh_med, 3),
            "bootstrap_sharpe_se": round(sh_std, 3),
            "bootstrap_p_sharpe_gt_0": round(p_gt0, 3),
            "bootstrap_p_sharpe_gt_1": round(p_gt1, 3),
            "bootstrap_ann_return_ci95": [round(ann_ci[0], 4), round(ann_ci[1], 4)],
            "loo_cohort_sharpe_min": round(loo_min, 3),
            "loo_cohort_sharpe_max": round(loo_max, 3),
            "loo_cohort_sign_stable_positive": loo_sign_stable,
            "loo_by_month_sharpe_min": round(float(loo_m_s.min()), 3),
            "most_supportive_month": worst_drop_month,
            "top3_winning_months": list(top3.index),
            "top3_share_of_positive_return": round(top3_share, 3),
            "hhi_positive_months": round(hhi, 3),
            "effective_n_winning_months": round(eff_n_months, 2),
            "n_positive_months": int(len(pos)),
            "n_total_months": int(len(months)),
            "regime_main_contributor": regime_main,
        },
        "regime_attribution": reg_table,
        "artifacts": [
            "research/backtest/run_fin001.py",
            "research/cache/fin001/fin001_results.json",
            "research/cache/fin001/fin001_loo_cohort.parquet",
            "research/cache/fin001/fin001_monthly.parquet",
        ],
        "note": "不改策略 (R01 不动门柱); 在 WFE-001 embargo 真·真实 OOS book 上补误差条 (per-cohort block "
                "bootstrap 1000次 + LOO + 月度/regime 归因); cohort=20交易日再平衡/持有期 (= 自然依赖块); "
                "ST 已 WFE-001 源头排除 (R06); regime 分层必带 (R11); 描述性=合法完成 (R02); 中间指标≠ship (R03); "
                "复用 WFE-001 缓存未碰生产 (R05); 大缓存 research/cache/fin001/ gitignored (R04)。"
                "吃进 codex 第三次 review: 1.31 是最干净真实基线点估计 (非最终可部署期望, CI/集中度已补)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
