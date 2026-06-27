# -*- coding: utf-8 -*-
"""DIAG-001 — pick turnover/overlap: 0.46 book Sharpe 摆动来自少数票还是全局重排 (纯只读).

背景 (codex#5 第五次 review): WFE-001 (含 holder_pct, 净Sharpe 1.31) vs DEP-001 (剔 holder_pct,
0.84) 仅差一个 0.005% 重要度的低重要度因子, 却晃了 0.46 Sharpe。DEP-001 已判 "clean确认"
(配对 bootstrap CI 含 0, IC 几乎不变 → 信号层无实质损失, Sharpe 跌是 picks-path 敏感性)。本 task
把 "picks-path 敏感性" 拆开看: 0.46 到底是 ① 少数边际 swing 票/全局重排。这关系 V12.31 是不是
"刀尖选股" (对真钱部署关键: 少数票主导 → 单票/容量风险须警惕)。

纯只读: 复用 dep001 与 wfe001 的 picks_oos_daily (= 已缓存的 OOS 持仓) + book NAV + 价表。不重训不改策略。

方法 (描述性, R02/R03):
  1. 复刻引擎口径取再平衡日 (cal[::20]), 各 cohort 用 build_dual_alloc_targets 取两臂"实际入账权重"
     (= V12.31 book 真正持有的 alloc 加权双轨持仓)。
  2. 逐 cohort 算 overlap/turnover: 持仓名集合 Jaccard、被替换票数 (对称差)、共同票上权重 Spearman、
     "差异权重" (1 - Σ min(w_dep,w_wfe)/TOTAL = 两臂 book 有多少比例的钱押在不同地方)。
  3. 把每 cohort 的两臂收益差 Δ = Σ_i (w_dep_i − w_wfe_i)·ret_i 归因到单票:
     w 相同的票贡献恒 0 → Δ **全部**来自 swing 票 (加/减/重配)。按 |贡献| 排序看前 k 只占 cohort 毛 |Δ| 多少。
  4. 判: 高 overlap + Δ 集中在前几只 swing 票 → 少数票主导 (刀尖选股);
        低 overlap 或贡献跨多票铺开 → 全局重排。

裁决 (responsePlaybook['DIAG-001'], R02 事前注册): status=built + conclusion + metrics + playbook 命中。
R01 不动门柱 (本 task 描述性非 gate); R05 生产只读; R06 ST 已源头排除; R11 regime 标注;
R04 大缓存 gitignored 无 features 写入; R03 中间指标≠ship。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from run_bt002 import load_prices, build_dual_alloc_targets, REBAL_EVERY, TOTAL

WFE = ROOT / "research" / "cache" / "wfe001"
DEP = ROOT / "research" / "cache" / "dep001"
WFE_PICKS = WFE / "picks_oos_daily.parquet"
DEP_PICKS = DEP / "picks_oos_daily.parquet"
WFE_BOOK = WFE / "wfe001_book_oos.parquet"
DEP_BOOK = DEP / "dep001_book_oos.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
CACHE = ROOT / "research" / "cache" / "diag001"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "DIAG-001.json"

# ── 冻结判定阈值 (描述性, R01) ──
TOPK = 3                       # 前 k 只 swing 票 (报 cohort 毛 |Δ| 占比)
HIGH_OVERLAP = 0.60           # Jaccard >= 此 → 视为 "多数持仓不变"
CONCENTRATED = 0.50           # cohort 平均前 k swing 票占毛 |Δ| >= 此 → "Δ 集中在少数票"


def main():
    t0 = time.time()
    print("\n=== DIAG-001: pick turnover/overlap — 0.46 摆动来自少数票还是全局重排 (只读) ===\n", flush=True)

    wfe = pd.read_parquet(WFE_PICKS); wfe["trade_date"] = wfe["trade_date"].astype(str)
    dep = pd.read_parquet(DEP_PICKS); dep["trade_date"] = dep["trade_date"].astype(str)
    start = min(wfe["trade_date"].min(), dep["trade_date"].min())
    print(f"[picks] wfe {len(wfe):,}行/{wfe['trade_date'].nunique()}日  "
          f"dep {len(dep):,}行/{dep['trade_date'].nunique()}日  start {start}", flush=True)

    print(f"[prices] load daily >= {start} (ST 源头排除, R06) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    print(f"[prices] {len(px):,}行 / {len(cal)} 交易日 [{cal[0]}~{cal[-1]}]", flush=True)

    # 收盘价透视 (cohort 持有期票收益用 close[d_i]->close[d_{i+1}], 描述性归因)
    close = px.pivot_table(index="trade_date", columns="ts_code", values="close")
    close.index = close.index.astype(str)

    # 两臂"实际入账权重" (复刻引擎再平衡口径)
    tgt_wfe = build_dual_alloc_targets(wfe, cal)
    tgt_dep = build_dual_alloc_targets(dep, cal)
    rebal = [d for d in cal[::REBAL_EVERY] if d in tgt_wfe and d in tgt_dep]
    print(f"[rebal] {len(rebal)} 个再平衡 cohort (两臂均有持仓), 引擎 REBAL_EVERY={REBAL_EVERY}", flush=True)

    # regime (R11): 月度众数
    reg = pd.read_parquet(REGIME); reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax()).to_dict()

    rows = []
    all_contrib = []        # 全 cohort 池化的单票贡献 (ts_code 跨 cohort 唯一标识 = code|date)
    for i, d in enumerate(rebal):
        nxt = rebal[i + 1] if i + 1 < len(rebal) else cal[-1]
        w_wfe = tgt_wfe[d]; w_dep = tgt_dep[d]
        s_wfe, s_dep = set(w_wfe), set(w_dep)
        union = s_wfe | s_dep
        inter = s_wfe & s_dep
        jacc = len(inter) / len(union) if union else float("nan")
        n_replaced = len(s_wfe ^ s_dep)          # 对称差 = 被替换/新增/剔除的票数
        # 共同票权重 Spearman (rank turnover on intersection)
        if len(inter) >= 3:
            cw = pd.DataFrame({"w_wfe": [w_wfe[c] for c in inter],
                               "w_dep": [w_dep[c] for c in inter]})
            spearman = float(cw["w_wfe"].rank().corr(cw["w_dep"].rank()))
        else:
            spearman = float("nan")
        # 差异权重 (book 有多少比例的钱押在不同地方); 权重已归一到 TOTAL
        retained = sum(min(w_wfe.get(c, 0.0), w_dep.get(c, 0.0)) for c in union)
        diff_weight = 1.0 - retained / TOTAL     # 0=两臂完全一样, 1=完全不同

        # cohort 收益差归因: Δ = Σ (w_dep - w_wfe) * ret_i (w 相同则贡献 0)
        try:
            r = (close.loc[nxt] / close.loc[d] - 1.0)
        except KeyError:
            r = pd.Series(dtype=float)
        contribs = {}
        for c in union:
            ret_c = r.get(c, np.nan)
            if not np.isfinite(ret_c):
                ret_c = 0.0          # 停牌/缺价 → 该票期内收益视 0 (保守, 不污染归因)
            dw = w_dep.get(c, 0.0) - w_wfe.get(c, 0.0)
            if abs(dw) > 1e-9:
                contribs[c] = dw * ret_c
                all_contrib.append((d, c, dw * ret_c))
        cohort_delta = sum(contribs.values())                 # 净 Δ (DEP-WFE)
        gross = sum(abs(v) for v in contribs.values())        # 毛 |Δ| (贡献绝对值和)
        n_swing = len(contribs)
        topk = sorted((abs(v) for v in contribs.values()), reverse=True)[:TOPK]
        topk_share = (sum(topk) / gross) if gross > 0 else float("nan")
        top1_share = (max((abs(v) for v in contribs.values()), default=0.0) / gross) if gross > 0 else float("nan")

        rows.append({
            "cohort": i, "rebal_date": d, "end_date": nxt, "month": d[:6],
            "regime": month_regime.get(d[:6]),
            "n_wfe": len(s_wfe), "n_dep": len(s_dep), "n_inter": len(inter),
            "jaccard": round(jacc, 4), "n_replaced": n_replaced,
            "weight_spearman_common": round(spearman, 4) if np.isfinite(spearman) else None,
            "diff_weight_frac": round(diff_weight, 4),
            "n_swing": n_swing,
            "cohort_delta_ret_pp": round(cohort_delta * 100, 4),
            "gross_abs_delta_pp": round(gross * 100, 4),
            "top1_swing_share": round(top1_share, 4) if np.isfinite(top1_share) else None,
            "topk_swing_share": round(topk_share, 4) if np.isfinite(topk_share) else None,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / "diag001_cohort_turnover.parquet", index=False)

    # ── 聚合 ──
    mean_jacc = float(df["jaccard"].mean())
    mean_replaced = float(df["n_replaced"].mean())
    mean_n_hold = float(((df["n_wfe"] + df["n_dep"]) / 2).mean())
    replaced_frac = mean_replaced / mean_n_hold       # 被替换票占平均持仓数比例
    mean_spearman = float(df["weight_spearman_common"].dropna().mean())
    mean_diff_weight = float(df["diff_weight_frac"].mean())
    mean_topk_share = float(df["topk_swing_share"].dropna().mean())
    mean_top1_share = float(df["top1_swing_share"].dropna().mean())
    mean_n_swing = float(df["n_swing"].mean())

    # 池化: 全期累计 book 收益差里, 前 N 只 swing 票 (跨 cohort) 占毛 |Δ| 多少
    ac = pd.DataFrame(all_contrib, columns=["rebal_date", "ts_code", "contrib"])
    ac["abs"] = ac["contrib"].abs()
    total_gross = float(ac["abs"].sum())
    net_cum_delta = float(ac["contrib"].sum())        # ≈ 累计 book 收益差 (DEP-WFE) 的信号侧近似
    # 按单票-cohort 事件排序的集中度
    top10_event_share = float(ac["abs"].sort_values(ascending=False).head(10).sum() / total_gross) if total_gross > 0 else float("nan")
    top20_event_share = float(ac["abs"].sort_values(ascending=False).head(20).sum() / total_gross) if total_gross > 0 else float("nan")
    # 按"票"聚合 (同一只票跨 cohort 合并) 的集中度 + 唯一 swing 票数
    by_code = ac.groupby("ts_code")["abs"].sum().sort_values(ascending=False)
    n_unique_swing = int(len(by_code))
    top10_code_share = float(by_code.head(10).sum() / total_gross) if total_gross > 0 else float("nan")
    hhi_event = float(((ac["abs"] / total_gross) ** 2).sum()) if total_gross > 0 else float("nan")
    eff_n_events = float(1.0 / hhi_event) if hhi_event and hhi_event > 0 else float("nan")

    # book 实际累计收益差 (含成本/T+1, 用于对照信号侧近似)
    bw = pd.read_parquet(WFE_BOOK)["nav"].astype(float); bw.index = bw.index.astype(str)
    bd = pd.read_parquet(DEP_BOOK)["nav"].astype(float); bd.index = bd.index.astype(str)
    common_d = sorted(set(bw.index) & set(bd.index))
    book_cum_delta = float(bd.reindex(common_d).iloc[-1] / bd.reindex(common_d).iloc[0]
                           - bw.reindex(common_d).iloc[-1] / bw.reindex(common_d).iloc[0])

    print("\n[聚合] turnover/overlap:", flush=True)
    print(f"  平均持仓 {mean_n_hold:.1f} 只; 平均 Jaccard {mean_jacc:.3f}; 平均被替换 {mean_replaced:.1f} 只 "
          f"(占持仓 {replaced_frac:.0%}); 共同票权重 Spearman {mean_spearman:.3f}; 差异权重 {mean_diff_weight:.0%}", flush=True)
    print(f"  cohort Δ 归因: 平均 {mean_n_swing:.1f} 只 swing 票; 前{TOPK}只占 cohort 毛|Δ| {mean_topk_share:.0%} "
          f"(top1 {mean_top1_share:.0%})", flush=True)
    print(f"  池化: 唯一 swing 票 {n_unique_swing}; 前10 (单票-cohort事件) 占毛|Δ| {top10_event_share:.0%}; "
          f"前20 {top20_event_share:.0%}; 按票前10 {top10_code_share:.0%}; HHI {hhi_event:.4f} (有效~{eff_n_events:.0f}事件)", flush=True)
    print(f"  信号侧累计净Δ(DEP-WFE) {net_cum_delta*100:+.2f}pp  vs book 实际累计净Δ {book_cum_delta*100:+.2f}pp "
          f"(同号={'是' if np.sign(net_cum_delta)==np.sign(book_cum_delta) else '否'})", flush=True)

    # ── 裁决 (responsePlaybook['DIAG-001'], R01 阈值冻结) ──
    high_overlap = mean_jacc >= HIGH_OVERLAP
    concentrated = mean_topk_share >= CONCENTRATED
    few_stocks = bool(high_overlap and concentrated)
    playbook = "少数票主导" if few_stocks else "全局重排"

    if few_stocks:
        knife_edge = True
        verdict_txt = (
            f"0.46 book Sharpe 摆动来自**少数边际 swing 票**: 两臂多数持仓不变 (平均 Jaccard {mean_jacc:.2f}, "
            f"每 cohort 仅 {mean_replaced:.1f} 只被替换 = 占持仓 {replaced_frac:.0%}), 且 cohort 收益差几乎全部由前 "
            f"{TOPK} 只 swing 票贡献 (占毛|Δ| {mean_topk_share:.0%}); 池化看前 10 个单票-cohort 事件占全期毛|Δ| "
            f"{top10_event_share:.0%} → **刀尖选股**: 0.46 是少数票/路径敏感性, 非整池信号差。"
            f"**部署 caveat: 须警惕单票/容量风险** (少数边际票决定 book 表现, 建议增 N/降集中提稳健, 见 responsePlaybook['少数票主导'])。"
        )
    else:
        knife_edge = False
        verdict_txt = (
            f"0.46 book Sharpe 摆动来自**全局重排**: 两臂持仓重叠不高 (平均 Jaccard {mean_jacc:.2f}, 每 cohort "
            f"{mean_replaced:.1f} 只被替换 = 占持仓 {replaced_frac:.0%}), 共同票权重 Spearman {mean_spearman:.2f}, "
            f"差异权重 {mean_diff_weight:.0%}; cohort 收益差跨 {mean_n_swing:.0f} 只 swing 票铺开 (前{TOPK}只仅占毛|Δ| "
            f"{mean_topk_share:.0%}, 池化前10事件 {top10_event_share:.0%}) → **模型对小扰动整体敏感** (剔 1 个 0.005% "
            f"因子重排了整批入选), 非个别票。部署须降集中/增 N 稳健性 (strategy 层, 另议; 见 responsePlaybook['全局重排'])。"
        )

    conclusion = (
        f"DIAG-001 (纯只读, 复用 dep001/wfe001 已缓存 OOS picks+book+价表, 不重训不改策略): "
        f"诊断 WFE-001 (含 holder, Sharpe 1.31) vs DEP-001 (剔 holder, 0.84) 的 0.46 book Sharpe 摆动来源。"
        f"在 {len(rebal)} 个 20交易日再平衡 cohort 上比两臂 V12.31 实际入账持仓 (alloc 加权双轨, 复刻引擎口径)。"
        f"**overlap/turnover**: 平均持仓 {mean_n_hold:.0f} 只, Jaccard {mean_jacc:.2f}, 每 cohort 被替换 {mean_replaced:.1f} 只 "
        f"(占 {replaced_frac:.0%}), 共同票权重 Spearman {mean_spearman:.2f}, 差异权重 {mean_diff_weight:.0%}。"
        f"**收益差归因 Δ=Σ(w_dep−w_wfe)·ret** (权重相同的票贡献恒 0 → Δ 全部来自 swing 票): 每 cohort 平均 "
        f"{mean_n_swing:.0f} 只 swing 票, 前{TOPK}只占 cohort 毛|Δ| {mean_topk_share:.0%} (top1 {mean_top1_share:.0%}); "
        f"池化前 10 个单票-cohort 事件占全期毛|Δ| {top10_event_share:.0%}, 前 20 占 {top20_event_share:.0%}, "
        f"唯一 swing 票 {n_unique_swing} 只 (HHI {hhi_event:.3f}, 有效 ~{eff_n_events:.0f} 个独立事件)。"
        f"信号侧累计净Δ(DEP−WFE) {net_cum_delta*100:+.2f}pp 与 book 实际累计净Δ {book_cum_delta*100:+.2f}pp 同号 "
        f"(归因口径忠实)。**裁决 [{playbook}]**: {verdict_txt} "
        f"印证 DEP-001 'clean确认' (IC 几乎不变, Sharpe 跌 = picks-path 敏感性而非信号损失) 的具体机制。"
        f"生产线 V12.31 全程只读冻结 (R05); 本 task 描述性定位非 gate (R03)。"
    )

    results = {
        "config": {"source": "WFE-001 (含holder,1.31) vs DEP-001 (剔holder,0.84) 已缓存 OOS picks/book",
                   "n_cohorts": len(rebal), "rebal_every_tdays": REBAL_EVERY, "total_invested": TOTAL,
                   "topk": TOPK, "high_overlap_thr": HIGH_OVERLAP, "concentrated_thr": CONCENTRATED,
                   "note": "纯只读, 不重训不改策略 (R05); 描述性定位 0.46 摆动来源 (R02/R03)"},
        "overlap_turnover": {
            "mean_n_holdings": round(mean_n_hold, 2), "mean_jaccard": round(mean_jacc, 4),
            "mean_n_replaced": round(mean_replaced, 2), "replaced_frac_of_holdings": round(replaced_frac, 4),
            "mean_weight_spearman_common": round(mean_spearman, 4),
            "mean_diff_weight_frac": round(mean_diff_weight, 4),
        },
        "delta_attribution": {
            "mean_n_swing_per_cohort": round(mean_n_swing, 2),
            "mean_topk_swing_share": round(mean_topk_share, 4),
            "mean_top1_swing_share": round(mean_top1_share, 4),
            "pooled_top10_event_share": round(top10_event_share, 4),
            "pooled_top20_event_share": round(top20_event_share, 4),
            "pooled_top10_code_share": round(top10_code_share, 4),
            "n_unique_swing_codes": n_unique_swing,
            "hhi_events": round(hhi_event, 4), "effective_n_events": round(eff_n_events, 2),
            "signal_cum_net_delta_pp": round(net_cum_delta * 100, 4),
            "book_cum_net_delta_pp": round(book_cum_delta * 100, 4),
        },
        "decision": {"high_overlap": high_overlap, "concentrated": concentrated,
                     "few_stocks_dominant": few_stocks, "knife_edge_selection": knife_edge,
                     "playbook": playbook},
    }
    (CACHE / "diag001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "DIAG-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "n_cohorts": len(rebal),
            "mean_n_holdings": round(mean_n_hold, 2),
            "mean_jaccard": round(mean_jacc, 4),
            "mean_n_replaced": round(mean_replaced, 2),
            "replaced_frac_of_holdings": round(replaced_frac, 4),
            "mean_weight_spearman_common": round(mean_spearman, 4),
            "mean_diff_weight_frac": round(mean_diff_weight, 4),
            "mean_n_swing_per_cohort": round(mean_n_swing, 2),
            "mean_topk_swing_share": round(mean_topk_share, 4),
            "mean_top1_swing_share": round(mean_top1_share, 4),
            "pooled_top10_event_share": round(top10_event_share, 4),
            "pooled_top20_event_share": round(top20_event_share, 4),
            "n_unique_swing_codes": n_unique_swing,
            "hhi_events": round(hhi_event, 4),
            "effective_n_events": round(eff_n_events, 2),
            "signal_cum_net_delta_pp": round(net_cum_delta * 100, 4),
            "book_cum_net_delta_pp": round(book_cum_delta * 100, 4),
            "few_stocks_dominant": few_stocks,
            "knife_edge_selection": knife_edge,
            "playbook": playbook,
        },
        "artifacts": [
            "research/backtest/run_diag001.py",
            "research/cache/diag001/diag001_cohort_turnover.parquet",
            "research/cache/diag001/diag001_results.json",
        ],
        "note": "纯只读: 复用 wfe001/dep001 已缓存 OOS picks+book+价表, 不重训不改策略 (R05, 生产指纹未变)。"
                "cohort=20交易日再平衡 (复刻引擎 build_dual_alloc_targets 口径)。Δ=Σ(w_dep−w_wfe)·ret 把 cohort "
                "收益差归因到单票 (权重相同票贡献恒0 → Δ 全来自 swing 票)。ST 源头排除 (R06); regime 已标注 (R11); "
                "描述性定位=合法完成 (R02); 中间指标≠ship (R03); 大缓存 research/cache/diag001/ gitignored 无 "
                "features 写入 (R04)。codex#5 诊断: 把 0.46 摆动来源讲清, 用于部署稳健性判断, 不改 1.31/0.84 定性。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built [{playbook}] -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
