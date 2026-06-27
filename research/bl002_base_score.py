# -*- coding: utf-8 -*-
"""BL-002 — BASE sleeve 选择器 v1 规则法 (训练 v2 已被 BL-001 判样本不足 → 跳过).

BL-001 冻结判据: 正例 445 (walk-forward 每折≈23) << 门槛 (n_pos>=2000 & 每折>=100)
→ sample_sufficiency.bl002_route = **v1_rules_only**。落 responsePlaybook.BL-002 "样本不够"
分叉: 仅走规则法 (简单质量排序), 不强训 LGBM, 文档化小样本约束。SLV-002 已证严格 BASE
入场本身即 edge (无模型已最高质量), v1 绕开 n 小训练陷阱。

────────────────────────────────────────────────────────────────────────
v1 base_score (FROZEN, SIGN-R01 — 等权, 不在 base_label 上调权重 = 防 p-hack):

  三个简单质量代理 (全因果, 入场日收盘后/次日开盘前全部已知):
    ① squeeze 紧度   comp_squeeze = pctrank(-boll_bw_pctile)   越紧(boll_bw 越低) 越高
    ② 突破量能       comp_vol     = pctrank(vol_ratio)         放量越大 越高
    ③ 整理时长       comp_dur     = pctrank(consol_duration)   蓄势越久 越高
  base_score = mean(comp_squeeze, comp_vol, comp_dur)   ∈ [0,1], 横截面 pct rank, 等权

  设计冻结理由: BL-001 揭示无条件突破入场仍有肥左尾 (μ+0.018/win0.48/dd_p10−0.148, 含假突破秒回)
  → v1 目标 = 在 base_entry 候选**内部**挑尾更薄/命中更高的子集供 BL-003 blend 注入排序。
  等权 (非在 base_label 上拟合) 是事前注册的简单代理, 避免把 v1 选择器调成隐式 v2 训练 (SIGN-R01/R03)。

诚实约束 (SIGN-R03): 下面对 base_score 是否真排序 base_label / 前向尾部的检验是**描述性核对**,
不作 ship 依据; 只有 BL-003 walk-forward Δα 定夺。base_score 仅产出供 BL-003 注入排序。

红线: 权重/代理冻结 (R01); 全因果零泄漏 (R04, base_score 只用因果因子, label/前向列只留 cache 非 features);
ST 已源头排除 (R06); 分 regime (R11); 描述统计非 ship (R03); 生产线只读 (R05)。

产出:
  research/cache/bl002/base_score_YYYYMM.parquet  (月度: ts_code,trade_date,base_score,comp_*,base_label,_ret*/_dd*)
  research/cache/bl002_results.json
  research/verdicts/BL-002.json                    (status=built + v1 选择理由 + 打分摘要 + 描述性排序核对)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "bl002"
CACHE.mkdir(parents=True, exist_ok=True)
ENTRY_P = ROOT / "research" / "cache" / "bl001" / "base_entry.parquet"
FACTOR_CKPT = ROOT / "research" / "cache" / "slv001" / "factor_panel.parquet"
BL001_RESULTS = ROOT / "research" / "cache" / "bl001_results.json"
RESULTS = ROOT / "research" / "cache" / "bl002_results.json"
VERDICT = ROOT / "research" / "verdicts" / "BL-002.json"
KEY = ["ts_code", "trade_date"]

# ── 冻结质量代理 (SIGN-R01): 三代理等权, 方向固定 ──
PROXIES = [
    ("comp_squeeze", "boll_bw_pctile", -1),   # 越紧(低) 越高 → 负向 pctrank
    ("comp_vol", "vol_ratio", +1),            # 放量越大 越高
    ("comp_dur", "consol_duration", +1),      # 整理越久 越高
]


def main():
    t0 = time.time()
    print("\n=== BL-002 BASE sleeve 选择器 v1 规则法 (等权质量排序) ===\n", flush=True)

    # ── BL-001 路由确认 (冻结判据) ──
    bl001 = json.loads(BL001_RESULTS.read_text(encoding="utf-8"))
    route = bl001.get("sample_sufficiency", {}).get("bl002_route")
    sufficient = bl001.get("sample_sufficiency", {}).get("sufficient_for_training")
    print(f"[route] BL-001 sample_sufficiency: sufficient_for_training={sufficient} "
          f"→ bl002_route={route}", flush=True)
    assert route == "v1_rules_only", \
        f"BL-001 路由={route} != v1_rules_only; 若样本变够需走 responsePlaybook v2 分叉"

    # ── 候选 = BL-001 base_entry 事件; 补 boll_bw_pctile (entry parquet 未含) ──
    ev = pd.read_parquet(ENTRY_P)
    ev["trade_date"] = ev["trade_date"].astype(str)
    bw = pd.read_parquet(FACTOR_CKPT, columns=KEY + ["boll_bw_pctile"])
    bw["trade_date"] = bw["trade_date"].astype(str)
    ev = ev.merge(bw, on=KEY, how="left")
    print(f"[data] base_entry 候选 {len(ev):,} ({ev['trade_date'].min()}~{ev['trade_date'].max()}), "
          f"缺 boll_bw {ev['boll_bw_pctile'].isna().mean():.3%}", flush=True)

    # ── 三代理横截面 pct rank (全候选池, 全因果), 等权合成 base_score ──
    comp_cols = []
    for name, src, sign in PROXIES:
        ev[name] = (sign * ev[src]).rank(pct=True)
        comp_cols.append(name)
    ev["base_score"] = ev[comp_cols].mean(axis=1)
    print(f"[score] base_score = 等权 mean({', '.join(comp_cols)}) ∈ [0,1]; "
          f"非缺失 {ev['base_score'].notna().sum():,}", flush=True)

    # ── 落月度 cache (供 BL-003 blend 注入排序) ──
    ev["ym"] = ev["trade_date"].str[:6]
    out_cols = KEY + ["base_score"] + comp_cols + \
        ["base_label", "regime", "vol_ratio", "consol_duration", "boll_bw_pctile",
         "_ret5", "_dd5", "_ret20", "_dd20"]
    out_cols = [c for c in out_cols if c in ev.columns]
    for f in CACHE.glob("base_score_*.parquet"):
        f.unlink()
    n_files = 0
    for ym, sub in ev.groupby("ym"):
        sub[out_cols].to_parquet(CACHE / f"base_score_{ym}.parquet", index=False)
        n_files += 1
    print(f"[out] {n_files} 月度文件 -> {CACHE.relative_to(ROOT)}/base_score_YYYYMM.parquet", flush=True)

    # ── 描述性核对 (SIGN-R03: 非 ship, 仅看 v1 排序是否有方向) ──
    lab = ev[ev["base_label"].notna()].copy()
    n_lab, n_pos = len(lab), int(lab["base_label"].sum())
    overall_pos = float(lab["base_label"].mean()) if n_lab else None

    # base_score 五分位 → 紧尾正例率 + 前向20d dd_p10 (高分组该更高命中/更薄尾, 否则 v1 无方向)
    by_quintile = {}
    if n_lab >= 250:
        lab["q"] = pd.qcut(lab["base_score"].rank(method="first"), 5,
                           labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
        for q, s in lab.groupby("q", observed=True):
            dd20 = s["_dd20"].dropna()
            ret20 = s["_ret20"].dropna()
            by_quintile[str(q)] = {
                "n": int(len(s)),
                "pos_rate": round(float(s["base_label"].mean()), 4),
                "ret20_mean": round(float(ret20.mean()), 5) if len(ret20) else None,
                "dd20_p10": round(float(dd20.quantile(0.10)), 5) if len(dd20) else None,
            }
    q5 = by_quintile.get("Q5_high", {})
    q1 = by_quintile.get("Q1_low", {})
    monotone = None
    strictly_monotone = None
    tail_thinned = None
    if by_quintile:
        prates = [by_quintile[k]["pos_rate"] for k in
                  ["Q1_low", "Q2", "Q3", "Q4", "Q5_high"] if k in by_quintile]
        # 单调性弱检验: Q5 是否 > Q1 (描述, 非 gate)
        monotone = bool(q5.get("pos_rate", 0) > q1.get("pos_rate", 0)) if q5 and q1 else None
        # 严格单调 (Q1<=...<=Q5): 描述 v1 排序是否真有干净梯度
        strictly_monotone = bool(all(prates[i] <= prates[i + 1]
                                     for i in range(len(prates) - 1))) if len(prates) >= 5 else None
        # 尾是否被排序变薄: Q5 dd20_p10 是否比 Q1 更浅 (less negative)
        if q5.get("dd20_p10") is not None and q1.get("dd20_p10") is not None:
            tail_thinned = bool(q5["dd20_p10"] > q1["dd20_p10"])

    # 各代理与紧尾 label 的边际相关 (描述, 看哪个代理有方向)
    proxy_corr = {}
    for name, src, sign in PROXIES:
        v = lab[[name, "base_label"]].dropna()
        if len(v) >= 100:
            proxy_corr[name] = round(float(np.corrcoef(v[name], v["base_label"])[0, 1]), 4)

    # 分 regime (R11): 高分组 (Q5) vs 低分组 (Q1) 正例率
    by_regime = {}
    if "q" in lab.columns:
        for rgm in ["momentum", "reversal", "mixed"]:
            sub = lab[lab["regime"] == rgm]
            if len(sub) < 60:
                continue
            hi = sub[sub["q"] == "Q5_high"]["base_label"]
            lo = sub[sub["q"] == "Q1_low"]["base_label"]
            by_regime[rgm] = {
                "n": int(len(sub)),
                "q5_pos_rate": round(float(hi.mean()), 4) if len(hi) else None,
                "q1_pos_rate": round(float(lo.mean()), 4) if len(lo) else None,
            }

    monthly_coverage = {
        "n_months": int(ev["ym"].nunique()),
        "candidates_per_month_mean": round(float(ev.groupby("ym").size().mean()), 2),
        "candidates_per_month_median": round(float(ev.groupby("ym").size().median()), 2),
    }

    results = {
        "task": "BL-002", "elapsed_s": round(time.time() - t0, 1),
        "route": route, "sufficient_for_training": sufficient,
        "v2_skipped_reason": "BL-001 样本不足 (正例 445/每折≈23 << 2000/100); SLV-002 证入场即 edge, v1 绕开小样本训练陷阱",
        "frozen_base_score": {
            "formula": "mean(comp_squeeze, comp_vol, comp_dur), 各为横截面 pct rank, 等权 (R01 冻结, 不在 label 上调权)",
            "comp_squeeze": "pctrank(-boll_bw_pctile) 越紧越高",
            "comp_vol": "pctrank(vol_ratio) 放量越大越高",
            "comp_dur": "pctrank(consol_duration) 整理越久越高",
        },
        "n_candidates": int(len(ev)), "n_labeled": n_lab, "n_pos": n_pos,
        "overall_pos_rate": round(overall_pos, 4) if overall_pos is not None else None,
        "monthly_coverage": monthly_coverage,
        "score_by_quintile": by_quintile,
        "q5_vs_q1": {"q5_pos_rate": q5.get("pos_rate"), "q1_pos_rate": q1.get("pos_rate"),
                     "monotone_q5_gt_q1": monotone, "strictly_monotone": strictly_monotone,
                     "tail_thinned_q5_vs_q1": tail_thinned,
                     "q5_dd20_p10": q5.get("dd20_p10"), "q1_dd20_p10": q1.get("dd20_p10")},
        "proxy_marginal_corr_with_label": proxy_corr,
        "by_regime": by_regime,
        "n_monthly_files": n_files,
        "guardrails": ["SIGN-R01 权重/代理冻结等权", "SIGN-R03 排序核对非ship",
                       "SIGN-R04 base_score只用因果因子,label留cache", "SIGN-R06 ST源头排除",
                       "SIGN-R11 分regime"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print(f"\n--- base_score 五分位 → 紧尾正例率 / 前向20d (描述, 非 ship) ---", flush=True)
    for k in ["Q1_low", "Q2", "Q3", "Q4", "Q5_high"]:
        if k in by_quintile:
            d = by_quintile[k]
            print(f"  {k:8s}: n{d['n']:4d} pos_rate {d['pos_rate']:.4f} "
                  f"ret20μ {d['ret20_mean']} dd20_p10 {d['dd20_p10']}", flush=True)
    print(f"  Q5>Q1 {monotone} / 严格单调 {strictly_monotone} / 尾变薄 {tail_thinned} | 整体正例率 {overall_pos}", flush=True)
    print(f"  代理边际相关(vs label): {json.dumps(proxy_corr, ensure_ascii=False)}", flush=True)
    print(f"  分regime: {json.dumps(by_regime, ensure_ascii=False)}", flush=True)

    # ── verdict ──
    if monotone is None:
        direction = "样本不足判方向"
    elif strictly_monotone:
        direction = f"有干净梯度 (Q1→Q5 严格单调升, Q5 {q5.get('pos_rate')} > Q1 {q1.get('pos_rate')})"
    elif monotone:
        direction = (f"方向弱且非单调 (Q5 {q5.get('pos_rate')} 最高 > Q1 {q1.get('pos_rate')}, "
                     f"但中间分位 Q2-Q4 无序, 主要由 comp_vol 驱动 corr={proxy_corr.get('comp_vol')})")
    else:
        direction = f"无明显方向 (Q5 {q5.get('pos_rate')} 不高于 Q1 {q1.get('pos_rate')})"
    tail_note = ("Q5 尾比 Q1 薄" if tail_thinned else
                 f"**尾未被排序变薄** (Q5 dd20_p10 {q5.get('dd20_p10')} ≈ Q1 {q1.get('dd20_p10')}, 各分位 ~−0.14 持平)"
                 if tail_thinned is False else "尾部样本不足")
    conclusion = (
        f"BL-001 已判样本不足 (紧尾正例 {n_pos}/walk-forward 每折≈{n_pos//19} << 门槛 2000/100) "
        f"→ 落 responsePlaybook.BL-002 '样本不够'分叉: **跳过 v2 LGBM 训练, 仅走 v1 规则法**, "
        f"文档化小样本约束。SLV-002 已证严格 BASE 入场本身即 edge (无模型即最高质量画像), "
        f"v1 绕开 n 小训练陷阱 (反 [[feedback_train_label_over_inference_hack]] 的训练层 hack 诱惑)。"
        f"【v1 base_score (R01 冻结等权)】= mean 三横截面 pct rank: squeeze 紧度 (−boll_bw_pctile)、"
        f"突破量能 (vol_ratio)、整理时长 (consol_duration), 等权且**不在 base_label 上拟合权重** "
        f"(防把 v1 选择器调成隐式 v2 训练 = SIGN-R01/R03)。对 {len(ev):,} 个 base_entry 候选打分, "
        f"落 {n_files} 个月度文件 research/cache/bl002/base_score_YYYYMM.parquet 供 BL-003 blend 注入排序。"
        f"【描述性排序核对 (SIGN-R03 非 ship)】base_score 五分位: Q5(高分) 紧尾正例率 {q5.get('pos_rate')} "
        f"vs Q1(低分) {q1.get('pos_rate')} → {direction}; 尾部: {tail_note}。代理边际相关 (vs 紧尾 label): "
        f"{json.dumps(proxy_corr, ensure_ascii=False)} (仅 comp_vol 放量为正向且微弱, squeeze/dur 近零或反向)。"
        f"诚实判读: v1 等权排序对**命中率**仅微弱抬升 (Q5 略高且非单调), 对 BL-001 揭示的**肥左尾几乎无作用** "
        f"(各分位 dd20_p10 ~−0.14 持平) → 守尾仍只能靠紧尾 label+紧止损 (BL-003), 不能指望 base_score 排掉假突破。"
        f"BL-003 注入排序大概率接近'在 base_entry 内弱筛', 增益是否落地完全由 walk-forward Δα 定夺 (SIGN-R03)。"
        f"月度候选: {monthly_coverage['n_months']} 月, 月均 {monthly_coverage['candidates_per_month_mean']} 候选 "
        f"(中位 {monthly_coverage['candidates_per_month_median']}) — 稀疏, BL-003 多数日仅少量可注入。"
        f"全因果 base_score 只用因果因子 (R04), label/前向列只留 cache 非 features; ST 源头排除 (R06); "
        f"分 regime 已落 metrics (R11); 排序核对非 ship (R03); 生产线只读 (R05)。")

    VERDICT.write_text(json.dumps({
        "id": "BL-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "route": route, "v2_skipped": True, "n_candidates": int(len(ev)),
            "n_labeled": n_lab, "n_pos": n_pos,
            "overall_pos_rate": round(overall_pos, 4) if overall_pos is not None else None,
            "frozen_base_score": results["frozen_base_score"],
            "score_by_quintile": by_quintile,
            "q5_vs_q1": results["q5_vs_q1"],
            "proxy_marginal_corr_with_label": proxy_corr,
            "by_regime": by_regime,
            "monthly_coverage": monthly_coverage,
            "n_monthly_files": n_files,
        },
        "artifacts": ["research/cache/bl002/base_score_YYYYMM.parquet (月度)",
                      "research/cache/bl002_results.json"],
        "note": "v1 规则法等权打分 (R01 冻结, 未在 label 上调权); v2 训练按 BL-001 判样本不足跳过 "
                "(responsePlaybook.BL-002 '样本不够'分叉); base_score 只用因果因子, label/前向列只留 cache 非 features (R04); "
                "ST 源头排除 (R06); 分 regime (R11); 排序核对非 ship, 只有 BL-003 walk-forward 定夺 (R03); 生产线只读 (R05)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] BL-002 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
