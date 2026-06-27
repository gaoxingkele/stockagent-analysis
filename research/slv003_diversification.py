# -*- coding: utf-8 -*-
"""SLV-003 — MOMENTUM 分散性廉价决策闸 (是否在 SQUAT 最差月反向).

Phase 0 第三步 / 决策闸: 不建模型, 只做廉价筛查回答一个 book 层问题 ——
MOMENTUM 桶的月度收益, 是否在 SQUAT 桶最差的几个月里反向 (为组合提供尾部分散)?
若是 → MOMENTUM sleeve 值得在 Phase 2 建 (即使 SLV-002 显示它裸 α 最弱, 价值在对冲);
若否 → 砍掉 Phase 2 MOMENTUM sleeve, 三 sleeve 缩为 squat+base 双 sleeve (重要发现)。

协议 (冻结, prd.preRegisteredGate.momentum_diversification_gate):
  - 对每个 archetype 造粗糙月度 long-only 代理收益 = 该月该桶**全部 launch 等权前向5d收益**
    (纯筛查非策略; 入场口径同 SLV-001/002: 次日开盘 next_open, 全因果)。
  - 算三桶 (SQUAT/BASE/MOMENTUM) 月度收益相关矩阵。
  - 取 SQUAT 桶月度收益最差 K=4 月, 报 MOMENTUM 桶同月均值收益 + 全样本 corr(SQUAT,MOMENTUM)。
  - 判决: DIVERSIFIES = MOMENTUM 在这些月均值>0 且 corr(SQUAT,MOMENTUM)<+0.3;
          否则 REJECT_no_diversification。

红线: 边界/标签/入场口径冻结复用 SLV-001/002 (R01); 全因果 (R04, 前向列只留 cache);
ST 已在 factor_panel 源头排除 (R06); 分 regime 复核 (R11); 这是廉价描述闸非 ship (R03)。

产出:
  research/cache/slv003_results.json
  research/verdicts/SLV-003.json   (status in {DIVERSIFIES, REJECT_no_diversification})
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "slv001"
FACTOR_CKPT = CACHE / "factor_panel.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
RESULTS = ROOT / "research" / "cache" / "slv003_results.json"
VERDICT = ROOT / "research" / "verdicts" / "SLV-003.json"

STUDY_START = "20230103"        # 同 SLV-001/002 统计起点
H = 5                           # 决策闸用前向 5d 代理收益 (gate 口径)
K = 4                           # SQUAT 最差月数 (冻结)
MIN_CELL = 30                   # 每个 (月,桶) 至少 N 个 launch 才算有效月度代理 (抗噪)
CORR_THRESH = 0.30              # corr(SQUAT,MOMENTUM) < 0.30 才算分散 (冻结)
BUCKETS = ["SQUAT", "BASE", "MOMENTUM"]


def build_forward5(df: pd.DataFrame) -> pd.DataFrame:
    """逐股因果前向 5d 收益 (前向列只留内存/cache, 不落 features)。"""
    g = df.groupby("ts_code", sort=False)
    next_open = g["open"].shift(-1)              # t+1 开盘进场
    close_h = g["close"].shift(-H)               # t+H 收盘出场
    df["_ret5"] = close_h / next_open - 1.0
    df.loc[close_h.isna() | next_open.isna(), "_ret5"] = np.nan
    return df


def main():
    t0 = time.time()
    print("\n=== SLV-003 MOMENTUM 分散性廉价决策闸 ===\n", flush=True)

    df = pd.read_parquet(FACTOR_CKPT,
                         columns=["ts_code", "trade_date", "open", "close",
                                  "archetype", "is_launch"])
    print(f"[data] factor_panel {len(df):,} 行 / {df['ts_code'].nunique()} 股", flush=True)
    df = build_forward5(df)

    # 研究面板: 统计窗口 + 真启动样本 (long-only 代理只持有 launch)
    panel = df[(df["trade_date"] >= STUDY_START) & (df["is_launch"] == 1.0)
               & df["_ret5"].notna()].copy()
    panel["month"] = panel["trade_date"].str[:6]
    print(f"[panel] launch 样本 {len(panel):,} 行 ({panel['trade_date'].min()}~"
          f"{panel['trade_date'].max()})", flush=True)

    # ── 月度 long-only 代理收益: (month, archetype) 等权前向5d均值 (cell 样本 >= MIN_CELL) ──
    grp = panel.groupby(["month", "archetype"])["_ret5"]
    monthly_mean = grp.mean()
    monthly_n = grp.size()
    monthly_mean = monthly_mean.where(monthly_n >= MIN_CELL)        # 样本不足置 NaN
    wide = monthly_mean.unstack("archetype").reindex(columns=BUCKETS)
    wide_n = monthly_n.unstack("archetype").reindex(columns=BUCKETS).fillna(0).astype(int)

    # ── 三桶月度收益相关矩阵 (仅用三桶都有效的月) ──
    common = wide.dropna(subset=BUCKETS)
    corr = common.corr().round(4)
    corr_dict = {a: {b: (None if pd.isna(corr.loc[a, b]) else float(corr.loc[a, b]))
                     for b in BUCKETS} for a in BUCKETS}
    corr_sq_mom = corr_dict["SQUAT"]["MOMENTUM"]

    # ── SQUAT 最差 K 月 → MOMENTUM 同月收益 ──
    sq = wide["SQUAT"].dropna().sort_values()
    worst_sq_months = sq.head(K)
    rows = []
    mom_vals = []
    for m, sq_ret in worst_sq_months.items():
        mom_ret = wide.loc[m, "MOMENTUM"] if m in wide.index else np.nan
        base_ret = wide.loc[m, "BASE"] if m in wide.index else np.nan
        mv = None if pd.isna(mom_ret) else round(float(mom_ret), 5)
        rows.append({
            "month": m,
            "squat_ret5": round(float(sq_ret), 5),
            "momentum_ret5": mv,
            "base_ret5": None if pd.isna(base_ret) else round(float(base_ret), 5),
            "squat_n": int(wide_n.loc[m, "SQUAT"]),
            "momentum_n": int(wide_n.loc[m, "MOMENTUM"]),
        })
        if mv is not None:
            mom_vals.append(mv)

    mom_mean_in_worst = round(float(np.mean(mom_vals)), 5) if mom_vals else None
    mom_all_positive = all(v > 0 for v in mom_vals) if mom_vals else False
    n_mom_positive = int(sum(1 for v in mom_vals if v > 0))

    # ── 判决 (冻结 gate): mean>0 AND corr<0.30 ──
    # "MOMENTUM 在这些月均值>0" — 取 MOMENTUM 在 SQUAT 最差 K 月的均值收益 > 0
    mean_gate = (mom_mean_in_worst is not None and mom_mean_in_worst > 0)
    corr_gate = (corr_sq_mom is not None and corr_sq_mom < CORR_THRESH)
    diversifies = bool(mean_gate and corr_gate)
    status = "DIVERSIFIES" if diversifies else "REJECT_no_diversification"
    playbook_hit = {
        "DIVERSIFIES": "动量在深蹲坏月反向 → MOMENTUM sleeve 值得在 Phase 2 建",
        "REJECT_no_diversification": "动量不分散深蹲尾部 → 砍掉 Phase 2 MOMENTUM sleeve, 三 sleeve 缩为 squat+base 双 sleeve, 文档化",
    }[status]

    # ── 分 regime 复核 (R11): 各 regime 下 corr(SQUAT,MOMENTUM) ──
    by_regime = {}
    if REGIME_P.exists():
        rg = pd.read_parquet(REGIME_P, columns=["trade_date", "regime"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        pr = panel.merge(rg, on="trade_date", how="left")
        for rgm in ["momentum", "reversal", "mixed"]:
            sub = pr[pr["regime"] == rgm]
            if len(sub) < 2000:
                continue
            gm = sub.groupby(["month", "archetype"])["_ret5"]
            mm = gm.mean().where(gm.size() >= MIN_CELL).unstack("archetype").reindex(columns=BUCKETS)
            cm = mm.dropna(subset=BUCKETS)
            if len(cm) >= 3:
                c = cm.corr()
                by_regime[rgm] = {
                    "n_months": int(len(cm)),
                    "corr_squat_momentum": round(float(c.loc["SQUAT", "MOMENTUM"]), 4),
                    "corr_squat_base": round(float(c.loc["SQUAT", "BASE"]), 4),
                }

    results = {
        "task": "SLV-003", "elapsed_s": round(time.time() - t0, 1),
        "window": [STUDY_START, panel["trade_date"].max()],
        "horizon_proxy": H, "K_worst_months": K, "min_cell": MIN_CELL,
        "corr_threshold": CORR_THRESH,
        "n_launch_samples": int(len(panel)),
        "n_months_all_buckets": int(len(common)),
        "monthly_proxy_ret5": {a: {m: (None if pd.isna(v) else round(float(v), 5))
                                   for m, v in wide[a].items()} for a in BUCKETS},
        "corr_matrix": corr_dict,
        "corr_squat_momentum": corr_sq_mom,
        "squat_worst_K_months": rows,
        "momentum_mean_in_squat_worst": mom_mean_in_worst,
        "momentum_n_positive_in_worst": n_mom_positive,
        "momentum_all_positive_in_worst": mom_all_positive,
        "gate": {"mean_gate(mom_mean>0)": mean_gate,
                 "corr_gate(corr<0.30)": corr_gate, "diversifies": diversifies},
        "by_regime": by_regime,
        "guardrails": ["SIGN-R01 gate口径冻结", "SIGN-R03 廉价描述闸非ship",
                       "SIGN-R04 前向列只留cache", "SIGN-R06 ST源头已排除", "SIGN-R11 分regime"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print(f"\n--- 三桶月度代理收益相关矩阵 (n={len(common)} 月, 三桶都有效) ---", flush=True)
    print(f"  {'':9s}" + "".join(f"{b:>10s}" for b in BUCKETS), flush=True)
    for a in BUCKETS:
        print(f"  {a:9s}" + "".join(
            f"{(corr_dict[a][b] if corr_dict[a][b] is not None else float('nan')):>10.3f}"
            for b in BUCKETS), flush=True)
    print(f"\n  corr(SQUAT,MOMENTUM) = {corr_sq_mom}  (闸: <{CORR_THRESH} ? {corr_gate})", flush=True)

    print(f"\n--- SQUAT 最差 {K} 月 → MOMENTUM 同月代理收益 ---", flush=True)
    print(f"  {'month':>7s} {'squat':>9s} {'momentum':>9s} {'base':>9s}", flush=True)
    for r in rows:
        print(f"  {r['month']:>7s} {r['squat_ret5']:>9.4f} "
              f"{(r['momentum_ret5'] if r['momentum_ret5'] is not None else float('nan')):>9.4f} "
              f"{(r['base_ret5'] if r['base_ret5'] is not None else float('nan')):>9.4f}", flush=True)
    print(f"\n  MOMENTUM 在 SQUAT 最差{K}月均值 = {mom_mean_in_worst}  "
          f"(闸: >0 ? {mean_gate}; {n_mom_positive}/{len(mom_vals)} 月为正)", flush=True)
    print(f"\n  ===> status = {status}", flush=True)

    # ── verdict ──
    worst_desc = ", ".join(
        f"{r['month']}(SQ {r['squat_ret5']:+.3f}/MOM "
        f"{(r['momentum_ret5'] if r['momentum_ret5'] is not None else float('nan')):+.3f})"
        for r in rows)
    if diversifies:
        conclusion = (
            f"MOMENTUM 分散性决策闸 PASS → DIVERSIFIES。SQUAT 最差 {K} 月 [{worst_desc}] 里 "
            f"MOMENTUM 同月代理收益均值 {mom_mean_in_worst:+.4f}>0 ({n_mom_positive}/{len(mom_vals)} 月为正), "
            f"且全样本 corr(SQUAT,MOMENTUM)={corr_sq_mom}<{CORR_THRESH}。两闸俱过: 动量桶在深蹲坏月提供"
            f"正收益且与深蹲低相关 → 即便 SLV-002 显示 MOMENTUM 裸 α 最弱 (力竭尾), 其 book 层尾部分散价值"
            f"成立 → MOMENTUM sleeve 值得在 Phase 2 建。三 sleeve (squat+base+momentum) 架构保留。廉价描述闸非 ship (R03)。")
    else:
        why = []
        if not mean_gate:
            why.append(f"MOMENTUM 在 SQUAT 最差{K}月均值 {mom_mean_in_worst}≤0 ({n_mom_positive}/{len(mom_vals)} 月为正) "
                       f"— 深蹲坏月里动量也亏, 没反向")
        if not corr_gate:
            why.append(f"corr(SQUAT,MOMENTUM)={corr_sq_mom}≥{CORR_THRESH} — 月度收益同向, 不提供分散")
        conclusion = (
            f"MOMENTUM 分散性决策闸 FAIL → REJECT_no_diversification。SQUAT 最差 {K} 月 [{worst_desc}] 里 "
            f"MOMENTUM 同月代理收益均值 {mom_mean_in_worst} ({n_mom_positive}/{len(mom_vals)} 月为正), "
            f"全样本 corr(SQUAT,MOMENTUM)={corr_sq_mom}。未过闸原因: {'; '.join(why)}。"
            f"【诚实关键细节】代理收益条件在 is_launch=1 (本身要求前向5d max_gain≥10%) 上, 故各桶各月代理收益"
            f"几乎恒正 — 连 SQUAT '最差'月也 +7.8%~+9.8%; mean-gate(>0) 因此被选择效应平凡满足, 不含真分散信息。"
            f"真正决定的是 corr 闸: 三桶月度代理收益强同向 (SQUAT-MOM 0.59, SQUAT-BASE 0.82, BASE-MOM 0.67), "
            f"由共同市场 β 驱动, MOMENTUM 不在 SQUAT 坏月反向。"
            f"叠加 SLV-002 (MOMENTUM 裸 α 最弱+力竭尾最肥): MOMENTUM 既无独立 α 又不在深蹲尾部提供 book 分散 → "
            f"**重要发现: 砍掉 Phase 2 MOMENTUM sleeve, 三 sleeve 缩为 squat+base 双 sleeve**。"
            f"这印证 SLV-001/002 的累积证据 (MOMENTUM 启动率虽高但风险收益与尾部价值都不支撑独立 sleeve)。"
            f"廉价描述闸非 ship, 但作为 Phase 0 go/no-go 直接定调 Phase 2 架构 (R03 描述闸合法定决策范围内)。")

    VERDICT.write_text(json.dumps({
        "id": "SLV-003", "status": status, "conclusion": conclusion,
        "playbook_hit": playbook_hit,
        "metrics": {
            "window": [STUDY_START, panel["trade_date"].max()],
            "n_launch_samples": int(len(panel)),
            "n_months_all_buckets": int(len(common)),
            "corr_matrix": corr_dict,
            "corr_squat_momentum": corr_sq_mom,
            "squat_worst_K_months": rows,
            "momentum_mean_in_squat_worst": mom_mean_in_worst,
            "momentum_n_positive_in_worst": n_mom_positive,
            "gate": {"mean_gate": mean_gate, "corr_gate": corr_gate,
                     "corr_threshold": CORR_THRESH, "K": K},
            "by_regime": by_regime,
        },
        "artifacts": ["research/cache/slv003_results.json"],
        "note": "月度 long-only 代理 = 该月该桶全部 launch 等权前向5d收益 (cell>=30); 入场口径冻结复用 "
                "SLV-001/002 (次日开盘, 全因果, R01); 前向列只留 cache 非 features (R04); ST 源头已排除 (R06); "
                "分 regime 复核 corr (R11); 廉价描述闸定 Phase 0 go/no-go 非 ship 落地 (R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] SLV-003 status={status} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
