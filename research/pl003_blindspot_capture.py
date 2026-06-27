# -*- coding: utf-8 -*-
"""PL-003 盲点捕捉验证 — debias pump 是否把 past_r5>=0 启动排上来?

承 PL-001/PL-002 (收紧后的北极星):
  - up 标签两臂逐位相同 (横盘/微涨启动早已是 up 正样本); 去偏全部杠杆在 down 类
    (去 past_r5<=8% 约束把高涨幅伪跌启动收回 down 正样本)。
  - 推理端 ratio 相关仅 0.576 → 池内排序确被搅动, 但全经 down 约束放松对 3way
    softmax 分母 (P_down) 的间接影响。**搅动方向**是否把 past_r5>=0 (尤其 flat-mid
    -5%~+5%) 启动排上来 = 本任务核心。

  机理预判 (待数据证伪): 去偏 down 把高 past_r5 (上涨) 样本纳入 down 正样本 →
  past_r5>=0 股推理 P_down 上升 → ratio=P_up/(P_down+eps) 反而**下降** → 去偏可能
  把 past_r5>=0 启动排得**更低**而非更高。若如此则"捕捉失败", 盲点未修到。

本任务 (PL-003, 中间验证非落地 SIGN-R03) 做的事:
  用 pl002_pump_scores.parquet (19 月 walk-forward 两臂分数, apples-to-apples),
  以 v3c up 正例 (v3c_label==2, 两臂共有 ground truth) 为"启动"事件, 比较两臂在
  past_r5>=0 / flat-mid / 分 base_type / 分 regime 启动上的:
    - 每日横截面 ratio 排名分位 (生产 V12.31 用 ratio 排序) + P_up 排名分位
    - recall@K% (启动落入 top-K% 的比例) + precision@K% (top-K% 里盲点启动占比)
    - debias - v3c 的 Δ → 捕捉成功/失败裁决

裁决规则 (描述性, PL-003 非 ship gate; ship 由 PL-004 walk-forward α 定):
  捕捉成功 = debias 把 flat-mid 启动的平均 ratio 排名分位抬升 >= +0.02 且
            recall@20% 改善 > 0 (盲点被排上来)。
  捕捉失败 = 否 (label 改了但模型/特征排序仍不偏向 flat-mid 启动) → 文档化,
            并按 PL-003 验收要求把 PL-004 标记 skip。

零泄漏 / 红线:
  - 仅消费 cache 分数 + cache base_type/regime, 不碰生产模型/文件 (SIGN-R05)。
  - 排名/recall 是中间指标, 仅描述捕捉, 不作 ship 依据 (SIGN-R03)。
  - 分 regime (SIGN-R11) + 分形态 (SIGN-R06 分层)。ST 已在 PL-002 源头排除。

产出:
  - research/cache/pl003_capture.parquet   (启动子集明细: 两臂 ratio/pup 排名分位)
  - research/cache/pl003_results.json
  - research/verdicts/PL-003.json
  - (若捕捉失败) plans/prd.json: PL-004 skip=true
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache"
SCORES = CACHE / "pl002_pump_scores.parquet"
LAUNCH_PANEL = CACHE / "cb002" / "launch_panel.parquet"
CAP_OUT = CACHE / "pl003_capture.parquet"
RESULTS = CACHE / "pl003_results.json"
VERDICT = ROOT / "research" / "verdicts" / "PL-003.json"
PRD = ROOT / "plans" / "prd.json"

# 捕捉成功阈值 (描述性, 非 ship gate)
RANK_LIFT_MIN = 0.02     # flat-mid 启动平均 ratio 排名分位抬升下限
KS = [0.10, 0.20]        # top-K% recall / precision


def daily_rank_pct(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date")[col].rank(pct=True, method="first")


def capture_table(sub: pd.DataFrame, name: str) -> dict:
    """对一个启动子集 (sub 已筛为 launch) 算两臂排名/recall/Δ。"""
    if len(sub) < 200:
        return {"subset": name, "n_launch": int(len(sub)), "skip": "n<200"}
    out = {"subset": name, "n_launch": int(len(sub))}
    # 排名分位均值 (ratio 是生产排序信号; P_up 辅证)
    for sig in ["ratio", "pup"]:
        v3c_r = float(sub[f"v3c_{sig}_rank"].mean())
        deb_r = float(sub[f"debias_{sig}_rank"].mean())
        out[f"{sig}_rank_v3c"] = round(v3c_r, 4)
        out[f"{sig}_rank_debias"] = round(deb_r, 4)
        out[f"{sig}_rank_delta"] = round(deb_r - v3c_r, 4)
    # recall@K (启动落入 top-K%): rank>= 1-K
    for k in KS:
        thr = 1.0 - k
        rv = float((sub["v3c_ratio_rank"] >= thr).mean())
        rd = float((sub["debias_ratio_rank"] >= thr).mean())
        out[f"recall@{int(k*100)}_v3c"] = round(rv, 4)
        out[f"recall@{int(k*100)}_debias"] = round(rd, 4)
        out[f"recall@{int(k*100)}_delta"] = round(rd - rv, 4)
    return out


def main():
    t0 = time.time()
    print("\n=== PL-003 盲点捕捉验证 (debias vs v3c, past_r5>=0 启动) ===\n", flush=True)

    df = pd.read_parquet(SCORES)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"[data] pl002 分数 {len(df):,} 行 / {df['ts_code'].nunique()} 股 "
          f"({df['trade_date'].min()}-{df['trade_date'].max()})", flush=True)

    # 合并 base_type + regime (来自 cb002 launch_panel)
    lp = pd.read_parquet(LAUNCH_PANEL, columns=["ts_code", "trade_date", "base_type", "regime"])
    lp["trade_date"] = lp["trade_date"].astype(str)
    n0 = len(df)
    df = df.merge(lp, on=["ts_code", "trade_date"], how="left")
    assert len(df) == n0, "merge 改变行数"
    print(f"[data] 合并 base_type/regime 覆盖 {df['base_type'].notna().mean():.1%}", flush=True)

    # 每日横截面排名分位 (ratio 主 / P_up 辅)
    print("[rank] 每日横截面排名分位 (ratio + P_up, 两臂)...", flush=True)
    df["v3c_ratio_rank"] = daily_rank_pct(df, "v3c_ratio")
    df["debias_ratio_rank"] = daily_rank_pct(df, "debias_ratio")
    df["v3c_pup_rank"] = daily_rank_pct(df, "v3c_pump_up")
    df["debias_pup_rank"] = daily_rank_pct(df, "debias_pump_up")

    # ground truth 启动 = v3c up 正例 (两臂共有 up 标签)
    df["is_launch"] = (df["v3c_label"] == 2)
    n_launch_all = int(df["is_launch"].sum())
    print(f"[gt] 启动事件 (v3c up 正例) {n_launch_all:,} / {len(df):,} "
          f"({n_launch_all/len(df):.2%})", flush=True)

    launches = df[df["is_launch"]].copy()

    # ── 启动子集定义 ──
    subsets = {}
    subsets["all_launch"] = launches
    subsets["past_r5_ge0"] = launches[launches["past_r5"] >= 0]
    subsets["past_r5_lt0"] = launches[launches["past_r5"] < 0]
    subsets["flat_mid(-5~5%)"] = launches[(launches["past_r5"] >= -0.05) &
                                          (launches["past_r5"] <= 0.05)]
    # 分形态 (SIGN-R06): 横盘 flat_base / 微涨 up_drift, 且 past_r5>=0
    for bt in ["flat_base", "up_drift", "pullback", "other"]:
        subsets[f"base={bt}&ge0"] = launches[(launches["base_type"] == bt) &
                                             (launches["past_r5"] >= 0)]
    # 分 regime (SIGN-R11): flat-mid 启动在各 regime
    for rg in ["momentum", "reversal", "mixed"]:
        subsets[f"flatmid&{rg}"] = launches[(launches["past_r5"] >= -0.05) &
                                            (launches["past_r5"] <= 0.05) &
                                            (launches["regime"] == rg)]

    tables = [capture_table(s, n) for n, s in subsets.items()]

    # ── precision@K: top-K% (按 debias/v3c ratio) 里盲点启动 (flat-mid 启动) 占比 ──
    prec = {}
    fm_mask = (df["past_r5"] >= -0.05) & (df["past_r5"] <= 0.05) & df["is_launch"]
    ge0_mask = (df["past_r5"] >= 0) & df["is_launch"]
    for k in KS:
        thr = 1.0 - k
        row = {}
        for arm in ["v3c", "debias"]:
            sel = df[df[f"{arm}_ratio_rank"] >= thr]
            row[f"{arm}_topK_n"] = int(len(sel))
            row[f"{arm}_prec_flatmid_launch"] = round(
                float((fm_mask & (df[f"{arm}_ratio_rank"] >= thr)).sum() / max(len(sel), 1)), 5)
            row[f"{arm}_prec_ge0_launch"] = round(
                float((ge0_mask & (df[f"{arm}_ratio_rank"] >= thr)).sum() / max(len(sel), 1)), 5)
        row["flatmid_prec_delta"] = round(
            row["debias_prec_flatmid_launch"] - row["v3c_prec_flatmid_launch"], 5)
        row["ge0_prec_delta"] = round(
            row["debias_prec_ge0_launch"] - row["v3c_prec_ge0_launch"], 5)
        prec[f"top{int(k*100)}pct"] = row

    # ── 捕捉裁决 (flat-mid 是 PL-001 锁定的真盲点) ──
    fm = next(t for t in tables if t["subset"] == "flat_mid(-5~5%)")
    fm_rank_lift = fm.get("ratio_rank_delta", float("nan"))
    fm_recall20_lift = fm.get("recall@20_delta", float("nan"))
    ge0 = next(t for t in tables if t["subset"] == "past_r5_ge0")
    ge0_rank_lift = ge0.get("ratio_rank_delta", float("nan"))

    captured = bool((fm_rank_lift >= RANK_LIFT_MIN) and (fm_recall20_lift > 0))
    playbook = "捕捉成功" if captured else "捕捉失败"

    results = {
        "task": "PL-003", "elapsed_s": round(time.time() - t0, 1),
        "sort_signal": "ratio = P_up/(P_down+eps) (生产 V12.31 池内排序信号); P_up 辅证",
        "ground_truth_launch": "v3c_label==2 (up 正例, 两臂共有 up 标签)",
        "n_launch_all": n_launch_all,
        "capture_tables": tables,
        "precision_at_K": prec,
        "decision": {
            "rule": f"捕捉成功 = flat-mid 启动 ratio 排名分位抬升 >= +{RANK_LIFT_MIN} 且 recall@20% 改善 > 0",
            "flatmid_ratio_rank_delta": fm_rank_lift,
            "flatmid_recall@20_delta": fm_recall20_lift,
            "ge0_ratio_rank_delta": ge0_rank_lift,
            "captured": captured, "playbook": playbook,
        },
        "guardrails": ["SIGN-R03 排名/recall 仅描述捕捉非 ship", "SIGN-R05 生产模型只读",
                       "SIGN-R06 分形态", "SIGN-R11 分 regime"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 明细落 cache (诊断用)
    keep = ["ts_code", "trade_date", "past_r5", "base_type", "regime",
            "v3c_ratio_rank", "debias_ratio_rank", "v3c_pup_rank", "debias_pup_rank"]
    launches[keep].to_parquet(CAP_OUT, index=False)

    # ── 打印 ──
    print("\n--- 捕捉表 (启动子集; ratio 排名分位 + recall@K, debias - v3c) ---", flush=True)
    print(f"  {'子集':22s} {'n':>7s} {'ratioR_v3c':>10s} {'ratioR_deb':>10s} "
          f"{'Δrank':>8s} {'rec@20Δ':>8s}", flush=True)
    for t in tables:
        if "skip" in t:
            print(f"  {t['subset']:22s} {t['n_launch']:>7,d}  (skip {t['skip']})", flush=True)
            continue
        print(f"  {t['subset']:22s} {t['n_launch']:>7,d} {t['ratio_rank_v3c']:>10.4f} "
              f"{t['ratio_rank_debias']:>10.4f} {t['ratio_rank_delta']:>+8.4f} "
              f"{t['recall@20_delta']:>+8.4f}", flush=True)
    print("\n--- precision@K (top-K% 里 flat-mid / ge0 启动占比) ---", flush=True)
    for k, r in prec.items():
        print(f"  {k}: flatmid prec v3c {r['v3c_prec_flatmid_launch']:.5f} → "
              f"debias {r['debias_prec_flatmid_launch']:.5f} (Δ {r['flatmid_prec_delta']:+.5f}); "
              f"ge0 Δ {r['ge0_prec_delta']:+.5f}", flush=True)
    print(f"\n[decision] flat-mid Δrank={fm_rank_lift:+.4f} (bar +{RANK_LIFT_MIN}) "
          f"recall@20Δ={fm_recall20_lift:+.4f} → 捕捉={captured} ({playbook})", flush=True)

    # ── verdict ──
    if captured:
        conclusion = (
            f"**捕捉成功**: debias 把 flat-mid (-5%~+5%) 启动的平均 ratio 排名分位抬升 "
            f"{fm_rank_lift:+.4f} (>= +{RANK_LIFT_MIN}) 且 recall@20% 改善 {fm_recall20_lift:+.4f} > 0 "
            f"→ 去偏确把 PL-001 锁定的真盲点 (flat-mid 启动) 排上来。进 PL-004 walk-forward gate "
            f"看捕捉了能否增 α (中间捕捉 ≠ 落地, SIGN-R03)。")
        pl004_skip = False
    else:
        conclusion = (
            f"**捕捉失败**: debias 对 flat-mid 启动 ratio 排名分位 Δ={fm_rank_lift:+.4f} "
            f"(未达 +{RANK_LIFT_MIN}), recall@20% Δ={fm_recall20_lift:+.4f}; past_r5>=0 启动整体 "
            f"Δrank={ge0_rank_lift:+.4f}。印证 PL-001/PL-002 机理预判: 去偏全部杠杆在 down 约束放松, "
            f"对 past_r5>=0 股推理抬高 P_down → ratio=P_up/(P_down+eps) 分母变大, **不偏向甚至压低** "
            f"flat-mid 启动排序 (label 改了但模型/特征排序仍不修这一段)。盲点未被排上来 → 按 PL-003 "
            f"验收把 PL-004 标记 skip (捕捉是 walk-forward 的前提, 前提不成立则无须跑 gate, "
            f"playbook 对应 REJECT_未捕捉)。重要发现: 纯 label 去偏 (只动 down past_r5 约束) "
            f"无法把 flat-mid 启动排上来 — 修盲点需动 up 端正样本定义或特征, 非本轮 down 约束所能及。")
        pl004_skip = True

    VERDICT.write_text(json.dumps({
        "id": "PL-003", "status": playbook, "conclusion": conclusion,
        "sort_signal": results["sort_signal"],
        "decision": results["decision"],
        "capture_tables": tables,
        "precision_at_K": prec,
        "artifacts": ["research/cache/pl003_capture.parquet",
                      "research/cache/pl003_results.json"],
        "note": "排名/recall 仅描述捕捉非 ship (SIGN-R03); 生产模型只读 (SIGN-R05); "
                "分形态 (SIGN-R06) + 分 regime (SIGN-R11)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 若捕捉失败, 按 PL-003 验收把 PL-004 标记 skip ──
    prd = json.loads(PRD.read_text(encoding="utf-8"))
    for feat in prd["features"]:
        if feat["id"] == "PL-004":
            if pl004_skip:
                feat["skip"] = True
                feat["skipReason"] = ("PL-003 捕捉失败: debias 未把 flat-mid 启动排上来 "
                                      "(ratio 排名分位无抬升), 捕捉是 walk-forward 的前提, "
                                      "前提不成立故 skip gate (playbook REJECT_未捕捉)。")
                feat["notes"] = (feat.get("notes", "") +
                                 " [PL-003 置 skip] 捕捉失败, 见 verdicts/PL-003.json。")
            break
    PRD.write_text(json.dumps(prd, ensure_ascii=False, indent=2), encoding="utf-8")
    if pl004_skip:
        print("\n[prd] PL-003 捕捉失败 → PL-004 已标记 skip", flush=True)

    print(f"\n[done] PL-003 status={playbook} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
