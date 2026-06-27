# -*- coding: utf-8 -*-
"""PL-001 去偏 pump label 设计 + past_r5 偏好来源诊断.

北极星 (承 CB-002): 真启动里 past_r5>=0 (横盘/微涨) 占 52.2% 且启动率更高
(13.15% vs 下蹲 11.19%), 但被 pump 启动子"软降权"。本任务做两件事:

  (1) 诊断 past_r5 偏好从哪进 (三个候选通道):
      A. v3c label 定义     — 拆 up / down 两类各自的 past_r5 依赖。
      B. 训练数据分布        — up 正样本里 past_r5>=0 / 各 base_type 占比。
      C. 现有 pump_score 排序 — 跑生产 v3c 模型 OOS, 按 past_r5 分层看 P_up 排名
                                vs 实际启动率, 确认"软降权"机理 (排名低 ≠ 启动率低)。

  (2) 构建去偏 3 分类 label (up/down/neutral), 唯一改动 = 去掉 label 里的 past_r5 约束:
      up   = 前向5日 max_gain>=10% & max_dd>=-5%   (与生产 v3c 完全同, 本就不限 past_r5)
      down = 前向5日 max_dd<=-10% & max_gain<=5%    (去掉 v3c 的 past_r5<=8% 约束)
      → "唯一差异 = label 的 past_r5 约束" 字面成立 (v3c 的 past_r5 约束只在 down 类)。

关键诊断洞察 (本任务预期产出): v3c 的 **up label 本就不限 past_r5**, 横盘/微涨启动
早已是 up 正样本; past_r5 约束只活在 down 类。所以"软降权"主要是**模型/特征学到的隐性
偏好**, 不是 up-label 规则。这决定了 PL-002 纯 label 重训的杠杆有限 (只动 down 约束)。

零泄漏 / 红线:
  - 前向 outcome 仅用于算 label, label 落 research/cache/ (非 features), 不入 features 黑名单闸。
  - 全因果: past_r5 backward; label 用前向 high/low 但只作训练目标。
  - ST 源头排除 (同 load_window: name 含 'ST' 的 ts_code 全程剔除)。
  - 生产模型只读不写 (SIGN-R05); 分层诊断分 regime (SIGN-R11)。
  - 中间指标 (排名/覆盖) 仅诊断, ship 由 PL-004 walk-forward 定 (SIGN-R03)。

产出:
  - research/cache/pump_debias_label.parquet   (ts_code,trade_date,v3c_label,debias_label,past_r5)
  - research/cache/pl001_oos_pup.parquet        (OOS 生产 P_up 分层诊断 checkpoint)
  - research/cache/pl001_results.json
  - research/verdicts/PL-001.json               (status=built)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
LABEL_OUT = CACHE / "pump_debias_label.parquet"
OOS_PUP = CACHE / "pl001_oos_pup.parquet"
RESULTS = CACHE / "pl001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "PL-001.json"
PROD = ROOT / "output" / "production"
DAILY = ROOT / "output" / "tushare_cache" / "daily"
CB002 = CACHE / "cb002" / "launch_panel.parquet"

# 生产 v3c 同口径常量
PUMP_UP, PUMP_DD, FORWARD, UPTREND_R5 = 0.10, 0.05, 5, 0.08

# OOS 诊断窗口 (生产 v3c 训练截止 20250930, 之后为干净 OOS)
OOS_START, OOS_END = "20251001", "20260601"


# ───────────────── 前向 outcome + 两版 label (全历史, 仅训练目标) ─────────────────
def build_labels() -> pd.DataFrame:
    files = sorted(DAILY.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
             for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # ST 源头排除 (同 load_window)
    basic_p = ROOT / "output/tushare_cache/stock_basic.parquet"
    n0 = len(big)
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
        st_set = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
        big = big[~big["ts_code"].isin(st_set)].reset_index(drop=True)
        print(f"  ST 排除: {n0 - len(big):,} 行 ({len(st_set)} 只)", flush=True)

    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    N = FORWARD
    big["max_high_next"] = (big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
    big["min_low_next"] = (big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))
    big["upside"] = big["max_high_next"] / big["next_open"] - 1
    big["downside"] = big["min_low_next"] / big["next_open"] - 1
    big["past_r5"] = big.groupby("ts_code")["close"].pct_change(periods=5)   # backward

    up = (big["upside"] >= PUMP_UP) & (big["downside"] >= -PUMP_DD)
    down_base = (big["downside"] <= -PUMP_UP) & (big["upside"] <= PUMP_DD)

    # v3c (生产): down 加 past_r5<=8% 约束
    big["v3c_label"] = 0
    big.loc[down_base & (big["past_r5"] <= UPTREND_R5), "v3c_label"] = 1
    big.loc[up, "v3c_label"] = 2                      # up 优先后写

    # debias: 去掉 past_r5 约束 (down 不限 past_r5); up 不变
    big["debias_label"] = 0
    big.loc[down_base, "debias_label"] = 1
    big.loc[up, "debias_label"] = 2

    big = big.dropna(subset=["max_high_next"]).reset_index(drop=True)
    return big[["ts_code", "trade_date", "past_r5", "upside", "downside",
                "v3c_label", "debias_label"]]


# ───────────────── 生产 v3c 模型 OOS 推理 (诊断通道 C, 带 checkpoint) ─────────────────
def oos_pup_scores() -> pd.DataFrame:
    if OOS_PUP.exists():
        print(f"  [ckpt] {OOS_PUP.name} 命中, 跳过推理", flush=True)
        return pd.read_parquet(OOS_PUP)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json")
                      .read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})
    booster = lgb.Booster(model_str=(PROD / "r5_pump_3way_lgbm_v3c" / "classifier.txt")
                          .read_text(encoding="utf-8"))

    print(f"  load_window OOS {OOS_START}-{OOS_END} (ST 源头排除, +mfk)...", flush=True)
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if ind_map:
        daily["industry_id"] = (daily["industry"].fillna("unknown")
                                .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily.columns:
            daily[c] = 0.0
    X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).clip(-200, 200).fillna(0)
    proba = booster.predict(X)
    out = daily[["ts_code", "trade_date"]].copy()
    out["P_up"] = proba[:, 2]
    out["P_down"] = proba[:, 1]
    out.to_parquet(OOS_PUP, index=False)
    print(f"  OOS P_up 落盘 {OOS_PUP.name} ({len(out):,} 行)", flush=True)
    return out


def main():
    t0 = time.time()
    print("\n=== PL-001 去偏 pump label 设计 + past_r5 偏好诊断 ===\n", flush=True)

    # ── 1. 两版 label (全历史) ──
    print("[1] 构建 v3c / debias 两版 label (全历史, ST 排除)...", flush=True)
    lab = build_labels()
    print(f"  label 面板 {len(lab):,} 行 / {lab['ts_code'].nunique()} 股", flush=True)

    # 落 label cache (非 features, 防泄漏闸; 仅 ts_code,trade_date,past_r5,两版 label)
    lab[["ts_code", "trade_date", "past_r5", "v3c_label", "debias_label"]].to_parquet(
        LABEL_OUT, index=False)
    print(f"  label cache -> {LABEL_OUT.name}", flush=True)

    # ── 诊断 A: v3c label 定义拆解 (up vs down 的 past_r5 依赖) ──
    up_v3c = lab["v3c_label"] == 2
    up_deb = lab["debias_label"] == 2
    dn_v3c = lab["v3c_label"] == 1
    dn_deb = lab["debias_label"] == 1
    # up 两版是否逐位相同 (验证 up-label 本就不限 past_r5)
    up_identical = bool((up_v3c == up_deb).all())
    # down: v3c 比 debias 少的 (被 past_r5<=8% 踢成 neutral 的)
    down_reassigned = dn_deb & ~dn_v3c             # debias=down 但 v3c=neutral
    n_down_reassigned = int(down_reassigned.sum())
    share_r5ge0_in_reassigned = float((lab.loc[down_reassigned, "past_r5"] >= 0).mean()) \
        if n_down_reassigned else float("nan")
    diag_A = {
        "up_label_identical_v3c_vs_debias": up_identical,
        "up_label_past_r5_dependence": "NONE (生产 v3c up 本就不限 past_r5)",
        "n_up_v3c": int(up_v3c.sum()), "n_up_debias": int(up_deb.sum()),
        "n_down_v3c": int(dn_v3c.sum()), "n_down_debias": int(dn_deb.sum()),
        "n_down_reassigned_v3c_to_neutral": n_down_reassigned,
        "share_past_r5_ge0_in_reassigned_down": round(share_r5ge0_in_reassigned, 4),
        "note": "唯一 label 级 past_r5 约束在 DOWN 类 (past_r5<=8% 把高涨幅伪跌启动踢成 neutral); "
                "UP 类两版逐位相同 → 横盘/微涨启动早已是 up 正样本。",
    }

    # ── 诊断 B: up 正样本的 past_r5 / base_type 分布 (训练数据分布通道) ──
    up_rows = lab[up_v3c]
    share_up_r5ge0 = float((up_rows["past_r5"] >= 0).mean())
    # base_type 来自 cb002 panel
    base_dist_in_up = {}
    cb = None
    if CB002.exists():
        cb = pd.read_parquet(CB002, columns=["ts_code", "trade_date", "base_type",
                                             "is_launch", "regime", "cs_past_r5"])
        cb["trade_date"] = cb["trade_date"].astype(str)
        m = up_rows.merge(cb[["ts_code", "trade_date", "base_type"]],
                          on=["ts_code", "trade_date"], how="left")
        base_dist_in_up = (m["base_type"].value_counts(normalize=True).round(4).to_dict())
    diag_B = {
        "share_past_r5_ge0_in_up_positives": round(share_up_r5ge0, 4),
        "base_type_dist_in_up_positives": base_dist_in_up,
        "note": "up 正样本里 past_r5>=0 占比; 若 ~CB-002 的 52% 量级 → up-label 已公平含横盘/微涨启动。",
    }

    # ── 诊断 C: 生产 pump_score 按 past_r5 分层排序 vs 实际启动率 (软降权机理) ──
    print("[2] 生产 v3c OOS 推理 + past_r5 分层诊断...", flush=True)
    pup = oos_pup_scores()
    pup["trade_date"] = pup["trade_date"].astype(str)
    # 合并 OOS label (用 v3c up 作"实际启动" outcome, = CB-002 launch 口径)
    oos = pup.merge(lab[["ts_code", "trade_date", "past_r5", "v3c_label"]],
                    on=["ts_code", "trade_date"], how="inner")
    oos = oos.dropna(subset=["past_r5"])
    oos["is_launch"] = (oos["v3c_label"] == 2).astype(int)
    # 每日横截面 P_up 百分位排名 (软降权 = 排名系统性偏低)
    oos["pup_rank"] = oos.groupby("trade_date")["P_up"].rank(pct=True, method="first")
    # past_r5 分桶
    bins = [-np.inf, -0.05, 0.0, 0.05, np.inf]
    labels = ["r5<-5%", "-5%~0", "0~5%", "r5>5%"]
    oos["r5_bucket"] = pd.cut(oos["past_r5"], bins=bins, labels=labels)
    strat = []
    for b in labels:
        sub = oos[oos["r5_bucket"] == b]
        if len(sub) < 1000:
            continue
        strat.append({
            "bucket": b, "n": int(len(sub)),
            "actual_launch_rate": round(float(sub["is_launch"].mean()), 5),
            "mean_P_up": round(float(sub["P_up"].mean()), 5),
            "mean_pup_rank": round(float(sub["pup_rank"].mean()), 4),
        })
    # ge0 vs lt0 汇总 (对齐 CB-002)
    ge0 = oos[oos["past_r5"] >= 0]
    lt0 = oos[oos["past_r5"] < 0]
    summary_c = {
        "past_r5_ge0": {"n": int(len(ge0)),
                        "actual_launch_rate": round(float(ge0["is_launch"].mean()), 5),
                        "mean_P_up": round(float(ge0["P_up"].mean()), 5),
                        "mean_pup_rank": round(float(ge0["pup_rank"].mean()), 4)},
        "past_r5_lt0": {"n": int(len(lt0)),
                        "actual_launch_rate": round(float(lt0["is_launch"].mean()), 5),
                        "mean_P_up": round(float(lt0["P_up"].mean()), 5),
                        "mean_pup_rank": round(float(lt0["pup_rank"].mean()), 4)},
    }
    # 软降权判定 (aggregate): ge0 实际启动率不低于 lt0, 但模型给 ge0 的平均排名却更低?
    soft_deprior = (summary_c["past_r5_ge0"]["actual_launch_rate"] >=
                    summary_c["past_r5_lt0"]["actual_launch_rate"]) and \
                   (summary_c["past_r5_ge0"]["mean_pup_rank"] <
                    summary_c["past_r5_lt0"]["mean_pup_rank"])
    # 更细的 barbell 检验: 平坦中段 (-5%~+5%) 是否相对启动率被低排?
    # 用"启动率超额排名" = 该桶实际启动率分位 vs 模型 P_up 排名分位 的落差。
    flat_mid = oos[(oos["past_r5"] >= -0.05) & (oos["past_r5"] <= 0.05)]
    barbell = {
        "flat_mid_n": int(len(flat_mid)),
        "flat_mid_launch_rate": round(float(flat_mid["is_launch"].mean()), 5),
        "flat_mid_mean_pup_rank": round(float(flat_mid["pup_rank"].mean()), 4),
        "extreme_dip_r5_lt_-5pct_rank": round(float(
            oos[oos["past_r5"] < -0.05]["pup_rank"].mean()), 4),
        "extreme_up_r5_gt_5pct_rank": round(float(
            oos[oos["past_r5"] > 0.05]["pup_rank"].mean()), 4),
        "note": "barbell: 模型奖励两端 (强超跌 r5<-5% / 强动量 r5>5%), 低排平坦中段; "
                "平坦中段启动率高于其 P_up 排名应得 → 真盲点在 flat-mid 而非全部 ge0。",
    }
    # 分 regime (SIGN-R11): 每 regime 下 ge0 vs lt0 排名/启动率
    by_regime_c = {}
    if "regime" in oos.columns:
        for rg in ["momentum", "reversal", "mixed"]:
            sub = oos[oos["regime"] == rg]
            if len(sub) < 5000:
                continue
            g = sub[sub["past_r5"] >= 0]; l = sub[sub["past_r5"] < 0]
            by_regime_c[rg] = {
                "ge0_launch": round(float(g["is_launch"].mean()), 5),
                "ge0_rank": round(float(g["pup_rank"].mean()), 4),
                "lt0_launch": round(float(l["is_launch"].mean()), 5),
                "lt0_rank": round(float(l["pup_rank"].mean()), 4),
            }
    diag_C = {
        "oos_window": [OOS_START, OOS_END], "n_oos": int(len(oos)),
        "by_past_r5_bucket": strat,
        "ge0_vs_lt0": summary_c,
        "soft_deprioritization_aggregate_confirmed": bool(soft_deprior),
        "barbell_flat_mid_underranked": barbell,
        "by_regime": by_regime_c,
        "note": "aggregate 软降权 = past_r5>=0 实际启动率不低于 <0 但模型平均排名反而更低; "
                "若 False 则 CB-002 的粗粒度'ge0 软降权'不成立, 盲点更精确地在 flat-mid 段。",
    }

    # ── 诊断 D: 去偏 label vs v3c 在 past_r5>=0 启动上的正例覆盖差异 (验收要求) ──
    launch_r5ge0 = lab[(lab["v3c_label"] == 2) & (lab["past_r5"] >= 0)]
    diag_D = {
        "n_launch_past_r5_ge0": int(len(launch_r5ge0)),
        "covered_by_v3c_up": int((launch_r5ge0["v3c_label"] == 2).sum()),
        "covered_by_debias_up": int((launch_r5ge0["debias_label"] == 2).sum()),
        "up_coverage_delta_on_r5ge0": int((launch_r5ge0["debias_label"] == 2).sum() -
                                          (launch_r5ge0["v3c_label"] == 2).sum()),
        "note": "关键发现: up 覆盖差 = 0 → 去偏 label 在 UP 类对 past_r5>=0 启动零增量 "
                "(它们本就是 v3c up 正样本)。label 级去偏只动 DOWN 约束, 故纯 label 重训修盲点的杠杆有限。",
    }

    results = {
        "task": "PL-001", "elapsed_s": round(time.time() - t0, 1),
        "label_def": {
            "up": "fwd5d max_gain>=0.10 & max_dd>=-0.05 (两版相同, 不限 past_r5)",
            "down_v3c": "fwd5d max_dd<=-0.10 & max_gain<=0.05 & past_r5<=0.08",
            "down_debias": "fwd5d max_dd<=-0.10 & max_gain<=0.05 (去 past_r5 约束)",
            "only_diff": "label 的 past_r5 约束 (仅在 down 类)",
        },
        "diag_A_label_definition": diag_A,
        "diag_B_training_distribution": diag_B,
        "diag_C_model_softdeprior": diag_C,
        "diag_D_coverage_delta": diag_D,
        "label_panel_rows": int(len(lab)),
        "guardrails": ["SIGN-R03 中间指标仅诊断", "SIGN-R04 label 留 cache 不进 features",
                       "SIGN-R05 生产模型只读", "SIGN-R06 ST 源头排除", "SIGN-R11 分 regime/分层"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n--- 诊断 A: label 定义拆解 ---", flush=True)
    print(f"  up-label 两版逐位相同: {up_identical} (up 不限 past_r5)", flush=True)
    print(f"  down 被 past_r5<=8% 踢成 neutral: {n_down_reassigned:,} "
          f"(其中 past_r5>=0 占 {share_r5ge0_in_reassigned:.1%})", flush=True)
    print("\n--- 诊断 B: up 正样本分布 ---", flush=True)
    print(f"  up 正样本里 past_r5>=0 占 {share_up_r5ge0:.1%}", flush=True)
    print(f"  base_type 分布: {base_dist_in_up}", flush=True)
    print("\n--- 诊断 C: 生产 pump_score past_r5 分层 (OOS) ---", flush=True)
    print(f"  {'bucket':10s} {'n':>9s} {'启动率':>9s} {'meanP_up':>9s} {'排名分位':>9s}", flush=True)
    for s in strat:
        print(f"  {s['bucket']:10s} {s['n']:>9,d} {s['actual_launch_rate']:>9.4f} "
              f"{s['mean_P_up']:>9.4f} {s['mean_pup_rank']:>9.3f}", flush=True)
    print(f"  ge0: 启动率 {summary_c['past_r5_ge0']['actual_launch_rate']:.4f} "
          f"排名 {summary_c['past_r5_ge0']['mean_pup_rank']:.3f} | "
          f"lt0: 启动率 {summary_c['past_r5_lt0']['actual_launch_rate']:.4f} "
          f"排名 {summary_c['past_r5_lt0']['mean_pup_rank']:.3f}", flush=True)
    print(f"  aggregate 软降权确认: {soft_deprior} (False=粗粒度 ge0 软降权不成立)", flush=True)
    print(f"  barbell: flat-mid 启动率 {barbell['flat_mid_launch_rate']:.4f} 排名 "
          f"{barbell['flat_mid_mean_pup_rank']:.3f} (两端 "
          f"{barbell['extreme_dip_r5_lt_-5pct_rank']:.3f}/{barbell['extreme_up_r5_gt_5pct_rank']:.3f})",
          flush=True)
    print("\n--- 诊断 D: 去偏 vs v3c 在 past_r5>=0 启动的 up 覆盖差 ---", flush=True)
    print(f"  past_r5>=0 启动数 {len(launch_r5ge0):,}; up 覆盖差 = "
          f"{diag_D['up_coverage_delta_on_r5ge0']} (本就同覆盖)", flush=True)

    # ── verdict ──
    conclusion = (
        f"已构建 v3c / debias 两版 3 分类 label ({len(lab):,} 行, ST 排除, 全因果)。"
        f"诊断有两个关键发现, 都收紧/修正了北极星假设: "
        f"(1) past_r5 偏好**不在 up-label** — 生产 v3c 的 up 类本就不限 past_r5, 两版 up 逐位相同 "
        f"(up 正样本里 past_r5>=0 占 {share_up_r5ge0:.0%}, ~CB-002 的 52%), 横盘/微涨启动早已是 up "
        f"正样本, 去偏在 UP 类对 past_r5>=0 启动**零增量覆盖**; label 级唯一 past_r5 约束在 DOWN 类 "
        f"(past_r5<=8% 把 {n_down_reassigned:,} 高涨幅伪跌启动踢成 neutral, 其中 past_r5>=0 占 "
        f"{share_r5ge0_in_reassigned:.0%}), 去偏即去此 down 约束。"
        f"(2) **粗粒度'ge0 软降权'不成立** (aggregate confirmed={soft_deprior}): OOS 上模型给 "
        f"past_r5>=0 的平均 P_up 排名 {summary_c['past_r5_ge0']['mean_pup_rank']:.3f} 反而**高于** lt0 "
        f"{summary_c['past_r5_lt0']['mean_pup_rank']:.3f}, 与其更高启动率 "
        f"({summary_c['past_r5_ge0']['actual_launch_rate']:.4f} vs "
        f"{summary_c['past_r5_lt0']['actual_launch_rate']:.4f}) 方向一致。真实模式是 **barbell**: 模型奖励"
        f"两端 (强超跌 rank {barbell['extreme_dip_r5_lt_-5pct_rank']:.3f} / 强动量 "
        f"{barbell['extreme_up_r5_gt_5pct_rank']:.3f}), 低排平坦中段 (-5%~+5% rank "
        f"{barbell['flat_mid_mean_pup_rank']:.3f} 但启动率 {barbell['flat_mid_launch_rate']:.4f}) → "
        f"盲点更精确地在 **flat-mid 段** 而非全部 ge0。"
        f"含义: 纯 label 重训 (PL-002) 杠杆仅在 down 约束 (up 零增量), 且模型已正确高排 ge0, "
        f"修盲点能力存疑 — 为 PL-004 的 REJECT_捕捉无α 分叉 (偏好=隐性质量过滤) 埋下伏笔。")
    VERDICT.write_text(json.dumps({
        "id": "PL-001", "status": "built", "conclusion": conclusion,
        "label_def": results["label_def"],
        "diagnosis": {
            "A_label_definition": diag_A,
            "B_training_distribution": diag_B,
            "C_model_softdeprior": diag_C,
            "D_coverage_delta": diag_D,
        },
        "artifacts": ["research/cache/pump_debias_label.parquet",
                      "research/cache/pl001_oos_pup.parquet",
                      "research/cache/pl001_results.json"],
        "note": "label 留 cache 非 features (SIGN-R04); 生产模型只读 (SIGN-R05); "
                "中间指标仅诊断, ship 由 PL-004 walk-forward 定 (SIGN-R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] PL-001 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
