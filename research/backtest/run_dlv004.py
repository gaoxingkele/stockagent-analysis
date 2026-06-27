# -*- coding: utf-8 -*-
"""DLV-004 — C1 r20 特征 block-ablation 剪枝 (卫生消融).

北极星 (Roadmap C1, 卫生交付物, 非研究 gate):
  对 PIT-clean r20 (dep001 剔 holder_pct) 的 246 特征**按语义分块**, 逐块剔除**重训 (embargo
  walk-forward)**, 报各块 ΔRankIC (块剔除后 vs 全特征基线), per-month 配对 bootstrap 噪声带, 找
  **净负块** (剔除反而 IC 升 = 该块净拖后腿)。卫生用途, 效应量大概率 21 月噪声内须标注 (prd 预注册)。

设计 (冻结 R01, 不在结果上调):
  - 全特征集 = dep001 月度 r20 模型 feature_meta 的 246 列 (PIT-clean, 已剔 holder_pct)。
  - 12 个语义块 (估值市值/基本面筹码/趋势均线/动量震荡/波动/量能流动性/K线形态/突破位置/
    资金流主力/金字塔/市场环境/横截面相对), 互斥且覆盖全部 246 列 (脚本启动 assert)。
  - 每个 arm (baseline=全特征, 每块=剔该块) 用**同一月度训练子样本** (同 embargo cut/同 N_TRAIN/
    同 SEED 子采样) 重训 → Δ 是受控差分 (共模相消, 仅测该块净贡献)。
  - 指标 = OOS RankIC (pred_r20 vs r20_fresh, 同 dep001 口径)。r20 是**回归器**故无 AUC, 用 RankIC
    作 ΔAUC 的天然类比 (prd 注: 报 ΔRankIC/Δbook)。
  - N_TRAIN_ABLATE 缩到 300k 提速 (卫生消融, baseline 同步在 300k 重算 → Δ 内部一致 apples-to-apples;
    另报 dep001 900k 基线 IC 作量级参照)。

裁决 (responsePlaybook, 冻结 R02):
  找到净负块 = 某块 mean ΔRankIC > 0 且 per-month bootstrap 95%CI 整段 > 0 (超 21 月噪声) → 剪枝候选。
  全块正/噪声内 = 无净负块或全部效应 CI 含 0 → r20 特征集已干净, 剪枝无实质收益, 文档化。

checkpoint (R08): 每 (month) 落 research/cache/dlv004/ic_by_month.csv (逐 arm IC, 断点续跑)。
ST 源头排除 (load_window exclude_st, R06)。前向列只留 cache (R04)。生产线只读 (R05)。regime 分层 (R11)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))

from train_v15_refresh import load_window
from walk_forward_validation import compute_r20_label
from t005_walk_forward_gate import TEST_MONTHS, EPS
from wfe001_gen_picks import (
    embargo_cut_for_month, DATA_START, DATA_END, N_TRAIN as N_TRAIN_FULL,
    N_TREES_FIXED, SEED, EMBARGO_TDAYS,
)
from walk_forward_validation import get_train_end_for_test_month, get_train_start
from t005_walk_forward_gate import TRAIN_LOOKBACK_MONTHS

OUT_DIR = ROOT / "research" / "cache" / "dlv004"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IC_CSV = OUT_DIR / "ic_by_month.csv"
RESULTS = OUT_DIR / "dlv004_results.json"
DEP_DIAG = ROOT / "research" / "cache" / "dep001" / "r20_oos_diagnostics.csv"
DEP_FMETA = ROOT / "research" / "cache" / "dep001" / "r20_models"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "DLV-004.json"

# ── 冻结参数 (R01) ──
N_TRAIN_ABLATE = 300_000     # 卫生消融提速 (baseline 同步重算 → Δ 内部一致)
N_BOOT = 2000
BOOT_SEED = 20260616
GATE_CI = 0.95               # bootstrap CI 水平

# ── 12 语义块 (互斥覆盖全 246, 启动 assert) ──
BLOCKS = {
    "valuation_size": ["total_mv", "pe", "pe_ttm", "pb"],
    "fundamental_chip": ["winner_rate"],
    "trend_ma": [
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60", "ma_ratio_120",
        "ma5_ma20", "ma20_ma60", "macd", "macd_signal", "macd_hist",
        "lr_slope_20", "lr_slope_60", "lr_angle_20", "channel_pos_60", "lr_r2_60",
        "kama_20", "kama_dist", "sar_signal", "trix", "aroon_up", "aroon_dn",
        "aroon_osc", "ht_trendmode", "ppo", "apo",
    ],
    "oscillator": [
        "adx", "rsi_6", "rsi_14", "rsi_24", "kdj_k", "kdj_d", "kdj_j",
        "roc_10", "roc_20", "cci_14", "wr_14", "mfi_14", "bias_5", "bias_10",
        "bias_20", "cmo_14", "stochrsi_k", "stochrsi_d", "bop",
    ],
    "volatility": ["atr_pct", "natr_14", "boll_pct", "boll_width", "amplitude", "vstd_20"],
    "volume_liquidity": [
        "vol_ratio_5", "vol_ratio_20", "amount_ratio_20", "vol_price_match", "obv",
        "obv_diff_20", "amount_1d", "amount_5d_avg", "amount_20d_avg", "amount_log",
        "amount_zscore_60d", "amount_breakout", "vma_20", "wvma_20", "ad", "adosc",
        "amihud_illiq_5d", "kyle_lambda_20d", "corr_close_vol_20", "cord_ret_vol_20",
    ],
    "candle_pattern": [
        "cdl2crows", "cdl3blackcrows", "cdl3inside", "cdl3linestrike", "cdl3outside",
        "cdl3starsinsouth", "cdl3whitesoldiers", "cdlabandonedbaby", "cdladvanceblock",
        "cdlbelthold", "cdlbreakaway", "cdlclosingmarubozu", "cdlconcealbabyswall",
        "cdlcounterattack", "cdldarkcloudcover", "cdldoji", "cdldojistar",
        "cdldragonflydoji", "cdlengulfing", "cdleveningdojistar", "cdleveningstar",
        "cdlgapsidesidewhite", "cdlgravestonedoji", "cdlhammer", "cdlhangingman",
        "cdlharami", "cdlharamicross", "cdlhighwave", "cdlhikkake", "cdlhikkakemod",
        "cdlhomingpigeon", "cdlidentical3crows", "cdlinneck", "cdlinvertedhammer",
        "cdlkicking", "cdlkickingbylength", "cdlladderbottom", "cdllongleggeddoji",
        "cdllongline", "cdlmarubozu", "cdlmatchinglow", "cdlmathold", "cdlmorningdojistar",
        "cdlmorningstar", "cdlonneck", "cdlpiercing", "cdlrickshawman", "cdlrisefall3methods",
        "cdlseparatinglines", "cdlshootingstar", "cdlshortline", "cdlspinningtop",
        "cdlstalledpattern", "cdlsticksandwich", "cdltakuri", "cdltasukigap", "cdlthrusting",
        "cdltristar", "cdlunique3river", "cdlupsidegap2crows", "cdlxsidegap3methods",
        "k_kmid", "k_klen", "k_kmid2", "k_kup", "k_kup2", "k_klow", "k_klow2",
        "k_ksft", "k_ksft2",
    ],
    "breakout_position": [
        "break_high_20", "break_high_60", "break_low_20", "gap_up", "gap_down",
        "limit_up", "limit_consec", "volume_spike_up", "volume_spike_dn",
        "qtlu_20", "qtld_20", "rank_20", "rsv_20", "imax_20", "imin_20", "imxd_20",
        "cntp_20", "cntn_20", "cntd_20", "sump_20", "sumn_20", "sumd_20",
    ],
    "moneyflow": [
        "mf_divergence", "mf_strength", "mf_consecutive", "sm_net_5d", "mid_net_5d",
        "lg_net_5d", "elg_net_5d", "main_net_5d", "lg_net_20d", "elg_net_20d",
        "main_net_20d", "main_consec_in", "main_consec_out", "dispersion_5d",
        "elg_ratio_5d", "buy_sell_imb_5d", "upward_impact_per_amount", "range_per_amount_5d",
        "tail_risk_5d", "range_compression", "drawdown_recovery", "price_impact_asym",
        "mfk_main_ma5", "mfk_main_ma20", "mfk_main_ma_diff", "mfk_main_cross_state",
        "mfk_main_days_in_cross", "mfk_main_cross_strength", "mfk_main_macd",
        "mfk_main_macd_hist", "mfk_main_macd_zero_dist", "mfk_smart_dumb_spread",
        "mfk_pyramid_top_heavy", "mfk_inst_acc_velocity", "mfk_main_price_sync_20d",
        "mfk_main_lead_5d", "mfk_main_velocity_5d", "mfk_main_consec_above_ma20",
        "mfk_main_acc_ratio_60d", "f1_neg1", "f2_pos1", "f1_neg5", "f2_pos5",
        "f1_neg8", "f2_pos8",
    ],
    "pyramid": [
        "pyr_5d", "pyr_10d", "pyr_20d", "pyr_60d", "pyr_acc_5_20",
        "pyr_velocity_10_60", "pyr_velocity_5_60", "pyr_velocity_10_30",
        "pyr_velocity_20_60", "pyr_velocity_5_30",
    ],
    "market_context": [
        "market_score_adj", "regime_id", "mkt_ret_5d", "mkt_ret_20d", "mkt_ret_60d",
        "mkt_rsi14", "mkt_vol_ratio", "regime_days_in", "regime_intensity",
        "hs300_ret60_z60", "cyb_rel_strength", "zz500_rel_strength",
    ],
    "cross_sectional_rel": [
        "industry_id", "rs_in_decile", "long_return_252d", "long_return_504d",
        "long_return_252d_decile", "long_return_504d_decile", "industry_return_252d",
        "industry_return_504d", "industry_return_504d_decile", "concept_return_504d_mean",
        "relative_strength_252d", "relative_strength_504d",
    ],
}


def load_full_feat_cols() -> list:
    """全特征集 = dep001 月度 r20 模型 feature_meta (246, PIT-clean 已剔 holder_pct)。"""
    metas = sorted(DEP_FMETA.glob("*/feature_meta.json"))
    if not metas:
        raise RuntimeError("缺 dep001 r20 feature_meta — 先跑 dep001_gen_picks.py")
    return json.loads(metas[0].read_text(encoding="utf-8"))["feature_cols"]


def fit_predict_ic(sub: pd.DataFrame, df_test: pd.DataFrame, feat_cols: list) -> float:
    """在子样本上 fit r20 回归器 (同 dep001 config), 对测试月预测, 返回 OOS RankIC。"""
    cat = ["industry_id"] if "industry_id" in feat_cols else "auto"
    Xtr = sub[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    ytr = sub["r20"].astype("float32")
    clf = lgb.LGBMRegressor(
        n_estimators=N_TREES_FIXED, learning_rate=0.04, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1, max_bin=127, force_col_wise=True,
        random_state=SEED, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(Xtr, ytr, categorical_feature=cat)
    Xte = df_test[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    pred = clf.booster_.predict(Xte)
    msk = df_test["r20_fresh"].notna()
    if msk.sum() <= 50:
        return float("nan")
    fresh = df_test.loc[msk, "r20_fresh"].clip(-30, 30)
    ic = pd.Series(pred[msk.values]).corr(pd.Series(fresh.values), method="spearman")
    return float(ic) if pd.notna(ic) else float("nan")


def main():
    t0 = time.time()
    print("\n=== DLV-004: r20 特征 block-ablation 剪枝 (卫生) ===\n", flush=True)

    full_fc = load_full_feat_cols()
    print(f"[feat] 全特征集 {len(full_fc)} (dep001 PIT-clean)", flush=True)

    # ── 块互斥覆盖自检 (R01 设计冻结) ──
    assigned = [c for blk in BLOCKS.values() for c in blk]
    assert len(assigned) == len(set(assigned)), "块间有重叠特征"
    miss = set(full_fc) - set(assigned)
    extra = set(assigned) - set(full_fc)
    assert not miss, f"未分块特征: {sorted(miss)}"
    assert not extra, f"分块含全集外特征: {sorted(extra)}"
    print(f"[blocks] {len(BLOCKS)} 块互斥覆盖全 {len(assigned)} 特征 OK "
          f"({', '.join(f'{k}:{len(v)}' for k,v in BLOCKS.items())})", flush=True)

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)
    for c in full_fc:
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    cal_all = sorted(daily["trade_date"].unique())
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / {len(cal_all)} 交易日", flush=True)

    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)

    arms = ["baseline"] + list(BLOCKS.keys())
    done = {}
    if IC_CSV.exists():
        prev = pd.read_csv(IC_CSV, dtype={"month": str})
        for _, r in prev.iterrows():
            done.setdefault(str(r["month"]), {})[r["arm"]] = r["rankic"]
    rows = [] if not IC_CSV.exists() else pd.read_csv(IC_CSV, dtype={"month": str}).to_dict("records")

    for m_ in TEST_MONTHS:
        if m_ in done and all(a in done[m_] for a in arms):
            continue
        embargo_cut = embargo_cut_for_month(cal_all, m_)
        if embargo_cut is None:
            print(f"  {m_}: 无 embargo 截止日, 跳过", flush=True)
            continue
        train_end = get_train_end_for_test_month(m_)
        train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
        df_tr = daily[(daily["trade_date"] >= train_start) & (daily["trade_date"] <= embargo_cut)]
        sub = df_tr.dropna(subset=["r20"]).copy()
        if len(sub) < 100_000:
            print(f"  {m_}: 训练样本不足 {len(sub)}, 跳过", flush=True)
            continue
        if len(sub) > N_TRAIN_ABLATE:
            sub = sub.sample(n=N_TRAIN_ABLATE, random_state=SEED).reset_index(drop=True)

        df_test = daily[daily["trade_date"].str.startswith(m_)].copy()
        df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
        if df_test["r20_fresh"].notna().sum() <= 50:
            print(f"  {m_}: r20_fresh 不足, 跳过", flush=True)
            continue

        mdone = done.get(m_, {})
        for arm in arms:
            if arm in mdone:
                continue
            fc = full_fc if arm == "baseline" else [c for c in full_fc if c not in set(BLOCKS[arm])]
            ic = fit_predict_ic(sub, df_test, fc)
            rows = [r for r in rows if not (str(r["month"]) == m_ and r["arm"] == arm)]
            rows.append({"month": m_, "arm": arm, "rankic": ic, "n_feat": len(fc)})
            pd.DataFrame(rows).to_csv(IC_CSV, index=False)
            mdone[arm] = ic
        done[m_] = mdone
        base_ic = mdone["baseline"]
        print(f"  {m_}: baseline IC={base_ic:+.4f} | "
              + " ".join(f"{a[:6]}Δ{mdone[a]-base_ic:+.3f}" for a in arms[1:])
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        del sub, df_test
        gc.collect()

    aggregate(full_fc)
    print(f"\n[done] 耗时 {(time.time()-t0)/60:.1f} min", flush=True)


def aggregate(full_fc: list):
    ic = pd.read_csv(IC_CSV, dtype={"month": str})
    piv = ic.pivot(index="month", columns="arm", values="rankic").sort_index()
    months = list(piv.index)
    base = piv["baseline"].values

    # month -> regime (OOS 月主导 regime)
    reg = pd.read_parquet(REGIME)[["trade_date", "regime"]]
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    mreg = reg.groupby("month")["regime"].agg(lambda s: s.value_counts().idxmax()).to_dict()

    rng = np.random.default_rng(BOOT_SEED)
    n = len(months)
    block_stats = {}
    for blk in BLOCKS:
        if blk not in piv.columns:
            continue
        arm = piv[blk].values
        delta = arm - base                      # >0 = 剔该块 IC 升 = 净负块 (拖后腿)
        valid = np.isfinite(delta)
        d = delta[valid]
        idx = np.arange(n)[valid]
        if len(d) < 3:
            continue
        boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [(1-GATE_CI)/2*100, (1+GATE_CI)/2*100])
        # regime 分层 (R11)
        reg_delta = {}
        for rg in ("momentum", "reversal", "mixed"):
            sel = [i for i in idx if mreg.get(months[i]) == rg]
            if sel:
                reg_delta[rg] = float(np.mean(delta[sel]))
        net_negative = bool(np.mean(d) > 0 and lo > 0)   # 超噪声的净负块
        block_stats[blk] = {
            "n_feat_in_block": len(BLOCKS[blk]),
            "mean_delta_rankic": float(np.mean(d)),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "net_negative_block": net_negative,
            "regime_delta": reg_delta,
            "n_months": int(len(d)),
        }

    # dep001 900k 基线 IC (量级参照)
    dep_full_ic = np.nan
    if DEP_DIAG.exists():
        dd = pd.read_csv(DEP_DIAG, dtype={"month": str})
        dep_full_ic = float(dd["r20_oos_rankic"].mean())

    net_neg = sorted([b for b, s in block_stats.items() if s["net_negative_block"]],
                     key=lambda b: -block_stats[b]["mean_delta_rankic"])
    any_ci_excl = [b for b, s in block_stats.items() if s["ci_excludes_zero"]]

    if net_neg:
        status = "找到净负块"
        playbook = "找到净负块"
        conclusion = (
            f"r20 246 特征 {len(BLOCKS)} 块 embargo-WF block-ablation: 净负块(剔除后 IC 升且超噪声) = "
            f"{net_neg} → 剪枝候选; 须 bootstrap 确认效应超噪声方纳入 V12.32。")
    else:
        status = "全块正/噪声内"
        playbook = "全块正/噪声内"
        ranked = sorted(block_stats.items(), key=lambda kv: -kv[1]["mean_delta_rankic"])
        top = ", ".join(f"{b}(Δ{s['mean_delta_rankic']:+.4f}, CI[{s['ci_low']:+.4f},{s['ci_high']:+.4f}])"
                        for b, s in ranked[:3])
        conclusion = (
            f"r20 246 特征 {len(BLOCKS)} 块 embargo-WF block-ablation (N_train={N_TRAIN_ABLATE//1000}k, "
            f"{len(months)} 月): **无净负块** (无块剔除后 IC 显著上升)。各块 ΔRankIC 95%CI 均{'含 0' if not any_ci_excl else '多数含 0'} "
            f"→ 效应落 21 月噪声带内, r20 特征集已干净, 剪枝无实质收益。最'拖后腿'(Δ最大但仍噪声内): {top}。"
            f" baseline(300k) 全期 IC={float(np.nanmean(base)):+.4f} (dep001 900k 参照 {dep_full_ic:+.4f})。")

    out = {
        "task": "DLV-004", "status": status, "playbook_hit": playbook,
        "conclusion": conclusion,
        "n_features_total": len(full_fc), "n_blocks": len(BLOCKS),
        "n_test_months": len(months), "n_train_ablate": N_TRAIN_ABLATE,
        "metric": "OOS RankIC (pred_r20 vs r20_fresh); r20 是回归器故无 AUC, RankIC 作 ΔAUC 类比",
        "baseline_ic_300k_mean": float(np.nanmean(base)),
        "dep001_baseline_ic_900k_mean": dep_full_ic,
        "net_negative_blocks": net_neg,
        "blocks_ci_excludes_zero": any_ci_excl,
        "block_stats": block_stats,
        "months": months,
        "frozen_params": {
            "n_train_ablate": N_TRAIN_ABLATE, "n_trees": N_TREES_FIXED,
            "embargo_tdays": EMBARGO_TDAYS, "n_boot": N_BOOT, "ci": GATE_CI, "seed": SEED,
        },
        "discipline": "R01 块/阈值冻结; R02 描述性消融=合法完成; R03 中间IC≠ship; "
                      "R05 仅 load 缓存 feature_meta 0 生产改动; R06 ST 源头排除; "
                      "R08 月度 checkpoint; R11 regime 分层",
    }
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    VERDICT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status}", flush=True)
    print(f"  {conclusion}", flush=True)
    print(f"  net_negative_blocks={net_neg}", flush=True)
    for b, s in sorted(block_stats.items(), key=lambda kv: -kv[1]["mean_delta_rankic"]):
        print(f"    {b:22s} ΔRankIC={s['mean_delta_rankic']:+.4f} "
              f"CI[{s['ci_low']:+.4f},{s['ci_high']:+.4f}] "
              f"{'NET-NEG' if s['net_negative_block'] else ('CI≠0' if s['ci_excludes_zero'] else 'noise')}",
              flush=True)
    print(f"  -> {RESULTS.relative_to(ROOT)} + {VERDICT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
