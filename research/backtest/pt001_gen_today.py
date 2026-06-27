# -*- coding: utf-8 -*-
"""PT-001 — 生成"今日"(最新可用交易日) V12.31-clean picks + append-only 落库 (启动前向 paper-trade).

北极星 (运维落库, 非研究 gate):
  dep001_gen_picks 是 walk-forward 批量器 (按 TEST_MONTHS 跑历史 OOS, s5 用 t005 缓存月度模型, 止于
  202604), 无"单日 / 当期"入口。本脚本 = dep001 的**单日 live 扩展**: 给最新可用交易日 (本地数据/特征到
  0611, Tushare 到 0615) 打分, 生成 V12.31-clean picks, 经 paper_trade_harness.log_picks_for_date
  **append-only** 落库, 作为前向 paper-trade 的第一笔真实记录。

口径 (逐字同 V12.31-clean / dep001, R01 冻结):
  V7c 池 (r20 top5% × pyr_velocity 过滤 × pump_down<0.6) → ratio_s5 排序 → 双轨 build_dual (70/20/10) →
  行业 cap (CAP_IN/CAP_CROSS 0.2, 每轨 MAX_A=8/MAX_B=15 → 行业 ≤4) → close-based → ST 源头排除 (R06)。

模型 (PIT-clean live):
  - r20 池 filter = **当期 PIT-clean r20** (复用 dep001 train_r20_month_clean 逻辑): 剔 holder_pct
    (PIT 前视), label-availability embargo (训练截止 = 预测日 − 21 交易日, 保证 r20 前向 20 日 label 在
    预测日前已可知, R04), 24m lookback, 固定 120 树, 同超参。**非生产 in-sample r20** (那是 +0.44 IC
    记忆来源, 含 holder_pct)。
  - s5 排序 = **生产 v3c pump 模型** (r5_pump_3way_lgbm_v3c, class {0:neutral,1:down,2:up})。对一个
    真正向 (forward) 的交易日, 生产模型 = 训练于过去数据、对未来打分, **因果无泄漏** (label 尚不存在,
    无从泄漏); 这正是 V12.31-clean 上线后每日 live 推理的 s5 来源 (dep001 用 t005 walk-forward s5 仅因
    它在回测历史月, 用生产 v3c 会共模)。class 序与 dep001 一致 (ratio=P_up/P_down, pump_down=P_down)。

生产线 v12_scoring.py / v12_dual_track.py **只读冻结** (R05, 本任务只 load 缓存模型/价表, 0 生产改动)。
落库 append-only, existing 拒覆盖 (SIGN-R01/R02 红线)。大缓存 gitignored (R04)。长任务 r20 模型 checkpoint (R08)。
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
from walk_forward_validation import compute_r20_label, compute_ind_mom, get_train_start
from t005_walk_forward_gate import build_dual, EPS, TRAIN_LOOKBACK_MONTHS
from wf001_gen_picks import build_r20_feat_cols
from wfe001_gen_picks import DATA_START, N_TRAIN, N_TREES_FIXED, SEED, EMBARGO_TDAYS, R20_HORIZON
from paper_trade_harness import log_picks_for_date, read_log

PROD = ROOT / "output" / "production"
OUT_DIR = ROOT / "research" / "cache" / "pt001"
OUT_DIR.mkdir(parents=True, exist_ok=True)
R20_MODEL_DIR = OUT_DIR / "r20_model"
R20_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 单日 live 入口参数化 (PT-001 默认 0611; PT-002 起指最新可用交易日, 如 20260615):
#   python pt001_gen_today.py [YYYYMMDD] [PT-XXX]
#   - 第一个 8 位数字 = 数据 as-of / 目标交易日 (默认 20260611, 保持 PT-001 行为不变)
#   - PT-\d+ token   = verdict id (默认 PT-001), 决定写哪个 verdicts/<id>.json
import re as _re
_args = sys.argv[1:]
TARGET_END = next((a for a in _args if a.isdigit() and len(a) == 8), "20260611")
VERDICT_ID = next((a for a in _args if _re.fullmatch(r"PT-\d+", a or "")), "PT-001")
VERDICT = ROOT / "research" / "verdicts" / f"{VERDICT_ID}.json"
DROP_LEAK = "holder_pct"
SOURCE = "v12.31-clean-live"


def train_r20_clean_for_date(daily_full: pd.DataFrame, feat_cols: list,
                             target_date: str, cal_all: list) -> tuple:
    """当期 PIT-clean r20: embargo cut = 预测日 − EMBARGO_TDAYS 交易日 (date-based, 非 month-based)。

    复用 dep001 train_r20_month_clean 的全部口径 (剔 holder_pct / 24m / 120 树 / 同超参), 唯一区别 =
    embargo 锚在**预测日**而非 test_month 月首 (单日 live 语义)。checkpoint: 模型落盘, existing 复用 (R08)。
    """
    idx = cal_all.index(target_date)
    cut_idx = idx - EMBARGO_TDAYS
    if cut_idx < 0:
        raise RuntimeError(f"{target_date}: 交易日历不足 {EMBARGO_TDAYS} 天做 embargo")
    embargo_cut = cal_all[cut_idx]
    train_start = get_train_start(embargo_cut, TRAIN_LOOKBACK_MONTHS)

    mfile = R20_MODEL_DIR / f"r20_clean_{target_date}.txt"
    meta = {
        "target_date": target_date, "embargo_cut": embargo_cut,
        "embargo_tdays": EMBARGO_TDAYS, "r20_horizon": R20_HORIZON,
        "train_window": [train_start, embargo_cut], "n_features": len(feat_cols),
        "dropped_leak_factor": DROP_LEAK, "n_trees_fixed": N_TREES_FIXED,
    }
    if mfile.exists():
        print(f"    [r20 clean] checkpoint 命中 {mfile.name}", flush=True)
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8")), meta

    df_tr = daily_full[(daily_full["trade_date"] >= train_start) &
                       (daily_full["trade_date"] <= embargo_cut)]
    sub = df_tr.dropna(subset=["r20"]).copy()
    if len(sub) < 100_000:
        raise RuntimeError(f"{target_date}: r20 训练样本不足 {len(sub)} (embargo 后)")
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    meta["n_train"] = int(len(sub))

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
    clf.fit(Xtr, ytr, categorical_feature=["industry_id"])
    clf.booster_.save_model(str(mfile))
    (R20_MODEL_DIR / f"meta_{target_date}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [r20 clean] 训练 {train_start}~{embargo_cut} (剔 {DROP_LEAK}, "
          f"n_feat={len(feat_cols)}), n_tr={len(sub):,} trees={N_TREES_FIXED}", flush=True)
    return clf.booster_, meta


def main():
    t0 = time.time()
    print("\n=== PT-001 生成今日 V12.31-clean picks + append-only 落库 (启动 paper-trade) ===\n",
          flush=True)

    # ── 生产 pump v3c (s5 排序) + 特征集 ──
    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    assert meta_p["class_meaning"]["2"].startswith("pump_up"), "v3c class 序异常"
    b5 = lgb.Booster(model_str=(PROD / "r5_pump_3way_lgbm_v3c" / "classifier.txt").read_text(encoding="utf-8"))

    # ── 数据 (ST 源头排除 R06) ──
    print(f"[data] load_window {DATA_START}-{TARGET_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, TARGET_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)
    for c in set(fc):
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    r20_fc_full = build_r20_feat_cols(daily)
    r20_fc = [c for c in r20_fc_full if c != DROP_LEAK]
    assert DROP_LEAK not in r20_fc
    print(f"[clean] r20 特征 {len(r20_fc_full)} -> {len(r20_fc)} (剔 {DROP_LEAK})", flush=True)

    cal_all = sorted(daily["trade_date"].unique())
    target_date = cal_all[-1]
    print(f"[asof] 最新可用交易日 (target) = {target_date} | 数据 as-of {TARGET_END} "
          f"| verdict={VERDICT_ID}", flush=True)

    df = daily[daily["trade_date"] == target_date].copy()
    print(f"[universe] {target_date} universe = {len(df)} 股 (ST 已源头排除)", flush=True)
    if len(df) < 100:
        raise RuntimeError(f"{target_date} universe 不足 ({len(df)})")

    # ── s5 ratio (生产 v3c, 因果无泄漏: forward 日 label 尚不存在) ──
    Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = b5.predict(Xf)
    df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)   # P_up / P_down
    df["pump_down_s5"] = proba[:, 1]                      # P_down (硬过滤)

    # ── pred_r20 (当期 PIT-clean r20, 剔 holder + embargo) ──
    print("[r20] 训当期 PIT-clean r20 (embargo cut = 预测日 − 21 交易日) ...", flush=True)
    b20, r20_meta = train_r20_clean_for_date(daily, r20_fc, target_date, cal_all)
    Xr = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan)
    df["pred_r20"] = b20.predict(Xr)

    # r20_fresh (forward, 评测用; target 日尚无 20d 前向 → NaN, build_dual 容忍)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

    # ── 双轨 build_dual (V7c 池 → ratio_s5 排序 → 行业 cap, 逐字同 V12.31-clean) ──
    print("[dual] 行业 60d 动量 rank + V7c 双轨 build_dual (sort=ratio_s5) ...", flush=True)
    ind_mom = compute_ind_mom(daily)
    hold = build_dual(df, ind_mom, sort_col="ratio_s5")
    if hold is None or hold.empty:
        raise RuntimeError(f"{target_date}: build_dual 无持仓")
    hold = hold.rename(columns={"entry_date": "trade_date"})

    # 补 ratio_s5 / pred_r20 / name 用于展示与落库
    cols = df[["ts_code", "ratio_s5", "pred_r20", "pump_down_s5"]].drop_duplicates("ts_code")
    hold = hold.merge(cols, on="ts_code", how="left")
    hold = hold.rename(columns={"pred_r20": "r20_pred_clean"})
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
        hold = hold.merge(basic, on="ts_code", how="left")
    hold = hold.sort_values("alloc_pct", ascending=False).reset_index(drop=True)

    print(f"\n[picks] {target_date} V12.31-clean: {len(hold)} 持仓 / "
          f"{hold['industry'].nunique()} 行业 / 总 alloc {hold['alloc_pct'].sum():.3f}\n", flush=True)
    print(f"  {'#':>2s} {'代码':11s} {'名称':10s} {'行业':10s} {'ratio_s5':>9s} "
          f"{'r20_clean':>9s} {'alloc%':>7s}", flush=True)
    for i, (_, p) in enumerate(hold.head(25).iterrows(), 1):
        nm = str(p.get("name", "") or "")[:10]
        ind = str(p.get("industry", "") or "")[:10]
        print(f"  {i:>2d} {p['ts_code']:11s} {nm:10s} {ind:10s} {p['ratio_s5']:>9.3f} "
              f"{p['r20_pred_clean']:>9.3f} {p['alloc_pct']*100:>6.2f}%", flush=True)

    # ── append-only 落库 (existing 拒覆盖 = 红线 SIGN-R01/R02) ──
    print(f"\n[log] append-only 落库 {target_date} → paper_trade_harness (source={SOURCE}) ...", flush=True)
    status = log_picks_for_date(target_date, hold, source=SOURCE)
    logged = read_log()
    today_rows = logged[logged["trade_date"] == target_date]
    print(f"    落库状态 = {status} | 读回确认 {target_date}: {len(today_rows)} 行 "
          f"| log 总计 {len(logged):,} 行 / {logged['trade_date'].nunique()} 日", flush=True)

    # ── picks 表 (verdict 用) ──
    picks_table = []
    for _, p in hold.iterrows():
        picks_table.append({
            "ts_code": p["ts_code"], "name": str(p.get("name", "") or ""),
            "industry": str(p.get("industry", "") or ""),
            "ratio_s5": round(float(p["ratio_s5"]), 4),
            "r20_pred_clean": round(float(p["r20_pred_clean"]), 4),
            "alloc_pct": round(float(p["alloc_pct"]), 5),
        })

    results = {
        "target_date": target_date, "data_asof": TARGET_END,
        "source": SOURCE, "log_status": status,
        "n_picks": int(len(hold)), "n_industries": int(hold["industry"].nunique()),
        "total_alloc": round(float(hold["alloc_pct"].sum()), 4),
        "r20_model": r20_meta,
        "config": {
            "pool": "V7c (r20 top5% × pyr_velocity<q35 × pump_down_s5<0.6)",
            "sort": "ratio_s5 = P_up/(P_down+eps) [生产 v3c]",
            "dual_track": "70/20/10 build_dual, 行业 cap (CAP_IN/CROSS 0.2, MAX_A 8/MAX_B 15)",
            "r20_pool_model": "当期 PIT-clean r20 (剔 holder_pct + embargo cut=预测日-21交易日)",
            "execution": "close-based, T+1, ST 源头排除 (R06)",
        },
        "picks": picks_table,
    }
    (OUT_DIR / "pt001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ── verdict (status=built, 运维落库非 gate) ──
    conclusion = (
        f"启动前向 paper-trade: 用 **当期 PIT-clean r20** (剔 holder_pct + embargo cut=预测日−21 交易日, "
        f"非生产 in-sample 模型) + 生产 v3c pump (s5 ratio 排序) 给最新可用交易日 **{target_date}** "
        f"(数据 as-of {TARGET_END}, 已刷 Tushare daily/moneyflow + 增量 factor_lab/衍生特征到该日) "
        f"打分, 口径逐字同 V12.31-clean (V7c 池 → ratio_s5 排序 → 双轨 70/20/10 → 行业 cap4 → close-based "
        f"→ ST 源头排除 R06), 产 **{len(hold)} 持仓 / {hold['industry'].nunique()} 行业 / 总 alloc "
        f"{hold['alloc_pct'].sum():.3f}**。经 paper_trade_harness.log_picks_for_date **append-only 落库** "
        f"(source={SOURCE}, existing 拒覆盖红线 SIGN-R01/R02), 落库状态 {status}, 读回确认 {len(today_rows)} 行 "
        f"(log 总计 {logged['trade_date'].nunique()} 日)。前向 paper-trade 第一笔真实记录就位, 后续每交易日续落, "
        f"满 20/40d 用 harness realize_book 对锚 FIN-001 净 Sharpe 1.31±CI[0.11,2.60] (分 regime R11)。"
        f"生产线 v12_scoring/v12_dual_track 只读冻结 (R05, 本任务用 PIT-clean 变体, 0 生产改动)。"
    )
    VERDICT.write_text(json.dumps({
        "id": VERDICT_ID,
        "title": f"生成 {target_date} V12.31-clean picks + append-only 落库 (前向 paper-trade)",
        "status": "built",
        "conclusion": conclusion,
        "metrics": {
            "target_date": target_date, "data_asof": TARGET_END,
            "n_picks": int(len(hold)), "n_industries": int(hold["industry"].nunique()),
            "total_alloc": round(float(hold["alloc_pct"].sum()), 4),
            "log_status": status, "n_rows_readback": int(len(today_rows)),
            "n_logged_days_total": int(logged["trade_date"].nunique()),
            "r20_embargo_cut": r20_meta["embargo_cut"],
            "r20_train_window": r20_meta["train_window"],
            "r20_n_features": r20_meta["n_features"],
            "r20_dropped_leak": DROP_LEAK,
            "source": SOURCE,
        },
        "picks": picks_table,
        "artifacts": {
            "generator": "research/backtest/pt001_gen_today.py",
            "results": "research/cache/pt001/pt001_results.json",
            "r20_model": f"research/cache/pt001/r20_model/r20_clean_{target_date}.txt",
            "picks_log": f"research/cache/paper_trade/picks/{target_date}.parquet (append-only)",
            "log_jsonl": "research/cache/paper_trade/log.jsonl",
            "anchor": "FIN-001 净 Sharpe 1.31 ± CI[0.11,2.60] (V12.31_BASELINE.md §3)",
        },
        "discipline": (
            "R01/R02 落库 append-only existing 拒覆盖 (前向数据不回改/不回流调参); R04 全因果无泄漏 "
            "(r20 embargo cut=预测日−21 交易日, forward 日 pump label 尚不存在; 大缓存 gitignored); "
            "R05 生产线只读冻结 (PIT-clean 变体, 0 生产改动); R06 ST 源头排除; R08 r20 模型 checkpoint; "
            "运维落库非研究 gate, status=built (preRegisteredGate.frozen=true)。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT.relative_to(ROOT)} (status=built)", flush=True)
    print(f"[done] PT-001 完成, 耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    del daily; gc.collect()


if __name__ == "__main__":
    main()
