# -*- coding: utf-8 -*-
"""WFE-001 — label-embargo r20 真·真实 walk-forward → 生成 V12.31 OOS 每日持仓.

codex 第二次 review (research/backtest/audits/codex_review_wf_0614_response.md) 抓到 WF-001 仍有
**label-availability 泄漏**: WF-001 的 r20 月度重训按 `trade_date < train_end` 切训练样本, 但 r20 标签
本身是前向 20 日收益 (next_open -> close_20d)。靠近训练截止日的样本, 其 r20 标签要到预测月才实现 —
模型没吃预测月特征, 但**吃了预测月之后才可知的标签信息** = label-availability lookahead。故 WF-001 的
1.84 仍偏高。

WFE-001 = WF-001 + **严格 label-availability embargo**:
  训练样本 trade_date <= 预测窗起始(P_start) 之前第 (HORIZON + 1) = 21 个交易日 (= cal[idx(P_start) - 21])。
  这保证训练样本 T 的 r20 标签 (实现于 T+20 交易日) 在预测窗起始前一交易日已全部可知。
  其余一切 (24m lookback 起点 / 固定 120 树 / 配置 / 引擎下游) 与 WF-001 逐字一致 → 唯一改动 = 截止日
  从"特征日 < train_end"收紧到"标签可知 (embargo)"。

下游 run_wfe001.py 把 embargo OOS picks 过同一 book 引擎 (同 BT-002/WF-001 配置), 报真·真实 Sharpe/
年化/maxDD + vs WF-001 (无 embargo 1.84) 的**额外缩水** + 分 regime + 因果自检。

checkpoint (SIGN-R08): 每月 r20 模型落 research/cache/wfe001/r20_models/{月}/; picks 落 picks_by_month/。
ST 源头排除 (load_window exclude_st=True, R06)。前向列只留 research/cache/ (R04)。生产线只读 (R05)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))

from train_v15_refresh import load_window, EXCLUDE
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom,
    get_train_end_for_test_month, get_train_start,
)
from t005_walk_forward_gate import build_dual, TEST_MONTHS, EPS, TRAIN_LOOKBACK_MONTHS
from wf001_gen_picks import build_r20_feat_cols

PROD = ROOT / "output" / "production"
T005_MODELS = ROOT / "research" / "cache" / "t005_wf_models"   # 已 walk-forward 的 s5 pump 模型
OUT_DIR = ROOT / "research" / "cache" / "wfe001"
OUT_DIR.mkdir(parents=True, exist_ok=True)
R20_MODELS = OUT_DIR / "r20_models"
R20_MODELS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "picks_by_month"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PICKS = OUT_DIR / "picks_oos_daily.parquet"
DIAG = OUT_DIR / "r20_oos_diagnostics.csv"

DATA_START, DATA_END = "20220801", "20260601"
N_TRAIN = 900_000        # 与 WF-001 一致 (子采样上限)
N_TREES_FIXED = 120      # 与 WF-001 一致 (固定树数无早停)
SEED = 42
R20_HORIZON = 20         # r20 标签前向交易日数 (next_open -> close_20d)
EMBARGO_TDAYS = R20_HORIZON + 1   # = 21; 训练截止须 <= P_start - 21 交易日 (标签全可知)


def embargo_cut_for_month(cal_all: list, test_month: str) -> str:
    """test_month 预测窗起始 P_start 之前第 EMBARGO_TDAYS 个交易日 = 训练样本截止日 (含)。

    保证: 截止日训练样本 T 的 r20 标签实现于 T + R20_HORIZON 交易日 = cal[idx(P_start) - 1]
    = 预测窗起始前一交易日 → 标签在预测前已全部可知 (label-availability embargo)。
    """
    month_days = [d for d in cal_all if d.startswith(test_month)]
    if not month_days:
        return None
    p_start = month_days[0]
    idx = cal_all.index(p_start)
    cut_idx = idx - EMBARGO_TDAYS
    if cut_idx < 0:
        return None
    return cal_all[cut_idx]


def train_r20_month_embargo(daily_full: pd.DataFrame, feat_cols: list, test_month: str,
                            cal_all: list):
    """月度 walk-forward 重训 r20 回归器 + label-availability embargo。checkpoint。

    与 WF-001 train_r20_month 唯一差异: 训练截止从 `trade_date < train_end (特征日)` 收紧到
    `trade_date <= embargo_cut (标签可知)`。24m lookback 起点保持 WF-001 同口径 (从原 train_end 倒推),
    故 embargo 仅削掉最近 ~21 交易日的训练样本 (恰是 label 最易泄漏的近截止段)。
    """
    mdir = R20_MODELS / test_month
    mfile = mdir / "classifier.txt"
    if mfile.exists():
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    train_end = get_train_end_for_test_month(test_month)          # WF-001 原截止 (特征日)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)  # 24m 起点, 不变
    embargo_cut = embargo_cut_for_month(cal_all, test_month)
    if embargo_cut is None:
        raise RuntimeError(f"{test_month}: 无法定位 embargo 截止日")

    df_tr = daily_full[(daily_full["trade_date"] >= train_start) &
                       (daily_full["trade_date"] <= embargo_cut)]   # ← embargo (含截止日)
    sub = df_tr.dropna(subset=["r20"]).copy()
    if len(sub) < 100_000:
        raise RuntimeError(f"{test_month}: r20 训练样本不足 {len(sub)} (embargo后)")
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

    def X(d):
        return d[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    Xtr, ytr = X(sub), sub["r20"].astype("float32")

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
    (mdir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": "r20", "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (mdir / "meta.json").write_text(json.dumps({
        "test_month": test_month,
        "train_window_wf001": [train_start, train_end],
        "embargo_cut": embargo_cut, "embargo_tdays": EMBARGO_TDAYS,
        "n_trees_fixed": N_TREES_FIXED, "n_train": int(len(sub)),
        "note": "WFE-001 月度 walk-forward r20 + label-availability embargo (训练截止 <= P_start-21交易日)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [r20 train+embargo] {test_month}: {train_start}~{embargo_cut} "
          f"(WF-001 原截止 {train_end}, 收紧 {EMBARGO_TDAYS}交易日), n_tr={len(sub):,} trees={N_TREES_FIXED}",
          flush=True)
    return clf.booster_


def main():
    t0 = time.time()
    print("\n=== WFE-001 gen_picks: label-embargo r20 月度 walk-forward → OOS 持仓 ===\n", flush=True)

    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    meta_r20p = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20p_fc = meta_r20p["feature_cols"]
    b20_prod = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)

    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)

    for c in set(fc) | set(r20p_fc):
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    r20_fc = build_r20_feat_cols(daily)
    cal_all = sorted(daily["trade_date"].unique())   # 交易日历 (union, 用于 embargo 定位)
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / r20 特征 {len(r20_fc)} / "
          f"{len(cal_all)} 交易日 / embargo={EMBARGO_TDAYS}交易日", flush=True)

    print("[label] r20_fresh 前向 (next_open->close_20d, 评测/IC 用) ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    print("[ind] 行业 60d 动量 rank ...", flush=True)
    ind_mom = compute_ind_mom(daily)

    done_months = {p.stem for p in CKPT_DIR.glob("*.parquet")}
    print(f"[ckpt] 已完成 {len(done_months)} 月: {sorted(done_months)}\n", flush=True)

    diag_rows = []
    if DIAG.exists():
        diag_rows = pd.read_csv(DIAG, dtype={"month": str}).to_dict("records")
    diag_done = {str(r["month"]) for r in diag_rows}

    for m_ in TEST_MONTHS:
        if m_ in done_months and m_ in diag_done:
            continue
        s5file = T005_MODELS / m_ / "pump_scale_5" / "classifier.txt"
        if not s5file.exists():
            print(f"  {m_}: 缺缓存 s5 模型, 跳过", flush=True)
            continue
        df = daily[daily["trade_date"].str.startswith(m_)].copy()
        if len(df) < 100:
            print(f"  {m_}: 测试样本不足 ({len(df)}), 跳过", flush=True)
            continue

        # ratio_s5 (已 walk-forward, 复用 t005 缓存, 与 WF-001 同)
        b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
        Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b5.predict(Xf)
        df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
        df["pump_down_s5"] = proba[:, 1]

        # de-lookahead + EMBARGO r20 (本月 walk-forward 重训)
        b20_oos = train_r20_month_embargo(daily, r20_fc, m_, cal_all)
        Xr_oos = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan)
        df["pred_r20"] = b20_oos.predict(Xr_oos)

        # in-sample 生产 r20 (对照, 因果自检) — 不进 picks
        Xr_p = df[r20p_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        pred_r20_prod = b20_prod.predict(Xr_p)

        df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

        msk = df["r20_fresh"].notna()
        if msk.sum() > 50:
            fresh = df.loc[msk, "r20_fresh"].clip(-30, 30)
            ic_oos = pd.Series(df.loc[msk, "pred_r20"].values).corr(
                pd.Series(fresh.values), method="spearman")
            ic_prod = pd.Series(pred_r20_prod[msk.values]).corr(
                pd.Series(fresh.values), method="spearman")
        else:
            ic_oos = ic_prod = np.nan

        hold = build_dual(df, ind_mom, sort_col="ratio_s5")
        if hold is None or hold.empty:
            print(f"  {m_}: 无持仓, 跳过", flush=True)
            continue
        hold = hold.rename(columns={"entry_date": "trade_date"})
        hold["month"] = m_
        hold.to_parquet(CKPT_DIR / f"{m_}.parquet", index=False)

        in_sample = m_ <= "202509"
        diag_rows = [r for r in diag_rows if str(r["month"]) != m_]
        diag_rows.append({"month": m_, "r20_oos_rankic": float(ic_oos) if pd.notna(ic_oos) else np.nan,
                          "r20_prod_rankic": float(ic_prod) if pd.notna(ic_prod) else np.nan,
                          "prod_in_train_window": bool(in_sample),
                          "n_picks_rows": int(len(hold)), "n_days": int(hold["trade_date"].nunique())})
        pd.DataFrame(diag_rows).to_csv(DIAG, index=False)
        print(f"  {m_}: {len(hold):,} 持仓行 / {hold['trade_date'].nunique()} 日 | "
              f"r20 IC OOS(embargo)={ic_oos:+.4f} vs 生产={ic_prod:+.4f} "
              f"({'in-sample' if in_sample else 'true-OOS'})  (累计 {time.time()-t0:.0f}s)", flush=True)
        del df, b5, b20_oos
        gc.collect()

    parts = [pd.read_parquet(p) for p in sorted(CKPT_DIR.glob("*.parquet"))]
    if not parts:
        print("[done] 无任何月完成", flush=True)
        return
    allh = pd.concat(parts, ignore_index=True)
    allh = allh[allh["month"].isin(TEST_MONTHS)].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    allh.to_parquet(PICKS, index=False)
    print(f"\n[done] embargo OOS picks -> {PICKS.relative_to(ROOT)}  "
          f"{len(allh):,} 行 / {allh['trade_date'].nunique()} 日 / {allh['month'].nunique()} 月 / "
          f"{allh['ts_code'].nunique()} 股  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
