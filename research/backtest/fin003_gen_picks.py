# -*- coding: utf-8 -*-
"""FIN-003 — triple-barrier V12.32 挑战者: 双屏障 label + embargo walk-forward → OOS 每日持仓.

承用户 "不限天数, 限幅度 + 回撤" 直觉, 把它落到**选股 label 层** (而非出场层, EX/WFE-002 已证出场层
止盈无 alpha)。设计 = WFE-001 embargo r20 walk-forward 的**独立挑战者** (FIN-003 是平行 V12.32 候选,
非 V12.31 基线补丁): 唯一改动 = r20 回归 label 换成 **triple-barrier label**, 其余 (24m lookback / 固定
120 树 / 池构造 / 双轨 / 引擎下游) 逐字同 WFE-001。

triple-barrier label (参数预注册冻结, R01, 不搜):
  entry = next_open (D+1 开盘, 与 r20 同口径); 持有窗 = D+1 .. D+40 (40 交易日 backstop)。
    · 上屏障 X = +15% (entry*1.15): 任一日 high >= 上屏障 → 先触上 → label = +15.0
    · 下屏障 Y = -8%  (entry*0.92): 任一日 low  <= 下屏障 → 先触下 → label = -8.0
    · 同日上下都触 → 保守判先触下 (label = -8.0)
    · 40 日内都不触 → 时间 backstop: label = (close_40d / entry - 1) * 100
  → label 是"按幅度/回撤先到为准结清"的实现收益 (%), 与 r20 (固定 20d 收益 %) 同尺度, 作回归目标。

embargo (label-availability, R04): 屏障 horizon = 40d → 训练截止 <= P_start - 41 交易日
  (= cal[idx(P_start) - 41]), 保证训练样本 T 的 triple-barrier label (最迟 T+40 交易日实现) 在预测前已
  全部可知。(WFE-001 r20 horizon 20d → embargo 21; 本任务 horizon 40d → embargo 41。)

score 排序替代 r20: TB-score 替代 build_dual 里的 pred_r20 (= r20 池 filter 信号), 池内仍按已 walk-forward
  的 ratio_s5 排序 (= V12.31 口径)。下游 run_fin003.py 过同一引擎 (close-based 成本 + embargo) apples-
  to-apples vs WFE-001 基线 1.31, 按 prd.gate_tb 断言。

checkpoint (R08): TB label 落 research/cache/fin003/tb_label.parquet; 每月模型落 r20_models/{月}/;
picks 落 picks_by_month/。ST 源头排除 (R06)。前向列只留 research/cache/ (R04)。生产线只读 (R05)。
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
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
T005_MODELS = ROOT / "research" / "cache" / "t005_wf_models"
OUT_DIR = ROOT / "research" / "cache" / "fin003"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TB_MODELS = OUT_DIR / "r20_models"          # TB-score 月度模型 (沿用目录名习惯)
TB_MODELS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "picks_by_month"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PICKS = OUT_DIR / "picks_oos_daily.parquet"
DIAG = OUT_DIR / "tb_oos_diagnostics.csv"
TB_LABEL_P = OUT_DIR / "tb_label.parquet"

DATA_START, DATA_END = "20220801", "20260601"
N_TRAIN = 900_000
N_TREES_FIXED = 120
SEED = 42

# ── triple-barrier 参数 (预注册冻结, R01, 不搜) ──
TB_X = 0.15              # 上屏障 +15%
TB_Y = 0.08             # 下屏障 -8%
TB_HORIZON = 40         # 时间 backstop 40 交易日
EMBARGO_TDAYS = TB_HORIZON + 1   # = 41; 训练截止 <= P_start - 41 交易日 (label 全可知)


def _tb_for_stock(o, h, l, c) -> np.ndarray:
    """单股 numpy 三屏障扫描 (40 offset 向量化)。返回每行 label (%) 或 nan。"""
    n = len(o)
    no = np.full(n, np.nan); no[:-1] = o[1:]          # next_open (entry)
    up = no * (1.0 + TB_X)
    dn = no * (1.0 - TB_Y)
    lab = np.full(n, np.nan)
    res = np.zeros(n, dtype=bool)
    valid_entry = ~np.isnan(no)
    for d in range(1, TB_HORIZON + 1):
        if d >= n:
            break
        hi = np.full(n, np.nan); hi[:n - d] = h[d:]    # high at i+d
        lo = np.full(n, np.nan); lo[:n - d] = l[d:]    # low  at i+d
        # 同日先判下屏障 (保守)
        dn_hit = (~res) & valid_entry & (~np.isnan(lo)) & (lo <= dn)
        lab[dn_hit] = -TB_Y * 100.0
        res[dn_hit] = True
        up_hit = (~res) & valid_entry & (~np.isnan(hi)) & (hi >= up)
        lab[up_hit] = TB_X * 100.0
        res[up_hit] = True
    if n > TB_HORIZON:
        cl = np.full(n, np.nan); cl[:n - TB_HORIZON] = c[TB_HORIZON:]   # close at i+40
        bs = (~res) & valid_entry & (~np.isnan(cl))
        lab[bs] = (cl[bs] / no[bs] - 1.0) * 100.0
    # n <= TB_HORIZON: 无完整 backstop 窗口的未触行保持 nan (= 不可判, 同 r20 序列末端)
    return lab


def compute_tb_label() -> pd.DataFrame:
    """从 daily cache 全市场算 triple-barrier label (checkpoint)。"""
    if TB_LABEL_P.exists():
        df = pd.read_parquet(TB_LABEL_P)
        df["trade_date"] = df["trade_date"].astype(str)
        return df
    files = sorted(DAILY_DIR.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
             for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    out = []
    for code, g in big.groupby("ts_code", sort=False):
        lab = _tb_for_stock(g["open"].to_numpy(float), g["high"].to_numpy(float),
                            g["low"].to_numpy(float), g["close"].to_numpy(float))
        out.append(pd.DataFrame({"ts_code": code, "trade_date": g["trade_date"].values, "tb_label": lab}))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(TB_LABEL_P, index=False)
    print(f"  [tb_label] {len(res):,} 行 / {res['ts_code'].nunique()} 股 / "
          f"非空 {res['tb_label'].notna().sum():,} "
          f"(上触 {(res['tb_label']==TB_X*100).sum():,} / 下触 {(res['tb_label']==-TB_Y*100).sum():,} / "
          f"backstop {((res['tb_label'].notna())&(res['tb_label']!=TB_X*100)&(res['tb_label']!=-TB_Y*100)).sum():,})",
          flush=True)
    return res


def embargo_cut_for_month(cal_all: list, test_month: str) -> str:
    month_days = [d for d in cal_all if d.startswith(test_month)]
    if not month_days:
        return None
    p_start = month_days[0]
    idx = cal_all.index(p_start)
    cut_idx = idx - EMBARGO_TDAYS
    if cut_idx < 0:
        return None
    return cal_all[cut_idx]


def train_tb_month_embargo(daily_full: pd.DataFrame, feat_cols: list, test_month: str, cal_all: list):
    """月度 walk-forward 重训 TB-score 回归器 + label-availability embargo (41交易日)。checkpoint。"""
    mdir = TB_MODELS / test_month
    mfile = mdir / "classifier.txt"
    if mfile.exists():
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    embargo_cut = embargo_cut_for_month(cal_all, test_month)
    if embargo_cut is None:
        raise RuntimeError(f"{test_month}: 无法定位 embargo 截止日")

    df_tr = daily_full[(daily_full["trade_date"] >= train_start) &
                       (daily_full["trade_date"] <= embargo_cut)]
    sub = df_tr.dropna(subset=["tb_label"]).copy()
    if len(sub) < 100_000:
        raise RuntimeError(f"{test_month}: TB 训练样本不足 {len(sub)} (embargo后)")
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

    def X(d):
        return d[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    Xtr, ytr = X(sub), sub["tb_label"].astype("float32")

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
        "feature_cols": feat_cols, "target": "tb_label", "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (mdir / "meta.json").write_text(json.dumps({
        "test_month": test_month, "train_window_base": [train_start, train_end],
        "embargo_cut": embargo_cut, "embargo_tdays": EMBARGO_TDAYS,
        "tb_X": TB_X, "tb_Y": TB_Y, "tb_horizon": TB_HORIZON,
        "n_trees_fixed": N_TREES_FIXED, "n_train": int(len(sub)),
        "note": "FIN-003 月度 walk-forward triple-barrier 回归器 + embargo (训练截止 <= P_start-41交易日)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [TB train+embargo] {test_month}: {train_start}~{embargo_cut} "
          f"(base 截止 {train_end}, 收紧 {EMBARGO_TDAYS}交易日), n_tr={len(sub):,} trees={N_TREES_FIXED}",
          flush=True)
    return clf.booster_


def main():
    t0 = time.time()
    print("\n=== FIN-003 gen_picks: triple-barrier label + embargo walk-forward → OOS 持仓 ===\n", flush=True)
    print(f"[tb] 屏障 X=+{TB_X:.0%} / Y=-{TB_Y:.0%} / backstop {TB_HORIZON}d / embargo {EMBARGO_TDAYS}交易日 (冻结 R01)", flush=True)

    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)

    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)

    for c in set(fc):
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 特征列 (= WFE-001 r20 特征集; 排除 tb_label, 它是 forward label 不进特征)
    r20_fc = [c for c in build_r20_feat_cols(daily) if c != "tb_label"]
    cal_all = sorted(daily["trade_date"].unique())

    print("[label] triple-barrier label (训练目标) ...", flush=True)
    tb_lab = compute_tb_label()
    daily = daily.merge(tb_lab, on=["ts_code", "trade_date"], how="left")
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / TB特征 {len(r20_fc)} / "
          f"{len(cal_all)} 交易日 / tb_label 非空 {daily['tb_label'].notna().sum():,}", flush=True)

    print("[label] r20_fresh 前向 (评测/IC 用, 与基线同口径) ...", flush=True)
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

        # ratio_s5 (已 walk-forward, 复用 t005 缓存, 与基线同)
        b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
        Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b5.predict(Xf)
        df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
        df["pump_down_s5"] = proba[:, 1]

        # TB-score (本月 walk-forward 重训 + embargo) → 替代 pred_r20 作池 filter 信号
        b_tb = train_tb_month_embargo(daily, r20_fc, m_, cal_all)
        Xr_oos = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan)
        df["pred_r20"] = b_tb.predict(Xr_oos)     # 复用 build_dual 的 pred_r20 接口 = TB-score

        df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

        # 诊断: TB-score 对 r20_fresh 的截面 rank-IC (描述性, 非 gate)
        msk = df["r20_fresh"].notna()
        if msk.sum() > 50:
            fresh = df.loc[msk, "r20_fresh"].clip(-30, 30)
            ic_tb = pd.Series(df.loc[msk, "pred_r20"].values).corr(
                pd.Series(fresh.values), method="spearman")
        else:
            ic_tb = np.nan

        hold = build_dual(df, ind_mom, sort_col="ratio_s5")
        if hold is None or hold.empty:
            print(f"  {m_}: 无持仓, 跳过", flush=True)
            continue
        hold = hold.rename(columns={"entry_date": "trade_date"})
        hold["month"] = m_
        hold.to_parquet(CKPT_DIR / f"{m_}.parquet", index=False)

        diag_rows = [r for r in diag_rows if str(r["month"]) != m_]
        diag_rows.append({"month": m_, "tb_oos_rankic_vs_r20fresh": float(ic_tb) if pd.notna(ic_tb) else np.nan,
                          "n_picks_rows": int(len(hold)), "n_days": int(hold["trade_date"].nunique())})
        pd.DataFrame(diag_rows).to_csv(DIAG, index=False)
        print(f"  {m_}: {len(hold):,} 持仓行 / {hold['trade_date'].nunique()} 日 | "
              f"TB-score IC vs r20_fresh={ic_tb:+.4f}  (累计 {time.time()-t0:.0f}s)", flush=True)
        del df, b5, b_tb
        gc.collect()

    parts = [pd.read_parquet(p) for p in sorted(CKPT_DIR.glob("*.parquet"))]
    if not parts:
        print("[done] 无任何月完成", flush=True)
        return
    allh = pd.concat(parts, ignore_index=True)
    allh = allh[allh["month"].isin(TEST_MONTHS)].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    allh.to_parquet(PICKS, index=False)
    print(f"\n[done] TB embargo OOS picks -> {PICKS.relative_to(ROOT)}  "
          f"{len(allh):,} 行 / {allh['trade_date'].nunique()} 日 / {allh['month'].nunique()} 月 / "
          f"{allh['ts_code'].nunique()} 股  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
