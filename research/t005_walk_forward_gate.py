# -*- coding: utf-8 -*-
"""T-005 walk-forward 决策 gate — 动态窗口 ratio_dyn vs V12.31 固定窗口 baseline.

研究问题 (承接 T-001~T-004):
  T-004 产出了"动态窗口"信号 ratio_dyn (regime/run_len 门控的多尺度 pump ratio)。
  本任务回答唯一的 ship 问题: 在严格 walk-forward 下, ratio_dyn 作为 V7c 池内排序信号,
  相对 V12.31 的固定窗口 pump_ratio, 是否带来事前注册 gate 要求的真 alpha?

apples-to-apples 设计 (两臂唯一差异 = 池内排序列):
  - 两臂共用同一 V7c 池构造 (r20 top + pyr_velocity 低 + 行业动量排除 + s5 pump_down 过滤)。
  - baseline 臂 (= V12.31 等价): 池内按 ratio_s5 排序。
      s5 = H5/g10% 3way pump 模型, 与生产 v3c 同档 (H=5, up: 5日 max_gain>=10% & max_dd>=-5%)。
  - dynamic 臂 (= T-004): 池内按 ratio_dyn (regime/run_len 门控选 s3/s5/s10) 排序。
  两臂的池/过滤/行业 cap/仓位分配/成本/r20 池排序模型完全一致 → 任何共模偏差 (含 r20 池模型
  的潜在 lookahead) 在 Δα 中相消; gate 只看相对量, 不看绝对 α (SIGN-R03)。

walk-forward 协议 (与 project_walk_forward_validation_0525 一致):
  - 测试月 202410-202604 共 19 月; 每月用前 24 月数据独立重训 (滚动, 不跨月 fine-tune)。
  - 每月重训 3 个尺度 pump 模型 (s3/s5/s10); r20 池模型沿用生产 r20_v16_long_nost (不重训,
    与 walk_forward_validation.py 同口径, 两臂共用)。
  - 训练特征全 causal (load_window backward); 标签 forward high/low 仅训练用。
  - ST 源头排除 (load_window exclude_st=True 默认, SIGN-R06)。

事前注册 gate (冻结, SIGN-R01; 全相对 baseline 臂同 harness):
  Δα(月化) >= +0.30pp  且  Sharpe 不低于 baseline  且  最差月不低于 baseline 最差月
  且  正 α 月占比不低于 baseline  → verdict=PASS, 否则 verdict=REJECT。
  REJECT 是合法完成, 禁止在同段 OOS 反复重调再试 (SIGN-R02)。

checkpoint (SIGN-R08, >5min 必须):
  - 每月每尺度模型落 research/cache/t005_wf_models/{month}/pump_scale_{H}/, 存在即跳过重训。
  - 每月两臂结果 append 到 research/cache/t005_monthly.csv, 已完成的月跳过 → 可断点续跑。

产出:
  - research/cache/t005_monthly.csv         (逐月两臂 α/Sharpe/...)
  - research/cache/t005_wf_models/          (月度尺度模型, checkpoint)
  - research/cache/t005_results.json        (汇总 + gate 判定)
  - research/verdicts/T-005.json            (status in {PASS, REJECT} + 四项指标对照表)
不碰任何生产文件 (SIGN-R05)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window, EXCLUDE
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
    get_train_end_for_test_month, get_train_start,
)
from t004_multiscale_gate import (
    compute_gate_inputs, gate_select, build_label, load_daily,
    SCALES, REGIME_BASE, EPS, DD_CAP,
)

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
WF_MODELS = CACHE / "t005_wf_models"
WF_MODELS.mkdir(parents=True, exist_ok=True)
MONTHLY = CACHE / "t005_monthly.csv"
RESULTS = CACHE / "t005_results.json"
VERDICT = ROOT / "research" / "verdicts" / "T-005.json"
PROD = ROOT / "output" / "production"

# 测试月 (19 月, 与 V11 walk-forward 同)
TEST_MONTHS = ["202410", "202411", "202412",
               "202501", "202502", "202503", "202504", "202505", "202506",
               "202507", "202508", "202509",
               "202510", "202511", "202512",
               "202601", "202602", "202603", "202604"]
TRAIN_LOOKBACK_MONTHS = 24
DATA_START, DATA_END = "20220801", "20260601"

# 训练 (轻量, walk-forward 每月重训; 两臂同设置, 模型质量偏差共模相消)
N_TRAIN = 400_000
SEED = 20260603
SCALE_LIST = [3, 5, 10]

# V7c / dual-track 池构造 (与 walk_forward_validation.py 同口径)
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60


# ───────────────────────── 训练 (3way 尺度模型, 月度) ─────────────────────────
def train_scale_month(df_train: pd.DataFrame, fc: list, label_col: str,
                       test_month: str, H: int) -> lgb.Booster:
    mdir = WF_MODELS / test_month / f"pump_scale_{H}"
    mfile = mdir / "classifier.txt"
    if mfile.exists():
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    sub = df_train.dropna(subset=[label_col]).copy()
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED + H).reset_index(drop=True)
    # 时间切分早停: 训练窗最后 ~1 月做 val (leak-free, val < train_end < 测试月)
    # 注: trade_date 是 arrow large_string, pyarrow quantile 无 string kernel,
    #     改按排序位置取 0.92 分位日期 (YYYYMMDD 字典序 == 时序)。
    if len(sub):
        _sd = np.sort(sub["trade_date"].astype(str).to_numpy())
        cut = str(_sd[min(int(len(_sd) * 0.92), len(_sd) - 1)])
    else:
        cut = "0"
    tr = sub[sub["trade_date"] < cut]
    va = sub[sub["trade_date"] >= cut]
    if len(va) < 5000 or len(tr) < 20000:
        tr, va = sub, sub.iloc[:0]

    def Xy(d):
        X = (d[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0))
        y = d[label_col].astype("int")
        return X, y

    Xtr, ytr = Xy(tr)
    params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                  num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=80,
                  seed=SEED + H, verbose=-1, num_threads=0)
    dtr = lgb.Dataset(Xtr, ytr)
    if len(va):
        Xva, yva = Xy(va)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False),
                                       lgb.log_evaluation(0)])
        best = booster.best_iteration
    else:
        booster = lgb.train(params, dtr, num_boost_round=150)
        best = 150
    mfile.write_text(booster.model_to_string(), encoding="utf-8")
    (mdir / "meta.json").write_text(json.dumps({
        "test_month": test_month, "scale_H": H, "gain_threshold": SCALES[H],
        "best_iter": int(best), "n_train": int(len(tr)),
        "note": "T-005 walk-forward 月度尺度模型, 非生产",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return booster


# ───────────────────────── 池构造 (两臂共用, sort_col 可配) ─────────────────────────
def build_dual(daily: pd.DataFrame, ind_mom: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    """V7c dual-track, 池内按 sort_col 排序。两臂仅 sort_col 不同, 其余完全一致。"""
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 100:
            continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 50:
            continue
        g["r20_rank"] = g["pred_r20"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0:
            continue
        v7c = v7c[v7c["pump_down_s5"] < PUMP_DOWN_THRESHOLD]   # 共享 s5 跌过滤
        if len(v7c) == 0:
            continue
        v7c = v7c.sort_values(sort_col, ascending=False)
        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col=sort_col)
        a_ind = industry_alloc(a_pool, A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col=sort_col)
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col=sort_col)
        for tag, pool, alloc in [("A", a_pool, A_PCT), ("B", b_pool, B_PCT)]:
            if len(pool) == 0:
                continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                             "industry": row.get("industry", ""),
                             "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                             "alloc_pct": per})
    return pd.DataFrame(rows)


def month_metrics(hold: pd.DataFrame, df_test: pd.DataFrame) -> dict | None:
    if hold is None or hold.empty:
        return None
    h = hold.dropna(subset=["r20_fresh"]).copy()
    if h.empty:
        return None
    h["r20_fresh"] = h["r20_fresh"].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h["r20_fresh"]
    total = A_PCT + B_PCT
    pnl = h.groupby("entry_date").agg(gross=("w", "sum"), n=("ts_code", "count")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = df_test.groupby("trade_date")["r20_fresh"].apply(
        lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * total
    alpha_avg = float(pnl["alpha"].mean())
    sharpe = float(alpha_avg / (pnl["alpha"].std() + 1e-9) * np.sqrt(12))
    return {"alpha": alpha_avg, "sharpe": sharpe,
            "n_days": int(pnl["entry_date"].nunique()),
            "avg_n_stocks": float(pnl["n"].mean())}


# ───────────────────────── 单月评估 (两臂) ─────────────────────────
def evaluate_month(test_month, daily_full, fc, ind_map, gate, ind_mom,
                   r20_lab, b20, r20_fc) -> dict | None:
    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    print(f"\n--- {test_month}: train {train_start}-{train_end} ---", flush=True)

    df_train = daily_full[(daily_full["trade_date"] >= train_start) &
                          (daily_full["trade_date"] < train_end)].copy()
    if len(df_train) < 100_000:
        print(f"  跳过 (训练样本不足 {len(df_train)})", flush=True)
        return None

    # 训练 3 个尺度模型 (checkpoint)
    boosters = {}
    for H in SCALE_LIST:
        lc = f"label_s{H}"
        npos = int((df_train[lc] == 2).sum())
        print(f"  训 s{H} (up 正样本 {npos:,}) ...", flush=True)
        boosters[H] = train_scale_month(df_train, fc, lc, test_month, H)

    # 测试月
    df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
    if len(df_test) < 100:
        print(f"  跳过 (测试样本不足 {len(df_test)})", flush=True)
        return None
    for c in fc:
        if c not in df_test.columns:
            df_test[c] = 0.0
    for c in r20_fc:
        if c not in df_test.columns:
            df_test[c] = 0.0

    Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    ratios = {}
    for H in SCALE_LIST:
        proba = boosters[H].predict(Xf)
        ratios[H] = proba[:, 2] / (proba[:, 1] + EPS)
        if H == 5:
            df_test["pump_down_s5"] = proba[:, 1]
        df_test[f"ratio_s{H}"] = ratios[H]

    # r20 池排序 (生产模型, 两臂共用)
    Xr = df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test["pred_r20"] = b20.predict(Xr)

    # 门控 -> ratio_dyn
    df_test = df_test.merge(gate, on=["ts_code", "trade_date"], how="left")
    cross = df_test["ma5_60_cross_60d"]
    p20, p80 = cross.quantile(0.20), cross.quantile(0.80)
    regime = pd.Series("mid", index=df_test.index)
    regime[cross <= p20] = "trend"
    regime[cross >= p80] = "choppy"
    df_test["regime"] = regime
    df_test["run_len"] = df_test["run_len"].fillna(0).astype(int)
    sel = gate_select(df_test["regime"], df_test["run_len"]).to_numpy()
    df_test["ratio_dyn"] = np.where(sel == 3, df_test["ratio_s3"],
                            np.where(sel == 5, df_test["ratio_s5"], df_test["ratio_s10"]))

    # r20 forward label
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

    # 两臂: baseline=ratio_s5 (V12.31 等价固定窗口), dynamic=ratio_dyn
    out = {"test_month": test_month, "train_start": train_start, "train_end": train_end}
    for arm, col in [("base", "ratio_s5"), ("dyn", "ratio_dyn")]:
        hold = build_dual(df_test, ind_mom, sort_col=col)
        m = month_metrics(hold, df_test)
        if m is None:
            print(f"  {arm}: 无持仓/无 r20, 跳过本月", flush=True)
            return None
        out[f"{arm}_alpha"] = m["alpha"]
        out[f"{arm}_sharpe"] = m["sharpe"]
        out[f"{arm}_n_days"] = m["n_days"]
        out[f"{arm}_n_stocks"] = m["avg_n_stocks"]
    frac_switch = float((sel != 5).mean())
    out["frac_switch"] = frac_switch
    print(f"  base α={out['base_alpha']:+.3f}pp Sh={out['base_sharpe']:+.2f} | "
          f"dyn α={out['dyn_alpha']:+.3f}pp Sh={out['dyn_sharpe']:+.2f} | "
          f"切换 {frac_switch:.0%}", flush=True)

    del df_train, df_test, boosters
    gc.collect()
    return out


# ───────────────────────── 主流程 ─────────────────────────
def main():
    t0 = time.time()
    print("\n=== T-005 walk-forward 决策 gate (ratio_dyn vs V12.31 固定窗口) ===\n", flush=True)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json")
                      .read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map:
        daily_full["industry_id"] = (daily_full["industry"].fillna("unknown")
                                     .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily_full.columns:
            daily_full[c] = 0.0
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[data] daily_full {len(daily_full):,} 行 / {daily_full['ts_code'].nunique()} 股", flush=True)

    # 价 (标签 + 门控)
    px = load_daily()
    slim = (daily_full[["ts_code", "trade_date"]]
            .merge(px, on=["ts_code", "trade_date"], how="left")
            .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    print("[label] 构建 3 尺度 forward 标签 (仅训练用) ...", flush=True)
    for H, g in SCALES.items():
        daily_full[f"label_s{H}"] = build_label(slim, H, g).values
    del slim
    gc.collect()

    print("[gate] 门控输入 (MA5/MA60 regime + run_len, causal) ...", flush=True)
    gate = compute_gate_inputs(px)
    gate["trade_date"] = gate["trade_date"].astype(str)

    print("[label] r20 forward + 行业动量 ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)

    # r20 池模型 (生产, 不重训, 两臂共用)
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt")
                      .read_text(encoding="utf-8"))
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json")
                          .read_text(encoding="utf-8"))
    r20_fc = meta_r20["feature_cols"]
    print(f"[prep] 完成 {time.time()-t0:.0f}s\n", flush=True)

    # checkpoint: 已完成月跳过
    done = set()
    results = []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records")
        done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月: {sorted(done)}", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, daily_full, fc, ind_map, gate, ind_mom, r20_lab, b20, r20_fc)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(MONTHLY, index=False)   # 每月落盘

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].copy()
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("test_month").reset_index(drop=True)

    # ── 汇总两臂 ──
    def agg(arm):
        a = df[f"{arm}_alpha"]
        return {"alpha_mean": float(a.mean()), "alpha_median": float(a.median()),
                "alpha_std": float(a.std()), "sharpe_mean": float(df[f"{arm}_sharpe"].mean()),
                "worst_month": float(a.min()),
                "worst_month_id": str(df.loc[a.idxmin(), "test_month"]),
                "pos_alpha_ratio": float((a > 0).mean()),
                "disaster_months": int((a < -1).sum()), "n_months": int(len(df))}
    base, dyn = agg("base"), agg("dyn")

    d_alpha = dyn["alpha_mean"] - base["alpha_mean"]
    # ── 事前注册 gate (SIGN-R01, 全相对 baseline) ──
    c1 = d_alpha >= 0.30
    c2 = dyn["sharpe_mean"] >= base["sharpe_mean"]
    c3 = dyn["worst_month"] >= base["worst_month"]
    c4 = dyn["pos_alpha_ratio"] >= base["pos_alpha_ratio"]
    conds = {"delta_alpha>=+0.30pp": bool(c1), "sharpe>=baseline": bool(c2),
             "worst_month>=baseline": bool(c3), "pos_alpha_ratio>=baseline": bool(c4)}
    passed = all(conds.values())
    status = "PASS" if passed else "REJECT"

    res = {
        "window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
        "protocol": f"walk-forward {TRAIN_LOOKBACK_MONTHS}m lookback, monthly retrain, "
                    "r20 pool model fixed (prod), apples-to-apples (sort col only diff)",
        "baseline": base, "dynamic": dyn,
        "delta_alpha_pp": d_alpha, "frac_switch_mean": float(df["frac_switch"].mean()),
        "gate_conditions": conds, "gate_status": status,
        "note": "Δα 等相对量是 gate; 绝对 α 含两臂共模 r20 池 lookahead, 不作判定 (SIGN-R03)。"
                "REJECT 合法完成, 禁止同段 OOS 重调 (SIGN-R02)。",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n\n=== T-005 walk-forward 汇总 ===\n", flush=True)
    print(f"  月数: {len(df)}  ({TEST_MONTHS[0]}-{TEST_MONTHS[-1]})", flush=True)
    print(f"  {'指标':22s} {'baseline(s5)':>14s} {'dynamic(dyn)':>14s}", flush=True)
    print(f"  {'月化 α(pp)':22s} {base['alpha_mean']:>+14.3f} {dyn['alpha_mean']:>+14.3f}", flush=True)
    print(f"  {'Sharpe':22s} {base['sharpe_mean']:>+14.2f} {dyn['sharpe_mean']:>+14.2f}", flush=True)
    print(f"  {'最差月(pp)':22s} {base['worst_month']:>+14.3f} {dyn['worst_month']:>+14.3f}", flush=True)
    print(f"  {'正 α 月占比':22s} {base['pos_alpha_ratio']:>14.1%} {dyn['pos_alpha_ratio']:>14.1%}", flush=True)
    print(f"\n  Δα = {d_alpha:+.3f}pp  (gate 需 >= +0.30pp)", flush=True)
    print(f"  门控平均切换比例: {df['frac_switch'].mean():.0%}", flush=True)
    print(f"\n  gate 四条件: {conds}", flush=True)
    print(f"  >>> verdict = {status} <<<\n", flush=True)
    print("  月度明细 (base α / dyn α):", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']}  base {r['base_alpha']:+.3f}  dyn {r['dyn_alpha']:+.3f}  "
              f"(Δ {r['dyn_alpha']-r['base_alpha']:+.3f})", flush=True)

    # ── verdict ──
    table = {
        "metric": ["alpha_monthly_pp", "sharpe", "worst_month_pp", "pos_alpha_month_ratio"],
        "baseline_v1231_fixed_s5": [round(base["alpha_mean"], 3), round(base["sharpe_mean"], 2),
                                    round(base["worst_month"], 3), round(base["pos_alpha_ratio"], 3)],
        "dynamic_ratio_dyn": [round(dyn["alpha_mean"], 3), round(dyn["sharpe_mean"], 2),
                              round(dyn["worst_month"], 3), round(dyn["pos_alpha_ratio"], 3)],
        "delta": [round(d_alpha, 3),
                  round(dyn["sharpe_mean"] - base["sharpe_mean"], 2),
                  round(dyn["worst_month"] - base["worst_month"], 3),
                  round(dyn["pos_alpha_ratio"] - base["pos_alpha_ratio"], 3)],
    }
    if passed:
        conclusion = (f"PASS: 动态窗口 ratio_dyn 在 19 月 walk-forward 上 Δα={d_alpha:+.3f}pp/月 "
                      f"(>=+0.30pp) 且 Sharpe/最差月/正α月占比均不低于 V12.31 固定窗口 baseline "
                      f"→ 真 alpha 成立, 解除 T-006 落地条件门。")
    else:
        fail = [k for k, v in conds.items() if not v]
        conclusion = (f"REJECT: 动态窗口 ratio_dyn 在 19 月 walk-forward 上 Δα={d_alpha:+.3f}pp/月, "
                      f"未满足事前注册 gate (未过: {fail})。相对 V12.31 固定窗口 baseline 无可落地真 "
                      f"alpha → 不落地, 负结论文档化 (SIGN-R02 合法完成)。门控切换 {df['frac_switch'].mean():.0%} "
                      f"但 ratio_dyn 与 s5 高度相关, 增量空间有限, 与 T-004 预判一致。")
    VERDICT.write_text(json.dumps({
        "id": "T-005", "status": status, "conclusion": conclusion,
        "metrics": {"alpha_delta_pp": round(d_alpha, 3),
                    "sharpe": round(dyn["sharpe_mean"], 2),
                    "worst_month": round(dyn["worst_month"], 3),
                    "pos_alpha_month_ratio": round(dyn["pos_alpha_ratio"], 3)},
        "comparison_table": table,
        "gate_conditions": conds,
        "n_months": int(len(df)), "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "frac_switch_mean": round(float(df["frac_switch"].mean()), 3),
        "artifacts": ["research/cache/t005_monthly.csv", "research/cache/t005_results.json",
                      "research/cache/t005_wf_models/"],
        "note": "apples-to-apples: 两臂唯一差异=池内排序列 (ratio_s5 vs ratio_dyn); 绝对 α 含共模 "
                "r20 池 lookahead, 仅 Δ 量作 gate (SIGN-R03)。REJECT 是合法完成。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] verdict={status} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    main()
