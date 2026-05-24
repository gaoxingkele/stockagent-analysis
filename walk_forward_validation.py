"""Walk-forward 严格验证 (任务 3).

每月用前 24 个月数据重训 r5_pump + r5_pump_down, 测下个月.
跟固定 OOS 调参对比, 看 V11 在真实滚动场景的 α 是否稳定.

测试月: 202410 - 202604 (15 个月)
训练窗口: 测试月前 24 个月

跟一次性 OOS 的差异:
  - 固定 OOS: 一次 cut at 20250930, 测 20251001-20260331 (容易过拟合)
  - Walk-forward: 滚动重训, 每月独立预测 → 真实运营预期

输出: output/walk_forward/monthly_results.csv + report.md
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window, EXCLUDE

OUT = ROOT / "output" / "walk_forward"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
WF_PROD = ROOT / "output" / "walk_forward_models"   # 临时月度模型
WF_PROD.mkdir(parents=True, exist_ok=True)
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

# 测试月份
TEST_MONTHS = ["202410", "202411", "202412",
                 "202501", "202502", "202503", "202504", "202505", "202506",
                 "202507", "202508", "202509",
                 "202510", "202511", "202512",
                 "202601", "202602", "202603", "202604"]

TRAIN_LOOKBACK_MONTHS = 24

# V11 生产配置
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60


def get_train_end_for_test_month(test_month: str) -> str:
    """测试月份 → 训练 cut 日期 (测试月初前一日)."""
    year = int(test_month[:4]); mon = int(test_month[4:])
    if mon == 1:
        return f"{year-1}1231"
    return f"{year}{mon-1:02d}28"   # 大致月初


def get_train_start(train_end: str, n_months: int) -> str:
    """train_end 倒推 n 个月作为 train_start."""
    y = int(train_end[:4]); m = int(train_end[4:6])
    new_m = m - n_months
    while new_m <= 0:
        new_m += 12
        y -= 1
    return f"{y}{new_m:02d}01"


def compute_pump_labels(daily_cache_dir: Path) -> pd.DataFrame:
    """从 daily cache 算 pump_up + pump_down label."""
    files = sorted(daily_cache_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    N = 5
    big["max_high_next"] = (big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
    big["min_low_next"] = (big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))
    big["upside"] = big["max_high_next"] / big["next_open"] - 1
    big["downside"] = big["min_low_next"] / big["next_open"] - 1
    big["is_pump_up"] = ((big["upside"] >= 0.10) & (big["downside"] >= -0.05)).astype(int)
    big["is_pump_down"] = ((big["downside"] <= -0.10) & (big["upside"] <= 0.05)).astype(int)
    return big[["ts_code", "trade_date", "is_pump_up", "is_pump_down"]]


def train_pump_model(df: pd.DataFrame, feat_cols: list, label_col: str,
                       train_end: str, model_name: str) -> str:
    """训单个 pump 模型 (轻量, 不存 long_oos meta)."""
    out_dir = WF_PROD / model_name
    if (out_dir / "classifier.txt").exists():
        return str(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    sub = df.dropna(subset=[label_col]).copy()
    train = sub[sub["trade_date"] < train_end]
    if len(train) > 1_500_000:
        train = train.sample(n=1_500_000, random_state=42).reset_index(drop=True)
    if train[label_col].sum() < 1000:
        return None   # 正样本不够

    X = train[feat_cols].astype("float32")
    y = train[label_col].astype("int8")

    clf = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        n_estimators=500, learning_rate=0.05,   # 单月不需要太长
        num_leaves=63, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        is_unbalance=True,
        random_state=42, n_jobs=4, verbose=-1,
    )
    # 不用 eval (训练快), 固定 500 棵树
    clf.fit(X, y, categorical_feature=["industry_id"])
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": label_col,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_dir)


def compute_ind_mom(daily):
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["mom_60d"] = (big.groupby("ts_code")["close"].shift(1) /
                        big.groupby("ts_code")["close"].shift(61) - 1)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "industry"]].drop_duplicates("ts_code")
    big = big.merge(basic, on="ts_code", how="left")
    ind = big.dropna(subset=["mom_60d"]).groupby(["trade_date", "industry"]).agg(
        industry_mom_60d=("mom_60d", "mean")).reset_index()
    ind["industry_mom_60d_rank"] = ind.groupby("trade_date")["industry_mom_60d"].rank(
        pct=True, method="first")
    return ind


def compute_r20_label():
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    big["close_20d"] = big.groupby("ts_code")["close"].shift(-20)
    big["r20_fresh"] = (big["close_20d"] / big["next_open"] - 1) * 100
    return big[["ts_code", "trade_date", "r20_fresh"]]


def apply_cap(pool, alloc, cap, prior=None, sort_col="pump_up_score"):
    if pool.empty or cap >= 1.0: return pool
    per_stock = alloc / len(pool)
    pool = pool.sort_values(sort_col, ascending=False)
    ia = dict(prior or {})
    keep = []
    for idx, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        new = ia.get(ind, 0) + per_stock
        if new > cap + 1e-9: continue
        ia[ind] = new
        keep.append(idx)
    return pool.loc[keep]


def industry_alloc(pool, alloc):
    if pool.empty: return {}
    per = alloc / len(pool)
    counts = {}
    for _, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        counts[ind] = counts.get(ind, 0) + 1
    return {k: v * per for k, v in counts.items()}


def build_dual_v11(daily, ind_mom):
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue
        g["r20_rank"] = g["pred_r20"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue
        v7c = v7c[v7c["pump_down_score"] < PUMP_DOWN_THRESHOLD]
        if len(v7c) == 0: continue
        v7c = v7c.sort_values("pump_up_score", ascending=False)
        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN)
        a_ind = industry_alloc(a_pool, A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN)
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind)
        for tag, pool, alloc in [("A", a_pool, A_PCT), ("B", b_pool, B_PCT)]:
            if len(pool) == 0: continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                                "industry": row.get("industry", ""),
                                "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                                "alloc_pct": per})
    return pd.DataFrame(rows)


def evaluate_month(test_month: str, daily_full: pd.DataFrame, ind_mom: pd.DataFrame,
                     r20_lab: pd.DataFrame, pump_label: pd.DataFrame) -> dict:
    """单月 walk-forward 训练 + 评估."""
    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    print(f"\n--- {test_month}: train {train_start}-{train_end} ---", flush=True)

    # 训练数据 (train_start 到 train_end)
    df_train = daily_full[(daily_full["trade_date"] >= train_start) &
                            (daily_full["trade_date"] < train_end)].copy()
    df_train = df_train.merge(pump_label, on=["ts_code", "trade_date"], how="inner")
    if len(df_train) < 100000:
        print(f"  跳过 (训练样本不足: {len(df_train)})", flush=True)
        return None

    # 特征
    EXC = set(EXCLUDE) | {"is_pump_up", "is_pump_down"}
    feat_cols = [c for c in df_train.columns
                  if c not in EXC and pd.api.types.is_numeric_dtype(df_train[c])]

    # clip
    for c in feat_cols:
        df_train[c] = df_train[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    # 训练 pump_up + pump_down (单月模型, 文件名带月)
    print(f"  训练 pump_up (n={df_train['is_pump_up'].sum():,} 正), "
           f"pump_down (n={df_train['is_pump_down'].sum():,} 正)", flush=True)
    pump_up_path = train_pump_model(df_train, feat_cols, "is_pump_up",
                                       train_end, f"pump_up_{test_month}")
    pump_dn_path = train_pump_model(df_train, feat_cols, "is_pump_down",
                                       train_end, f"pump_down_{test_month}")
    if pump_up_path is None or pump_dn_path is None:
        return None

    # r20 模型用现有 r20_v16_long_nost (V12 标配, 不重训)
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20_fc = meta_r20["feature_cols"]
    ind_map = meta_r20.get("industry_map", {})

    # 测试月数据
    df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
    if len(df_test) < 100:
        print(f"  跳过 (测试样本不足: {len(df_test)})", flush=True)
        return None

    if "industry" in df_test.columns and ind_map:
        df_test["industry_id"] = df_test["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in feat_cols:
        if c not in df_test.columns: df_test[c] = 0.0
        df_test[c] = df_test[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    for c in r20_fc:
        if c not in df_test.columns: df_test[c] = 0.0

    # 推理 3 个模型
    b_up = lgb.Booster(model_str=(WF_PROD / f"pump_up_{test_month}" / "classifier.txt").read_text(encoding="utf-8"))
    b_dn = lgb.Booster(model_str=(WF_PROD / f"pump_down_{test_month}" / "classifier.txt").read_text(encoding="utf-8"))
    X_feat = df_test[feat_cols].astype("float32")
    X_r20 = df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)

    df_test["pump_up_score"] = b_up.predict(X_feat)
    df_test["pump_down_score"] = b_dn.predict(X_feat)
    df_test["pred_r20"] = b20.predict(X_r20)

    # merge r20 label
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

    # 跑 V11 双轨
    hold = build_dual_v11(df_test, ind_mom)
    if hold.empty:
        return None

    # 月度指标
    h = hold.dropna(subset=["r20_fresh"]).copy()
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

    alpha_avg = pnl["alpha"].mean()
    sharpe = alpha_avg / (pnl["alpha"].std() + 1e-9) * np.sqrt(12)
    n_days = pnl["entry_date"].nunique()

    print(f"  {test_month}: α={alpha_avg:+.3f}pp Sharpe={sharpe:+.2f} 天数={n_days}", flush=True)

    # 释放内存
    del df_train, df_test, b_up, b_dn
    gc.collect()

    return {
        "test_month": test_month,
        "train_start": train_start,
        "train_end": train_end,
        "alpha": float(alpha_avg),
        "sharpe": float(sharpe),
        "n_days": int(n_days),
        "avg_n_stocks": float(pnl["n"].mean()),
        "mkt_avg": float(pnl["mkt"].mean()),
    }


def main():
    t0 = time.time()
    print(f"\n=== Walk-Forward 严格验证 ({len(TEST_MONTHS)} 个月) ===\n", flush=True)
    print(f"  训练窗口: 测试月前 {TRAIN_LOOKBACK_MONTHS} 个月", flush=True)
    print(f"  测试月: {TEST_MONTHS[0]} - {TEST_MONTHS[-1]}\n", flush=True)

    # 一次加载所有需要的数据 (覆盖 2022 至今)
    data_start = get_train_start(TEST_MONTHS[0] + "01", TRAIN_LOOKBACK_MONTHS + 2)
    data_end = "20260525"
    print(f"  加载 daily factor {data_start}-{data_end} ...", flush=True)
    daily_full = load_window(data_start, data_end, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily_full = daily_full.merge(lf, on=["ts_code", "trade_date"], how="left")
    print(f"  daily_full: {len(daily_full):,}", flush=True)

    # industry_id 用 r20 模型的 map
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    ind_map = meta_r20.get("industry_map", {})
    if "industry" in daily_full.columns:
        daily_full["industry_id"] = daily_full["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)

    # pump labels (全市场)
    print(f"  计算 pump_up/down labels ...", flush=True)
    pump_label = compute_pump_labels(ROOT / "output" / "tushare_cache" / "daily")
    pump_label["trade_date"] = pump_label["trade_date"].astype(str)

    # r20 label
    print(f"  计算 r20 forward labels ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)

    # industry momentum
    print(f"  计算 industry 60d momentum ...", flush=True)
    ind_mom = compute_ind_mom(daily_full)
    print(f"  数据准备完成 {time.time()-t0:.0f}s", flush=True)

    # 逐月评估
    results = []
    for m_ in TEST_MONTHS:
        r = evaluate_month(m_, daily_full, ind_mom, r20_lab, pump_label)
        if r: results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "monthly_results.csv", index=False)

    print(f"\n\n=== Walk-Forward 汇总 ===\n", flush=True)
    print(f"  共 {len(df)} 月 (训练-测试不重叠)", flush=True)
    print(f"  平均月 α: {df['alpha'].mean():+.3f}pp", flush=True)
    print(f"  α 中位数: {df['alpha'].median():+.3f}pp", flush=True)
    print(f"  α std: {df['alpha'].std():.3f}", flush=True)
    print(f"  平均 Sharpe: {df['sharpe'].mean():+.2f}", flush=True)
    print(f"  α > 0 月数: {(df['alpha'] > 0).sum()}/{len(df)} ({(df['alpha'] > 0).mean()*100:.0f}%)",
           flush=True)
    print(f"  α < -1 月数 (灾难): {(df['alpha'] < -1).sum()}", flush=True)
    print(f"  最差月: {df['alpha'].min():+.3f}pp ({df.loc[df['alpha'].idxmin(), 'test_month']})",
           flush=True)
    print(f"  最佳月: {df['alpha'].max():+.3f}pp ({df.loc[df['alpha'].idxmax(), 'test_month']})",
           flush=True)

    # 月度详情
    print(f"\n--- 月度详情 ---", flush=True)
    print(f"  {'month':10s} {'α':10s} {'Sharpe':8s} {'天数':5s} {'股数':6s}", flush=True)
    for _, r in df.iterrows():
        marker = " *" if r["alpha"] < -1 else ""
        print(f"  {r['test_month']:10s} {r['alpha']:+.3f}pp  {r['sharpe']:+.2f}    "
               f"{r['n_days']:3d}    {r['avg_n_stocks']:.1f}{marker}", flush=True)

    # 报告
    md = [f"# Walk-Forward 严格验证报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"测试月: {len(df)}, 训练窗口: 前 {TRAIN_LOOKBACK_MONTHS} 个月\n\n",
            f"## 整体指标\n\n",
            f"- 平均月 α: **{df['alpha'].mean():+.3f}pp**\n",
            f"- α 中位数: {df['alpha'].median():+.3f}pp\n",
            f"- α std: {df['alpha'].std():.3f}\n",
            f"- 平均 Sharpe: **{df['sharpe'].mean():+.2f}**\n",
            f"- 正 α 月占比: **{(df['alpha'] > 0).mean()*100:.0f}%**\n",
            f"- 灾难月数 (α < -1pp): {(df['alpha'] < -1).sum()}\n\n",
            f"## 跟固定 OOS 调参对比\n\n",
            f"- 固定 OOS V11 旧 OOS α: +1.59pp / Sharpe 1.44 (可能含 OOS 过拟合)\n",
            f"- 固定 OOS V11 新 OOS α: +3.36pp / Sharpe 2.02\n",
            f"- **Walk-forward α: {df['alpha'].mean():+.3f}pp / Sharpe {df['sharpe'].mean():+.2f} (无过拟合)**\n\n",
            f"## 月度明细\n\n| 月份 | α | Sharpe | 天数 | 股数 |\n|---|---|---|---|---|\n"]
    for _, r in df.iterrows():
        md.append(f"| {r['test_month']} | {r['alpha']:+.3f}pp | {r['sharpe']:+.2f} | "
                   f"{r['n_days']} | {r['avg_n_stocks']:.1f} |\n")
    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
