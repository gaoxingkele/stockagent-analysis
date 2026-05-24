"""α 衰减检测系统 (任务 2).

每日运行, 跟踪 V12 双轨架构的"模拟实战 α" 滚动表现.
触发预警/告警, 防止 alpha 死亡而不知.

逻辑:
  1. 历史 N 日 (默认最近 60 日) 每日:
     - 假设按 V12 双轨推荐建仓
     - 持有 5/10/20 日各算一次 α (vs 全市场)
  2. 计算最近 W 日 (默认 20 日) 滚动 α 均值
  3. 状态机:
     - normal: 滚动 α > -0.5pp
     - warning: 滚动 α ∈ [-1.0, -0.5pp]
     - alert: 滚动 α < -1.0pp (建议减仓到 50%)
     - critical: 滚动 α < -2.0pp (建议停止策略)
  4. 输出每日报告 + 状态变化日志

用法:
  python scripts/alpha_decay_monitor.py                    # 跑最近 60 日
  python scripts/alpha_decay_monitor.py --lookback 90      # 跑 90 日
  python scripts/alpha_decay_monitor.py --hold 5           # 用 5 日持仓 (更敏感)

输出:
  output/alpha_decay/daily_alpha_<date>.csv  每日 α 序列
  output/alpha_decay/alert_log.csv           状态变化日志
  output/alpha_decay/report_<date>.md        最新日报
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

OUT = ROOT / "output" / "alpha_decay"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

# V11 生产配置
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60

# α 衰减状态阈值 (滚动均值)
ROLLING_WINDOW = 20  # 滚动 20 日
ALPHA_WARNING = -0.5    # pp/期, 警告线
ALPHA_ALERT = -1.0      # 告警线
ALPHA_CRITICAL = -2.0   # 临界线


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


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


def compute_forward_label(daily_cache_dir, hold_days):
    files = sorted(daily_cache_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    big[f"close_{hold_days}d"] = big.groupby("ts_code")["close"].shift(-hold_days)
    big[f"r{hold_days}_fresh"] = (big[f"close_{hold_days}d"] / big["next_open"] - 1) * 100
    return big[["ts_code", "trade_date", f"r{hold_days}_fresh"]]


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


def build_dual_v11(daily, ind_mom, hold_col):
    """V11 生产配置双轨构建."""
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue
        # V11 pump_down 硬过滤
        v7c = v7c[v7c["pump_down_score"] < PUMP_DOWN_THRESHOLD]
        if len(v7c) == 0: continue
        # 按 pump_up 排序
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
                                hold_col: float(row[hold_col]) if pd.notna(row.get(hold_col)) else np.nan,
                                "alloc_pct": per})
    return pd.DataFrame(rows)


def state_of_alpha(rolling_alpha: float) -> str:
    """根据滚动 α 决定状态."""
    if pd.isna(rolling_alpha):
        return "init"
    if rolling_alpha < ALPHA_CRITICAL:
        return "critical"
    if rolling_alpha < ALPHA_ALERT:
        return "alert"
    if rolling_alpha < ALPHA_WARNING:
        return "warning"
    return "normal"


def action_for_state(state: str) -> str:
    return {
        "init": "尚未足够数据 (累积中)",
        "normal": "保持当前仓位 (90%)",
        "warning": "提高警惕, 密切跟踪",
        "alert": "建议减仓到 50% (alpha 持续低于 -1pp)",
        "critical": "建议停止策略 / 全部转现金 (alpha 已死亡)",
    }.get(state, "未知")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60,
                     help="跟踪最近 N 个交易日 (默认 60)")
    ap.add_argument("--hold", type=int, default=20,
                     help="模拟持仓周期 (5/10/20, 默认 20)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\n=== α 衰减检测系统 ===\n", flush=True)
    print(f"参数: lookback={args.lookback} 日, hold={args.hold} 日", flush=True)
    print(f"阈值: warning < {ALPHA_WARNING}pp, alert < {ALPHA_ALERT}pp, "
           f"critical < {ALPHA_CRITICAL}pp\n", flush=True)

    # 加载需要的数据范围 (lookback + hold_days + 60 日提前量算 mom)
    days_to_load = args.lookback + args.hold + 100
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    all_dates = sorted([f.stem for f in daily_dir.glob("*.parquet")])
    if len(all_dates) < days_to_load:
        print(f"!! daily 数据不够 ({len(all_dates)} < {days_to_load})", flush=True)
        return
    start_date = all_dates[-days_to_load]
    end_date = all_dates[-1]
    print(f"  数据范围: {start_date} - {end_date}", flush=True)

    # load_window + 推理
    daily = load_window(start_date, end_date, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    for name in ["r20_v16_long_nost", "r5_v17_long_nost",
                   "r5_pump_lgbm_v1", "r5_pump_down_lgbm_v1"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        if name == "r5_pump_lgbm_v1":
            daily["pump_up_score"] = b.predict(X)
        elif name == "r5_pump_down_lgbm_v1":
            daily["pump_down_score"] = b.predict(X)
        else:
            daily[f"pred_{name}"] = b.predict(X)
    print(f"  推理完成 {time.time()-t0:.0f}s", flush=True)

    ind_mom = compute_ind_mom(daily)
    hold_col = f"r{args.hold}_fresh"
    forward_label = compute_forward_label(daily_dir, args.hold)
    forward_label["trade_date"] = forward_label["trade_date"].astype(str)
    daily = daily.merge(forward_label, on=["ts_code", "trade_date"], how="left")

    # 跑 v11 双轨
    print(f"\n  跑 V11 生产配置 (pump_up sort + pump_down @{PUMP_DOWN_THRESHOLD} 过滤) ...",
           flush=True)
    hold = build_dual_v11(daily, ind_mom, hold_col)

    if hold.empty:
        print("  !! 无持仓数据", flush=True); return

    # 每日 α
    h = hold.dropna(subset=[hold_col]).copy()
    h[hold_col] = h[hold_col].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h[hold_col]
    total = A_PCT + B_PCT
    daily_pnl = h.groupby("entry_date").agg(gross=("w", "sum"),
                                              n=("ts_code", "count")).reset_index()
    daily_pnl["net"] = daily_pnl["gross"] - total * COST_BPS * 200

    mkt = daily.groupby("trade_date")[hold_col].apply(
        lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    daily_pnl = daily_pnl.merge(mkt, on="entry_date", how="left")
    daily_pnl["alpha"] = daily_pnl["net"] - daily_pnl["mkt"] * total

    # 滚动 α (W 日均)
    daily_pnl["rolling_alpha"] = daily_pnl["alpha"].rolling(ROLLING_WINDOW,
                                                              min_periods=5).mean()
    daily_pnl["state"] = daily_pnl["rolling_alpha"].apply(state_of_alpha)
    daily_pnl["action"] = daily_pnl["state"].apply(action_for_state)
    daily_pnl["state_change"] = daily_pnl["state"] != daily_pnl["state"].shift(1)

    # 只保 lookback 期
    daily_pnl = daily_pnl.sort_values("entry_date")
    daily_pnl = daily_pnl.tail(args.lookback).reset_index(drop=True)

    # 输出每日 csv
    today = datetime.now().strftime("%Y%m%d")
    daily_csv = OUT / f"daily_alpha_{today}.csv"
    daily_pnl.to_csv(daily_csv, index=False)

    # 状态变化日志 (累积)
    alert_log_p = OUT / "alert_log.csv"
    changes = daily_pnl[daily_pnl["state_change"]].copy()
    if alert_log_p.exists():
        existing = pd.read_csv(alert_log_p)
        all_changes = pd.concat([existing, changes], ignore_index=True).drop_duplicates(
            subset=["entry_date", "state"])
    else:
        all_changes = changes
    all_changes.to_csv(alert_log_p, index=False)

    # 当前状态
    last_row = daily_pnl.iloc[-1]
    print(f"\n=== 当前 α 衰减状态 ===\n", flush=True)
    print(f"  最新日: {last_row['entry_date']}", flush=True)
    print(f"  当日 α: {last_row['alpha']:+.3f}pp", flush=True)
    print(f"  滚动 {ROLLING_WINDOW} 日 α: {last_row['rolling_alpha']:+.3f}pp", flush=True)
    print(f"  状态: {last_row['state'].upper()}", flush=True)
    print(f"  建议: {last_row['action']}", flush=True)

    # 状态历史
    print(f"\n--- 最近 {min(args.lookback, len(daily_pnl))} 日状态分布 ---", flush=True)
    state_counts = daily_pnl["state"].value_counts()
    for s in ["normal", "warning", "alert", "critical", "init"]:
        if s in state_counts:
            print(f"  {s:10s}: {state_counts[s]:3d} 日 "
                   f"({state_counts[s]/len(daily_pnl)*100:.1f}%)", flush=True)

    # 状态变化日志
    if not changes.empty:
        print(f"\n--- 状态变化日志 ({len(changes)} 次) ---", flush=True)
        for _, r in changes.iterrows():
            print(f"  {r['entry_date']}: → {r['state'].upper()} "
                   f"(rolling_α = {r['rolling_alpha']:+.3f}pp)", flush=True)

    # 报告
    md = [f"# α 衰减检测报告 ({today})\n\n",
            f"## 当前状态: **{last_row['state'].upper()}**\n\n",
            f"- 最新日: {last_row['entry_date']}\n",
            f"- 滚动 {ROLLING_WINDOW} 日 α: **{last_row['rolling_alpha']:+.3f}pp**\n",
            f"- 当日 α: {last_row['alpha']:+.3f}pp\n",
            f"- 建议: **{last_row['action']}**\n\n",
            f"## 阈值参考\n\n",
            f"| 状态 | 滚动 α 范围 | 建议 |\n|---|---|---|\n",
            f"| normal | > {ALPHA_WARNING} pp | 保持仓位 90% |\n",
            f"| warning | [{ALPHA_ALERT}, {ALPHA_WARNING}] pp | 密切跟踪 |\n",
            f"| alert | [{ALPHA_CRITICAL}, {ALPHA_ALERT}] pp | 减仓到 50% |\n",
            f"| critical | < {ALPHA_CRITICAL} pp | 停止策略 |\n\n",
            f"## 最近 {args.lookback} 日 α 序列 (尾部 20 日)\n\n",
            "| 日期 | 当日 α | 滚动 α | 状态 |\n|---|---|---|---|\n"]
    for _, r in daily_pnl.tail(20).iterrows():
        ra = f"{r['rolling_alpha']:+.3f}" if pd.notna(r['rolling_alpha']) else "init"
        md.append(f"| {r['entry_date']} | {r['alpha']:+.3f}pp | {ra}pp | {r['state']} |\n")
    Path(OUT / f"report_{today}.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出:")
    print(f"  {daily_csv}")
    print(f"  {alert_log_p}")
    print(f"  {OUT / f'report_{today}.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
