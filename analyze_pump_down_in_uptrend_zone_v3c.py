"""验证用户假设: 正启动子真样本后的 r10-r20 上涨期内, 负启动子是否真的都低分?

逻辑:
  1. 找所有 pump_up=1 真样本 (t 时刻 5 日涨 ≥ 10%, 回撤 ≤ 5%)
  2. 对每个锚点 t, 看 t+1, t+5, t+10, t+15, t+20 这些时刻的:
     - 该股 P_down 分数
     - 实际未来 5 日跌幅 (检验模型是否对)
  3. 跟全市场 P_down 均值对比, 看锚点之后的"上涨区"内是否 P_down 偏高

如果 P_down 在上涨区内系统性高于全市场 → 用户洞察成立, 需重训
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

OOS_START = "20251001"
OOS_END = "20260331"


def main():
    print(f"\n=== 验证: 上涨区内 P_down 是否系统性偏高? ===\n", flush=True)

    # 1. 加载 daily 算 forward label + 锚点
    print("[1] 加载 daily, 算 pump_up/down label ...", flush=True)
    daily_dir = ROOT / "output/tushare_cache/daily"
    files = sorted(daily_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    # 未来 5 日 max/min
    big["max5"] = big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(5, min_periods=5).max().shift(-5)).reset_index(level=0, drop=True)
    big["min5"] = big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(5, min_periods=5).min().shift(-5)).reset_index(level=0, drop=True)
    big["up5"] = big["max5"] / big["next_open"] - 1
    big["dn5"] = big["min5"] / big["next_open"] - 1
    big["pump_up_true"] = ((big["up5"] >= 0.10) & (big["dn5"] >= -0.05)).astype(int)
    big["pump_dn_true"] = ((big["dn5"] <= -0.10) & (big["up5"] <= 0.05)).astype(int)
    # 未来 20 日累计收益 (用 close 计算)
    big["close_20d"] = big.groupby("ts_code")["close"].shift(-20)
    big["r20_fwd"] = (big["close_20d"] / big["close"] - 1) * 100

    # 2. v3b 推理 OOS
    print("\n[2] v3b 推理 OOS 期 P_up / P_down ...", flush=True)
    df = load_window(OOS_START, OOS_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    lf = pd.read_parquet(ROOT / "output/long_return_features/features.parquet")
    lf["trade_date"] = lf["trade_date"].astype(str)
    df = df.merge(lf, on=["ts_code", "trade_date"], how="left")

    prod = ROOT / "output" / "production" / "r5_pump_3way_lgbm_v3c"
    print(f"  [v3c 测试 - label 时序约束]", flush=True)
    b = lgb.Booster(model_str=(prod / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((prod / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})
    if "industry" in df.columns:
        df["industry_id"] = df["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in fc:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    proba = b.predict(df[fc].astype("float32"))
    df = df.assign(P_neutral=proba[:, 0], P_down=proba[:, 1], P_up=proba[:, 2])

    # merge 真实 label + r20_fwd
    df = df.merge(big[["ts_code", "trade_date", "pump_up_true", "pump_dn_true", "r20_fwd"]],
                    on=["ts_code", "trade_date"], how="left")
    df = df.dropna(subset=["P_up", "P_down", "pump_up_true"])
    print(f"  样本: {len(df):,}, 全市场 P_down 均: {df['P_down'].mean():.4f}", flush=True)

    # 3. 找锚点 (pump_up_true=1 的样本)
    print("\n[3] 锚点分析 ...", flush=True)
    anchors = df[df["pump_up_true"] == 1].copy()
    print(f"  pump_up_true=1 锚点数: {len(anchors):,}", flush=True)

    # 4. 对每个锚点 (ts_code, t), 找 t+1, t+5, t+10, t+15, t+20 的 P_down
    # 用 groupby + shift 高效实现
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 给每个 (ts_code, trade_date) 一个标记: 是否在锚点的 +1 到 +20 范围内
    # 简化: 用 groupby 内的 rolling forward
    # 先标记 anchor_flag
    df["is_anchor"] = (df["pump_up_true"] == 1).astype(int)

    # 用 shift 取锚点后 1/5/10/15/20 日的 P_down
    print("  计算锚点后 +1/+5/+10/+15/+20 日的 P_down 均值 ...", flush=True)
    for offset in [1, 3, 5, 10, 15, 20]:
        # 取每股 trade_date 后 offset 天的 P_down
        df[f"P_down_t+{offset}"] = df.groupby("ts_code")["P_down"].shift(-offset)
        df[f"r20_fwd_t+{offset}"] = df.groupby("ts_code")["r20_fwd"].shift(-offset)

    # 锚点 (现在含 t+N P_down)
    anc = df[df["pump_up_true"] == 1].dropna(subset=[f"P_down_t+5"])
    print(f"  有效锚点 (有 t+5 数据): {len(anc):,}\n", flush=True)

    print(f"=== 锚点 (pump_up=1) 后续 P_down 均值 vs 全市场 ===\n", flush=True)
    mkt_pd = df["P_down"].mean()
    print(f"  全市场 P_down 均: {mkt_pd:.4f}", flush=True)
    print(f"\n  {'offset':10s} {'P_down 均':>10s} {'vs 全市场':>10s} {'r20_fwd 均%':>10s}")
    for offset in [0, 1, 3, 5, 10, 15, 20]:
        if offset == 0:
            pd_mean = anc["P_down"].mean()
            r20_mean = anc["r20_fwd"].mean()
        else:
            pd_mean = anc[f"P_down_t+{offset}"].mean()
            r20_mean = anc[f"r20_fwd_t+{offset}"].mean()
        diff = pd_mean - mkt_pd
        marker = "↓ 偏低" if diff < -0.005 else ("↑ 偏高" if diff > 0.005 else "≈ 相当")
        print(f"  t+{offset:<6d} {pd_mean:>10.4f} {diff:>+9.4f} {marker:8s} "
               f"{r20_mean if pd.notna(r20_mean) else 0:>+9.2f}%", flush=True)

    # 5. 分位数对比 (更细致)
    print(f"\n=== 锚点后 P_down 分位数对比 (t+5) ===\n", flush=True)
    print(f"  {'qt':>6s} {'锚点后 t+5':>10s} {'全市场':>10s} {'差异':>10s}")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        a_q = anc["P_down_t+5"].quantile(q)
        m_q = df["P_down"].quantile(q)
        print(f"  q={q:.2f} {a_q:>10.4f} {m_q:>10.4f} {a_q-m_q:>+9.4f}", flush=True)

    # 6. 关键: 锚点之后 P_down 高 (Top 20%) 的占比
    p_down_high_threshold = df["P_down"].quantile(0.80)   # 全市场 P_down Top 20%
    print(f"\n=== 锚点后 P_down 误高占比 (≥ 全市场 80% 分位 {p_down_high_threshold:.4f}) ===\n",
           flush=True)
    print(f"  {'offset':10s} {'误高%':>10s}")
    for offset in [0, 1, 3, 5, 10, 15, 20]:
        if offset == 0:
            pct = (anc["P_down"] >= p_down_high_threshold).mean()
        else:
            pct = (anc[f"P_down_t+{offset}"] >= p_down_high_threshold).mean()
        print(f"  t+{offset:<6d} {pct*100:>9.2f}%", flush=True)


if __name__ == "__main__":
    main()
