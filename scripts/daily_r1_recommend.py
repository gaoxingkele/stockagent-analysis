"""P2 r1 T+1 实盘脚本.

每日 EOD (15:00 收盘后) 用 r1_next_open_v3_long 推全市场, 输出 Top N 隔夜推荐.

策略:
  - 入场: 当日 EOD 推荐, 次日 09:30 集合竞价 / 09:30 开盘买入
  - 退出: 次日 09:30 集合竞价直接卖出 (T+1 持有 1 个交易日 ≈ 23 小时)
  - 涨跌停过滤: dist_to_upper_limit_pct < 3% 排除 (买不到 + 滑点重)
  - cap r1 at +3% (集合竞价滑点真实约束, 不是模型 cap, 是预期收益评估)

长 OOS 142 日严格验证:
  - Top 5  月化 +28.8% Sharpe 23.65 MDD -2.8% α 胜率 92%
  - Top 10 月化 +21.0% Sharpe 18.48 MDD -3.5% α 胜率 90%  ⭐ 推荐
  - Top 20 月化 +10.4% Sharpe 11.09 MDD -4.0% α 胜率 84%
  - 实盘 25% 折扣后预期: Top10 月化 ~+5% Sharpe ~4.5

用法:
  python scripts/daily_r1_recommend.py                # 推最新可用日期
  python scripts/daily_r1_recommend.py 20260515       # 指定日期
  python scripts/daily_r1_recommend.py --top 20       # 调整 TopN

输出:
  output/daily_r1/<date>_top<N>.json   原始数据
  output/daily_r1/<date>_top<N>.md     人类可读推荐表 + 平仓清单
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT = ROOT / "output" / "daily_r1"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "r1_next_open_v3_long_nost"  # 2026-05-22 切 nost (ST 源头排除)
DIST_THRESHOLD = 3.0   # 距涨停 < 3% 排除
R1_CAP = 3.0           # 集合竞价滑点 cap (评估用, 不影响推荐排序)
DEFAULT_TOP = 10
MIN_AMOUNT_WAN = 5000  # 当日成交额下限 (5000 万元), 排除低流动性

# ⚠️ 重要警告:
# r1 模型即使训练时排除 ST, 实战长 OOS 142 日 Top10 月化仍 -3.7% Sharpe -4.0
# (vs 旧版 ST 偏见 -4.6%, 改善但仍负). 见 output/backtest_t1_long_nost/report.md
# 本脚本仅作研究参考, 不作单独实盘信号. 主信号请用 V12 双轨架构 (R5 反向过滤 +
# V7c) - 见 scripts/daily_v12_dual_track.py


def load_model():
    d = PROD / MODEL_NAME
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta


def map_anchored(v: np.ndarray, p5: float, p50: float, p95: float) -> np.ndarray:
    """与 v12_scoring 同款锚定 0-100 映射."""
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5 + 1e-9) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50 + 1e-9) * 50, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", help="YYYYMMDD, 默认取 factors_v3 最新日")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP)
    ap.add_argument("--dist", type=float, default=DIST_THRESHOLD,
                     help="距涨停过滤阈值 (百分点)")
    ap.add_argument("--keep-st", action="store_true",
                     help="保留 ST 股 (默认排除, ST 模型预测偏高且实盘风险大)")
    ap.add_argument("--min-amount-wan", type=float, default=MIN_AMOUNT_WAN,
                     help="当日成交额下限 (万元), 排除低流动性")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\n=== r1 T+1 EOD 推荐 ===\n", flush=True)
    print(f"模型: {MODEL_NAME}", flush=True)
    print(f"过滤: 距涨停 < {args.dist}%", flush=True)
    print(f"TopN: {args.top}\n", flush=True)

    print(f"加载 factors_v3 (~3.5 GB) ...", flush=True)
    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)

    if args.date is None:
        date = df["trade_date"].max()
        print(f"  自动选最新日: {date}", flush=True)
    else:
        date = args.date
        if date not in df["trade_date"].unique():
            print(f"!! 日期 {date} 不在 factors_v3 (max={df['trade_date'].max()})", flush=True)
            sys.exit(2)

    # EOD bar (hour=15)
    bar = df[(df["trade_date"] == date) & (df["trade_time"].dt.hour == 15)].copy()
    del df
    print(f"  {date} EOD bar: {len(bar):,} 股", flush=True)
    if len(bar) < 100:
        print(f"!! 数据太少, 检查 factors_v3 是否包含 {date} 的 15:00 K 线", flush=True)
        sys.exit(3)

    print(f"模型推理 ...", flush=True)
    booster, meta = load_model()
    feat_cols = meta["feature_cols"]
    for c in feat_cols:
        if c not in bar.columns: bar[c] = 0.0
        bar[c] = bar[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X = bar[feat_cols].astype("float32")
    bar["pred_r1"] = booster.predict(X)
    print(f"  推理 OK, pred 均值 {bar['pred_r1'].mean():+.3f}, std {bar['pred_r1'].std():.3f}",
           flush=True)

    # 锚定 0-100 评分
    p5 = meta.get("anchor_p5"); p50 = meta.get("anchor_p50"); p95 = meta.get("anchor_p95")
    if p5 is not None and p50 is not None and p95 is not None:
        bar["r1_score"] = map_anchored(bar["pred_r1"].values, p5, p50, p95)
    else:
        bar["r1_score"] = bar["pred_r1"].rank(pct=True) * 100

    # 过滤
    bar["is_near_upper"] = bar["dist_to_upper_limit_pct"].fillna(100) < args.dist
    bar["is_at_lower"] = bar["dist_to_lower_limit_pct"].fillna(100) < 1.0
    bar["bad_filter"] = bar["is_near_upper"] | bar["is_at_lower"]
    n_filtered = int(bar["bad_filter"].sum())
    print(f"  涨/跌停过滤: 排除 {n_filtered} 股", flush=True)

    # ST 过滤 + 流动性过滤 (merge stock_basic 拿 name)
    n_st = 0
    n_low_amt = 0
    if BASIC_P.exists():
        basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
        bar = bar.merge(basic, on="ts_code", how="left")
        if not args.keep_st:
            bar["is_st"] = bar["name"].fillna("").str.contains("ST", regex=False)
            n_st = int(bar["is_st"].sum())
            bar["bad_filter"] = bar["bad_filter"] | bar["is_st"]
            print(f"  ST 过滤: 排除 {n_st} 股", flush=True)

    # 流动性过滤: 从 daily cache 读当日 amount (Tushare amount 单位 = 千元)
    if args.min_amount_wan > 0:
        daily_p = ROOT / "output" / "tushare_cache" / "daily" / f"{date}.parquet"
        if daily_p.exists():
            dly = pd.read_parquet(daily_p)[["ts_code", "amount"]]
            dly["amount_wan"] = dly["amount"] / 10.0  # 千元 -> 万元
            bar = bar.merge(dly[["ts_code", "amount_wan"]], on="ts_code", how="left")
            bar["low_liq"] = bar["amount_wan"].fillna(0) < args.min_amount_wan
            n_low_amt = int(bar["low_liq"].sum())
            bar["bad_filter"] = bar["bad_filter"] | bar["low_liq"]
            print(f"  流动性过滤 (当日成交额 < {args.min_amount_wan:.0f} 万): "
                   f"排除 {n_low_amt} 股", flush=True)
        else:
            print(f"  (skip 流动性过滤: 找不到 daily cache {daily_p.name})", flush=True)

    valid = bar[~bar["bad_filter"]].dropna(subset=["pred_r1"]).copy()
    valid["pred_r1_capped"] = valid["pred_r1"].clip(-R1_CAP, R1_CAP)
    print(f"  有效候选: {len(valid):,}", flush=True)

    top = valid.nlargest(args.top, "pred_r1").reset_index(drop=True)

    # 补 industry (name 上面 ST 过滤时已 merge)
    if BASIC_P.exists() and "industry" not in top.columns:
        basic = pd.read_parquet(BASIC_P)[["ts_code", "industry"]].drop_duplicates("ts_code")
        top = top.merge(basic, on="ts_code", how="left")

    # JSON 输出
    out_json = OUT / f"{date}_top{args.top}.json"
    payload = {
        "trade_date": date,
        "model": MODEL_NAME,
        "top_n": args.top,
        "dist_threshold": args.dist,
        "r1_cap_for_eval": R1_CAP,
        "n_candidates": int(len(valid)),
        "n_filtered_limit_up_down": int(n_filtered),
        "n_filtered_st": int(n_st),
        "n_filtered_low_liquidity": int(n_low_amt),
        "exclude_st": not args.keep_st,
        "min_amount_wan": float(args.min_amount_wan),
        "pred_mean": float(bar["pred_r1"].mean()),
        "pred_std": float(bar["pred_r1"].std()),
        "recommendations": [
            {
                "rank": i + 1,
                "ts_code": r["ts_code"],
                "name": r.get("name", "") if pd.notna(r.get("name", "")) else "",
                "industry": r.get("industry", "") if pd.notna(r.get("industry", "")) else "",
                "pred_r1_raw": float(r["pred_r1"]),
                "pred_r1_capped_pct": float(r["pred_r1_capped"]),
                "r1_score": float(r["r1_score"]),
                "dist_to_upper_pct": float(r.get("dist_to_upper_limit_pct", 0))
                    if pd.notna(r.get("dist_to_upper_limit_pct", np.nan)) else None,
            }
            for i, r in top.iterrows()
        ],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出 JSON: {out_json}", flush=True)

    # Markdown 报告
    md = [f"# r1 T+1 EOD 推荐报告 ({date})\n\n",
            f"生成: {payload['generated_at']}\n",
            f"模型: {MODEL_NAME} (长 OOS IC 0.77)\n",
            f"候选池: {len(valid):,} 股 (排除涨跌停 {n_filtered})\n\n",
            f"## 入场计划 (次日 09:30 开盘买入, 次日 09:30 平仓 - 实际是 T+1 一日持有)\n\n",
            "| # | 代码 | 名称 | 行业 | r1_score | 预测 r1 % (cap 3%) | 距涨停 % |\n",
            "|---|---|---|---|---|---|---|\n"]
    for i, r in top.iterrows():
        nm = r.get("name", "") if pd.notna(r.get("name", "")) else ""
        ind = r.get("industry", "") if pd.notna(r.get("industry", "")) else ""
        dist = r.get("dist_to_upper_limit_pct", None)
        dist_s = f"{dist:.2f}" if pd.notna(dist) else "-"
        md.append(f"| {i+1} | {r['ts_code']} | {nm} | {ind} | "
                   f"{r['r1_score']:.1f} | {r['pred_r1_capped']:+.2f} | {dist_s} |\n")

    # 平仓清单 (次日 09:30 全部市价卖)
    md.append(f"\n## 次日 09:30 平仓清单 (无脑全卖)\n\n")
    md.append("| 代码 | 名称 |\n|---|---|\n")
    for _, r in top.iterrows():
        nm = r.get("name", "") if pd.notna(r.get("name", "")) else ""
        md.append(f"| {r['ts_code']} | {nm} |\n")

    md.append(f"\n## 关键警告\n\n")
    md.append(f"1. 长 OOS Top {args.top} 月化 ~+21% Sharpe 18.5 (理论值)\n")
    md.append(f"2. **实盘按 25% 折扣理解**: 预期月化 ~+5% Sharpe ~4.5\n")
    md.append(f"3. 涨停股已排除, 但近涨停 ({args.dist}-5%) 仍可能滑点\n")
    md.append(f"4. 流动性: Top {args.top} 中若有小盘股, 实际下单可买入比例可能更低\n")

    out_md = OUT / f"{date}_top{args.top}.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"输出 MD:   {out_md}", flush=True)

    # 终端打印 Top
    print(f"\n--- Top {args.top} ---")
    for i, r in top.iterrows():
        nm = r.get("name", "") if pd.notna(r.get("name", "")) else ""
        ind = r.get("industry", "") if pd.notna(r.get("industry", "")) else ""
        print(f"  {i+1:2d}. {r['ts_code']} {nm:<8s} {ind:<12s} "
              f"score={r['r1_score']:5.1f}  pred_r1={r['pred_r1_capped']:+.2f}%")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
