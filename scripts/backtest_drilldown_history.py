"""1H 钻取信号历史验证 (跨多日 + 跟踪后续 5 日实际表现).

流程:
  1. 取过去 N 个评测日 (有 V12 推荐 csv 的日子)
  2. 每日 V12 Top 30 做 1H v1 + v2 钻取
  3. 跟踪 5/10 日后实际收益
  4. 按 dual_consensus 分组统计 alpha

输出: output/drill_down/backtest_history.csv + .md
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.intraday_filter import drill_down_analyze
from stockagent_analysis.intraday_filter_v2 import drill_down_v2

DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"


def get_price_change(ts_code: str, start_date: str, days_after: int) -> Optional[float]:
    """从 daily cache 算 start_date 持有 days_after 个交易日的累计涨幅."""
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    dates = [f.stem for f in files]
    if start_date not in dates: return None
    idx = dates.index(start_date)
    if idx + days_after >= len(dates): return None
    end_date = dates[idx + days_after]
    p_start = pd.read_parquet(DAILY_CACHE / f"{start_date}.parquet")
    p_end = pd.read_parquet(DAILY_CACHE / f"{end_date}.parquet")
    s = p_start[p_start["ts_code"] == ts_code]
    e = p_end[p_end["ts_code"] == ts_code]
    if s.empty or e.empty: return None
    return float(e["close"].iloc[0]) / float(s["close"].iloc[0]) - 1


def main():
    snap_dates = ["20260508", "20260511", "20260512", "20260513"]
    end_cur = "20260515"
    all_rows = []

    for snap in snap_dates:
        v12_csv = ROOT / "output" / "v7c_full_inference" / f"v7c_inference_{snap}.csv"
        if not v12_csv.exists():
            print(f"  跳过 {snap} (no v12 csv)")
            continue
        v12 = pd.read_csv(v12_csv, dtype={"ts_code": str})
        if "v7c_recommend" in v12.columns:
            top = v12[v12["v7c_recommend"] == True].sort_values("r20_pred", ascending=False).head(30)
        elif "pool" in v12.columns:
            top = v12[v12["pool"].notna()].sort_values("r20_pred", ascending=False).head(30)
        else:
            top = v12.sort_values("r20_pred", ascending=False).head(30)

        if len(top) == 0: continue
        print(f"\n=== {snap} V12 Top {len(top)} 1H 钻取 ===", flush=True)

        # v12 r5 score (有则用, 无则 buy_score)
        score_col = "buy_r5_score" if "buy_r5_score" in top.columns else "buy_score"
        v12_scores = dict(zip(top["ts_code"], top[score_col])) if score_col in top.columns else {}

        t0 = time.time()
        # v1
        v1 = drill_down_analyze(top["ts_code"].tolist(), end_date=snap,
                                  v12_scores=v12_scores, lookback_days=10)
        # v2
        v2 = drill_down_v2(top["ts_code"].tolist(), end_date=snap, lookback_days=10)
        print(f"  v1+v2 耗时 {time.time()-t0:.0f}s", flush=True)

        # 实际 5 日 / 10 日表现 (从 daily cache)
        for _, r in top.iterrows():
            ts = r["ts_code"]
            ret_5d = get_price_change(ts, snap, 5)
            ret_max = get_price_change(ts, snap, 10) if snap < "20260508" else None
            v1_row = v1[v1["ts_code"] == ts]
            v2_row = v2[v2["ts_code"] == ts]
            row = {
                "snap_date": snap,
                "ts_code": ts,
                "v12_r20_pred": float(r.get("r20_pred", 0)),
                "v12_buy_r5": float(r.get(score_col, 0)) if score_col in r else None,
                "v1_intraday_score": float(v1_row["intraday_score"].iloc[0]) if len(v1_row) and "intraday_score" in v1_row.columns else None,
                "v1_consensus": str(v1_row["dual_consensus"].iloc[0]) if len(v1_row) and "dual_consensus" in v1_row.columns else None,
                "v2_score": float(v2_row["v2_score"].iloc[0]) if len(v2_row) and "v2_score" in v2_row.columns else None,
                "actual_ret_5d_pct": round(ret_5d * 100, 2) if ret_5d is not None else None,
            }
            all_rows.append(row)

    if not all_rows:
        print("无数据"); return
    out_df = pd.DataFrame(all_rows)
    out_dir = ROOT / "output" / "drill_down"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "history_drill_down.csv", index=False, encoding="utf-8-sig")

    # 统计
    valid = out_df.dropna(subset=["actual_ret_5d_pct"])
    print(f"\n=== 跨 {len(snap_dates)} 日历史验证 ({len(valid)} 个有效样本) ===")

    # 按 v1 consensus 分组
    print("\n[v1 dual_consensus → 5 日实际收益]")
    for cons, g in valid.groupby("v1_consensus"):
        if len(g) < 3: continue
        print(f"  {cons:<18} n={len(g):3d}, mean={g['actual_ret_5d_pct'].mean():+5.2f}%, "
              f"median={g['actual_ret_5d_pct'].median():+5.2f}%, "
              f"win={(g['actual_ret_5d_pct'] > 0).mean()*100:.0f}%")

    # 按 v1 score 分桶
    print("\n[v1 score 分桶 → 5 日实际收益]")
    for lo, hi in [(-1, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 1.01)]:
        g = valid[(valid["v1_intraday_score"] >= lo) & (valid["v1_intraday_score"] < hi)]
        if len(g) < 3: continue
        print(f"  [{lo:+.1f}, {hi:+.2f}) n={len(g):3d}, mean={g['actual_ret_5d_pct'].mean():+5.2f}%, win={(g['actual_ret_5d_pct']>0).mean()*100:.0f}%")

    # 按 v2 score 分桶
    print("\n[v2 score 分桶 → 5 日实际收益]")
    valid_v2 = valid.dropna(subset=["v2_score"])
    for lo, hi in [(-1, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 1.01)]:
        g = valid_v2[(valid_v2["v2_score"] >= lo) & (valid_v2["v2_score"] < hi)]
        if len(g) < 3: continue
        print(f"  [{lo:+.1f}, {hi:+.2f}) n={len(g):3d}, mean={g['actual_ret_5d_pct'].mean():+5.2f}%, win={(g['actual_ret_5d_pct']>0).mean()*100:.0f}%")

    print(f"\n输出: {out_dir}/history_drill_down.csv")


if __name__ == "__main__":
    from typing import Optional
    main()
