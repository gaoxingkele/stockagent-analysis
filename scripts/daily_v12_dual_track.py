"""P3 V12 双轨架构日报 (实盘整合).

整合 v12_scoring.score_market (含 R5 反向过滤) → v12_dual_track 双轨持仓.

用法:
  python scripts/daily_v12_dual_track.py                # 自动选 factor_lab 最新日
  python scripts/daily_v12_dual_track.py 20260515       # 指定日期

输出:
  output/daily_dual_track/<date>.json   原始 + 持仓清单
  output/daily_dual_track/<date>.md     人类可读
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output" / "daily_dual_track"
OUT.mkdir(parents=True, exist_ok=True)


def cb_print(phase: str, percent: int, message: str, data):
    print(f"  [{percent:3d}%] {phase:20s} {message}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", help="YYYYMMDD, 默认取 factor_lab 最新可用")
    ap.add_argument("--track-a-pct", type=float, default=0.70)
    ap.add_argument("--track-b-pct", type=float, default=0.20)
    ap.add_argument("--cash-pct", type=float, default=0.10)
    ap.add_argument("--a-bottom-pct", type=float, default=1.0,
                     help="A 轨集中过滤 R5 Bot 比例 (v3 默认 1.0 不过滤, 用 --max-a 控制数量)")
    ap.add_argument("--b-bottom-pct", type=float, default=1.0,
                     help="B 轨分散过滤 R5 Bot 上限 (v3 默认 1.0 不过滤)")
    ap.add_argument("--max-a", type=int, default=8)
    ap.add_argument("--max-b", type=int, default=15)
    args = ap.parse_args()

    from stockagent_analysis.v12_scoring import V12Scorer
    from stockagent_analysis.v12_dual_track import build_dual_track, render_dual_track_md

    t0 = time.time()
    print(f"\n=== V12 双轨日报 ===\n", flush=True)
    scorer = V12Scorer.get(ROOT)
    if args.date is None:
        dates = scorer.list_available_dates()
        if not dates:
            print("!! factor_lab 无可用日期, 先跑 update_factor_lab_from_tushare.py", flush=True)
            sys.exit(2)
        date = dates[-1]
        print(f"  自动选最新日: {date}", flush=True)
    else:
        date = args.date

    print(f"\n[1] V12 全市场评分 ...", flush=True)
    df = scorer.score_market(date, cb=cb_print)
    n_v7c = int(df["v7c_recommend"].sum())
    n_r5filt = int(df["v7c_recommend_r5filtered"].sum())
    print(f"  V7c 主推 {n_v7c}, R5 反向过滤后 {n_r5filt}", flush=True)

    print(f"\n[2] 构建双轨持仓 ...", flush=True)
    res = build_dual_track(
        df,
        track_a_pct=args.track_a_pct,
        track_b_pct=args.track_b_pct,
        cash_pct=args.cash_pct,
        a_bottom_pct=args.a_bottom_pct,
        b_bottom_pct=args.b_bottom_pct,
        max_a_stocks=args.max_a,
        max_b_stocks=args.max_b,
    )
    s = res["summary"]
    print(f"  轨 A (集中 Bot {s['a_bottom_pct']*100:.0f}%): {s['n_a']} 股, 单仓 {s['per_stock_a_pct']*100:.2f}%",
           flush=True)
    print(f"  轨 B (分散 Bot {s['a_bottom_pct']*100:.0f}-{s['b_bottom_pct']*100:.0f}%): "
           f"{s['n_b']} 股, 单仓 {s['per_stock_b_pct']*100:.2f}%", flush=True)
    print(f"  现金: {s['alloc']['cash']*100:.0f}%", flush=True)

    # 输出
    md = render_dual_track_md(res, date)
    out_md = OUT / f"{date}.md"
    out_md.write_text(md, encoding="utf-8")

    payload = {
        "trade_date": date,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": s,
        "track_a": [{
            "ts_code": r["ts_code"], "name": r.get("name", ""),
            "industry": r.get("industry", ""),
            "buy_score": float(r["buy_score"]),
            "r5_long_rank": float(r["r5_long_rank_in_pool"]),
            "alloc_pct": float(r["alloc_pct"]),
        } for _, r in res["track_a"].iterrows()],
        "track_b": [{
            "ts_code": r["ts_code"], "name": r.get("name", ""),
            "industry": r.get("industry", ""),
            "buy_score": float(r["buy_score"]),
            "r5_long_rank": float(r["r5_long_rank_in_pool"]),
            "alloc_pct": float(r["alloc_pct"]),
        } for _, r in res["track_b"].iterrows()],
    }
    out_json = OUT / f"{date}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n输出: {out_md}", flush=True)
    print(f"     {out_json}", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
