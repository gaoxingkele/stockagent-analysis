# -*- coding: utf-8 -*-
"""一次性: V12.31 给自定义清单打分, 按 ratio 降序。生产线只读。"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.v12_scoring import V12Scorer

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260618"
CODES = ["600160.SH","600549.SH","688257.SH","002171.SZ","603588.SH","600909.SH","300475.SZ",
"300566.SZ","688525.SH","600176.SH","000100.SZ","688498.SH","600206.SH","688766.SH","301099.SZ",
"300031.SZ","600522.SH","001696.SZ","600458.SH","601208.SH","300726.SZ","688233.SH","301536.SZ",
"688549.SH","300976.SZ","603225.SH","600498.SH","600459.SH","601211.SH","002842.SZ","600392.SH",
"300304.SZ","605589.SH","300666.SZ","300014.SZ","603271.SH","600378.SH","688809.SH","688378.SH",
"688002.SH","603045.SH","000725.SZ","002297.SZ","688388.SH","002378.SZ","920357.BJ","300975.SZ",
"920394.BJ","688146.SH"]


def main():
    s = V12Scorer.get(ROOT)
    print(f"\n=== V12.31 自定义清单评分 {DATE} ({len(CODES)}股) ===\n", flush=True)
    df = s.score_market(DATE, cb=None)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    for c in ["name", "industry"]:
        if c not in df.columns:
            df = df.merge(basic[["ts_code", c]], on="ts_code", how="left")
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
    sub = df[df["ts_code"].isin(CODES)].sort_values("ratio", ascending=False).reset_index(drop=True)

    print(f"  {'#':>3s} {'代码':11s}{'名称':9s}{'行业':9s}{'r20':>5s}{'pump↑':>7s}{'pump↓':>7s}{'↑/↓':>7s}{'past5':>8s} 主推", flush=True)
    for i, (_, p) in enumerate(sub.iterrows(), 1):
        rec = "★" if p.get("v7c_recommend") else ""
        print(f"  {i:>3d} {p['ts_code']:11s}{str(p.get('name',''))[:8]:9s}{str(p.get('industry',''))[:8]:9s}"
              f"{p.get('buy_r20_score',0):>5.0f}{p['pump_score']:>7.3f}{p['pump_down_score']:>7.3f}"
              f"{p['ratio']:>6.1f}x{p.get('past_r5',0)*100:>+7.1f}%  {rec}", flush=True)

    miss = [c for c in CODES if c not in set(sub["ts_code"])]
    if miss:
        print(f"\n  (缺数据/未评分 {len(miss)}: {miss})", flush=True)
    nrec = int(sub["v7c_recommend"].sum()) if len(sub) else 0
    print(f"\n  入V7c主推★: {nrec}/{len(sub)}", flush=True)
    out = ROOT / "output/daily_pick" / f"watchlist_{DATE}.csv"
    cols = ["ts_code","name","industry","buy_r20_score","pump_score","pump_down_score","ratio","past_r5","v7c_recommend"]
    sub[[c for c in cols if c in sub.columns]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  CSV: {out}", flush=True)


if __name__ == "__main__":
    main()
