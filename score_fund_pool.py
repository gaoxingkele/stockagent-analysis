# -*- coding: utf-8 -*-
"""好基金优选池 V12.31 评分排序: top70高收益基金重仓股(≥MIN_FUNDS只持有) → V12.31 ratio排序。生产只读。"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.v12_scoring import V12Scorer

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260618"
MIN_FUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Q = "20260331"


def main():
    top = pd.read_parquet(ROOT / "research/cache/fund_portfolio_cache.parquet")
    top["end_date"] = top["end_date"].astype(str); top["symbol"] = top["symbol"].astype(str)
    L = top[top["end_date"] == Q]
    nfund = L["fund"].nunique()
    cnt = L.groupby("symbol").agg(n_funds=("fund", "nunique"), avg_w=("stk_mkv_ratio", "mean")).reset_index()
    pool = cnt[cnt["n_funds"] >= MIN_FUNDS].copy()
    codes = set(pool["symbol"])
    print(f"\n=== 好基金优选池 V12.31 评分 {DATE} ===", flush=True)
    print(f"  池: top{nfund}高收益基金 {Q}季报中被≥{MIN_FUNDS}只持有的股 = {len(codes)}只\n", flush=True)

    s = V12Scorer.get(ROOT)
    df = s.score_market(DATE, cb=None)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    for c in ["name", "industry"]:
        if c not in df.columns:
            df = df.merge(basic[["ts_code", c]], on="ts_code", how="left")
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
    sub = df[df["ts_code"].isin(codes)].merge(pool, left_on="ts_code", right_on="symbol", how="left")
    sub = sub.sort_values("ratio", ascending=False).reset_index(drop=True)

    print(f"  {'#':>3s} {'代码':11s}{'名称':9s}{'行业':9s}{'持有基金':>6s}{'r20':>5s}{'pump↑':>7s}{'pump↓':>7s}{'↑/↓':>7s}{'past5':>8s} 主推", flush=True)
    for i, (_, p) in enumerate(sub.iterrows(), 1):
        rec = "★" if p.get("v7c_recommend") else ""
        print(f"  {i:>3d} {p['ts_code']:11s}{str(p.get('name',''))[:8]:9s}{str(p.get('industry',''))[:8]:9s}"
              f"{int(p['n_funds']):>6d}{p.get('buy_r20_score',0):>5.0f}{p['pump_score']:>7.3f}{p['pump_down_score']:>7.3f}"
              f"{p['ratio']:>6.1f}x{p.get('past_r5',0)*100:>+7.1f}%  {rec}", flush=True)
    miss = codes - set(sub["ts_code"])
    nrec = int(sub["v7c_recommend"].sum()) if len(sub) else 0
    print(f"\n  入V7c主推★: {nrec}/{len(sub)} | 缺评分: {len(miss)}只", flush=True)
    cols = ["ts_code", "name", "industry", "n_funds", "buy_r20_score", "pump_score", "pump_down_score", "ratio", "past_r5", "v7c_recommend"]
    sub[[c for c in cols if c in sub.columns]].to_csv(ROOT / f"output/daily_pick/fund_pool_score_{DATE}.csv", index=False, encoding="utf-8-sig")
    print(f"  CSV: output/daily_pick/fund_pool_score_{DATE}.csv", flush=True)


if __name__ == "__main__":
    main()
