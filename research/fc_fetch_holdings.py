# -*- coding: utf-8 -*-
"""FC-0 — 宽基基金宇宙(随机500业绩无关) + 历史持仓拉取 (PIT, 含 ann_date).

避幸存者偏差: 从 2744 只成立<2023 股票/混合型 随机抽 500 (业绩无关), 拉 fund_portfolio
20231001-20260430 (top10持仓, 含 ann_date 公告日供 PIT 对齐)。checkpoint 续跑。
输出 research/cache/fc_holdings_broad.parquet。
"""
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
UNI = ROOT / "research/cache/fc_universe_raw.parquet"
OUT = ROOT / "research/cache/fc_holdings_broad.parquet"
VERDICT = ROOT / "research/verdicts/FC-0.json"
N_SAMPLE = 500
SEED = 20260619


def _pro():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
    import tushare as ts; ts.set_token(os.environ["TUSHARE_TOKEN"]); return ts.pro_api()


def main():
    pro = _pro()
    uni = pd.read_parquet(UNI)
    rng = np.random.default_rng(SEED)
    samp = uni.iloc[rng.permutation(len(uni))[:N_SAMPLE]].copy()
    codes = samp["ts_code"].tolist()
    print(f"[uni] 随机抽 {len(codes)} 只 (业绩无关) 拉历史持仓 ...", flush=True)

    done_codes, parts = set(), []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done_codes = set(prev["fund"].unique()); parts = [prev]
        print(f"[ckpt] 已拉 {len(done_codes)} 只", flush=True)
    t0, ok, fail = time.time(), 0, 0
    batch = []
    for i, c in enumerate(codes):
        if c in done_codes:
            continue
        try:
            d = pro.fund_portfolio(ts_code=c, start_date="20231001", end_date="20260430")
            time.sleep(0.22)
        except Exception as e:
            fail += 1
            if "每分钟" in str(e) or "limit" in str(e).lower():
                time.sleep(15)
            continue
        if d is not None and len(d):
            d = d.copy(); d["fund"] = c
            batch.append(d); ok += 1
        if (i + 1) % 50 == 0:
            # 周期性落盘 checkpoint
            cur = pd.concat(parts + batch, ignore_index=True).drop_duplicates(["fund", "end_date", "symbol"])
            cur.to_parquet(OUT, index=False)
            print(f"  {i+1}/{len(codes)} (ok新{ok} fail{fail}, {time.time()-t0:.0f}s)", flush=True)
    allp = pd.concat(parts + batch, ignore_index=True).drop_duplicates(["fund", "end_date", "symbol"])
    allp.to_parquet(OUT, index=False)
    nq = sorted(allp["end_date"].astype(str).unique())
    nfund = allp["fund"].nunique()
    verdict = {"id": "FC-0", "status": "built", "n_funds": int(nfund), "n_rows": int(len(allp)),
               "quarters": nq, "n_stocks_covered": int(allp["symbol"].nunique()),
               "conclusion": f"宽基随机{nfund}只(业绩无关)历史持仓: {len(allp):,}行, {len(nq)}季度({nq[0]}~{nq[-1]}), "
                             f"覆盖{allp['symbol'].nunique()}股. ann_date供PIT对齐.",
               "guardrails": ["业绩无关随机抽样避幸存者", "含ann_date供PIT", "R05生产冻结"]}
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {nfund}只 {len(allp):,}行 {len(nq)}季度 {nq} -> {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
