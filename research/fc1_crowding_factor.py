# -*- coding: utf-8 -*-
"""FC-1 — 基金抱团度正交因子 Phase-1 (PIT ann_date 对齐, 扣动量残差, NW-t).

宽基500基金持仓 → PIT 抱团度: 每调仓日 d, 每基金取 ann_date<=d 的最近季度持仓(零前视),
crowding_level(s,d)=持有s的基金数, crowding_weight=Σstk_mkv_ratio, crowding_delta=环比变化。
扣 FULL_CTRL[动量+size+value+roe/margin]残差 rank-IC vs 前向r20(复用oa_screen, NW-t)。

廉价闸(R16): |orth IC|>=0.02 & |t|>=3 → A进FC-2; 否则C/D廉价REJECT(抱团≈动量镜像)。
诚实flag: 季度低频+stale; 基金后进已涨票→大概率动量镜像(如概念lead-lag). 宽基避幸存者(R04 PIT).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import oa_screen
HOLD = ROOT / "research/cache/fc_holdings_broad.parquet"
CTRL = ROOT / "research/features/fundamental_factors.parquet"
OUTF = ROOT / "research/features/fc_crowding_factors.parquet"
KEY = ["ts_code", "trade_date"]


def build_pit_crowding(hold, rebal_dates):
    """每调仓日 PIT 抱团度: 每基金取 ann_date<=d 的最近季度, 统计每股被多少基金持有。"""
    hold = hold.copy()
    hold["ann_date"] = hold["ann_date"].astype(str)
    hold["end_date"] = hold["end_date"].astype(str)
    hold["symbol"] = hold["symbol"].astype(str)
    rows = []
    for d in rebal_dates:
        h = hold[hold["ann_date"] <= d]
        if h.empty:
            continue
        # 每基金取 ann_date 最大的那个季度 (最近已公告)
        idx = h.groupby("fund")["ann_date"].transform("max")
        cur = h[h["ann_date"] == idx]
        g = cur.groupby("symbol").agg(cl=("fund", "nunique"),
                                      cw=("stk_mkv_ratio", "sum")).reset_index()
        g["trade_date"] = d
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True).rename(columns={"symbol": "ts_code"})
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # crowding_delta: 环比(上一个调仓日)变化
    panel["cl_delta"] = panel.groupby("ts_code")["cl"].diff()
    return panel.rename(columns={"cl": "crowding_level", "cw": "crowding_weight",
                                 "cl_delta": "crowding_delta"})


def main():
    print("\n=== FC-1 基金抱团度正交因子 Phase-1 (PIT) ===\n", flush=True)
    hold = pd.read_parquet(HOLD)
    print(f"[hold] 宽基持仓 {len(hold):,}行 / {hold['fund'].nunique()}基金 / "
          f"{hold['symbol'].nunique()}股 / 季度{sorted(hold['end_date'].astype(str).unique())}", flush=True)
    rebal = sorted(pd.read_parquet(CTRL, columns=["trade_date"])["trade_date"].astype(str).unique())
    # 只用持仓数据覆盖到的调仓日 (首个 ann_date 之后)
    first_ann = hold["ann_date"].astype(str).min()
    rebal = [d for d in rebal if d >= first_ann]
    print(f"[pit] {len(rebal)}个调仓日 PIT 抱团度 (首公告 {first_ann}) ...", flush=True)
    panel = build_pit_crowding(hold, rebal)
    # 全市场未持有股 crowding=0: 不显式补0(oa_screen inner join控制面板, 未持有=NaN→该日横截面只在持有股内排序)
    # 为让"持有vs未持有"也进入排序, 补0到控制面板全集
    cov = panel["trade_date"].nunique()
    OUTF.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTF, index=False)
    print(f"[factor] {len(panel):,}行 / {cov}调仓日, 落 {OUTF.name}", flush=True)

    factor_cols = ["crowding_level", "crowding_weight", "crowding_delta"]
    oa_screen.assert_backward(factor_cols)
    # screen: oa_screen inner join 控制面板, 未持有股不在panel→自动只在持有股横截面算IC
    verdict = oa_screen.screen_factors(panel, factor_cols, "FC-1", primary="crowding_level", write=False)
    cl = verdict["classifications"]
    survivors = verdict["survivors"]
    print(f"\n=== FC-1 汇总 (PIT, 持有股横截面) ===", flush=True)
    for f, c in cl.items():
        print(f"  {f:18s} {c['status']:10s} orth_ic={c['orth_ic']:+.4f} t={c['orth_t']} raw={c['raw_ic']:+.4f} "
              f"n月={c['n_months']}", flush=True)
    status = "PROCEED_FC2" if survivors else "REJECT_cheap"
    concl = (f"PIT基金抱团度({verdict['n_rebalance_months']}调仓日, 宽基500避幸存者): "
             f"存活(A/B)={survivors if survivors else '无'}; "
             f"crowding_level orth IC={cl['crowding_level']['orth_ic']:+.4f} t={cl['crowding_level']['orth_t']}, "
             f"crowding_delta orth IC={cl['crowding_delta']['orth_ic']:+.4f} t={cl['crowding_delta']['orth_t']}. → {status}. "
             + ("存活→FC-2池walk-forward。" if survivors
                else "抱团度扣动量残差不显著→廉价REJECT: 基金抱团≈动量镜像(后进已涨票), 季度低频stale无独立横截面alpha. FC-2/3仍按用户要求做(不同用途)。"))
    out = {"id": "FC-1", "status": status, "survivors": survivors,
           "n_rebalance_months": verdict["n_rebalance_months"],
           "classifications": cl, "conclusion": concl,
           "honest_flags": ["季度低频+stale(ann_date间无更新)", "抱团多与动量同期(基金后进已涨票)=可能动量镜像",
                            "宽基500随机避按收益选的幸存者偏差", "PIT ann_date对齐零前视(R04)",
                            "持有股横截面(未持有不入), 测'抱团多vs少'是否预测"],
           "guardrails": ["R01 gate冻结", "R03 IC仅go/no-go", "R04 PIT ann_date零前视",
                          "R12++ 扣FULL_CTRL残差", "R11 分regime", "R16 廉价闸", "R05 生产冻结"]}
    (ROOT / "research/verdicts/FC-1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[决策] {status}; {concl}", flush=True)


if __name__ == "__main__":
    main()
