"""多策略 backtest 对比 (Sprint 2.2).

跑 60 日 OOS 期 (20260214 → 20260413, 与 V15 OOS 一致):
  A. V7c 5 铁律 baseline (无 regime / 无池子 / 无 Kelly)
  B. V12.12 完整版 (6 池 + Kelly + regime + zombie + 行业分散)

输出 output/backtest/compare_{end_date}.json + nav 曲线 CSV.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.backtest_engine import V12Backtester

OUT_DIR = ROOT / "output" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(name: str, start: str, end: str, **kwargs) -> dict:
    print(f"\n=== {name} 开始 ===", flush=True)
    bt = V12Backtester(start_date=start, end_date=end, **kwargs)

    def cb(i, n, d, nav, dt):
        if i % 10 == 0:
            print(f"  [{name}] [{i}/{n}] {d}: nav={nav:.4f}", flush=True)

    res = bt.run(progress_cb=cb)
    # 保存 nav csv
    nav_csv = OUT_DIR / f"nav_{name}_{start}_{end}.csv"
    res["nav_df"].to_csv(nav_csv, index=False)
    trade_csv = OUT_DIR / f"trades_{name}_{start}_{end}.csv"
    res["trade_df"].to_csv(trade_csv, index=False)
    return {k: v for k, v in res.items() if k not in ("nav_df", "trade_df")}


def main():
    start = "20260214"; end = "20260413"
    t0 = time.time()

    summary = {}
    summary["A_baseline"] = run_one("baseline", start, end,
                                      use_regime_sizing=False,
                                      use_industry_diversification=False,
                                      use_zombie_filter=False)
    summary["B_v12_12_full"] = run_one("v12_12", start, end,
                                       use_regime_sizing=True,
                                       use_industry_diversification=True,
                                       use_zombie_filter=True)
    # 保存
    out = OUT_DIR / f"compare_{start}_{end}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 对比完成, 总耗时 {time.time()-t0:.0f}s ===", flush=True)
    print(f"输出: {out}", flush=True)

    print(f"\n{'策略':<25} {'final_nav':>10} {'total_ret%':>12} {'max_dd%':>10} {'trades':>8} {'win%':>6}")
    print("-"*80)
    for name, s in summary.items():
        print(f"{name:<25} {s['final_nav']:>10.4f} {s['total_return_pct']:>11.2f}% {s['max_drawdown_pct']:>9.2f}% {s['n_trades']:>8} {s['win_rate_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
