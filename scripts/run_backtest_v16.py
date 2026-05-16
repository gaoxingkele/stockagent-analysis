"""V12.16 (含 9 池) 60 日 OOS backtest - 与之前 V12.12 对比."""
from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.backtest_engine import V12Backtester

OUT_DIR = ROOT / "output" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

start = "20260214"; end = "20260413"
print(f"=== V12.16 (9 池) backtest ===", flush=True)
bt = V12Backtester(start_date=start, end_date=end,
                    use_regime_sizing=True,
                    use_industry_diversification=True,
                    use_zombie_filter=True)

def cb(i, n, d, nav, dt):
    if i % 10 == 0:
        print(f"  [v12_16] [{i}/{n}] {d}: nav={nav:.4f}", flush=True)

t0 = time.time()
res = bt.run(progress_cb=cb)
nav_csv = OUT_DIR / f"nav_v12_16_{start}_{end}.csv"
res["nav_df"].to_csv(nav_csv, index=False)
trade_csv = OUT_DIR / f"trades_v12_16_{start}_{end}.csv"
res["trade_df"].to_csv(trade_csv, index=False)
print(f"\n=== V12.16 完成, 耗时 {time.time()-t0:.0f}s ===")
print(f"final_nav: {res['final_nav']}, total_ret: {res['total_return_pct']:+.2f}%, max_dd: {res['max_drawdown_pct']:+.2f}%")
print(f"n_trades: {res['n_trades']}, n_sells: {res['n_sells']}, win_rate: {res['win_rate_pct']:.1f}%")
