"""Score frozen Top20/Top50 as 20-session holds that may exit at a take-profit."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.diagnostic_train import load_quotes, _pivot
from research.s20_harness.execution import market_horizon
from research.s20_harness.policy_replay import select_pool
from research.s20_harness.runtime import atomic_json, load_plan
from research.s20_harness.take_profit import (
    DRAWDOWN_FLOORS, PRIMARY_DRAWDOWN, PRIMARY_TP, TARGET_RISK_PAIRS,
    breached_drawdown, first_take_profit, pool_take_profit, window_mae,
)

CAMPAIGN = ROOT / "output/experiments/s20_safe_v4/sources/v4-rsi-campaign"


def _online_frame(pred, samples):
    part = pred[pred.segment == "outer-test"].copy()
    meta = samples.set_index("sample_id").loc[part.sample_id, ["entity_id", "signal_date"]].reset_index()
    out = meta.copy()
    for cls in "ABCD":
        out[f"p_{cls}"] = part[f"cal_p_{cls}"].to_numpy()
    return out


def _windows(frame, calendar, opens, highs, lows, closes, stocks):
    index = {s: i for i, s in enumerate(stocks)}
    cal_i = {d: i for i, d in enumerate(calendar)}
    out = []
    for row in frame.itertuples(index=False):
        i = cal_i[str(row.signal_date)]
        j = index[row.entity_id]
        horizon = market_horizon(calendar, str(row.signal_date), horizon=20)
        sl = slice(i + 1, i + 21)
        out.append(dict(
            sample_id=row.sample_id,
            entry=float(opens[i + 1, j]),
            high=highs[sl, j].copy(),
            low=lows[sl, j].copy(),
            close=closes[sl, j].copy(),
            horizon_end=horizon["horizon_end"],
        ))
    return out


def _score(windows, take_profit, drawdown):
    results = []
    for item in windows:
        if not np.isfinite(item["entry"]) or item["entry"] <= 0:
            results.append(dict(sample_id=item["sample_id"], exit="unresolved",
                                profit=None, path_risk=None, net=None, exit_day=None,
                                take_profit=take_profit, keep_ex_ante=True, mae=None))
            continue
        payload = first_take_profit(item["entry"], item["high"], item["low"], item["close"],
                                    take_profit=take_profit, b5=drawdown)
        payload["sample_id"] = item["sample_id"]
        results.append(payload)
    return results


def main():
    summary = load_plan(CAMPAIGN / "campaign_summary.json")
    base = next(m for m in summary["models"] if m["family"] == "multinomial_base")
    pred = pd.read_parquet(Path(base["run"]["directory"]) / "calibrated_predictions.parquet")
    samples = pd.read_parquet(CAMPAIGN / "samples.parquet")
    labels = pd.read_parquet(CAMPAIGN / "labels.parquet")
    online = _online_frame(pred, samples)
    calendar, daily = load_quotes(ROOT / "output/tushare_cache/daily")
    stocks = sorted(online.entity_id.unique())
    opens = _pivot(daily, calendar, stocks, "open")
    highs = _pivot(daily, calendar, stocks, "high")
    lows = _pivot(daily, calendar, stocks, "low")
    closes = _pivot(daily, calendar, stocks, "close")
    policies = {
        "universe": online,
        "top20": select_pool(online, n_cap=20, ranking="penalized_utility", max_risk=1.0),
        "top50": select_pool(online, n_cap=50, ranking="penalized_utility", lambda_=1.0, mu=3.0, nu=0.25, max_risk=0.6),
    }
    policies["top20"] = policies["top20"].loc[policies["top20"].selected]
    policies["top50"] = policies["top50"].loc[policies["top50"].selected]
    report = dict(
        pairs=[{"take_profit": tp, "drawdown": dd} for tp, dd in TARGET_RISK_PAIRS],
        primary_take_profit=PRIMARY_TP, primary_drawdown=PRIMARY_DRAWDOWN,
        note="+15% profit expectation pairs with -10% path risk; +25% pairs with -15%",
        rule="T+1 then first touch of paired target or D+20 close; same-day dual touch unresolved",
        formal_training_authorized=False,
    )
    for name, frame in policies.items():
        windows = _windows(frame, calendar, opens, highs, lows, closes, stocks)
        joined = frame.merge(labels, on="sample_id", how="left")
        hold_profit = float(joined.target.isin(["A", "B"]).mean())
        hold_risk = float(joined.target.isin(["B", "D"]).mean())
        maes = [window_mae(w["entry"], w["low"]) for w in windows]
        floors = {str(fl): float(np.mean([breached_drawdown(m, fl) for m in maes]))
                  for fl in DRAWDOWN_FLOORS}
        block = dict(n=len(frame), hold20_profit_rate=hold_profit,
                     hold20_drawdown_rate_b5=hold_risk,
                     hold20_mae_breach=floors, by_pair={})
        for tp, dd in TARGET_RISK_PAIRS:
            scored = _score(windows, tp, dd)
            stats = pool_take_profit(scored)
            resolved = [r for r in scored if r.get("mae") is not None and r["mae"] == r["mae"]]
            stats["paired_drawdown"] = dd
            stats["paired_drawdown_rate"] = (
                float(np.mean([breached_drawdown(r["mae"], dd) for r in resolved])) if resolved else None
            )
            block["by_pair"][f"tp{int(tp*100)}_dd{int(dd*100)}"] = stats
        report[name] = block
    out = CAMPAIGN / "take_profit_eval.json"
    atomic_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
