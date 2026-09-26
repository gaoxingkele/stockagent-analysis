"""Retrain diagnostic joints on user-paired +15%/-10% and +25%/-15% labels."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.diagnostic_train import (
    FEATURE_COLUMNS, _pivot, evaluate_ledger, load_quotes, plan_for, evaluate_saved,
)
from research.s20_harness.execution import market_horizon
from research.s20_harness.joint_run import build
from research.s20_harness.policy_replay import select_pool
from research.s20_harness.runtime import atomic_json, digest, load_plan
from research.s20_harness.take_profit import TARGET_RISK_PAIRS, first_take_profit, paired_class, pool_take_profit

SOURCE = ROOT / "output/experiments/s20_safe_v4/sources/v4-rsi-campaign"
OUT = ROOT / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"


def _windows(samples, calendar, opens, highs, lows, closes, stocks):
    index = {s: i for i, s in enumerate(stocks)}
    cal_i = {d: i for i, d in enumerate(calendar)}
    rows = []
    for row in samples.itertuples(index=False):
        i = cal_i[str(row.signal_date)]
        j = index[row.entity_id]
        sl = slice(i + 1, i + 21)
        rows.append((row.sample_id, float(opens[i + 1, j]), highs[sl, j], lows[sl, j], closes[sl, j]))
    return rows


def relabel(windows, take_profit, drawdown):
    records = []
    for sid, entry, high, low, close in windows:
        if not np.isfinite(entry) or entry <= 0:
            records.append(dict(sample_id=sid, target=None, exit="unresolved"))
            continue
        payload = first_take_profit(entry, high, low, close, take_profit=take_profit, b5=drawdown)
        records.append(dict(sample_id=sid, target=paired_class(payload),
                            exit=payload["exit"], profit=payload["profit"],
                            path_risk=payload["path_risk"], net=payload["net"]))
    return pd.DataFrame(records)


def top20_paired(pred, samples, labels):
    part = pred[pred.segment == "outer-test"].copy()
    frame = samples.set_index("sample_id").loc[part.sample_id, ["entity_id", "signal_date"]].reset_index()
    for cls in "ABCD":
        frame[f"p_{cls}"] = part[f"cal_p_{cls}"].to_numpy()
    picked = select_pool(frame, n_cap=20, ranking="penalized_utility", max_risk=1.0)
    selected = picked.loc[picked.selected]
    metrics = evaluate_ledger(selected.assign(selected=True), labels)
    joined = selected.merge(labels, on="sample_id", how="left")
    metrics["paired_profit"] = float(joined.target.isin(["A", "B"]).mean()) if len(joined) else None
    metrics["paired_drawdown"] = float(joined.target.isin(["B", "D"]).mean()) if len(joined) else None
    metrics["class_counts"] = joined.target.fillna("UNKNOWN").value_counts().to_dict()
    return metrics


def main():
    samples = pd.read_parquet(SOURCE / "samples.parquet")
    features = pd.read_parquet(SOURCE / "features.parquet")
    old_labels = pd.read_parquet(SOURCE / "labels.parquet")
    calendar, daily = load_quotes(ROOT / "output/tushare_cache/daily")
    stocks = sorted(samples.entity_id.unique())
    opens = _pivot(daily, calendar, stocks, "open")
    highs = _pivot(daily, calendar, stocks, "high")
    lows = _pivot(daily, calendar, stocks, "low")
    closes = _pivot(daily, calendar, stocks, "close")
    windows = _windows(samples, calendar, opens, highs, lows, closes, stocks)
    OUT.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(OUT / "samples.parquet", index=False)
    features.to_parquet(OUT / "features.parquet", index=False)
    old_summary = load_plan(SOURCE / "campaign_summary.json")
    old_dir = Path(next(m for m in old_summary["models"] if m["family"] == "multinomial_base")["run"]["directory"])
    old_pred = pd.read_parquet(old_dir / "calibrated_predictions.parquet")
    results = []
    for tp, dd in TARGET_RISK_PAIRS:
        labels = relabel(windows, tp, dd)
        name = f"tp{int(tp*100)}_dd{int(dd*100)}"
        labels.to_parquet(OUT / f"labels_{name}.parquet", index=False)
        realized = labels.dropna(subset=["target"])[["sample_id", "target"]].copy()
        realized["target"] = realized["target"].astype(str)
        keep = set(realized.sample_id)
        samples_p = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
        features_p = features[features.sample_id.isin(keep)].reset_index(drop=True)
        mix = realized.target.value_counts().to_dict()
        old_on_new = top20_paired(old_pred, samples_p, realized)
        pair_runs = []
        for family in ("mature_frequency", "multinomial"):
            plan = plan_for(samples_p, features_p, realized, calendar, family, columns=FEATURE_COLUMNS)
            path = OUT / f"plan_{name}_{family}.json"
            atomic_json(path, plan)
            report = build(ROOT, path, digest(path))
            pred = pd.read_parquet(Path(report["directory"]) / "calibrated_predictions.parquet")
            pair_runs.append(dict(family=family, run=report, top20=top20_paired(pred, samples, realized)))
        results.append(dict(pair={"take_profit": tp, "drawdown": dd}, class_counts=mix,
                            n_realized=int(len(realized)), n_unlabeled=int(labels.target.isna().sum()),
                            old_b5_model_on_new_labels=old_on_new, models=pair_runs))
    summary = dict(
        status="COMPLETED_DIAGNOSTIC",
        formal_training_authorized=False, formal_H04_accepted=False,
        production_eligible=False,
        note="retrained on user-paired TP/drawdown classes; B5 four-class is not the training target",
        pairs=results,
    )
    atomic_json(OUT / "campaign_summary.json", summary)
    print(json.dumps({
        "formal_training_authorized": False,
        "pairs": [
            dict(pair=p["pair"], class_counts=p["class_counts"],
                 old_top20=p["old_b5_model_on_new_labels"],
                 models=[{k: m[k] for k in ("family", "top20")} for m in p["models"]])
            for p in results
        ],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
