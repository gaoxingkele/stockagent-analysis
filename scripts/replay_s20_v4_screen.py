"""Causal silence screen + specified +15% pool replay. Not formal H04."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.diagnostic_train import load_quotes, _pivot
from research.s20_harness.independent_risk import fit_p_b10, rolling_calibrate_p_b10
from research.s20_harness.runtime import atomic_json, load_plan
from research.s20_harness.silence import (
    fit_p_hit15, fit_p_silent, hit15_labels, matured_silence_cooldown,
    replay_screen, rolling_calibrate_binary, silence_labels,
)
from research.s20_harness.take_profit import (
    PRIMARY_DRAWDOWN, PRIMARY_TP, first_take_profit, is_silent, specified_rise,
)

PAIRED = ROOT / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
RANK_SRC = ROOT / "output/experiments/s20_safe_v4/sources/v4-rsi-campaign"


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


def _screen_labels(windows, four_class):
    records = []
    for sid, entry, high, low, close in windows:
        if not np.isfinite(entry) or entry <= 0:
            records.append(dict(sample_id=sid, silent=None, hit15=None, max_gain=None))
            continue
        payload = first_take_profit(entry, high, low, close, take_profit=PRIMARY_TP, b5=PRIMARY_DRAWDOWN)
        records.append(dict(
            sample_id=sid,
            silent=is_silent(payload["max_gain"], payload["path_risk"]),
            hit15=specified_rise(payload),
            max_gain=payload["max_gain"],
            exit=payload["exit"],
        ))
    extra = pd.DataFrame(records)
    out = four_class.merge(extra, on="sample_id", how="left")
    return out


def _rank_frame(pred, samples):
    part = pred.merge(samples[["sample_id", "entity_id", "signal_date"]], on="sample_id")
    out = part[["sample_id", "entity_id", "signal_date", "segment"]].copy()
    for cls in "ABCD":
        out[f"p_{cls}"] = part[f"cal_p_{cls}"].to_numpy()
    return out


def _view(score):
    keys = ("n_selected", "specified_rise_rate", "hold_profit_rate", "hold_drawdown_rate",
            "coverage_vs_pi0", "coverage", "class_counts")
    return {k: score.get(k) for k in keys}


def main():
    samples = pd.read_parquet(PAIRED / "samples.parquet")
    features = pd.read_parquet(PAIRED / "features.parquet")
    four = pd.read_parquet(PAIRED / "labels_tp15_dd10.parquet")
    calendar, daily = load_quotes(ROOT / "output/tushare_cache/daily")
    stocks = sorted(samples.entity_id.unique())
    opens = _pivot(daily, calendar, stocks, "open")
    highs = _pivot(daily, calendar, stocks, "high")
    lows = _pivot(daily, calendar, stocks, "low")
    closes = _pivot(daily, calendar, stocks, "close")
    labels = _screen_labels(_windows(samples, calendar, opens, highs, lows, closes, stocks), four)
    labels.to_parquet(PAIRED / "labels_tp15_dd10_screen.parquet", index=False)
    keep = labels.dropna(subset=["target", "silent", "hit15"]).copy()
    keep["target"] = keep["target"].astype(str)
    ids = set(keep.sample_id)
    samples = samples[samples.sample_id.isin(ids)].reset_index(drop=True)
    features = features[features.sample_id.isin(ids)].reset_index(drop=True)
    keep = keep[keep.sample_id.isin(ids)].reset_index(drop=True)
    silent_tbl = keep[["sample_id", "silent"]].merge(
        samples[["sample_id", "label_available_at"]], on="sample_id")
    cooldown = matured_silence_cooldown(samples, silent_tbl)
    b10_pred, b10_card = fit_p_b10(samples, features, keep[["sample_id", "target"]])
    b10_cal = rolling_calibrate_p_b10(samples, b10_pred, keep[["sample_id", "target"]])
    silent_pred, silent_card = fit_p_silent(samples, features, keep)
    silent_cal = rolling_calibrate_binary(
        samples, silent_pred, silence_labels(keep), raw_col="p_silent", cal_col="p_silent_cal")
    hit_pred, hit_card = fit_p_hit15(samples, features, keep)
    hit_cal = rolling_calibrate_binary(
        samples, hit_pred, hit15_labels(keep), raw_col="p_hit15", cal_col="p_hit15_cal")
    old = load_plan(RANK_SRC / "campaign_summary.json")
    old_dir = Path(next(m for m in old["models"] if m["family"] == "multinomial_base")["run"]["directory"])
    rank_pred = pd.read_parquet(old_dir / "calibrated_predictions.parquet")
    frame = _rank_frame(rank_pred, samples)
    frame = frame.merge(b10_cal[["sample_id", "p_b10", "p_b10_cal"]], on="sample_id", how="left")
    frame = frame.merge(silent_cal[["sample_id", "p_silent", "p_silent_cal"]], on="sample_id", how="left")
    frame = frame.merge(hit_cal[["sample_id", "p_hit15", "p_hit15_cal"]], on="sample_id", how="left")
    frame = frame.merge(cooldown, on="sample_id", how="left")
    frame["silent_cooldown"] = frame["silent_cooldown"].fillna(False).astype(bool)
    dream = frame[frame.segment == "selection-policy"].copy()
    online = frame[frame.segment == "outer-test"].copy()
    eval_labels = keep[["sample_id", "target", "hit15", "silent"]]
    report = replay_screen(dream, eval_labels, online, eval_labels)
    report["heads"] = dict(
        ranking_source="frozen multinomial_base scores",
        b10=dict(target_id=b10_card["target_id"], predicted_rows=b10_card["predicted_rows"]),
        silent=dict(target_id=silent_card["target_id"], predicted_rows=silent_card["predicted_rows"]),
        hit15=dict(target_id=hit_card["target_id"], predicted_rows=hit_card["predicted_rows"]),
        silence_is_not_data_cleaning=True,
        current_window_unused_at_prediction=True,
    )
    rec = report["recommended_policy"]
    summary = {
        "recommended_policy": rec,
        "champion_policy": report["champion_policy"],
        "online_transfer_ok": report["online_transfer_ok"],
        "shipped_equals_pi0": report["shipped_equals_pi0"],
        "n_dream_improvers": report["n_dream_improvers"],
        "replay_mass_taus": report["replay_mass_taus"],
        "dream_pi0": _view(report["dream_pi0"]),
        "dream_champion": _view(report["dream_champion"]),
        "online_pi0": _view(report["online_pi0"]),
        "online_recommended": _view(report["online_recommended"]),
        "specified_rise": report["specified_rise"],
        "silence": report["silence"],
        "formal_training_authorized": False,
        "production_eligible": False,
        "pair": "+15%/-10%",
    }
    atomic_json(PAIRED / "screen_pred.json", report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
