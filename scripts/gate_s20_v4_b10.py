"""Independent P(B10) gate on frozen ranking scores. Not formal H04."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.independent_risk import (
    attach_gate, calibration_snapshot, fit_p_b10, replay_b10_gate, rolling_calibrate_p_b10,
)
from research.s20_harness.policy_replay import evaluate_policy
from research.s20_harness.runtime import atomic_json, load_plan

PAIRED = ROOT / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
RANK_SRC = ROOT / "output/experiments/s20_safe_v4/sources/v4-rsi-campaign"


def _rank_frame(pred, samples):
    part = pred.merge(samples[["sample_id", "entity_id", "signal_date"]], on="sample_id")
    out = part[["sample_id", "entity_id", "signal_date", "segment"]].copy()
    for cls in "ABCD":
        out[f"p_{cls}"] = part[f"cal_p_{cls}"].to_numpy()
    return out


def _view(score):
    keys = ("n_selected", "hold_profit_rate", "hold_drawdown_rate", "coverage", "class_counts")
    return {k: score.get(k) for k in keys}


def main():
    samples = pd.read_parquet(PAIRED / "samples.parquet")
    features = pd.read_parquet(PAIRED / "features.parquet")
    labels = pd.read_parquet(PAIRED / "labels_tp15_dd10.parquet")
    labels = labels.dropna(subset=["target"])[["sample_id", "target"]]
    labels["target"] = labels["target"].astype(str)
    keep = set(labels.sample_id)
    samples = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
    features = features[features.sample_id.isin(keep)].reset_index(drop=True)
    risk_pred, card = fit_p_b10(samples, features, labels)
    cal = rolling_calibrate_p_b10(samples, risk_pred, labels)
    old = load_plan(RANK_SRC / "campaign_summary.json")
    old_dir = Path(next(m for m in old["models"] if m["family"] == "multinomial_base")["run"]["directory"])
    rank_pred = pd.read_parquet(old_dir / "calibrated_predictions.parquet")
    rank = _rank_frame(rank_pred, samples)
    frame = attach_gate(rank, cal, risk_col="p_b10_cal")
    dream = frame[frame.segment == "selection-policy"].copy()
    online = frame[frame.segment == "outer-test"].copy()
    report = replay_b10_gate(dream, labels, online, labels)
    report["risk_model"] = dict(target_id=card["target_id"], model=card["model"],
                                fit_positive_count=card["fit_positive_count"],
                                predicted_rows=card["predicted_rows"],
                                ranking_source="frozen multinomial_base scores",
                                risk_source="rolling-calibrated independent logistic P(B or D) on +15/-10 labels")
    report["online_universe_b10"] = float(labels.merge(online[["sample_id"]], on="sample_id").target.isin(["B", "D"]).mean())
    report["calibration_snapshot"] = calibration_snapshot(frame, labels)
    sec = pd.read_parquet(PAIRED / "labels_tp25_dd15.parquet")
    sec = sec.dropna(subset=["target"])[["sample_id", "target"]]
    sec["target"] = sec["target"].astype(str)
    rec_policy = report["recommended_policy"]
    report["secondary_pair"] = dict(
        labels="+25%/-15%",
        used_to_pick=False,
        dream_pi0=_view(evaluate_policy(dream, sec, report["pi0"])),
        dream_recommended=_view(evaluate_policy(dream, sec, rec_policy)),
        online_pi0=_view(evaluate_policy(online, sec, report["pi0"])),
        online_recommended=_view(evaluate_policy(online, sec, rec_policy)),
    )
    cal[["sample_id", "segment", "p_b10", "p_b10_cal", "calibrator_n"]].to_parquet(
        PAIRED / "b10_calibrated.parquet", index=False)
    out = PAIRED / "b10_gate.json"
    atomic_json(out, report)
    summary = {
        "recommended_policy": report["recommended_policy"],
        "champion_policy": report["champion_policy"],
        "online_transfer_ok": report["online_transfer_ok"],
        "shipped_equals_pi0": report["shipped_equals_pi0"],
        "n_dream_improvers": report["n_dream_improvers"],
        "replay_mass_taus": report["replay_mass_taus"],
        "risk_col": rec_policy.get("risk_col"),
        "dream_pi0": _view(report["dream_pi0"]),
        "dream_champion": _view(report["dream_champion"]),
        "online_pi0": _view(report["online_pi0"]),
        "online_recommended": _view(report["online_recommended"]),
        "secondary_pair": report["secondary_pair"],
        "formal_training_authorized": report["formal_training_authorized"],
        "production_eligible": report["production_eligible"],
        "pair": "+15%/-10%",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
