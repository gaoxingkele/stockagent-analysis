"""Dream-RSI policy replay for 20-day hold Top20/Top50. Frozen model, no H04 claim."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.policy_replay import replay, evaluate_policy, pool_metrics, HOLD_PROFIT, HOLD_DRAWDOWN
from research.s20_harness.runtime import atomic_json, load_plan

CAMPAIGN = ROOT / "output/experiments/s20_safe_v4/sources/v4-rsi-campaign"


def _frame(pred, samples, segment):
    part = pred[pred.segment == segment].copy()
    meta = samples.set_index("sample_id").loc[part.sample_id, ["entity_id", "signal_date"]].reset_index()
    out = meta.copy()
    for cls in "ABCD":
        out[f"p_{cls}"] = part[f"cal_p_{cls}"].to_numpy()
    return out


def universe_hold(labels, sample_ids):
    sub = labels[labels.sample_id.isin(sample_ids)]
    return pool_metrics(sub.target, n_cap=1, n_dates=1)


def main():
    summary = load_plan(CAMPAIGN / "campaign_summary.json")
    base = next(m for m in summary["models"] if m["family"] == "multinomial_base")
    directory = Path(base["run"]["directory"])
    pred = pd.read_parquet(directory / "calibrated_predictions.parquet")
    samples = pd.read_parquet(CAMPAIGN / "samples.parquet")
    labels = pd.read_parquet(CAMPAIGN / "labels.parquet")
    dream = _frame(pred, samples, "selection-policy")
    online = _frame(pred, samples, "outer-test")
    report = replay(dream, labels, online, labels)
    report["label_audit"] = dict(
        meaning={
            "A": "20d hold net profit and never -5%",
            "B": "20d hold net profit but path went -5% (still profitable if held)",
            "C": "20d hold not profitable, never -5% (not a win)",
            "D": "20d hold not profitable and path went -5%",
            "hold_profit": "A or B — money made after buying the recommendation and holding 20 sessions",
            "hold_drawdown": "B or D — recommended name saw >5% adverse excursion while held",
        },
        full_sample=labels.target.value_counts().to_dict(),
        dream_universe=universe_hold(labels, dream.sample_id),
        online_universe=universe_hold(labels, online.sample_id),
        rejected_old_meanings=["O-track first-touch +20% is not 20d hold profit",
                               "class C is not a successful recommendation",
                               "class B is profitable if held; path risk is separate"],
    )
    out = CAMPAIGN / "dream_rsi_top50.json"
    atomic_json(out, report)
    print(json.dumps({
        "champion_policy": report["champion_policy"],
        "shipped_equals_pi0": report["shipped_equals_pi0"],
        "online_transfer_ok": report["online_transfer_ok"],
        "recommended_policy": report["recommended_policy"],
        "n_dream_improvers": report["n_dream_improvers"],
        "dream_pi0": {k: report["dream_pi0"][k] for k in
                      ("n_cap", "hold_profit_rate", "hold_drawdown_rate", "safe_profit_rate", "n_selected", "class_counts")
                      if k in report["dream_pi0"] or True},
        "online_pi0": {k: report["online_pi0"].get(k) for k in
                       ("n_cap", "hold_profit_rate", "hold_drawdown_rate", "safe_profit_rate", "n_selected", "coverage", "class_counts")},
        "online_champion": {k: report["online_champion"].get(k) for k in
                            ("n_cap", "hold_profit_rate", "hold_drawdown_rate", "safe_profit_rate", "n_selected", "coverage", "class_counts")},
        "label_audit_full": report["label_audit"]["full_sample"],
        "online_universe_hold_profit": report["label_audit"]["online_universe"]["hold_profit_rate"],
        "online_universe_hold_drawdown": report["label_audit"]["online_universe"]["hold_drawdown_rate"],
        "formal_training_authorized": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
