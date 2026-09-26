import pandas as pd
import pytest

from research.s20_harness.policy_search import compare
from tests.s20_harness.test_baseline_run import plan


def fixture():
    p = plan()
    samples = pd.DataFrame(p["samples"])
    candidates = samples.iloc[6:8][["sample_id", "entity_id", "prediction_at", "signal_date"]].copy()
    candidates["score"] = [0.7, 0.9]
    candidates["risk"] = [0.05, 0.8]
    policies = []
    for name, mode in [("control", "score_only_control"), ("gated", "risk_gated")]:
        policy = dict(p["policy"], policy_id=name, frozen_at="2024-03-01T00:00:00Z", mode=mode)
        if mode == "risk_gated":
            policy.update(risk_target_id="P.B5.v4", max_risk=.1)
        policies.append(policy)
    registry = dict(registry_id="search1", target_id="P.safe.v4", registered_at="2024-03-02T00:00:00Z",
                    policies=policies, min_active_coverage=.5, min_selected=1, min_mature_selected=1)
    outcomes = pd.DataFrame({"sample_id": candidates.sample_id, "target": [True, False],
                             "label_available_at": samples.iloc[6:8].label_available_at})
    return samples, candidates, outcomes, p["boundaries"], ["20240403", "20240404"], registry


def test_all_trials_kept_and_risk_gate_can_win():
    ledgers, report = compare(*fixture())
    assert report["selected_policy_id"] == "gated"
    assert report["policy_evaluations"] == 2 and all(len(v) == 2 for v in ledgers.values())
    assert not report["formal_H05_accepted"] and not report["outer_outcomes_consumed"]


def test_unknown_selection_remains_denominator():
    args = list(fixture())
    args[0].loc[7, "label_available_at"] = None
    args[2] = args[2].iloc[:1]
    _, report = compare(*args)
    control = report["trials"][0]["selected_event_bounds"]
    assert control["denominator"] == 1 and control["unknown"] == 1
    assert report["selected_policy_id"] == "gated"


def test_count_floor_can_reject_every_policy():
    args = fixture()
    args[-1]["min_mature_selected"] = 10
    _, report = compare(*args)
    assert report["selected_policy_id"] is None and report["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("kind", ["outer", "late_registry", "budget", "partial", "maturity", "calendar", "target"])
def test_registry_firewall(kind):
    args = list(fixture())
    if kind == "outer":
        args[2] = pd.concat([args[2], pd.DataFrame({"sample_id": ["outer-test0"], "target": [True],
                                                   "label_available_at": ["2024-05-28T21:00:00Z"]})])
    elif kind == "late_registry":
        args[-1]["registered_at"] = "2024-04-01T00:00:00Z"
    elif kind == "budget":
        args[-1]["policies"] *= 4
    elif kind == "partial":
        args[1] = args[1].iloc[:1]
    elif kind == "maturity":
        args[2].loc[6, "label_available_at"] = "2024-04-29T00:00:00Z"
    elif kind == "calendar":
        args[4] = ["20240503"]
    else:
        args[-1]["policies"][0]["target_id"] = "O.other"
    with pytest.raises(ValueError):
        compare(*args)
