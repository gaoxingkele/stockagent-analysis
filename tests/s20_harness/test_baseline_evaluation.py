import pandas as pd
import pytest

from research.s20_harness.baseline_evaluation import evaluate, build
from research.s20_harness.baseline_run import build as baseline
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_run import plan


def fixture():
    p = plan()
    samples = pd.DataFrame(p["samples"])
    candidates = samples.iloc[-2:][["sample_id", "prediction_at", "signal_date"]].copy()
    candidates["selected"] = True
    candidates["score"] = [0.8, 0.9]
    outcomes = pd.DataFrame({"sample_id": candidates.sample_id, "target": [True, False],
                             "label_available_at": ["2024-05-29T00:00:00Z", "2024-06-02T00:00:00Z"]})
    return candidates, samples, outcomes


def test_unknown_bounds_not_known_only_win_rate():
    rows, report = evaluate(*fixture(), evaluation_at="2024-06-01T00:00:00Z", calendar=plan()["calendar"])
    bounds = report["selected_candidates"]["event_bounds"]
    assert bounds["denominator"] == 2 and bounds["unknown"] == 1
    assert bounds["rate_lower"] == .5 and bounds["rate_upper"] == 1
    assert bounds["known_only_rate"] == 1
    assert rows.evaluation_status.tolist() == ["mature", "not_mature_at_evaluation"]
    assert report["daily"][1]["selected"]["event_bounds"]["known_only_rate"] is None


def test_reliability_retains_unmatured_in_same_bucket_without_reading_future():
    c, s, o = fixture()
    c['score'] = [.81, .89]
    _, report = evaluate(c, s, o, evaluation_at="2024-06-01T00:00:00Z", calendar=plan()['calendar'])
    bucket = report['selected_candidates']['reliability_bins'][8]
    assert bucket['known_count'] == 1 and bucket['scored_count'] == 2
    assert bucket['mean_prediction'] == .81 and bucket['observed_rate'] == 1
    assert bucket['mean_prediction_all_scored'] == pytest.approx(.85)
    assert bucket['known_fraction'] == .5
    assert bucket['event_bounds']['rate_lower'] == .5
    assert bucket['event_bounds']['rate_upper'] == 1
    assert bucket['prediction_minus_rate_lower'] == pytest.approx(-.15)
    assert bucket['prediction_minus_rate_upper'] == pytest.approx(.35)
    o.loc[o.index[1], 'target'] = True
    _, changed = evaluate(c, s, o, evaluation_at="2024-06-01T00:00:00Z", calendar=plan()['calendar'])
    assert changed == report  # unseen outcome cannot change any diagnostic


@pytest.mark.parametrize('score,bucket_index', [(0., 0), (.1, 1), (.8, 8), (1., 9)])
def test_reliability_fixed_edges_and_missing_predictions(score, bucket_index):
    c, s, o = fixture()
    c['selected'] = [True, False]
    c['score'] = [score, float('nan')]
    _, report = evaluate(c, s, o, evaluation_at="2024-06-01T00:00:00Z", calendar=plan()['calendar'])
    all_rows = report['all_candidates']
    assert all_rows['scored_rows'] == all_rows['missing_prediction_rows'] == 1
    assert sum(b['scored_count'] for b in all_rows['reliability_bins']) == 1
    assert all_rows['reliability_bins'][bucket_index]['scored_count'] == 1
    assert all_rows['event_bounds']['denominator'] == 2
    assert not all_rows['reliability_bounds_are_confidence_intervals']


def test_reliability_empty_bins_are_null_not_perfect():
    c, s, o = fixture()
    c['selected'] = False
    _, report = evaluate(c, s, o.iloc[:0], evaluation_at="2024-06-01T00:00:00Z", calendar=plan()['calendar'])
    for b in report['selected_candidates']['reliability_bins']:
        assert b['scored_count'] == b['known_count'] == 0
        assert b['known_fraction'] is b['mean_prediction_all_scored'] is None
        assert b['prediction_minus_rate_lower'] is b['prediction_minus_rate_upper'] is None
    unknown = report['all_candidates']['reliability_bins'][8]
    assert unknown['known_count'] == 0 and unknown['scored_count'] == 1
    assert unknown['event_bounds']['rate_lower'] == 0
    assert unknown['event_bounds']['rate_upper'] == 1


def test_empty_selection_and_missing_outcomes_retained():
    c, s, o = fixture()
    c["selected"] = False
    rows, report = evaluate(c, s, o.iloc[:0], evaluation_at="2024-06-01T00:00:00Z", calendar=plan()["calendar"])
    assert len(rows) == 2 and report["all_candidates"]["event_bounds"]["unknown"] == 2
    assert report["selected_candidates"]["brier_known_only"] is None


@pytest.mark.parametrize("kind", ["early", "numeric", "extra", "duplicate", "evaluation"])
def test_invalid(kind):
    c, s, o = fixture()
    cutoff = "2024-06-01T00:00:00Z"
    if kind == "early":
        o.loc[o.index[0], "label_available_at"] = "2024-05-10T00:00:00Z"
    elif kind == "numeric":
        o["target"] = [1, 0]
    elif kind == "extra":
        o.loc[o.index[0], "sample_id"] = "fit0"
    elif kind == "duplicate":
        o["sample_id"] = "outer-test0"
    else:
        cutoff = "2024-05-01T00:00:00Z"
    with pytest.raises(ValueError):
        evaluate(c, s, o, evaluation_at=cutoff, calendar=plan()["calendar"])


def test_persisted_run_evaluation(tmp_path):
    source = tmp_path/"input.json"
    atomic_json(source, plan())
    run = baseline(tmp_path, source, digest(source))
    out = __import__("pathlib").Path(run["directory"])
    outcome = tmp_path/"outcomes.json"
    atomic_json(outcome, {"target_id": "P.safe.v4", "evaluation_at": "2024-06-01T00:00:00Z",
                          "outcomes": fixture()[2].to_dict("records")})
    report = build(tmp_path, out, digest(out/"summary.json"), outcome, digest(outcome))
    assert report["rows"] == 2 and report["selected"] == 1
    assert report["evidence_mode"] == "synthetic" and not report["formal_promotion_authorized"]
