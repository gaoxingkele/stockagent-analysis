import pandas as pd
import pytest

from research.s20_harness.baseline_model import run
from tests.s20_harness.test_feature_pipeline import fixture


def inputs():
    samples, features, boundaries, contract = fixture()
    labels = pd.DataFrame({"sample_id": ["fit1", "fit0"], "target": [True, False]})
    return samples, features, labels, boundaries, contract


@pytest.mark.parametrize('seed', [True, False, -1, 2**32, 1.5, '20', None])
def test_invalid_seed_rejected_before_training(seed):
    from research.s20_harness.baseline_model import pipeline_costs
    with pytest.raises(ValueError, match='random_seed'):
        run(*inputs(), target_id='P.safe_profit.v4', random_seed=seed)
    with pytest.raises(ValueError, match='random_seed'):
        pipeline_costs({'random_seed': seed})


@pytest.mark.parametrize('family', ['logistic', 'shallow_tree', 'mature_frequency'])
def test_seed_recorded_and_same_seed_reproducible(family):
    first, card = run(*inputs(), target_id='P.safe_profit.v4', model_family=family, random_seed=71)
    second, other = run(*inputs(), target_id='P.safe_profit.v4', model_family=family, random_seed=71)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert card == other
    assert card['randomness']['requested_seed'] == 71
    assert card['randomness']['independent_market_evidence'] is False
    if family != 'mature_frequency':
        assert card['parameters']['random_state'] == 71
    if family in {'logistic', 'mature_frequency'}:
        changed, _ = run(*inputs(), target_id='P.safe_profit.v4', model_family=family, random_seed=72)
        pd.testing.assert_frame_equal(first, changed, check_exact=True)


def test_real_fit_forward_only_and_unknown_outer_prediction():
    args = inputs()
    args[0].loc[8:, "label_available_at"] = None
    predictions, card = run(*args, target_id="P.safe_profit.v4")
    assert predictions.raw_probability.iloc[:2].isna().all()
    assert predictions.raw_probability.iloc[2:].between(0, 1).all()
    assert card["predicted_rows"] == 8 and card["model_level_fits"] == 1
    assert card["coefficients"][0][0] > 0
    assert not card["calibration_performed"] and not card["formal_H03_accepted"]


def test_future_features_do_not_change_fitted_model():
    args = inputs()
    _, first = run(*args, target_id="O.safe20.v1")
    args[1].loc[2:, "x"] = -9999.
    _, second = run(*args, target_id="O.safe20.v1")
    assert first["coefficients"] == second["coefficients"]
    assert first["intercept"] == second["intercept"]
    assert first["preprocessing"]["median"] == second["preprocessing"]["median"]


def test_missing_feature_keeps_candidate_without_prediction():
    args = inputs()
    args[0].loc[8, "feature_available_at"] = None
    predictions, card = run(*args, target_id="P.safe_profit.v4")
    assert len(predictions) == 10 and card["predicted_rows"] == 7
    assert predictions.loc[8, "prediction_status"] == "feature_unavailable"
    assert pd.isna(predictions.loc[8, "raw_probability"])


def test_shallow_tree_uses_same_firewall_and_forward_denominator():
    args=inputs()
    predictions,first=run(*args,target_id='P.safe_profit.v4',model_family='shallow_tree')
    assert len(predictions)==len(args[0]) and predictions.raw_probability.iloc[:2].isna().all()
    assert predictions.raw_probability.iloc[2:].between(0,1).all()
    assert first['parameters']==dict(max_depth=3,min_samples_leaf=5,random_state=20)
    args[1].loc[2:,'x']=-9999.
    _,second=run(*args,target_id='P.safe_profit.v4',model_family='shallow_tree')
    assert first['tree_state']==second['tree_state']
    assert not second['formal_H03_accepted']
    leaked=pd.concat([args[2],pd.DataFrame(dict(sample_id=['outer-test0'],target=[True]))])
    with pytest.raises(ValueError,match='eligible fit labels'):
        run(args[0],args[1],leaked,args[3],args[4],target_id='P.safe_profit.v4',model_family='shallow_tree')


@pytest.mark.parametrize("kind", ["outer", "purged", "unknown", "single_class", "numeric", "duplicate"])
def test_label_firewall(kind):
    args = list(inputs())
    if kind == "outer":
        args[2] = pd.concat([args[2], pd.DataFrame({"sample_id": ["outer-test0"], "target": [True]})])
    elif kind == "purged":
        args[0].loc[1, "label_available_at"] = args[3][1]["start_at"]
    elif kind == "unknown":
        args[2]["target"] = None
    elif kind == "single_class":
        args[2]["target"] = True
    elif kind == "numeric":
        args[2]["target"] = [1, 0]
    else:
        args[2].loc[0, "sample_id"] = "fit0"
    with pytest.raises(ValueError):
        run(*args, target_id="P.safe_profit.v4")
