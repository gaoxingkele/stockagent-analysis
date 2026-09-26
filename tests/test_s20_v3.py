import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stockagent_analysis.s20_v3 import path_labels, daily_labels, probability_outputs, select_low_correlation


def test_six_classes_and_post_target_drop_does_not_undo_success():
    high = np.array([[121,105,100], [115,118,110], [101,121,110],
                     [110,111,105], [110,111,105], [118,119,110], [121,100,100]], float)
    low = np.array([[100,85,80], [95,91,99], [89,100,99],
                    [95,91,99], [89,95,99], [89,95,99], [89,95,99]], float)
    out = path_labels([100]*7, high, low)
    assert out.s20_class.tolist() == [0,1,2,3,4,5,-1]
    assert out.immediate.tolist() == [1,1,0,0,0,0,-1]
    assert out.down_risk.tolist() == [0,0,1,0,1,1,-1]


def test_touch_boundaries_and_target_day_below_entry_is_now_valid():
    out = path_labels([100]*3, [[120,105],[115,118],[120,105]], [[95,80],[90,90],[90,89]])
    assert out.s20_class.tolist() == [0,1,0]


def test_entry_and_maturity_no_off_by_one():
    d = pd.DataFrame(dict(ts_code=["X"]*4, trade_date=["1","2","3","4"],
                          open=[200,100,100,100],high=[210,110,120,105],
                          low=[190,95,100,80],close=[200,100,110,90]))
    out = daily_labels(d, horizon=3)
    assert out.entry_open.iloc[0] == 100
    assert out.entry_date.iloc[0] == "2"
    assert out.horizon_end_date.iloc[0] == "4"
    assert out.s20_class.iloc[0] == 0


def test_delay_penalizes_entry_score_and_outputs_are_coherent():
    out = probability_outputs(np.eye(6))
    assert out.score.tolist() == [100,100,0,50,0,0]
    np.testing.assert_allclose(out.p_opportunity + out.p_negative, 1)
    with pytest.raises(ValueError):
        probability_outputs(np.ones((2,6)))


def test_low_correlation_selection_removes_redundant_features():
    c = pd.DataFrame([[1,.95,.2],[.95,1,.3],[.2,.3,1]],index=list('abc'),columns=list('abc'))
    assert select_low_correlation(list('abc'),dict(a=3,b=2,c=1),c) == ['a','c']


def test_risk_weight_score_bounds_and_flat_negative_penalty():
    out = probability_outputs(np.eye(6),risk_weight=.5)
    np.testing.assert_allclose(out.score, [100,100,0,100/3,0,0])
    with pytest.raises(ValueError):
        probability_outputs(np.eye(6),risk_weight=-1)
