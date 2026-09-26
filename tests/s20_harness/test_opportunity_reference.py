import pytest
from research.s20_harness.opportunity_reference import classify
from research.s20_harness.opportunity_reference_probe import evaluate


@pytest.mark.parametrize('highs,lows,expected',[
    ([120,110],[95,85],0),([110,120],[85,95],2),([120],[85],-1),
    ([115],[90],1),([115],[89],5),([110],[90],3),([110],[89],4)])
def test_reference_rules(highs,lows,expected):
    assert classify(100,highs,lows)['s20_class']==expected


def test_finite_permutations_match_vectorized_implementation():
    contract,result=evaluate()
    assert contract['horizon']==20 and result['cases']==12546
    assert contract['first_touch_day_grid']==list(range(21))
    assert result['all_equal'] and not result['mismatches']
    assert set(result['class_counts'])=={-1,0,1,2,3,4,5}


def test_probe_detects_wrong_vectorized_class(monkeypatch):
    from research.s20_harness import opportunity_reference_probe as probe
    original=probe.path_labels
    def altered(*args):
        result=original(*args)
        result.loc[0,'s20_class']=99
        return result
    monkeypatch.setattr(probe,'path_labels',altered)
    _,result=probe.evaluate()
    assert not result['all_equal'] and len(result['mismatches'])==1
    assert result['mismatches'][0]['case_index']==0
