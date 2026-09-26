import pytest

from research.s20_harness.bounded_baseline import run
from research.s20_harness.process_runner import Limits
from research.s20_harness.trial_budget import Budget
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_run import plan
from tests.s20_harness.test_trial_budget import contract


@pytest.mark.parametrize('family',['logistic','shallow_tree','mature_frequency'])
def test_owned_child_and_repeated_attempt(tmp_path,family,monkeypatch):
    source = tmp_path/"input.json"
    value=plan();value['model_family']=family
    for i in range(8):
        key='extra_fit'+str(i)
        value['samples'].append(dict(value['samples'][i%2],sample_id=key))
        value['features'].append(dict(sample_id=key,x=1. if i%2==0 else 3.))
        value['fit_labels'].append(dict(sample_id=key,target=bool(i%2)))
    atomic_json(source, value)
    pin = digest(source)
    from research.s20_harness.baseline_model import pipeline_costs
    costs=pipeline_costs(value);spec=contract(pin)
    spec['limits']=costs;spec['trials'][0]['costs']=costs
    budget = Budget(tmp_path/"budget.sqlite", spec)
    result = run(tmp_path, source, pin, budget, "baseline", "one", Limits(120, 2*1024**3, 1))
    assert result["process"]["exit_code"] == 0 and result["process"]["peak_rss_bytes"] > 0
    from research.s20_harness.trial_budget import verify_attempt
    before=digest(budget.path)
    checked=verify_attempt(tmp_path,budget.path,budget.sha,'baseline','one',source,pin,result['artifact'],Limits(120,2*1024**3,1))
    assert checked['recorded_prelaunch_reservation_verified']
    assert checked['charged_costs']==costs and digest(budget.path)==before
    with pytest.raises(ValueError,match='contract pin'):
        verify_attempt(tmp_path,budget.path,'0'*64,'baseline','one',source,pin,result['artifact'],Limits(120,2*1024**3,1))
    with pytest.raises(ValueError,match='attempt missing'):
        verify_attempt(tmp_path,budget.path,budget.sha,'baseline','absent',source,pin,result['artifact'],Limits(120,2*1024**3,1))
    reused=run(tmp_path, source, pin, budget, "baseline", "one", Limits(120, 2*1024**3, 1))
    assert not reused['executed'] and reused['reusable']
    assert reused['budget']['reserved_counts']==costs
    with pytest.raises(ValueError,match='limits differ'):
        run(tmp_path, source, pin, budget, 'baseline','one',Limits(119,2*1024**3,1))
    from pathlib import Path
    from research.s20_harness import bounded_baseline
    original_digest=bounded_baseline.digest
    with monkeypatch.context() as patch:
        patch.setattr(bounded_baseline,'digest',lambda p:'f'*64 if Path(p).name=='baseline_model.py' else original_digest(p))
        with pytest.raises(ValueError,match='computational sources changed'):
            run(tmp_path,source,pin,budget,'baseline','one',Limits(120,2*1024**3,1))
    atomic_json(Path(result['artifact']['directory'])/'baseline_card.json',{})
    with pytest.raises(ValueError,match='artifact pin'):
        run(tmp_path, source, pin, budget, 'baseline','one',Limits(120,2*1024**3,1))
    assert budget.status()['reserved_counts']==costs


def test_timeout_keeps_charge_and_process_record(tmp_path):
    source = tmp_path/"input.json"
    atomic_json(source, plan())
    pin = digest(source)
    budget = Budget(tmp_path/"budget.sqlite", contract(pin))
    with pytest.raises(RuntimeError, match="wall_limit"):
        run(tmp_path, source, pin, budget, "baseline", "one", Limits(.001, 2*1024**3, 1, .001))
    assert budget.status()["attempts"][0]["state"] == "FAILED"
    assert budget.status()["reserved_counts"]["model_fits"] == 1


@pytest.mark.parametrize("limits", [(float('inf'), 1024, 1), (float('nan'), 1024, 1), (10, True, 1), (10, 1024, True)])
def test_nonfinite_or_boolean_limits_rejected(limits):
    with pytest.raises(ValueError):
        Limits(*limits)
