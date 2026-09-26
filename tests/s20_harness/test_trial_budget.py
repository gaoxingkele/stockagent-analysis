from concurrent.futures import ThreadPoolExecutor

import pytest

from research.s20_harness.trial_budget import Budget, run_baseline
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_run import plan


def contract(sha="a"*64):
    costs = dict(model_fits=1, underlying_fits=2, calibrator_fits=1, policy_evaluations=1)
    return dict(budget_id="diagnostic", limits=costs,
                trials=[dict(trial_id="baseline", input_sha256=sha, costs=costs, max_attempts=2)])


def test_failure_consumes_budget_and_duplicate_does_not_launch(tmp_path):
    b = Budget(tmp_path/"budget.sqlite", contract())
    assert b.reserve("baseline", "one", "a"*64)["newly_reserved"]
    b.finish("baseline", "one", "FAILED", {"error": "synthetic"})
    assert not b.reserve("baseline", "one", "a"*64)["newly_reserved"]
    with pytest.raises(ValueError, match="exhausted"):
        b.reserve("baseline", "two", "a"*64)
    assert b.status()["reserved_counts"]["model_fits"] == 1


def test_transaction_prevents_concurrent_overspend(tmp_path):
    b = Budget(tmp_path/"budget.sqlite", contract())
    def reserve(name):
        try:
            return b.reserve("baseline", name, "a"*64)["newly_reserved"]
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reserve, ["one", "two"])) == 1


def test_contract_and_input_cannot_change(tmp_path):
    path = tmp_path/"budget.sqlite"
    b = Budget(path, contract())
    with pytest.raises(ValueError, match="unregistered"):
        b.reserve("baseline", "one", "b"*64)
    changed = contract()
    changed["limits"] = dict(changed["limits"], model_fits=2)
    with pytest.raises(ValueError, match="contract changed"):
        Budget(path, changed)


def test_budgeted_pipeline_and_idempotent_attempt(tmp_path):
    source = tmp_path/"input.json"
    atomic_json(source, plan())
    sha = digest(source)
    b = Budget(tmp_path/"budget.sqlite", contract(sha))
    result = run_baseline(tmp_path, source, sha, b, "baseline", "one")
    assert result["executed"] and b.status()["attempts"][0]["state"] == "SUCCEEDED_DIAGNOSTIC"
    assert not run_baseline(tmp_path, source, sha, b, "baseline", "one")["executed"]


def test_pipeline_failure_stays_charged(tmp_path):
    source = tmp_path/"input.json"
    payload = plan()
    payload["calibration_labels"][1]["target"] = False
    atomic_json(source, payload)
    sha = digest(source)
    b = Budget(tmp_path/"budget.sqlite", contract(sha))
    with pytest.raises(ValueError):
        run_baseline(tmp_path, source, sha, b, "baseline", "one")
    assert b.status()["attempts"][0]["state"] == "FAILED"
    assert b.status()["reserved_counts"]["calibrator_fits"] == 1


@pytest.mark.parametrize('fail',[False,True])
def test_frequency_pipeline_actual_costs_and_failed_charge(tmp_path,fail):
    from research.s20_harness.baseline_model import pipeline_costs
    value=plan();value['model_family']='mature_frequency'
    if fail:value['calibration_labels'][1]['target']=False
    source=tmp_path/'frequency.json';atomic_json(source,value);sha=digest(source)
    spec=contract(sha);costs=pipeline_costs(value)
    spec['limits']=costs;spec['trials'][0]['costs']=costs
    budget=Budget(tmp_path/'budget.sqlite',spec)
    if fail:
        with pytest.raises(ValueError):run_baseline(tmp_path,source,sha,budget,'baseline','one')
    else:
        result=run_baseline(tmp_path,source,sha,budget,'baseline','one')
        assert result['run']['model_level_fits']==0
    status=budget.status()
    assert status['reserved_counts']==dict(model_fits=0,underlying_fits=1,calibrator_fits=1,policy_evaluations=1)
    assert status['attempts'][0]['state']==('FAILED' if fail else 'SUCCEEDED_DIAGNOSTIC')
    assert not run_baseline(tmp_path,source,sha,budget,'baseline','one')['executed']


def test_frequency_overcharge_rejected_without_reservation(tmp_path):
    source=tmp_path/'frequency.json';atomic_json(source,dict(plan(),model_family='mature_frequency'))
    sha=digest(source);budget=Budget(tmp_path/'budget.sqlite',contract(sha))
    with pytest.raises(ValueError,match='costs mismatch'):run_baseline(tmp_path,source,sha,budget,'baseline','one')
    assert budget.status()['attempts']==[]
