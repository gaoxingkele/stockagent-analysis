import json
from pathlib import Path
import pandas as pd
import pytest

from research.s20_harness.baseline_model import run,pipeline_costs
from research.s20_harness.baseline_run import build
from research.s20_harness.baseline_replay import replay
from research.s20_harness.multifold_run import run as campaign
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_baseline_model import inputs
from tests.s20_harness.test_baseline_run import plan
from tests.s20_harness.test_multifold_run import same_fold_models


@pytest.mark.parametrize('labels,expected',[([True,False],.5),([True,True],1.),([False,False],0.)])
def test_frequency_consumes_only_fit_targets_and_never_fits_classifier(monkeypatch,labels,expected):
    def fail(*a,**k):raise AssertionError('classifier fit forbidden')
    monkeypatch.setattr('research.s20_harness.baseline_model.LogisticRegression.fit',fail)
    monkeypatch.setattr('research.s20_harness.baseline_model.DecisionTreeClassifier.fit',fail)
    a=list(inputs());a[2]['target']=labels
    predictions,card=run(*a,target_id='P.safe.v4',model_family='mature_frequency')
    assert card['probability']==expected and card['model_level_fits']==0
    assert predictions.raw_probability.iloc[:2].isna().all()
    assert predictions.raw_probability.iloc[2:].eq(expected).all()
    a[1].loc[2:,'x']=-999
    changed,_=run(*a,target_id='P.safe.v4',model_family='mature_frequency')
    pd.testing.assert_frame_equal(predictions,changed)


def test_outer_label_injection_rejected():
    a=list(inputs());a[2]=pd.concat([a[2],pd.DataFrame(dict(sample_id=['outer-test0'],target=[True]))])
    with pytest.raises(ValueError,match='eligible fit labels'):run(*a,target_id='P.safe.v4',model_family='mature_frequency')


def test_saved_frequency_chain_replay_reports_calibration_separately(tmp_path):
    p=tmp_path/'frequency.json';atomic_json(p,dict(plan(),model_family='mature_frequency'))
    result=build(tmp_path,p,digest(p));out=Path(result['directory'])
    assert result['model_level_fits']==0 and result['calibrator_fits']==1
    repeated=replay(tmp_path,out,digest(out/'summary.json'))
    assert repeated['replay_model_fits']==0 and repeated['replay_calibrator_fits']==1


def test_owned_campaign_charges_frequency_zero_model_fit(tmp_path):
    path=same_fold_models(tmp_path);manifest=json.loads(path.read_text())
    job=manifest['jobs'][1];input_path=Path(job['input_path']);value=json.loads(input_path.read_text())
    value['model_family']='mature_frequency';atomic_json(input_path,value)
    job['input_sha256']=digest(input_path)
    trial=manifest['budget']['trials'][1];trial['input_sha256']=digest(input_path);trial['costs']=pipeline_costs(value)
    atomic_json(path,manifest)
    result=campaign(tmp_path,path,digest(path))
    counts=result['budget']['reserved_counts']
    assert counts['model_fits']==1 and counts['underlying_fits']==3 and counts['calibrator_fits']==2
    assert len(result['jobs'])==2 and result['distinct_outer_folds']==1
