"""Adversarial bundle checks with explicit isolated reconstruction oracle.

Unmocked campaign integration is in test_campaign_evaluation.py.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness import baseline_bundle as module
from research.s20_harness.runtime import atomic_json,digest


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    split=[dict(job_id='job',fold_id='fold',boundaries={'fit':['a','b']})]
    predictions=pd.DataFrame(dict(job_id=['job','job'],sample_id=['a','b'],score=[.7,.2],
        evaluation_target=[True,None],evaluation_status=['mature','not_mature_at_evaluation'],
        recorded_oof_provenance_valid=[True,True]))
    metrics=pd.DataFrame(dict(job_id=['job'],brier_known_only=[.09]))
    cards=[dict(job_id='job',fold_id='fold',model_family='logistic',target_id='P.safe.v4',evidence_mode='synthetic')]
    monkeypatch.setattr(module,'reconstruct',lambda *a:(split,predictions.copy(deep=True),metrics.copy(deep=True),cards))
    monkeypatch.setattr(module,'verify_evaluation',lambda *a:None)
    report=module.build(tmp_path,tmp_path/'evaluation','a'*64)
    return tmp_path,Path(report['directory'])


def rehash(directory):
    inner=json.loads((directory/'baseline_bundle.json').read_text())
    for name in inner['artifacts']: inner['artifacts'][name]=digest(directory/name)
    atomic_json(directory/'baseline_bundle.json',inner)
    outer=json.loads((directory/'summary.json').read_text())
    for name in outer['artifacts']: outer['artifacts'][name]=digest(directory/name)
    atomic_json(directory/'summary.json',outer)


def test_exact_isolated_bundle_reconstruction(bundle):
    root,d=bundle
    assert module.verify(root,d,digest(d/'summary.json'))['rows']==2


@pytest.mark.parametrize('fault',['score','unknown','split','cards','count','gaps','promotion','coverage'])
def test_coordinated_rehash_cannot_hide_semantic_change(bundle,fault):
    root,d=bundle
    if fault in ['score','unknown']:
        p=pd.read_parquet(d/'baseline_oof.parquet')
        if fault=='score':p.loc[0,'score']=.99
        else:p.loc[1,'evaluation_target']=True
        p.to_parquet(d/'baseline_oof.parquet',index=False)
    elif fault=='split':
        s=json.loads((d/'split_manifest.json').read_text());s['jobs'][0]['boundaries']['fit']=['a','future']
        atomic_json(d/'split_manifest.json',s)
    else:
        s=json.loads((d/'baseline_bundle.json').read_text())
        if fault=='cards':s['jobs'][0]['model_family']='claimed-champion'
        if fault=='count':s['prediction_rows']=999
        if fault=='promotion':s['formal_H03_accepted']=True
        if fault=='coverage':s['baseline_coverage']['observed_scope'][0]['missing_families']=[]
        if fault=='gaps':
            s['acceptance_gaps']=[]
            outer=json.loads((d/'summary.json').read_text());outer['acceptance_gaps']=[]
            atomic_json(d/'summary.json',outer)
        atomic_json(d/'baseline_bundle.json',s)
    rehash(d)
    with pytest.raises((ValueError,AssertionError)):module.verify(root,d,digest(d/'summary.json'))


def test_parent_verification_failure_propagates(bundle,monkeypatch):
    root,d=bundle
    def fail(*a):raise ValueError('upstream invalid')
    monkeypatch.setattr(module,'reconstruct',fail)
    with pytest.raises(ValueError,match='upstream invalid'):module.verify(root,d,digest(d/'summary.json'))
