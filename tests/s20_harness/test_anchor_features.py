import pandas as pd
import pytest

from research.s20_harness.anchor_features import attach
from research.s20_harness.oof_audit import membership_hash
from tests.s20_harness.test_oof_audit import fixture


def test_chronological_anchor_and_missing_rows_retained(tmp_path):
    samples,pred,models,deps=fixture(tmp_path)
    candidates=pd.DataFrame({'sample_id':['train','future'],'segment':['fit','tune']})
    out,report=attach(candidates,samples,pred,models,deps)
    assert len(out)==2 and report['admitted']==1
    assert pd.isna(out.anchor_score.iloc[0]) and out.anchor_score.iloc[1]==.6
    assert not report['all_anchor_features_admitted']
    empty,report=attach(candidates,samples,pred.iloc[:0],models,deps)
    assert len(empty)==2 and empty.anchor_score.isna().all()


def test_refit_tune_overlap_is_not_a_usable_feature(tmp_path):
    samples,pred,models,deps=fixture(tmp_path)
    samples.loc[1,'label_available_at']='2024-03-01T21:00:00+08:00'
    deps['m'].append('future')
    models['m']['dependency_sha256']=membership_hash(deps['m'])
    models['m']['information_cutoff_at']='2024-03-02T21:00:00+08:00'
    candidates=pd.DataFrame({'sample_id':['future'],'segment':['tune']})
    out,report=attach(candidates,samples,pred,models,deps)
    assert out.anchor_score.isna().all() and report['admitted']==0
    assert 'prediction_entity_time_used_as_model_dependency' in out.anchor_provenance_reasons.iloc[0]
    with pytest.raises(ValueError,match='overwrite'):
        attach(out,samples,pred,models,deps)
