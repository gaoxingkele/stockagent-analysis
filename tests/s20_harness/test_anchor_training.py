import pandas as pd
import pytest
from research.s20_harness.baseline_model import run
from research.s20_harness.feature_pipeline import prepare
from research.s20_harness.oof_audit import membership_hash
from research.s20_harness.runtime import digest
from tests.s20_harness.test_baseline_model import inputs


def setup(tmp_path):
    args=list(inputs());samples=args[0]
    samples['entity_id']=samples.sample_id
    earlier=dict(sample_id='earlier',entity_id='earlier',prediction_at='2023-10-02T21:00:00Z',
        feature_available_at='2023-10-02T20:00:00Z',horizon_close_at='2023-10-28T15:00:00Z',
        label_available_at='2023-10-28T21:00:00Z')
    history=pd.concat([pd.DataFrame([earlier]),samples],ignore_index=True)
    artifact=tmp_path/'anchor.txt';artifact.write_text('synthetic anchor artifact')
    context=dict(samples=history,predictions=pd.DataFrame(dict(sample_id=samples.sample_id[:-1],
        model_id='m',score=[.2,.8]+[.5]*7)),
        models={'m':dict(model_path=str(artifact),model_sha256=digest(artifact),
            information_cutoff_at='2023-12-01T00:00:00Z',dependency_sha256=membership_hash(['earlier']))},
        dependencies={'m':['earlier']},feature_name='anchor_score')
    args[4]['columns'].append('anchor_score')
    return args,context


def test_real_downstream_fit_does_not_impute_unavailable_anchor(tmp_path):
    args,context=setup(tmp_path)
    predictions,card=run(*args,target_id='P.safe.v4',anchor_context=context)
    assert card['preprocessing']['anchor_admission']['admitted']==9
    assert card['predicted_rows']==7 and len(predictions)==10
    assert pd.isna(predictions.raw_probability.iloc[-1])
    assert predictions.prediction_status.iloc[-1]=='feature_unavailable'
    assert args[0].feature_available_at.notna().all()


def test_missing_fit_anchor_excluded_from_fitted_statistics(tmp_path):
    args,context=setup(tmp_path)
    context['predictions']=context['predictions'].loc[lambda f:f.sample_id.ne('fit1')]
    matrix,assignment,report=prepare(args[0],args[1],args[3],args[4],anchor_context=context)
    assert report['fit_sample_ids']==['fit0']
    assert matrix.loc[matrix.sample_id.eq('fit1'),'anchor_score'].isna().all()
    with pytest.raises(ValueError,match='exact eligible fit labels'):
        run(*args,target_id='P.safe.v4',anchor_context=context)


def test_anchor_context_cannot_change_downstream_provenance(tmp_path):
    args,context=setup(tmp_path)
    context['samples'].loc[1,'prediction_at']='2024-01-03T21:00:00Z'
    with pytest.raises(AssertionError):
        run(*args,target_id='P.safe.v4',anchor_context=context)


@pytest.mark.parametrize('change',[.1,1e-12])
def test_calibration_rechecks_anchor_and_retains_missing_outer(tmp_path,change):
    from research.s20_harness.calibration_model import run as calibrate
    args,context=setup(tmp_path)
    predictions,card=run(*args,target_id='P.safe.v4',anchor_context=context)
    labels=pd.DataFrame(dict(sample_id=['calibration0','calibration1'],target=[False,True]))
    result,_=calibrate(args[0],predictions,labels,args[3],card,target_id='P.safe.v4',anchor_context=context)
    assert len(result)==10 and pd.isna(result.calibrated_probability.iloc[-1])
    assert result.calibrated_probability.iloc[6:9].notna().all()
    with pytest.raises(ValueError,match='anchor evidence'):
        calibrate(args[0],predictions,labels,args[3],card,target_id='P.safe.v4')
    saved=context['predictions'].copy()
    context['predictions'].loc[0,'score']+=change
    with pytest.raises(ValueError,match='admission changed'):
        calibrate(args[0],predictions,labels,args[3],card,target_id='P.safe.v4',anchor_context=context)
    context['predictions']=saved
    context['predictions']=context['predictions'].loc[lambda f:f.sample_id.ne('calibration0')]
    with pytest.raises(ValueError,match='admission changed'):
        calibrate(args[0],predictions,labels,args[3],card,target_id='P.safe.v4',anchor_context=context)
