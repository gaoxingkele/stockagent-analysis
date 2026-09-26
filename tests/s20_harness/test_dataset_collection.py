import pandas as pd
import pytest

from research.s20_harness.dataset_collection import combine


def frame(date):
    return pd.DataFrame(dict(sample_id=[date+':A',date+':B'],signal_date=[date]*2,
        feature=[1.,float('nan')],target=pd.array([True,None],dtype='boolean'),
        formal_training_eligible=[False,False]))


def test_all_unknown_and_missing_rows_retained():
    result=combine([frame('20240102'),frame('20240103')])
    assert len(result)==4 and result.target.isna().sum()==2 and result.feature.isna().sum()==2


@pytest.mark.parametrize('case',['empty','duplicate_date','reversed','schema','duplicate_sample','formal'])
def test_invalid_collections_rejected(case):
    a,b=frame('20240102'),frame('20240103')
    frames=[a,b]
    if case=='empty': frames=[]
    if case=='duplicate_date': frames=[a,a]
    if case=='reversed': frames=[b,a]
    if case=='schema': frames=[a,b.rename(columns={'feature':'future_feature'})]
    if case=='duplicate_sample': b.loc[0,'sample_id']=a.sample_id.iloc[0]
    if case=='formal': b.loc[0,'formal_training_eligible']=True
    with pytest.raises(ValueError): combine(frames)


@pytest.mark.parametrize('tamper',[None,'table','summary'])
def test_collection_consumer_replays_values(tmp_path,monkeypatch,tamper):
    from research.s20_harness import dataset_collection as module
    from research.s20_harness.runtime import atomic_json,digest
    originals={'a':frame('20240102'),'b':frame('20240103')}
    monkeypatch.setattr(module,'verify_saved',lambda root,directory,sha:(originals[directory].copy(),{}))
    combined=combine(list(originals.values()))
    combined.to_parquet(tmp_path/'samples.parquet',index=False)
    atomic_json(tmp_path/'inputs.json',dict(assemblies=[dict(directory=d,summary_sha256=d) for d in originals]))
    summary=dict(rows=4,dates=['20240102','20240103'],rows_by_date={'20240102':2,'20240103':2},
        all_candidates_retained=True,formal_training_authorized=False,historical_availability_proven=False)
    if tamper=='table':
        combined.loc[0,'feature']=100.
        combined.to_parquet(tmp_path/'samples.parquet',index=False)
    if tamper=='summary': summary['historical_availability_proven']=True
    summary['artifacts']={n:digest(tmp_path/n) for n in ['samples.parquet','inputs.json']}
    atomic_json(tmp_path/'summary.json',summary)
    if tamper:
        with pytest.raises((AssertionError,ValueError)):
            module.verify_collection(tmp_path,tmp_path,digest(tmp_path/'summary.json'))
    else:
        table,evidence=module.verify_collection(tmp_path,tmp_path,digest(tmp_path/'summary.json'))
        pd.testing.assert_frame_equal(table,combined)
        assert evidence['member_assemblies']==2 and not evidence['formal_training_authorized']
