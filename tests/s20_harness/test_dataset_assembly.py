import pandas as pd
import pytest

from research.s20_harness.dataset_assembly import assemble, FEATURES


def fixture():
    labels=pd.DataFrame(dict(sample_id=['a','b'],entity_id=['A','B'],signal_date=['20240102']*2,
        horizon_end=['20240130']*2,p_class=['A',None],label_realized=[True,False],label_status=['complete','unfilled']))
    features=labels[['sample_id','entity_id','signal_date']].copy()
    features['trading_code']=['A','B'];features['price_window_status']=['incomplete_market_window','vendor_factor_price_diagnostic']
    for f in FEATURES: features[f]=[float('nan'),1.]
    return labels,features


def test_missing_features_and_unknown_labels_both_retained():
    labels,features=fixture(); result=assemble(labels,features,'2026-09-14T00:00:00+00:00')
    assert len(result)==2 and result.sample_id.tolist()==['a','b']
    assert result.gross_safe_diagnostic_target.iloc[0] and pd.isna(result.gross_safe_diagnostic_target.iloc[1])
    assert result.all_feature_values_present.tolist()==[False,True]
    assert pd.isna(result.label_available_at.iloc[1]) and result.feature_available_at.str.startswith('2026').all()
    assert not result.formal_training_eligible.any()


def test_reordered_or_removed_candidates_rejected():
    labels,features=fixture()
    for f in (features.iloc[::-1],features.iloc[:1]):
        with pytest.raises(ValueError,match='identity or order'): assemble(labels,f,'2026-09-14T00:00:00+00:00')


def test_saved_replay_rejects_rehashed_semantic_tampering(tmp_path,monkeypatch):
    from research.s20_harness import dataset_assembly as module
    from research.s20_harness.runtime import atomic_json,digest,load_plan
    labels,features=fixture()
    observed='2026-09-14T00:00:00+00:00'
    expected=assemble(labels,features,observed)
    monkeypatch.setattr(module,'reconstruct',lambda *a,**kw:(expected.copy(),{}, {},observed))
    report=module.build(tmp_path,tmp_path,'label',tmp_path,'feature')
    directory=module.Path(report['directory'])
    saved,evidence=module.verify_saved(tmp_path,directory,digest(directory/'summary.json'))
    pd.testing.assert_frame_equal(saved,expected)
    assert evidence['replayed'] and not evidence['formal_training_authorized']
    saved.loc[0,'gross_safe_diagnostic_target']=False
    saved.to_parquet(directory/'samples.parquet',index=False)
    summary=load_plan(directory/'summary.json')
    summary['artifacts']['samples.parquet']=digest(directory/'samples.parquet')
    atomic_json(directory/'summary.json',summary)
    with pytest.raises(AssertionError):
        module.verify_saved(tmp_path,directory,digest(directory/'summary.json'))


def test_saved_replay_rejects_summary_promotion(tmp_path,monkeypatch):
    from research.s20_harness import dataset_assembly as module
    from research.s20_harness.runtime import atomic_json,digest
    labels,features=fixture(); observed='2026-09-14T00:00:00+00:00'
    monkeypatch.setattr(module,'reconstruct',lambda *a,**kw:(assemble(labels,features,observed),{}, {},observed))
    report=module.build(tmp_path,tmp_path,'label',tmp_path,'feature')
    directory=module.Path(report['directory'])
    report['formal_training_authorized']=True
    atomic_json(directory/'summary.json',report)
    with pytest.raises(ValueError,match='summary semantics'):
        module.verify_saved(tmp_path,directory,digest(directory/'summary.json'))
