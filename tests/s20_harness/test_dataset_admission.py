import pandas as pd
import pytest

from research.s20_harness import dataset_admission as module
from research.s20_harness.runtime import atomic_json,digest


def test_late_features_never_become_historical_training(tmp_path,monkeypatch):
    collection=tmp_path/'collection';collection.mkdir()
    atomic_json(collection/'summary.json',{'fixture':True})
    samples=pd.DataFrame(dict(sample_id=['a','b'],prediction_at=['2024-01-02T21:00:00+08:00']*2,
        feature_available_at=['2026-09-14T00:00:00+00:00']*2,
        horizon_close_at=['2024-01-30T15:00:00+08:00']*2,
        label_available_at=['2026-09-14T00:00:00+00:00',None]))
    monkeypatch.setattr(module,'verify_collection',lambda *a:(samples,{'fixture':True}))
    boundaries=[dict(name=n,start_at=f'2024-{i:02d}-01T00:00:00+00:00',
                     end_at=f'2024-{i+1:02d}-01T00:00:00+00:00')
                for i,n in enumerate(['fit','tune','calibration','selection-policy','outer-test'],1)]
    split=tmp_path/'split.json'
    atomic_json(split,dict(boundaries=boundaries,evaluation_at='2026-09-15T00:00:00+00:00'))
    result=module.build(tmp_path,collection,digest(collection/'summary.json'),split,digest(split))
    saved=pd.read_parquet(module.Path(result['directory'])/'assignments.parquet')
    assert saved.sample_id.tolist()==['a','b']
    assert not saved.feature_ready.any() and not saved.supervised_eligible.any()
    assert result['supervised_timestamp_ready_rows']==0
    assert result['reason_counts']=={'feature_unknown_or_not_prior':2}
    assert not result['formal_training_authorized']


def test_split_hash_mismatch_refused_before_reading_collection(tmp_path):
    split=tmp_path/'split.json';atomic_json(split,{})
    with pytest.raises(ValueError,match='split pin'):
        module.build(tmp_path,tmp_path,'x',split,'0'*64)
