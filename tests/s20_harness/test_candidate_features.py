import pandas as pd
import pytest

from research.s20_harness.candidate_features import attach, build
from research.s20_harness.runtime import digest


def frames():
    candidates = pd.DataFrame(dict(sample_id=['a','b','c'], entity_id=['X','Y','X'],
        trading_code=['000001.SZ','000002.SZ','000001.SZ'], signal_date=['20240102']*3,
        future_label=[None,True,False]))
    factors = pd.DataFrame(dict(ts_code=['000001.SZ'], trade_date=['20240102'], x=[2.],
                               source_path=['raw'], source_sha256=['a'*64]))
    return candidates, factors


def test_unknown_outcomes_and_missing_features_retained():
    candidates, factors = frames()
    result = attach(candidates, factors, ['x'])
    assert result.sample_id.tolist() == ['a','b','c']
    assert result.feature_row_present.tolist() == [True,False,True]
    assert pd.isna(result.x.iloc[1]) and 'future_label' not in result


def test_duplicate_factor_keys_rejected():
    candidates, factors = frames()
    with pytest.raises(ValueError, match='duplicate'):
        attach(candidates, pd.concat([factors,factors]), ['x'])


def test_actual_files_and_source_pins(tmp_path):
    candidates, factors = frames()
    path=tmp_path/'candidates.parquet'; candidates.to_parquet(path,index=False)
    folder=tmp_path/'factors'; folder.mkdir()
    factors[['ts_code','trade_date','x']].to_parquet(folder/'group_001.parquet',index=False)
    report=build(tmp_path,path,digest(path),folder,['x'])
    assert report['output_rows']==3 and report['missing_rows']==1
    assert not report['formal_training_authorized']
