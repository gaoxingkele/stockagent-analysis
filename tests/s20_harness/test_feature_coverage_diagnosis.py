import pandas as pd
import pytest

from research.s20_harness.feature_coverage_diagnosis import classify


def test_missing_stock_distinct_from_missing_date():
    joined=pd.DataFrame(dict(sample_id=['a','b','c'], trading_code=['A','B','C'],
                             signal_date=['20240102']*3, feature_row_present=[True,False,False]))
    keys=pd.DataFrame(dict(ts_code=['A','B'],trade_date=['20240102','20240103']))
    result=classify(joined,keys)
    assert result.coverage_reason.tolist()==['matched','code_present_but_signal_date_absent','absent_from_all_source_groups']
    assert not result.cause_established.any() and len(result)==3
    joined.loc[2,'feature_row_present']=True
    with pytest.raises(ValueError,match='disagrees'): classify(joined,keys)
