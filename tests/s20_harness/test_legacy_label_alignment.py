import pandas as pd
import pytest
from research.s20_harness.legacy_label_alignment import align


def test_unknowns_and_unmatched_rows_not_erased():
    current=pd.DataFrame(dict(trading_code=['A','B'],signal_date=['20240102']*2,
        compat_class=[0,-99],compat_horizon_end=['20240130',None],
        p_class=['A',None],o_class=[0,-99],o_label_realized=[True,False]))
    old=pd.DataFrame(dict(ts_code=['A','C'],trade_date=['20240102']*2,
        s20_class=[0,3],entry_date=['20240103']*2,horizon_end_date=['20240130']*2))
    rows,report=align(current,old)
    assert len(rows)==3 and report['current_P_unknown']==1 and report['current_O_unresolved']==1
    assert report['alignment_counts']=={'both':1,'left_only':1,'right_only':1}
    assert report['comparable_legacy_compat']==1 and report['legacy_compat_mismatches']==0
    assert rows.legacy_compat_equal.isna().sum()==2
    with pytest.raises(ValueError,match='unique'):
        align(current,pd.concat([old,old]))
