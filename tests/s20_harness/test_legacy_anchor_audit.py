import pandas as pd
import pytest
from research.s20_harness.legacy_anchor_audit import compose


def test_daily_ties_and_saved_mismatch():
    frame=pd.DataFrame(dict(trade_date=['d','d','e'],ts_code=['A','B','A'],
        stage1_probability=[.5,.5,.2],lambdarank_score=[1,2,1],s20_20r_rank=[.625,.875,1]))
    assert compose(frame).absolute_error.eq(0).all()
    frame.loc[0,'s20_20r_rank']=0
    assert compose(frame).absolute_error.iloc[0]==.625
    with pytest.raises(ValueError,match='duplicate'):
        compose(pd.concat([frame,frame]))
