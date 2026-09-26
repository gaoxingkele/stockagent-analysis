import pandas as pd
import pytest
from research.s20_harness.legacy_metric_replay import replay


def test_ties_and_complete_denominator():
    frame=pd.DataFrame(dict(trade_date=['20260101']*2,ts_code=['B','A'],immediate=[0,1],
        down_risk=[1,0],old_s20=[50,50],old_r20=[50,50],portable_full=[50,50],
        portable_full_up=[.5,.5],lowcorr24=[50,50]))
    metrics,selected=replay(frame,k=1)
    assert len(metrics)==5 and selected.ts_code.eq('A').all()
    assert all(r['immediate_rate']==1 and r['down_rate']==0 for r in metrics)
    frame.loc[0,'old_s20']=float('nan')
    with pytest.raises(ValueError,match='score missing'):
        replay(frame)
