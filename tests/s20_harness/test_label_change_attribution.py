import json
import pandas as pd
import pytest
from research.s20_harness.label_change_attribution import attribute


def test_unknowns_not_transitions_and_event_free_changes_flagged():
    frame=pd.DataFrame([dict(sample_id='a',signal_date='d',trading_code='A',compat_class=4,
        raw_calendar_class=4,o_class=3,raw_calendar_payload_json=json.dumps({'o_class':4}),
        o_payload_json=json.dumps({'o_class':3})),
        dict(sample_id='b',signal_date='d',trading_code='B',compat_class=4,raw_calendar_class=4,
        o_class=None,raw_calendar_payload_json=json.dumps({'o_class':4}),o_payload_json=json.dumps({'o_class':None}))])
    rows,report=attribute(frame)
    assert len(rows)==2 and report['economic_comparable']==1
    assert report['economic_class_changes']==1 and report['changed_without_recorded_events']==1
    frame.loc[0,'o_payload_json']=json.dumps({'o_class':0})
    with pytest.raises(ValueError,match='payload mismatch'): attribute(frame)
