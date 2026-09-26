import pandas as pd

from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.event_selection_audit import compare
from research.s20_harness.event_index import EventIndex


def test_alias_boundary_unknown_and_empty_selection_retained(monkeypatch):
    frame = pd.DataFrame([
        dict(ts_code='old', normalized_event_id='a', record_date='20240102'),
        dict(ts_code='new', normalized_event_id='b', record_date='20240105'),
        dict(ts_code='unknown', normalized_event_id='u', record_date=None),
        dict(ts_code='old', normalized_event_id='outside', record_date='20240101')])
    events = [Distribution(key, date, '20240108', '20231231',
        beneficiary_scope='existing_shareholders_verified')
        for key, date in [('b','20240105'), ('a','20240102'), ('outside','20240101')]]
    decisions = [dict(normalized_event_id=key, accepted_for_accounting=key!='u')
        for key in ['a','b','u','outside']]
    candidates = pd.DataFrame(dict(sample_id=['aliases','unknown','none'],
        event_codes=[['old','new'],['old','unknown'],['absent']]))
    result = compare(candidates, frame, events, decisions, '20240102','20240105')
    assert result.sample_id.tolist() == candidates.sample_id.tolist()
    assert result.matched.all()
    assert result.reference_event_ids.iloc[0] == '["b", "a"]'
    assert result.unresolved_event_ids.iloc[1] == '["u"]'
    assert result.reference_event_ids.iloc[2] == '[]'
    original = EventIndex.window
    def corrupted(self, *args):
        result = original(self, *args)
        return dict(result, events=[])
    monkeypatch.setattr(EventIndex, 'window', corrupted)
    changed = compare(candidates, frame, events, decisions, '20240102','20240105')
    assert changed.matched.tolist() == [False, True, True]
