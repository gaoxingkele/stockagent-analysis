import pandas as pd
import pytest

from research.s20_harness.event_index import EventIndex
from research.s20_harness.distribution_adapter import window_events
from research.s20_harness.corporate_actions import Distribution


def test_index_matches_reference_including_unknown_dates_aliases_and_boundaries():
    frame = pd.DataFrame([dict(ts_code=code, normalized_event_id=str(i), record_date=date)
                          for i, (code, date) in enumerate([("A", "20240103"), ("A", "20240105"),
                                                           ("B", None), ("C", "invalid")])])
    events = [Distribution(str(i), date, "20240108", "20240101", beneficiary_scope="existing_shareholders_verified")
              for i, date in [(1, "20240105"), (0, "20240103")]]
    decisions = [dict(normalized_event_id=str(i), accepted_for_accounting=i < 2) for i in range(4)]
    index = EventIndex(frame, events, decisions)
    for codes in [["A"], ["B"], ["A", "B"], ["C"], ["none"]]:
        for start, end in [("20240103", "20240105"), ("20240104", "20240104"), ("20240105", "20240110")]:
            assert index.window(codes, start, end) == window_events(frame, events, decisions, codes, start, end)
    with pytest.raises(ValueError, match="duplicate decision"):
        EventIndex(frame, events, decisions + decisions[:1])
    broken = EventIndex(frame, [], decisions)
    with pytest.raises(ValueError, match="payload missing"):
        broken.window(["A"], "20240103", "20240105")
