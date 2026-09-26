import pandas as pd

from research.s20_harness.name_timeline import timeline
from research.s20_harness.name_history import name_asof


def test_compact_timeline_matches_daily_asof_with_late_old_records():
    events = pd.DataFrame([
        dict(ts_code="A", name="old", start_date="20200101", ann_date="20191231"),
        dict(ts_code="A", name="STnew", start_date="20240103", ann_date="20240103"),
        dict(ts_code="A", name="late_old", start_date="20210101", ann_date="20240105"),
        dict(ts_code="A", name="missing", start_date="20240107", ann_date=None)])
    states = timeline(events, ["A"], "20240101", "20240110")
    for day in pd.date_range("2024-01-01", "2024-01-10"):
        date = day.strftime("%Y%m%d")
        state = next(s for s in states if s["valid_from"] <= date < s["valid_until_exclusive"])
        reference = name_asof(events, ["A"], date)
        assert state["name"] == reference["name"]
        assert state["name_st_proxy"] == reference["name_st_proxy"]
    assert states[-1]["name"] == "STnew"


def test_empty_history_produces_unknown_full_interval():
    events = pd.DataFrame(columns=["ts_code", "name", "start_date", "ann_date"])
    result = timeline(events, ["A"], "20240101", "20240110")
    assert len(result) == 1 and result[0]["status"] == "unknown"
    assert result[0]["valid_until_exclusive"] == "20240111"
