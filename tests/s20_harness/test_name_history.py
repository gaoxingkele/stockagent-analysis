import pandas as pd

from research.s20_harness.name_history import name_asof


def events():
    return pd.DataFrame([
        dict(ts_code="000001.SZ", name="旧名称", start_date="20200101", end_date="20240103", ann_date="20191231"),
        dict(ts_code="000001.SZ", name="*ST示例", start_date="20240104", end_date=None, ann_date="20240104")])


def test_same_day_announcement_not_assumed_before_decision():
    assert not name_asof(events(), ["000001.SZ"], "20240104")["name_st_proxy"]
    assert name_asof(events(), ["000001.SZ"], "20240105")["name_st_proxy"]


def test_no_current_name_fallback_for_unknown_code():
    result = name_asof(events(), ["000002.SZ"], "20240105")
    assert result["status"] == "unknown"
    assert result["name_st_proxy"] is None


def test_blank_name_is_unknown_and_whitespace_cannot_hide_st():
    frame = events()
    frame.loc[1, "name"] = "  "
    assert name_asof(frame, ["000001.SZ"], "20240105")["status"] == "unknown"
    frame.loc[1, "name"] = "  *ST示例  "
    assert name_asof(frame, ["000001.SZ"], "20240105")["name_st_proxy"] is True


def test_conflicting_latest_names_rejected_and_future_end_ignored():
    frame = events()
    extra = frame.iloc[1].copy()
    extra["name"] = "正常名称"
    frame = pd.concat([frame, extra.to_frame().T], ignore_index=True)
    assert name_asof(frame, ["000001.SZ"], "20240105")["status"] == "unknown"
    assert name_asof(frame, ["000001.SZ"], "20240104")["name"] == "旧名称"
