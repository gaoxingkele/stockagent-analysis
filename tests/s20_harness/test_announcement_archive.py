import pandas as pd
import pytest

from research.s20_harness.announcement_archive import match_events


def test_candidate_matching_retains_unmatched_and_ambiguity():
    events = pd.DataFrame([dict(normalized_event_id=str(i), ts_code=code, imp_ann_date="20240628")
                           for i, code in enumerate(["603081.SH", "000001.SZ"])])
    notice = dict(secCode="603081", query_date="2024-06-28", announcementTitle="2023年年度<em>权益分派实施</em>公告",
                  adjunctUrl="finalpage/2024-06-28/123.PDF")
    result = match_events(events, [notice], ["2024-06-28"])
    assert [r["status"] for r in result] == ["unique_candidate", "unmatched"]
    assert not any(r["beneficiary_approved"] for r in result)
    other = dict(notice, adjunctUrl="finalpage/2024-06-28/124.PDF")
    assert match_events(events, [notice, other], ["2024-06-28"])[0]["status"] == "ambiguous"
    with pytest.raises(ValueError, match="unsafe"):
        match_events(events, [dict(notice, adjunctUrl="../../escape.pdf")], ["2024-06-28"])


@pytest.mark.parametrize("title,matched", [
    ("关于2023年年度权益分派实施的公告", True),
    ("2023年度分红派息实施公告", True),
    ("2023年度分红派息实施公告的更正公告", False),
    ("龙溪股份2023年年度权益分派实施公告（2024-029）", True),
    ("权益分派实施公告的更正公告", False),
    ("关于取消权益分派实施公告的公告", False),
    ("权益分派实施公告（修订）", False),
])
def test_known_title_variants_do_not_swallow_corrections(title, matched):
    frame = pd.DataFrame([dict(normalized_event_id="e", ts_code="600592.SH", imp_ann_date="20240621")])
    item = dict(secCode="600592", query_date="2024-06-21", announcementTitle=title,
                adjunctUrl="finalpage/2024-06-21/123.PDF")
    assert (match_events(frame, [item], ["2024-06-21"])[0]["status"] == "unique_candidate") == matched
