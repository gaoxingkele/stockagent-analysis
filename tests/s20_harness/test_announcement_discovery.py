import pytest

from research.s20_harness.announcement_discovery import validate_page, collect


def test_page_count_not_trusted_over_row_coverage():
    rows, total, more = validate_page(dict(announcements=[dict(announcementId="123")],
                                         totalAnnouncement=93, totalpages=3, hasMore=True))
    assert len(rows) == 1 and total == 93 and more
    with pytest.raises(ValueError, match="duplicate"):
        validate_page(dict(announcements=rows + rows, totalAnnouncement=2, hasMore=False))
    with pytest.raises(ValueError, match="schema"):
        validate_page(dict(announcements=[], totalAnnouncement=0, hasMore="false"))


def test_discovery_unknown_keyword_refused_before_io(tmp_path):
    with pytest.raises(ValueError, match="keyword"):
        collect(tmp_path, ["2024-06-20"], keyword="unbounded search")
    assert not (tmp_path / "output").exists()


def test_null_rows_only_valid_for_explicit_terminal_zero():
    assert validate_page(dict(announcements=None, totalAnnouncement=0, hasMore=False)) == ([], 0, False)
    for total, more in [(1, False), (0, True), (False, False)]:
        with pytest.raises(ValueError):
            validate_page(dict(announcements=None, totalAnnouncement=total, hasMore=more))
