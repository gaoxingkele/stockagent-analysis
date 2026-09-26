import hashlib
import pytest

from research.s20_harness.suspension_monthly import parse_table


def parse(end="9999/12/31 00:00", duplicate=False):
    row = f"<tr><td>000046</td><td>name</td><td>reason</td><td>2023/12/28 09:30</td><td>{end}</td></tr>"
    data = ("<table><tr><th>Susp. Time Resume Time</th></tr>" + row * (2 if duplicate else 1) + "</table>").encode()
    return parse_table(data, encoding="utf-8", source_url="https://example.org/source",
                       observed_at="2026-09-14T00:00:00+08:00", expected_sha256=hashlib.sha256(data).hexdigest())


def test_unknown_end_not_infinite_or_backdated():
    frame = parse()
    assert frame.iloc[0].resumed_at is None
    assert frame.iloc[0].historical_available_at is None
    assert frame.iloc[0].interval_status == "unknown_end"
    assert parse("2024/01/05 09:30").iloc[0].interval_status == "closed_reported"


def test_bad_dates_and_duplicate_intervals_rejected():
    with pytest.raises(ValueError, match="nonpositive"):
        parse("2023/12/27 09:30")
    with pytest.raises(ValueError, match="duplicate"):
        parse(duplicate=True)
    with pytest.raises(ValueError, match="timestamp"):
        parse("unknown")
