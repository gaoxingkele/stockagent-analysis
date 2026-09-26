import json

from research.s20_harness.suspension_collect import collect


def test_failed_source_retained_and_valid_month_parsed(tmp_path, monkeypatch):
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"sources": [
        {"month": "2024.01", "url": "https://docs.static.szse.cn/good"},
        {"month": "2024.02", "url": "https://docs.static.szse.cn/wrong"}]}))
    class Response:
        status_code = 200
        content = ('证券停牌情况 2024.01 <table><tr><th>Susp. Time Resume Time</th></tr>'
                   '<tr><td>000046</td><td>name</td><td>reason</td><td>2024/01/02 09:30</td>'
                   '<td>9999/12/31 00:00</td></tr></table>').encode()
        def raise_for_status(self):
            pass
    monkeypatch.setattr("research.s20_harness.suspension_collect.requests.get", lambda *a, **k: Response())
    result = collect(tmp_path, config)
    assert result["requested_months"] == 2 and result["parsed_months"] == 1
    from pathlib import Path
    receipts = json.loads((Path(result["directory"]) / "receipts.json").read_text())
    assert receipts[1]["status"] == "FAILED_VALIDITY"
    assert Path(receipts[1]["archive_path"]).is_file()
    assert not result["full_history_coverage_verified"]
