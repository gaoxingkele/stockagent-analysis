import json
import pytest

from research.s20_harness import announcement_campaign as campaign
from research.s20_harness.runtime import atomic_json, digest


def test_campaign_persists_incomplete_discovery_and_never_archives(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    atomic_json(source, {})
    def incomplete(root, dates):
        child = tmp_path / "child"
        child.mkdir()
        report = dict(directory=str(child), queries=[dict(complete_search=False)])
        atomic_json(child / "summary.json", report)
        return report
    monkeypatch.setattr(campaign.announcement_discovery, "collect", incomplete)
    monkeypatch.setattr(campaign.announcement_archive, "build", lambda *a, **kw: pytest.fail("must not archive"))
    with pytest.raises(ValueError, match="incomplete discovery"):
        campaign.run(tmp_path, ["2024-07-03"], source, digest(source), source, digest(source), source, digest(source))
    out = next((tmp_path / "output/experiments/s20_safe_v4/sources").glob("announcement-campaign-*"))
    assert json.loads((out/"state.json").read_text())["status"] == "FAILED"
    stages = json.loads((out/"stages.json").read_text())
    assert len(stages) == 1 and stages[0]["stage"] == "discovery"
    assert not (out/"summary.json").exists()


def test_campaign_scope_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        campaign.run(tmp_path, ["2024-07-03"]*2, "missing", "", "missing", "", "missing", "")
