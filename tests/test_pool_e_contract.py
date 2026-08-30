import copy
import json
from types import SimpleNamespace

import pytest

import daily_dashboard as dashboard


def _payload():
    return {
        "result_group": "strategy",
        "signal_date": "2026-08-28",
        "unique_count": 100,
        "horizon_quotas": {"H5": 30, "H10": 35, "H20": 35},
        "strategies": [{"strategy_id": f"strategy_{i}"} for i in range(15)],
        "strategy_weights": {f"strategy_{i}": 1 / 15 for i in range(15)},
        "stocks": [
            {
                "overall_rank": rank,
                "ts_code": f"{rank:06d}.SZ",
                "name": f"stock_{rank}",
            }
            for rank in range(100, 0, -1)
        ],
        "published_at": "2026-08-28T21:21:38+08:00",
        "publication_policy": "two independent groups; no strategy-SEMAS fusion",
    }


def test_pool_e_contract_accepts_complete_strategy_publication():
    codes, meta = dashboard._validate_pool_e_payload(_payload())

    assert len(codes) == dashboard.POOL_E_TOPN
    assert codes[:2] == ["000001.SZ", "000002.SZ"]
    assert meta["signal_date"] == "20260828"
    assert meta["unique_count"] == 100
    assert meta["strategy_count"] == 15
    assert meta["strategy_weight_count"] == 15


@pytest.mark.parametrize("mutation", ["duplicate", "quota", "strategies"])
def test_pool_e_contract_rejects_incomplete_publication(mutation):
    payload = copy.deepcopy(_payload())
    if mutation == "duplicate":
        payload["stocks"][1]["ts_code"] = payload["stocks"][0]["ts_code"]
    elif mutation == "quota":
        payload["horizon_quotas"]["H5"] = 29
    else:
        payload["strategies"].pop()

    with pytest.raises(ValueError):
        dashboard._validate_pool_e_payload(payload)


def test_pool_e_fetch_uses_stable_json_cli(monkeypatch, tmp_path):
    exporter = tmp_path / "scripts" / "export_daily_top100_list.py"
    exporter.parent.mkdir()
    exporter.touch()
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps(_payload()))

    monkeypatch.setattr(dashboard, "POOL_E_EXPORT_SCRIPT", exporter)
    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)

    codes, meta = dashboard._fetch_pool_e_external()

    assert seen["command"][-4:] == ["--group", "strategy", "--format", "json"]
    assert seen["kwargs"]["cwd"] == str(exporter.parents[1])
    assert len(codes) == dashboard.POOL_E_TOPN
    assert meta["signal_date"] == "20260828"
