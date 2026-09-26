import pandas as pd
import pytest

from research.s20_harness import close_mark_source as source
from research.s20_harness.runtime import digest


def fixture(tmp_path, **changes):
    row = dict(ts_code="000001.SZ", trade_date="20240729", close=10., adj_close=99.)
    row.update(changes)
    p = tmp_path/"daily.parquet"
    pd.DataFrame([row]).to_parquet(p, index=False)
    return p


def test_raw_close_and_missing_identity_kept_with_present_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "now", lambda: "2026-09-14T00:00:00+00:00")
    p = fixture(tmp_path)
    marks, report = source.build(tmp_path, p, digest(p), "20240729", ["000001.SZ", "600027.SH"])
    assert marks[0].price == 10 and marks[1].price is None
    assert marks[1].status == "missing" and marks[0].available_at.startswith("2026-09-14")
    assert report["source_bytes_verified"] and not report["historical_availability_proven"]
    assert not report["formal_training_authorized"]


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf")])
def test_invalid_price_retains_requested_mark_as_unknown(tmp_path, price):
    p = fixture(tmp_path, close=price)
    marks, report = source.read_marks(p, digest(p), "20240729", ["000001.SZ"])
    assert marks[0].price is None and report["decisions"][0]["reason"] == "invalid_raw_close"


@pytest.mark.parametrize("fault", ["hash", "date", "duplicate", "codes", "future"])
def test_bad_inputs_rejected_without_output(tmp_path, monkeypatch, fault):
    p = fixture(tmp_path)
    codes = ["000001.SZ"]
    if fault == "duplicate":
        f = pd.read_parquet(p); pd.concat([f, f]).to_parquet(p, index=False)
    if fault == "codes": codes *= 2
    if fault == "future": monkeypatch.setattr(source, "now", lambda: "2024-07-29T01:00:00+00:00")
    with pytest.raises(ValueError):
        source.build(tmp_path, p, "0"*64 if fault == "hash" else digest(p),
                     "20240730" if fault == "date" else "20240729", codes)
    assert not (tmp_path/"output").exists()


def test_midread_source_mutation_rejected(tmp_path, monkeypatch):
    p = fixture(tmp_path); sha = digest(p); original = source.pd.read_parquet
    def mutate(path):
        f = original(path)
        f.assign(close=11).to_parquet(path, index=False)
        return f
    monkeypatch.setattr(source.pd, "read_parquet", mutate)
    with pytest.raises(ValueError, match="changed during"):
        source.build(tmp_path, p, sha, "20240729", ["000001.SZ"])
    assert not (tmp_path/"output").exists()


def test_pinned_marks_feed_valuation_without_backdating_availability(tmp_path, monkeypatch):
    from research.s20_harness.portfolio_book import create
    from research.s20_harness.portfolio_valuation import value_portfolio
    from tests.s20_harness.test_portfolio_book import buy, fill, CAL
    monkeypatch.setattr(source, "now", lambda: "2026-09-14T00:00:00+00:00")
    p = fixture(tmp_path)
    marks, report = source.read_marks(p, digest(p), "20240729", ["000001.SZ"])
    book = fill(buy(create(1000, CAL), ts_code="000001.SZ"))
    result = value_portfolio(book, marks, valuation_at=report["valuation_at"],
                             knowledge_at=report["observed_at"])
    assert result["nav"] == 1299
    with pytest.raises(ValueError, match="availability"):
        value_portfolio(book, marks, valuation_at=report["valuation_at"],
                        knowledge_at="2024-07-29T15:02:00+08:00")
