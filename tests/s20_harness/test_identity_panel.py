import pandas as pd
import pytest

from research.s20_harness.identity_panel import build, entity_window_frame
from research.s20_harness.runtime import atomic_json, digest
from research.s20_harness.identity_panel_verify import verify
from pathlib import Path
from research.s20_harness.security_identity import canonicalize_aliases
from research.s20_harness.labels import label_p_track


ALIAS = dict(entity_id="continuing", old_code="000001.SZ", new_code="000002.SZ",
             effective_date="20240103", share_conversion_ratio=1.)


def sample():
    return pd.DataFrame([dict(ts_code=code, trade_date=date, open=10., high=11., low=10., close=11.)
                         for date, code in [("20240102", "000001.SZ"), ("20240102", "000002.SZ"),
                                            ("20240103", "000002.SZ"), ("20240104", "000002.SZ")]])


def test_cross_code_window_uses_one_entity_without_rewriting_trading_codes():
    canonical, _, _ = canonicalize_aliases(sample(), [ALIAS])
    frame = entity_window_frame(canonical, "continuing")
    assert frame.trading_code.tolist() == ["000001.SZ", "000002.SZ", "000002.SZ"]
    label = label_p_track(frame, ["20240101", "20240102", "20240103", "20240104"],
                         "20240101", "continuing", horizon=3)
    assert label["p_class"] == "A"
    with pytest.raises(ValueError, match="duplicate"):
        entity_window_frame(pd.concat([canonical, canonical]), "continuing")


def test_build_preserves_source_and_hashes_outputs(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    atomic_json(tmp_path / "config/s20_v4_security_aliases.json", {"aliases": [ALIAS]})
    for date, frame in sample().groupby("trade_date"):
        frame.to_parquet(daily / (date + ".parquet"), index=False)
    result = build(tmp_path)
    assert result["raw_rows"] == 4 and result["canonical_rows"] == 3
    assert result["identity_consistent"]
    assert not result["formal_training_eligible"]
    assert len(pd.read_parquet(daily / "20240102.parquet")) == 2
    output = Path(result["directory"])
    pinned = digest(output / "summary.json")
    assert verify(tmp_path, output, pinned)["semantic_reconstruction_verified"]
    (output / "daily/20240102.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path, output, pinned)
    with pytest.raises(ValueError, match="pinned summary"):
        verify(tmp_path, output, "0" * 64)
