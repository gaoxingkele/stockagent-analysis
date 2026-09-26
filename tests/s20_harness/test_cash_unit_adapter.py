import pandas as pd
import pytest

from research.s20_harness.cash_unit_adapter import normalize_units
from research.s20_harness.distribution_adapter import row_fingerprint


def test_exact_event_unit_conversion_preserves_source_and_no_spillover():
    frame = pd.DataFrame([{"normalized_event_id": key, "ts_code": "689009.SH", "record_date": "20251030",
                           "ex_date": "20251031", "cash_div_tax": 4.2073} for key in ("reviewed", "other")])
    review = {"row_sha256": row_fingerprint(frame.iloc[0]), "ts_code": "689009.SH", "record_date": "20251030",
              "ex_date": "20251031", "source": "issuer", "source_unit": "share", "quote_unit": "CDR",
              "quoted_units_per_source_unit": 10, "source_gross_cash": 4.2073, "gross_cash_per_quote_unit": .42073}
    output, lineage = normalize_units(frame, {"reviewed": review})
    assert output.gross_cash_per_quote_unit.iloc[0] == .42073
    assert pd.isna(output.gross_cash_per_quote_unit.iloc[1])
    pd.testing.assert_frame_equal(output[frame.columns], frame)
    assert len(lineage) == 1
    frame.loc[0, "cash_div_tax"] = 42.073
    with pytest.raises(ValueError, match="row changed"):
        normalize_units(frame, {"reviewed": review})
