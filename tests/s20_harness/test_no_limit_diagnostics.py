import pandas as pd
import pytest

from research.s20_harness.no_limit_diagnostics import classify


def test_sentinel_alone_never_authorizes_unrestricted_trading():
    frame = pd.DataFrame({"ts_code": ["920001.BJ", "920002.BJ", "600001.SH"],
                          "trade_date": ["20240102"] * 3, "up_limit": [99999.99] * 3, "down_limit": [0.] * 3})
    basic = pd.DataFrame({"ts_code": frame.ts_code, "list_date": ["20240102", "20230101", "20240102"]})
    result = classify(frame, basic)
    assert result.diagnostic.str.startswith("consistent").tolist() == [True, False, False]
    assert not result.unrestricted_trading_authorized.any()
    with pytest.raises(ValueError, match="unique"):
        classify(frame, pd.concat([basic, basic]))
