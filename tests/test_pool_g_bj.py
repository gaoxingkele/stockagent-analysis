import numpy as np
import pandas as pd

import daily_dashboard as dashboard


def _market_frame():
    sh = pd.DataFrame({
        "ts_code": [f"600{i:03d}.SH" for i in range(100)],
        "r20_pred": np.arange(1000, 1100, dtype=float),
        "buy_r20_score": np.linspace(0, 100, 100),
        "ratio": 2.0,
        "industry": "沪深行业",
        "is_zombie": False,
        "industry_mom_60d_rank": 0.5,
        "v7c_recommend": False,
    })
    bj = pd.DataFrame({
        "ts_code": [f"920{i:03d}.BJ" for i in range(100)],
        "r20_pred": np.arange(100, dtype=float),
        "buy_r20_score": np.linspace(0, 100, 100),
        "ratio": np.linspace(1, 3, 100),
        "industry": [f"北交行业{i % 6}" for i in range(100)],
        "is_zombie": False,
        "industry_mom_60d_rank": 0.5,
        "v7c_recommend": False,
    })
    return pd.concat([sh, bj], ignore_index=True)


def test_pool_g_ranks_only_inside_bj_market():
    df = _market_frame()

    pool, stats = dashboard._build_bj_pool(df)

    assert stats["universe"] == 100
    assert stats["eligible"] == 100
    assert stats["pre_cap"] == 6
    assert len(pool) == 6
    assert pool["ts_code"].str.endswith(".BJ").all()
    assert set(pool["ts_code"]) == {f"920{i:03d}.BJ" for i in range(94, 100)}
    assert pool["v7c_recommend"].all()


def test_pool_g_applies_available_risk_gates():
    df = _market_frame()
    df.loc[df["ts_code"] == "920099.BJ", "is_zombie"] = True
    df.loc[df["ts_code"] == "920098.BJ", "industry_mom_60d_rank"] = 0.05

    pool, stats = dashboard._build_bj_pool(df)

    assert stats["pre_cap"] == 4
    assert "920099.BJ" not in set(pool["ts_code"])
    assert "920098.BJ" not in set(pool["ts_code"])
