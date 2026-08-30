import json
import hashlib
from pathlib import Path

import pandas as pd

import daily_dashboard as dashboard


def test_frozen_pool_a_contract_matches_runtime_constants():
    contract = json.loads(
        (Path(__file__).parents[1] / "config" / "pool_a_r20_target_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["contract_version"] == "pool-a-r20-target-v1"
    assert contract["selection"]["any_of"]["predicted_t20_close_return_pct_gte"] == dashboard.POOL_A_R20_MIN
    assert contract["selection"]["any_of"]["predicted_max_gain_within_20d_pct_gte"] == dashboard.POOL_A_MAX_GAIN_MIN
    assert contract["selection"]["and"]["predicted_max_adverse_excursion_20d_pct_gte"] == dashboard.POOL_A_DD_MIN
    assert contract["industry_cap"] is None
    assert contract["top_n"] is None
    root = Path(__file__).parents[1]
    for model in contract["models"].values():
        model_path = root / model["path"]
        if model_path.exists():
            assert hashlib.sha256(model_path.read_bytes()).hexdigest().upper() == model["sha256"]


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": [
            "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ",
            "000005.SZ", "000006.SZ", "000007.SZ", "920001.BJ",
        ],
        "industry": ["软件服务"] * 8,
        "r20_pred": [31.0, 30.0, 29.0, 28.0, 27.0, 25.0, 24.9, 40.0],
        "pred_max_gain_20": [10.0] * 8,
        "pred_max_dd_20": [-14.0, -12.0, -10.0, -8.0, -15.1, -15.0, -5.0, -5.0],
        "ratio": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "v7c_recommend": [False] * 8,
        # These deliberately disagree with the old reversal-style rules.
        "pyr_velocity_20_60": [1.0] * 8,
        "f1_neg1": [1.0] * 8,
        "f2_pos1": [1.0] * 8,
        "is_zombie": [True] * 8,
        "industry_mom_60d_rank": [0.01] * 8,
    })


def test_pool_a_uses_absolute_r20_and_drawdown_targets_only():
    pool, stats = dashboard._build_pool_a(_frame())

    assert list(pool["ts_code"]) == [
        "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000006.SZ"
    ]
    assert stats["n_pool"] == 5
    assert pool["a_recommend"].all()
    assert not pool["v7c_recommend"].any()


def test_pool_a_has_no_industry_cap_or_topn_and_excludes_bj():
    df = pd.DataFrame({
        "ts_code": [f"300{i:03d}.SZ" for i in range(30)] + ["920999.BJ"],
        "industry": ["同一行业"] * 31,
        "r20_pred": list(range(60, 30, -1)) + [100.0],
        "pred_max_gain_20": [10.0] * 31,
        "pred_max_dd_20": [-10.0] * 31,
        "ratio": [2.0] * 31,
        "v7c_recommend": [False] * 31,
    })

    pool, stats = dashboard._build_pool_a(df)

    assert len(pool) == 30
    assert stats["n_pool"] == 30
    assert pool["ts_code"].str.endswith((".SH", ".SZ")).all()
    assert "920999.BJ" not in set(pool["ts_code"])


def test_pool_a_accepts_intrawindow_gain_sleeve_when_terminal_r20_is_lower():
    df = pd.DataFrame({
        "ts_code": ["600001.SH", "600002.SH"],
        "industry": ["趋势", "趋势"],
        "r20_pred": [12.0, 12.0],
        "pred_max_gain_20": [25.0, 24.9],
        "pred_max_dd_20": [-15.0, -5.0],
        "ratio": [1.0, 1.0],
        "v7c_recommend": [False, False],
    })

    pool, _ = dashboard._build_pool_a(df)

    assert list(pool["ts_code"]) == ["600001.SH"]
