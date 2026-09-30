import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stockagent_analysis.s20_pure import (  # noqa: E402
    ExitRule,
    PureConfig,
    exit_return,
    first_touch,
    natr,
    select,
    three_state,
)

RULE = ExitRule("U15D10", 15.0, 10.0, 0.40, shakeout=5.0)


def test_first_touch_is_one_based_and_zero_when_untouched():
    path = np.array([[0.01, -0.02], [0.16, -0.05], [0.20, -0.11]])
    np.testing.assert_array_equal(first_touch(path, 0.15, up=True), [2, 0])
    np.testing.assert_array_equal(first_touch(path, -0.10, up=False), [0, 3])


def test_three_state_orders_barriers_and_flags_shakeout():
    up = [3, 0, 5, 0, 4, 4]
    dn = [0, 2, 2, 0, 4, 0]
    shake = [0, 1, 1, 0, 2, 2]
    got = three_state(up, dn, shake)
    assert list(got) == ["pure_up", "pure_down", "pure_down", "chop", "ambiguous", "dirty_up"]


def test_three_state_respects_horizon():
    assert list(three_state([12], [0], horizon=10)) == ["chop"]


def test_exit_return_books_take_profit_stop_and_time_exit():
    r = exit_return([3, 0, 0, 4], [0, 2, 0, 4], [1.0, -3.0, 2.5, 0.0], RULE, cost_pct=0.3)
    np.testing.assert_allclose(r, [14.7, -10.3, 2.2, -10.3])


def test_natr_matches_manual_true_range():
    d = pd.DataFrame({
        "ts_code": ["A"] * 3, "trade_date": ["1", "2", "3"],
        "high": [11.0, 12.0, 11.0], "low": [9.0, 10.0, 9.5],
        "close": [10.0, 11.0, 10.0], "pre_close": [10.0, 10.0, 11.0],
    })
    got = natr(d, window=2)
    assert np.isnan(got.iloc[0])
    assert np.isclose(got.iloc[1], ((2.0 + 2.0) / 2) / 11.0)
    assert np.isclose(got.iloc[2], ((2.0 + 1.5) / 2) / 10.0)


def test_select_drops_highest_amplitude_then_takes_top_k():
    n = 10
    frame = pd.DataFrame({
        "trade_date": ["20260101"] * n,
        "ts_code": [f"S{i}" for i in range(n)],
        "stage1_probability": np.linspace(0.9, 0.1, n),
        "natr14": [0.09, 0.01, 0.08, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, np.nan],
    })
    cfg = PureConfig(pool_size=8, top_k=3)
    out = select(frame, cfg)
    # pool = S0..S7; cap 40% drops natr pct-rank > 0.6 inside the pool: S0, S2, S6, S7
    assert list(out.ts_code) == ["S1", "S3", "S4"]
    assert list(out.list_rank) == [1, 2, 3]
    assert set(out.rule) == {"U15D10"}


def test_frozen_contract_matches_default_config():
    path = Path(__file__).parents[1] / "config/s20_pure_v1.json"
    if not path.exists():
        return
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["status"] == "shadow_preregistered"
    assert contract["funnel"] == json.loads(json.dumps(PureConfig().to_dict()))
    assert [r["amplitude_cap"] for r in contract["funnel"]["rules"]] == [0.4, 0.2]
    assert contract["stage1"]["reproduction_vs_run_v1"]["daily_top20_overlap_mean"] >= 0.95


def _day(inds, scores=None):
    n = len(inds)
    return pd.DataFrame({
        "trade_date": ["20260101"] * n,
        "ts_code": [f"S{i}" for i in range(n)],
        "industry": inds,
        "stage1_probability": scores if scores is not None else np.linspace(0.9, 0.1, n),
        "natr14": np.linspace(0.01, 0.02, n),
    })


def test_industry_expand_keeps_top_k_and_adds_substitutes_for_the_excess():
    frame = _day(["chip"] * 6 + ["bank", "chip", "food", "auto", "oil"])
    cfg = PureConfig(pool_size=100, top_k=6, rules=(ExitRule("X", 15.0, 10.0, 0.0),), primary_rule="X")
    out = select(frame, cfg, industry_cap=4, industry_mode="expand")
    # top 6 are all chip -> 2 over the cap -> 2 substitutes from other industries, chip #7 skipped
    assert list(out.ts_code) == ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S8"]
    assert list(out.fill) == ["top"] * 6 + ["industry_substitute"] * 2


def test_industry_replace_enforces_a_strict_cap():
    frame = _day(["chip"] * 6 + ["bank", "chip", "food", "auto", "oil"])
    cfg = PureConfig(pool_size=100, top_k=6, rules=(ExitRule("X", 15.0, 10.0, 0.0),), primary_rule="X")
    out = select(frame, cfg, industry_cap=4, industry_mode="replace")
    assert list(out.ts_code) == ["S0", "S1", "S2", "S3", "S6", "S8"]


def test_select_safe_keeps_only_the_calm_part_of_the_market():
    from stockagent_analysis.s20_pure import SafeConfig, select_safe
    frame = _day(["a", "b", "c", "d", "e"], scores=[0.9, 0.8, 0.7, 0.6, 0.5])
    frame["natr14"] = [0.09, 0.01, 0.02, 0.08, 0.03]
    out = select_safe(frame, SafeConfig(natr_pct_max=0.6, top_k=2))
    # natr pct: S1 .2, S2 .4, S4 .6 kept; ranked by stage1 -> S1, S2
    assert list(out.ts_code) == ["S1", "S2"]


def test_safe_contract_matches_default_config():
    from stockagent_analysis.s20_pure import SafeConfig
    path = Path(__file__).parents[1] / "config/s20_pure_v1_1_safe.json"
    if not path.exists():
        return
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["status"] == "shadow_preregistered"
    assert contract["list"] == json.loads(json.dumps(SafeConfig().to_dict()))


def test_market_valve_levels_use_frozen_cutoffs():
    from stockagent_analysis.market_valve import ValveConfig, daily_breadth, valve_levels
    daily = pd.DataFrame({
        "ts_code": ["600000.SH", "300001.SZ", "300002.SZ"] * 6,
        "trade_date": [f"2026010{i}" for i in range(1, 7) for _ in range(3)],
        "pct_chg": [-10.0, -10.0, -20.0] * 6,   # per day: 600000 and 300002 are limit-down, 300001 is not
    })
    b = daily_breadth(daily)
    assert list(b.limit_down) == [2] * 6
    lv = valve_levels(b, ValveConfig(yellow_at=5, orange_at=9, red_at=11))
    assert list(lv.limit_down_5d.iloc[4:]) == [10.0, 10.0]
    assert list(lv.level) == ["unknown"] * 4 + ["orange", "orange"]
    assert list(valve_levels(b, ValveConfig(yellow_at=5, orange_at=9, red_at=10)).level)[4:] == ["red", "red"]


def test_valve_contract_matches_config():
    from stockagent_analysis.market_valve import VALVE_CONTRACT_PATH, ValveConfig
    if not VALVE_CONTRACT_PATH.exists():
        return
    c = json.loads(VALVE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert c["status"] == "monitor_preregistered"
    assert c["config"] == ValveConfig().to_dict()
