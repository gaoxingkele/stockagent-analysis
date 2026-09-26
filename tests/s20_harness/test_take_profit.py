import numpy as np
import pytest

from research.s20_harness.take_profit import first_take_profit, pool_take_profit


def _path(high, low=None, close=None, n=20, entry=100.0):
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    for i, v in high.items():
        h[i] = v
    for i, v in (low or {}).items():
        l[i] = v
    if close:
        for i, v in close.items():
            c[i] = v
    return first_take_profit(entry, h, l, c, take_profit=0.10, buy_cost=0.0, sell_cost=0.0)


def test_t_plus_one_cannot_take_profit_on_buy_day():
    out = _path(high={0: 112.0}, close={19: 99.0})
    assert out["exit"] != "take_profit"
    assert out["profit"] is False


def test_take_profit_on_day_two_does_not_need_full_20d():
    out = _path(high={1: 111.0}, close={19: 90.0})
    assert out["exit"] == "take_profit"
    assert out["exit_day"] == 2
    assert out["profit"] is True
    assert out["net"] == pytest.approx(0.10)


def test_same_day_tp_and_drawdown_stays_ex_ante_with_bounds():
    out = _path(high={2: 111.0}, low={2: 90.0})
    assert out["exit"] == "ambiguous_same_day"
    assert out["keep_ex_ante"] is True
    assert out["profit"] is None
    assert out["bound_optimistic_net"] == pytest.approx(0.10)


def test_horizon_fallback_when_target_never_prints():
    out = _path(high={}, close={19: 103.0})
    assert out["exit"] == "horizon"
    assert out["exit_day"] == 20
    assert out["profit"] is True
    assert out["net"] == pytest.approx(0.03)


def test_paired_class_follows_profit_and_matched_drawdown():
    from research.s20_harness.take_profit import paired_class
    assert paired_class(dict(profit=True, path_risk=False)) == "A"
    assert paired_class(dict(profit=True, path_risk=True)) == "B"
    assert paired_class(dict(profit=False, path_risk=False)) == "C"
    assert paired_class(dict(profit=False, path_risk=True)) == "D"
    assert paired_class(dict(profit=None, path_risk=True)) is None


def test_user_pairs_15_with_minus_10_and_25_with_minus_15():
    from research.s20_harness.take_profit import TARGET_RISK_PAIRS
    assert TARGET_RISK_PAIRS == ((0.15, 0.10), (0.25, 0.15))


def test_minus_five_is_tighter_than_user_band_10_to_15():
    from research.s20_harness.take_profit import (
        PRIMARY_DRAWDOWN, USER_DRAWDOWN_BAND, breached_drawdown, window_mae,
    )
    mae = window_mae(100.0, np.array([97.0, 92.0, 96.0]))
    assert mae == pytest.approx(-0.08)
    assert breached_drawdown(mae, 0.05) is True
    assert breached_drawdown(mae, 0.10) is False
    assert USER_DRAWDOWN_BAND == (0.10, 0.15)
    assert PRIMARY_DRAWDOWN == pytest.approx(0.10)
    assert not breached_drawdown(-0.12, 0.15)
    assert breached_drawdown(-0.12, 0.10)


def test_silence_is_max_gain_below_eight_without_minus_ten():
    from research.s20_harness.take_profit import is_silent, window_max_gain
    entry = 100.0
    quiet = np.array([101.0, 104.0, 107.0, 103.0])
    mover = np.array([101.0, 108.0, 110.0])
    assert window_max_gain(entry, quiet) == pytest.approx(0.07)
    assert is_silent(window_max_gain(entry, quiet), False) is True
    assert is_silent(window_max_gain(entry, quiet), True) is False
    assert is_silent(0.08, False) is False
    assert is_silent(window_max_gain(entry, mover), False) is False
    assert is_silent(float("nan"), False) is False


def test_specified_rise_is_take_profit_not_horizon_grind():
    from research.s20_harness.take_profit import specified_rise
    high = np.full(20, 101.0)
    high[1] = 116.0
    close = np.full(20, 100.0)
    close[19] = 90.0
    hit = first_take_profit(100.0, high, np.full(20, 99.0), close, take_profit=0.15, buy_cost=0.0, sell_cost=0.0)
    grind_close = np.full(20, 100.0)
    grind_close[19] = 103.0
    grind = first_take_profit(100.0, np.full(20, 104.0), np.full(20, 99.0), grind_close,
                              take_profit=0.15, buy_cost=0.0, sell_cost=0.0)
    assert specified_rise(hit) is True
    assert hit["exit"] == "take_profit"
    assert specified_rise(grind) is False
    assert grind["exit"] == "horizon" and grind["profit"] is True


def test_pool_counts_take_profit_as_profit_even_if_later_close_is_red():
    hit = _path(high={3: 112.0}, close={19: 80.0})
    miss = _path(high={}, close={19: 80.0})
    stats = pool_take_profit([hit, miss])
    assert stats["profit_rate"] == pytest.approx(0.5)
    assert stats["take_profit_exit_rate"] == pytest.approx(0.5)
