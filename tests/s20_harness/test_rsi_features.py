import numpy as np
import pytest

from research.s20_harness.rsi_features import (
    RSI_PERIOD,
    compute,
    rsi_from_averages,
    shuffle_time,
    split_close_moves,
    wilder_avg,
)


def _window(kind, n=21, names=3):
    t = np.arange(n, dtype=float)[:, None]
    if kind == "up":
        close = 100 + t
    elif kind == "down":
        close = 120 - t
    else:
        close = np.where(t % 2 == 0, 100.0, 101.0)
    high = close + 0.5
    low = close - 0.5
    return np.repeat(close, names, axis=1), np.repeat(high, names, axis=1), np.repeat(low, names, axis=1)


def test_all_gains_map_to_one_all_losses_to_zero():
    c, h, l = _window("up")
    up = compute(c, h, l)
    assert np.all(up["rsi14_wilder"] == pytest.approx(1.0))
    assert np.all(up["rsi_avg_gain14"] > 0)
    assert np.all(up["rsi_avg_loss14"] == pytest.approx(0.0))
    c, h, l = _window("down")
    down = compute(c, h, l)
    assert np.all(down["rsi14_wilder"] == pytest.approx(0.0))
    assert np.all(down["rsi_avg_loss14"] > 0)
    assert np.all(down["rsi_avg_gain14"] == pytest.approx(0.0))


def test_ratio_is_bounded_and_zero_move_is_neutral():
    assert rsi_from_averages(np.array([0.0]), np.array([0.0])) == pytest.approx(0.5)
    assert 0 <= float(np.asarray(rsi_from_averages(np.array([0.02]), np.array([0.01]))).reshape(-1)[0]) <= 1
    c, h, l = _window("flat")
    out = compute(c, h, l)
    assert np.all((out["rsi14_wilder"] > 0.3) & (out["rsi14_wilder"] < 0.7))


def test_wilder_memory_is_not_a_plain_mean():
    values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert wilder_avg(values, period=3) == pytest.approx((16 / 3 + 5) / 3)
    assert wilder_avg(values, period=3) != pytest.approx(values[-3:].mean())


def test_lookback_only_and_shuffle_breaks_trend():
    c, h, l = _window("up")
    sequential = compute(c, h, l)["rsi14_wilder"][0]
    rng = np.random.default_rng(0)
    shuffled = compute(*shuffle_time(c, h, l, rng))["rsi14_wilder"][0]
    assert sequential == pytest.approx(1.0)
    assert shuffled < sequential
    gain, loss, _ = split_close_moves(c)
    assert gain.shape[0] == c.shape[0] - 1
    with pytest.raises(ValueError):
        compute(c[:RSI_PERIOD], h[:RSI_PERIOD], l[:RSI_PERIOD])
