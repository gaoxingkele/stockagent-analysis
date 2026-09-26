import numpy as np
import pandas as pd
import pytest

from research.s20_harness.labels import label_o_track, label_p_track


def _calendar(n=22):
    return [f"{20260105 + i}" for i in range(n)]


def _daily(ts_code, dates, open_=100.0, high=101.0, low=99.0, close=100.0):
    n = len(dates)
    def _vec(v):
        return np.full(n, v, float) if np.ndim(v) == 0 else np.asarray(v, float)
    return pd.DataFrame({
        "ts_code": [ts_code] * n,
        "trade_date": list(dates),
        "open": _vec(open_),
        "high": _vec(high),
        "low": _vec(low),
        "close": _vec(close),
    })


def test_p_track_safe_profit_class_a_on_calendar_window():
    cal = _calendar()
    n = len(cal)
    high = np.full(n, 104.0)
    low = np.full(n, 98.0)
    close = np.full(n, 100.0)
    close[20] = 106.0
    high[20] = 106.0
    daily = _daily("A.SZ", cal, open_=100.0, high=high, low=low, close=close)
    out = label_p_track(daily, cal, cal[0], "A.SZ", buy_cost=0.0, sell_cost=0.0)
    assert out["p_class"] == "A"
    assert out["up_event"] is True
    assert out["b5"] is False
    assert out["terminal_net"] > 0
    assert out["mae"] > -0.05
    assert out["first_up_day"] >= 1
    assert out["first_down_day"] >= 1
    assert out["time_to_b5"] is None
    assert out["max_drawdown"] is not None
    assert set(out) >= {"first_up_day", "first_down_day", "time_to_b5", "mae", "max_drawdown", "terminal_net"}


def test_p_track_dangerous_profit_class_b():
    cal = _calendar()
    n = len(cal)
    high = np.full(n, 106.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    low[5] = 90.0
    close[20] = 108.0
    high[20] = 108.0
    daily = _daily("B.SZ", cal, open_=100.0, high=high, low=low, close=close)
    out = label_p_track(daily, cal, cal[0], "B.SZ", buy_cost=0.0, sell_cost=0.0)
    assert out["p_class"] == "B"
    assert out["up_event"] is True
    assert out["b5"] is True
    assert out["time_to_b5"] == 5
    assert out["mae"] < -0.05


def test_p_track_safe_unprofitable_class_c_and_dangerous_loss_d():
    cal = _calendar()
    n = len(cal)
    high = np.full(n, 101.0)
    low = np.full(n, 98.0)
    close = np.full(n, 100.0)
    close[20] = 99.0
    daily_c = _daily("C.SZ", cal, open_=100.0, high=high, low=low, close=close)
    c_lab = label_p_track(daily_c, cal, cal[0], "C.SZ", buy_cost=0.0, sell_cost=0.0)
    assert c_lab["p_class"] == "C"
    assert c_lab["up_event"] is False
    assert c_lab["b5"] is False
    low_d = low.copy()
    low_d[4] = 90.0
    daily_d = _daily("D.SZ", cal, open_=100.0, high=high, low=low_d, close=close)
    d_lab = label_p_track(daily_d, cal, cal[0], "D.SZ", buy_cost=0.0, sell_cost=0.0)
    assert d_lab["p_class"] == "D"
    assert d_lab["up_event"] is False
    assert d_lab["b5"] is True


def test_unfilled_d1_keeps_recommendation_without_success_class():
    cal = _calendar()
    stock_dates = [d for d in cal if d != cal[1]]
    daily = _daily("U.SZ", stock_dates)
    out = label_p_track(daily, cal, cal[0], "U.SZ")
    assert out["recommendation_kept"] is True
    assert out["fill_status"] == "unfilled"
    assert out["p_class"] is None
    assert out["up_event"] is None
    assert out["reject_postpone_entry"] is True


def test_calendar_horizon_does_not_follow_stock_record_20_rows():
    cal = _calendar(23)
    stock_dates = [d for d in cal if d != cal[5]]
    n = len(stock_dates)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    daily = _daily("C.SZ", stock_dates, high=high, low=low, close=close)
    p_lab = label_p_track(daily, cal, cal[0], "C.SZ", buy_cost=0.0, sell_cost=0.0)
    o_stock = label_o_track(daily, cal, cal[0], "C.SZ", mode="stock_session_v3_compat")
    o_cal = label_o_track(daily, cal, cal[0], "C.SZ", mode="calendar_pit_v4")
    assert p_lab["horizon_end"] == cal[20]
    assert o_stock["horizon_end"] == stock_dates[stock_dates.index(cal[0]) + 20]
    assert o_stock["horizon_end"] != p_lab["horizon_end"]
    assert o_cal["horizon_end"] == p_lab["horizon_end"]
    assert p_lab["track"] != o_stock["track"]


def test_o_track_success_is_not_replaced_by_p_track_class():
    cal = _calendar()
    n = len(cal)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    high[2] = 121.0
    low[6] = 89.0
    close[20] = 92.0
    low[20] = 92.0
    daily = _daily("O.SZ", cal, open_=100.0, high=high, low=low, close=close)
    p_lab = label_p_track(daily, cal, cal[0], "O.SZ", buy_cost=0.0, sell_cost=0.0)
    o_lab = label_o_track(daily, cal, cal[0], "O.SZ", mode="stock_session_v3_compat")
    assert o_lab["opportunity"] == 1
    assert o_lab["immediate"] == 1
    assert p_lab["p_class"] == "D"
    assert p_lab["up_event"] is False
    assert p_lab["b5"] is True


@pytest.mark.parametrize("missing_index", [5, 20])
def test_missing_quotes_never_imputed_to_safe_profit(missing_index):
    cal = _calendar()
    daily = _daily("X.SZ", [d for i, d in enumerate(cal) if i != missing_index],
                   high=111, low=99, close=110)
    p = label_p_track(daily, cal, cal[0], "X.SZ")
    o = label_o_track(daily, cal, cal[0], "X.SZ", mode="calendar_pit_v4")
    assert p["p_class"] is None and p["up_event"] is None and p["b5"] is None
    assert not p["label_realized"] and p["recommendation_kept"]
    assert p["exit_pending"] == (missing_index == 20)
    assert o["o_class"] is None
    assert o["invalid_or_missing_dates"] == [cal[missing_index]]


def test_duplicate_quotes_are_not_silently_selected():
    cal = _calendar()
    daily = _daily("X.SZ", cal)
    daily = pd.concat([daily, daily.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="identity reconciliation"):
        label_p_track(daily, cal, cal[0], "X.SZ")


def test_invalid_bar_and_costs_rejected():
    cal = _calendar()
    daily = _daily("X.SZ", cal)
    daily.loc[4, "close"] = 200
    assert label_p_track(daily, cal, cal[0], "X.SZ")["p_class"] is None
    with pytest.raises(ValueError, match="costs"):
        label_p_track(daily, cal, cal[0], "X.SZ", sell_cost=float("nan"))
