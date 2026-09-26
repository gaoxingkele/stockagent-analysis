from research.s20_harness.execution import (
    decide_open_buy,
    market_horizon,
    ohlc_path_bounds,
    t_plus_one_can_sell,
)
from research.s20_harness.labels import label_p_track
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize("price", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_open_is_not_fillable(price):
    assert not decide_open_buy(price).filled


@pytest.mark.parametrize("upper,lower", [(float("nan"), 9), (float("inf"), 9), ("bad", 9), (8, 9), (11, 0)])
def test_invalid_limit_fields_do_not_allow_fills(upper, lower):
    result = decide_open_buy(10, limit_up=upper, limit_down=lower)
    assert not result.filled and result.reason == "invalid_limit_data"


def test_t_plus_one_forbids_sell_on_buy_day():
    cal = [f"{20260105 + i}" for i in range(22)]
    buy = cal[1]
    assert t_plus_one_can_sell(buy, buy, cal) is False
    assert t_plus_one_can_sell(buy, cal[2], cal) is True
    assert t_plus_one_can_sell(buy, cal[20], cal) is True


def test_suspension_does_not_extend_market_horizon():
    cal = [f"{20260105 + i}" for i in range(22)]
    info = market_horizon(cal, cal[0], horizon=20)
    assert info["entry_date"] == cal[1]
    assert info["horizon_end"] == cal[20]
    assert len(info["window"]) == 20
    stock_dates = [d for d in cal if d != cal[4]]
    n = len(stock_dates)
    daily = pd.DataFrame({
        "ts_code": ["S.SZ"] * n,
        "trade_date": stock_dates,
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.0),
    })
    labeled = label_p_track(daily, cal, cal[0], "S.SZ", buy_cost=0.0, sell_cost=0.0)
    assert labeled["horizon_end"] == cal[20]
    assert labeled["horizon_end"] != stock_dates[stock_dates.index(cal[0]) + 20]


def test_open_fill_ignores_final_day_pct_chg():
    limit_up, open_px = 110.0, 101.0
    a = decide_open_buy(open_px, limit_up=limit_up, pct_chg=9.9)
    b = decide_open_buy(open_px, limit_up=limit_up, pct_chg=0.5)
    assert a == b
    assert a.filled is True
    assert a.used_pct_chg is False
    blocked = decide_open_buy(110.0, limit_up=110.0, pct_chg=0.0)
    assert blocked.filled is False
    assert blocked.reason == "limit_up_open"
    halted = decide_open_buy(101.0, suspended=True, pct_chg=2.0)
    assert halted.filled is False
    assert halted.reason == "suspended"


def test_ambiguous_same_day_ohlc_stays_ex_ante_with_bounds():
    bounds = ohlc_path_bounds(100.0, 120.0, 89.0)
    assert bounds["ambiguous"] is True
    assert bounds["keep_ex_ante"] is True
    assert bounds["drop_using_future_path"] is False
    assert bounds["bound_optimistic"] == "up_first"
    assert bounds["bound_pessimistic"] == "down_first"
    clear = ohlc_path_bounds(100.0, 121.0, 95.0)
    assert clear["ambiguous"] is False
    assert clear["keep_ex_ante"] is True


@pytest.mark.parametrize('dates',[
    ['20240102','20240102','20240103'],['20240103','20240102'],
    ['20240230','20240301'],['2024012','20240103'],[]])
def test_invalid_calendar_cannot_authorize_horizon_or_t1(dates):
    with pytest.raises(ValueError): market_horizon(dates,'20240102',horizon=1)
    with pytest.raises(ValueError): t_plus_one_can_sell('20240102','20240103',dates)


@pytest.mark.parametrize('horizon',[True,1.5,0,-1])
def test_horizon_requires_positive_integer(horizon):
    with pytest.raises(ValueError): market_horizon(['20240102','20240103'],'20240102',horizon=horizon)


@pytest.mark.parametrize('args',[
    (float('nan'),120,90),(100,float('inf'),90),(100,120,float('nan')),
    (0,120,90),(100,120,-1),(100,120,90,1.,.9),(100,120,90,1.2,1.),
    (100,120,90,float('nan'),.9)])
def test_invalid_paths_cannot_be_classified_as_neither(args):
    with pytest.raises(ValueError): ohlc_path_bounds(*args)
