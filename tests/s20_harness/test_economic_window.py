import pandas as pd
import pytest

from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.economic_window import value_window


def quotes(prices, dates=None):
    dates = dates or ["20240102", "20240103", "20240104"]
    return pd.DataFrame({key: prices for key in ("open", "high", "low", "close")}, index=dates)


def cash_event():
    return Distribution("cash", "20240102", "20240103", "20240101", 1.,
                        pay_date="20240105", beneficiary_scope="existing_shareholders_verified")


def test_cash_ex_drop_is_not_economic_loss_but_not_spendable():
    bars, state = value_window(quotes([10., 9., 9.]), [cash_event()])
    assert bars.close.tolist() == [10., 10., 10.]
    assert state["cash"] == 0 and state["receivable_cash"] == 1
    assert state["settlement_pending"]
    assert not state["formal_training_eligible"]


def test_bonus_entitlement_does_not_become_sellable_early():
    event = Distribution("bonus", "20240102", "20240103", "20240101", bonus_per_share=1.,
                         bonus_list_date="20240104", beneficiary_scope="existing_shareholders_verified")
    bars, state = value_window(quotes([10., 5., 5.]), [event])
    assert bars.close.tolist() == [10., 10., 10.]
    assert bars.sellable_shares.tolist() == [1., 1., 2.]
    assert bars.pending_bonus_shares.tolist() == [0., 1., 0.]
    assert not state["settlement_pending"]


def test_ex_date_entry_cannot_receive_previous_holder_cash():
    bars, state = value_window(quotes([9., 9.], ["20240103", "20240104"]), [cash_event()])
    assert bars.close.tolist() == [9., 9.]
    assert state["ledger"] == []


def test_missing_record_day_and_duplicate_events_rejected():
    event = cash_event()
    with pytest.raises(ValueError, match="duplicate corporate"):
        value_window(quotes([10., 9., 9.]), [event, event])
    with pytest.raises(ValueError, match="record-date"):
        value_window(quotes([10., 9.], ["20240101", "20240103"]), [event])


def test_p_labels_use_entitlements_and_do_not_charge_sell_fee_on_cash():
    from research.s20_harness.labels import label_p_track
    raw = quotes([10., 9., 9.]).reset_index(names="trade_date")
    raw["ts_code"] = "X.SZ"
    calendar = ["20240101", "20240102", "20240103", "20240104"]
    unadjusted = label_p_track(raw, calendar, "20240101", "X.SZ", horizon=3, buy_cost=0., sell_cost=0.)
    assert unadjusted["b5"] is True
    adjusted = label_p_track(raw, calendar, "20240101", "X.SZ", horizon=3,
                             buy_cost=0., sell_cost=.01, distributions=[cash_event()])
    assert adjusted["b5"] is False
    assert adjusted["terminal_net"] == pytest.approx(-.009)
    assert adjusted["settlement_pending"]
    assert not adjusted["formal_training_eligible"]


def test_o_economic_target_counts_entitlement_without_rewriting_compat():
    from research.s20_harness.labels import label_o_track
    event = Distribution("large_cash", "20240102", "20240103", "20240101", 3.,
                         pay_date="20240105", beneficiary_scope="existing_shareholders_verified")
    raw = quotes([10., 7., 9.1]).reset_index(names="trade_date")
    raw["ts_code"] = "X.SZ"
    calendar = ["20240101", "20240102", "20240103", "20240104"]
    raw_result = label_o_track(raw, calendar, "20240101", "X.SZ", horizon=3, mode="calendar_pit_v4")
    economic = label_o_track(raw, calendar, "20240101", "X.SZ", horizon=3,
                             mode="calendar_pit_v4", distributions=[event])
    assert raw_result["down_risk"] == 1
    assert economic["immediate"] == 1
    assert economic["max_gain20"] == pytest.approx(21.)
    assert economic["settlement_pending"]
    with pytest.raises(ValueError, match="compatibility labels are frozen"):
        label_o_track(raw, calendar, "20240101", "X.SZ", horizon=3, distributions=[event])
