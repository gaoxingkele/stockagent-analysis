"""Unit-notional fill and market-calendar horizon helpers. No portfolio engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import math


@lru_cache(maxsize=32)
def _validated_calendar(dates):
    if not dates or list(dates)!=sorted(set(dates)):
        raise ValueError('nonempty unique chronological market calendar required')
    for date in dates:
        if len(date)!=8 or datetime.strptime(date,'%Y%m%d').strftime('%Y%m%d')!=date:
            raise ValueError('canonical market calendar dates required')
    return dates


@dataclass(frozen=True)
class FillDecision:
    filled: bool
    price: float | None
    reason: str
    used_pct_chg: bool = False


def market_horizon(calendar: list[str], signal_date: str, horizon: int = 20):
    """D+1 open date through D+{horizon} close date on the *market* calendar.

    Stock suspension does not stretch the forecast horizon: the end date is
    calendar[i+horizon] even if the name has fewer bars.
    """
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    dates = _validated_calendar(tuple(str(d) for d in calendar))
    try:
        i = dates.index(str(signal_date))
    except ValueError as exc:
        raise ValueError(f"signal_date {signal_date} not on calendar") from exc
    if i + horizon >= len(dates):
        raise ValueError("calendar shorter than D+horizon")
    entry_date = dates[i + 1]
    horizon_end = dates[i + horizon]
    window = list(dates[i + 1 : i + 1 + horizon])
    if len(window) != horizon:
        raise ValueError("window length is not the market-session horizon")
    return {"entry_date": entry_date, "horizon_end": horizon_end, "window": window}


def t_plus_one_can_sell(buy_date: str, sell_date: str, calendar: list[str]) -> bool:
    """A share bought at D+1 open cannot be sold on the buy day."""
    dates = _validated_calendar(tuple(str(d) for d in calendar))
    buy, sell = str(buy_date), str(sell_date)
    if buy not in dates or sell not in dates:
        return False
    return dates.index(sell) > dates.index(buy)


def decide_open_buy(
    open_price,
    *,
    suspended: bool = False,
    limit_up=None,
    limit_down=None,
    pct_chg=None,
) -> FillDecision:
    """Open fill uses open vs halt/limit prices. ``pct_chg`` is ignored if present.

    The old book engine used the day's final pct_chg to decide whether the open
    was tradable. That is future information and is not an input here.
    """
    del pct_chg  # explicitly unused; close-to-close return is not known at open
    if suspended:
        return FillDecision(False, None, "suspended", used_pct_chg=False)
    try:
        open_px = float(open_price)
    except (TypeError, ValueError):
        return FillDecision(False, None, "no_open", used_pct_chg=False)
    if open_px <= 0 or not math.isfinite(open_px):
        return FillDecision(False, None, "no_open", used_pct_chg=False)
    try:
        limit_up = float(limit_up) if limit_up is not None else None
        limit_down = float(limit_down) if limit_down is not None else None
    except (TypeError, ValueError):
        return FillDecision(False, None, "invalid_limit_data", used_pct_chg=False)
    if any(x is not None and (not math.isfinite(x) or x <= 0) for x in (limit_up, limit_down)):
        return FillDecision(False, None, "invalid_limit_data", used_pct_chg=False)
    if limit_up is not None and limit_down is not None and limit_up < limit_down:
        return FillDecision(False, None, "invalid_limit_data", used_pct_chg=False)
    if limit_up is not None and open_px >= float(limit_up) - 1e-12:
        return FillDecision(False, None, "limit_up_open", used_pct_chg=False)
    if limit_down is not None and open_px <= float(limit_down) + 1e-12:
        return FillDecision(False, None, "limit_down_open", used_pct_chg=False)
    return FillDecision(True, open_px, "filled_open", used_pct_chg=False)


def ohlc_path_bounds(entry, high, low, up_ratio=1.20, down_ratio=0.90) -> dict:
    """Same-day OHLC order is unknown. Keep the ex-ante name; report bounds."""
    entry = float(entry)
    high = float(high)
    low = float(low)
    up_ratio,down_ratio=float(up_ratio),float(down_ratio)
    if (not all(math.isfinite(v) and v>0 for v in (entry,high,low,up_ratio,down_ratio))
        or up_ratio<=1 or down_ratio>=1):
        raise ValueError('finite positive prices and meaningful up/down barriers required')
    if low > high:
        raise ValueError("low above high")
    hit_up = high >= entry * up_ratio
    hit_down = low < entry * down_ratio
    ambiguous = bool(hit_up and hit_down)
    optimistic = "up_first" if hit_up else ("down_first" if hit_down else "neither")
    pessimistic = "down_first" if hit_down else ("up_first" if hit_up else "neither")
    if ambiguous:
        optimistic, pessimistic = "up_first", "down_first"
    return {
        "ambiguous": ambiguous,
        "keep_ex_ante": True,
        "hit_up": hit_up,
        "hit_down": hit_down,
        "bound_optimistic": optimistic,
        "bound_pessimistic": pessimistic,
        "drop_using_future_path": False,
    }
