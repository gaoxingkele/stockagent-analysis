"""Clocked reference exit requests, isolated from production and fill inference.

One aggregate outcome per scheduled slot. Slot times identify research price
events, not exchange order-submission deadlines or queue/capacity guarantees.
Unobserved slots cannot be skipped to cherry-pick a later price.
"""
from dataclasses import dataclass, replace

import pandas as pd

from .exit_book import ExitLot, apply_exit, validate_calendar
from .label_availability import _instant


@dataclass(frozen=True)
class ExitPolicy:
    policy_id: str
    frozen_at: str
    kind: str = "due_close_then_each_market_open_reference_v1"


@dataclass(frozen=True)
class ExitRequest:
    order_id: str
    policy: ExitPolicy
    lot: ExitLot
    submitted_at: str
    slot_at: str
    phase: str
    calendar: tuple[str, ...]


def request_exit(lot, policy, *, order_id, submitted_at, calendar):
    validate_calendar(calendar)
    if policy.kind != "due_close_then_each_market_open_reference_v1" or not policy.policy_id:
        raise ValueError("unsupported/missing exit policy")
    frozen, submitted = _instant(policy.frozen_at), _instant(submitted_at)
    if lot.exit_policy_id is not None and (lot.exit_policy_id != policy.policy_id or
            lot.exit_policy_frozen_at != frozen.isoformat()):
        raise ValueError("exit policy changed during pending lifecycle")
    if lot.last_attempt_date is not None and lot.exit_policy_id is None:
        raise ValueError("previous attempts lack policy binding")
    if lot.exit_calendar is not None and lot.exit_calendar != tuple(calendar):
        raise ValueError("exit calendar changed during pending lifecycle")
    if frozen >= submitted:
        raise ValueError("policy not frozen before request")
    if not order_id or order_id in lot.attempt_ids:
        raise ValueError("missing/duplicate order")
    if lot.remaining_quantity <= 0:
        raise ValueError("closed position")
    if lot.last_attempt_date is None:
        day, phase, time = lot.due_date, "due_close", "15:00"
    else:
        if lot.last_attempt_date not in calendar:
            raise ValueError("last attempt outside calendar")
        index = calendar.index(lot.last_attempt_date) + 1
        if index >= len(calendar):
            raise ValueError("next session unavailable")
        day, phase, time = calendar[index], "retry_open", "09:30"
    if day not in calendar:
        raise ValueError("due date outside calendar")
    if lot.explicit_sellable_from is not None and day < lot.explicit_sellable_from:
        raise ValueError('credited shares not yet sellable')
    slot = pd.Timestamp(day + " " + time, tz="Asia/Shanghai").tz_convert("UTC")
    if submitted.tz_convert("Asia/Shanghai").strftime("%Y%m%d") != day or submitted >= slot:
        raise ValueError("request must precede its next scheduled slot on the same date")
    return ExitRequest(order_id, policy, lot, submitted.isoformat(), slot.isoformat(), phase, tuple(calendar))


def settle_exit(lot, request, *, outcome_at, received_at, processing_at,
                filled_quantity, price=None, fee=0.0, reason, evidence_id):
    """Apply supplied aggregate result after receipt; timestamps are checked,
    not authenticated. External adapters must independently validate evidence.
    """
    if lot != request.lot:
        raise ValueError("stale order position snapshot")
    # Reconstruct rather than trusting an editable request's slot/phase.
    rebuilt = request_exit(lot, request.policy, order_id=request.order_id,
                           submitted_at=request.submitted_at, calendar=list(request.calendar))
    if rebuilt != request:
        raise ValueError("request schedule mismatch")
    outcome, received, processing = map(_instant, (outcome_at, received_at, processing_at))
    if outcome != _instant(request.slot_at):
        raise ValueError("outcome outside requested slot")
    if not outcome <= received <= processing:
        raise ValueError("receipt/processing precedes outcome")
    day = outcome.tz_convert("Asia/Shanghai").strftime("%Y%m%d")
    state, event = apply_exit(lot, attempt_id=request.order_id, position_id=lot.position_id,
        ts_code=lot.ts_code, trade_date=day, calendar=list(request.calendar),
        filled_quantity=filled_quantity, price=price, fee=fee, reason=reason, evidence_id=evidence_id)
    state = replace(state, exit_policy_id=request.policy.policy_id,
                    exit_policy_frozen_at=_instant(request.policy.frozen_at).isoformat(),
                    exit_calendar=request.calendar)
    event.update(policy_id=request.policy.policy_id, policy_kind=request.policy.kind,
                 submitted_at=request.submitted_at, outcome_at=outcome.isoformat(),
                 received_at=received.isoformat(), processed_at=processing.isoformat(),
                 phase=request.phase, timestamp_authenticity_proven=False)
    return state, event
