"""积分相关 schemas。"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TransactionItem(BaseModel):
    id: int
    delta: int
    reason: str
    related_result_id: int | None
    related_invite_id: int | None
    related_user_id: int | None
    note: str | None
    balance_before: int
    balance_after: int
    created_at: datetime


class TransactionListResponse(BaseModel):
    items: list[TransactionItem]
    total: int
    limit: int
    offset: int


class RechargeRequest(BaseModel):
    target_user_id: int
    amount: int   # 正数加 / 负数扣
    note: str | None = None
