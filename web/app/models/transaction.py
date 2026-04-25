"""积分流水表 - 强一致到 result_id 或 invite_id。"""
from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.db import Base, TimestampMixin

if TYPE_CHECKING:
    from .user import User


class TransactionReason(str, enum.Enum):
    register_bonus = "register_bonus"              # 注册赠送
    invite_new_user_bonus = "invite_new_user_bonus"  # 通过邀请码注册得 50
    invite_referrer_bonus = "invite_referrer_bonus"  # 介绍人得 100
    analyze_full = "analyze_full"                  # LLM 全量评分 -20
    analyze_full_cache_hit = "analyze_full_cache_hit"  # LLM 缓存命中 -10
    analyze_quant = "analyze_quant"                # 量化评分 -1
    refund = "refund"                              # 失败退款
    recharge = "recharge"                          # 管理员充值
    admin_revoke = "admin_revoke"                  # 管理员撤销


class PointTransaction(Base, TimestampMixin):
    __tablename__ = "point_transactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    delta: Mapped[int] = mapped_column(Integer, nullable=False)   # 正=加 / 负=扣

    reason: Mapped[TransactionReason] = mapped_column(
        Enum(TransactionReason, name="transaction_reason_enum",
             values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    # 强关联 (任一 NULL 都可)
    related_result_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_results.id", ondelete="SET NULL"))
    related_invite_id: Mapped[int | None] = mapped_column(ForeignKey("invite_relations.id", ondelete="SET NULL"))
    related_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))   # 充值场景: 操作者

    note: Mapped[str | None] = mapped_column(Text)   # 备注 (充值原因, 撤销说明)

    balance_before: Mapped[int] = mapped_column(Integer, nullable=False)
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_transactions_user_created", "user_id", "created_at"),
        Index("ix_transactions_reason_created", "reason", "created_at"),
    )
