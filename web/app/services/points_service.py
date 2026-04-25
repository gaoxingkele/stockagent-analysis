"""积分服务: 通用扣款/退款/充值/查询。强一致到 result_id 或 invite_id。"""
from __future__ import annotations

import logging
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import PointTransaction, TransactionReason, User

logger = logging.getLogger(__name__)


class InsufficientPointsError(Exception):
    def __init__(self, need: int, have: int):
        self.need = need
        self.have = have
        super().__init__(f"积分不足: 需要 {need}, 当前 {have}")


async def deduct_points(
    db: AsyncSession, user: User, amount: int, *,
    reason: TransactionReason,
    related_result_id: int | None = None,
    related_invite_id: int | None = None,
    related_user_id: int | None = None,
    note: str | None = None,
    auto_commit: bool = True,
) -> PointTransaction:
    """扣减积分。amount 为正数, 内部转负。"""
    if amount <= 0:
        raise ValueError("扣减金额必须为正")
    if user.points < amount:
        raise InsufficientPointsError(need=amount, have=user.points)

    before = user.points
    user.points -= amount
    tx = PointTransaction(
        user_id=user.id, delta=-amount, reason=reason,
        related_result_id=related_result_id,
        related_invite_id=related_invite_id,
        related_user_id=related_user_id,
        note=note,
        balance_before=before, balance_after=user.points,
    )
    db.add(tx)
    if auto_commit:
        await db.commit()
        await db.refresh(user)
    else:
        await db.flush()
    logger.info("[points] -%d  user=%d reason=%s balance %d→%d",
                amount, user.id, reason.value, before, user.points)
    return tx


async def refund_points(
    db: AsyncSession, user: User, amount: int, *,
    related_result_id: int | None = None,
    note: str | None = None,
    auto_commit: bool = True,
) -> PointTransaction:
    """退款 (失败分析自动退)。"""
    if amount <= 0:
        raise ValueError("退款金额必须为正")
    before = user.points
    user.points += amount
    tx = PointTransaction(
        user_id=user.id, delta=amount, reason=TransactionReason.refund,
        related_result_id=related_result_id, note=note,
        balance_before=before, balance_after=user.points,
    )
    db.add(tx)
    if auto_commit:
        await db.commit()
        await db.refresh(user)
    else:
        await db.flush()
    logger.info("[points] +%d (refund)  user=%d balance %d→%d",
                amount, user.id, before, user.points)
    return tx


async def admin_recharge(
    db: AsyncSession, *,
    target_user: User, amount: int, operator: User, note: str | None = None,
) -> PointTransaction:
    """管理员充值 (正数加 / 负数扣)。"""
    if not operator.is_admin:
        raise PermissionError("仅管理员可充值")
    if amount == 0:
        raise ValueError("金额不能为 0")
    if amount < 0 and target_user.points + amount < 0:
        raise InsufficientPointsError(need=-amount, have=target_user.points)

    before = target_user.points
    target_user.points += amount
    tx = PointTransaction(
        user_id=target_user.id, delta=amount,
        reason=TransactionReason.recharge if amount > 0 else TransactionReason.admin_revoke,
        related_user_id=operator.id, note=note or f"管理员 {operator.id} 操作",
        balance_before=before, balance_after=target_user.points,
    )
    db.add(tx)
    await db.commit()
    await db.refresh(target_user)
    logger.warning("[points] admin recharge: target=%d %+d  by admin=%d  balance %d→%d",
                   target_user.id, amount, operator.id, before, target_user.points)
    return tx


async def list_transactions(
    db: AsyncSession, user_id: int, *,
    limit: int = 50, offset: int = 0,
    reason_filter: TransactionReason | None = None,
) -> tuple[list[PointTransaction], int]:
    """获取用户流水, 返回 (records, total_count)。"""
    q = select(PointTransaction).where(PointTransaction.user_id == user_id)
    if reason_filter:
        q = q.where(PointTransaction.reason == reason_filter)
    total = await db.scalar(
        select(func.count()).select_from(PointTransaction).where(
            PointTransaction.user_id == user_id,
            PointTransaction.reason == reason_filter if reason_filter else True,
        )
    )
    q = q.order_by(desc(PointTransaction.created_at)).limit(limit).offset(offset)
    res = await db.execute(q)
    return list(res.scalars().all()), int(total or 0)
