"""邀请服务: 邀请码生成 + 注册时绑定 + 物化路径。"""
from __future__ import annotations

import logging
import random
import string

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import InviteMethod, InviteRelation, PointTransaction, TransactionReason, User

logger = logging.getLogger(__name__)


def generate_invite_code() -> str:
    """1 字母 + 6 数字 (A123456)"""
    letter = random.choice(string.ascii_uppercase)
    digits = "".join(random.choices(string.digits, k=6))
    return f"{letter}{digits}"


async def generate_unique_invite_code(db: AsyncSession, max_tries: int = 20) -> str:
    """避免冲突, 最多尝试 N 次。"""
    for _ in range(max_tries):
        code = generate_invite_code()
        res = await db.execute(select(User.id).where(User.invite_code == code))
        if res.scalar_one_or_none() is None:
            return code
    raise RuntimeError("邀请码冲突, 请重试")


async def find_inviter_by_code(db: AsyncSession, code: str) -> User | None:
    res = await db.execute(select(User).where(User.invite_code == code))
    return res.scalar_one_or_none()


async def bind_invite_relation(
    db: AsyncSession, *,
    inviter: User, invitee: User,
    inviter_reward: int, invitee_reward: int,
    method: InviteMethod = InviteMethod.code,
    poster_result_id: int | None = None,
) -> InviteRelation:
    """建立邀请关系 + 双方加积分 + 流水。

    调用者要确保 invitee 已 commit (有 id), inviter 也已存在。
    本函数不重复扣加, 全部在事务内操作。
    """
    # 物化路径
    invitee.invited_by_user_id = inviter.id
    invitee.invite_path = f"{inviter.invite_path or ''}{inviter.id}/"

    # 介绍人增加奖励
    inviter.points += inviter_reward
    inviter.invite_count += 1
    inviter.invite_earned_points += inviter_reward

    # 新人额外奖励 (注册基础积分由 register_user 函数加, 这里只加邀请额外)
    inviter_bal_before = inviter.points - inviter_reward   # 已加过, 推算 before
    invitee.points += invitee_reward

    relation = InviteRelation(
        inviter_user_id=inviter.id, invitee_user_id=invitee.id,
        invite_method=method,
        inviter_reward_points=inviter_reward,
        invitee_reward_points=invitee_reward,
        poster_result_id=poster_result_id,
    )
    db.add(relation)
    await db.flush()    # 拿 relation.id

    # 双方流水
    db.add(PointTransaction(
        user_id=inviter.id, delta=inviter_reward,
        reason=TransactionReason.invite_referrer_bonus,
        related_invite_id=relation.id, related_user_id=invitee.id,
        balance_before=inviter_bal_before, balance_after=inviter.points,
        note=f"邀请 {invitee.phone[-4:]} 注册",
    ))
    db.add(PointTransaction(
        user_id=invitee.id, delta=invitee_reward,
        reason=TransactionReason.invite_new_user_bonus,
        related_invite_id=relation.id, related_user_id=inviter.id,
        balance_before=invitee.points - invitee_reward,
        balance_after=invitee.points,
        note=f"通过 {inviter.phone[-4:]} 邀请码注册",
    ))

    await db.commit()
    await db.refresh(invitee)
    await db.refresh(inviter)

    logger.info("[invite] 绑定成功: %s → %s (inviter +%d, invitee +%d)",
                inviter.phone, invitee.phone, inviter_reward, invitee_reward)

    return relation
