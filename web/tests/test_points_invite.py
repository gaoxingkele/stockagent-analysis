"""P3 积分 + 邀请测试。"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models import PointTransaction, TransactionReason, User
from app.services.auth_service import register_or_get_user
from app.services.invite_service import bind_invite_relation
from app.services.points_service import (
    InsufficientPointsError, admin_recharge, deduct_points,
    list_transactions, refund_points,
)
from app.services.seed import ensure_admin_user

pytestmark = pytest.mark.asyncio


async def test_deduct_points(db_session):
    user, _ = await register_or_get_user(db_session, "13800010001")
    assert user.points == 100
    await deduct_points(db_session, user, 20, reason=TransactionReason.analyze_full,
                        note="test")
    await db_session.refresh(user)
    assert user.points == 80


async def test_deduct_insufficient(db_session):
    user, _ = await register_or_get_user(db_session, "13800010002")
    with pytest.raises(InsufficientPointsError) as ex:
        await deduct_points(db_session, user, 9999,
                             reason=TransactionReason.analyze_full)
    assert ex.value.have == 100
    assert ex.value.need == 9999


async def test_refund(db_session):
    user, _ = await register_or_get_user(db_session, "13800010003")
    await deduct_points(db_session, user, 20, reason=TransactionReason.analyze_full)
    await refund_points(db_session, user, 20, note="测试退款")
    await db_session.refresh(user)
    assert user.points == 100
    # 流水: register +100 / analyze -20 / refund +20
    items, total = await list_transactions(db_session, user.id)
    assert total == 3
    reasons = sorted(tx.reason.value for tx in items)
    assert "analyze_full" in reasons
    assert "refund" in reasons
    assert "register_bonus" in reasons


async def test_admin_recharge(db_session):
    admin = await ensure_admin_user(db_session)
    target, _ = await register_or_get_user(db_session, "13800010004")
    assert target.points == 100

    tx = await admin_recharge(db_session, target_user=target, amount=500,
                                operator=admin, note="活动奖励")
    await db_session.refresh(target)
    assert target.points == 600
    assert tx.reason == TransactionReason.recharge
    assert tx.related_user_id == admin.id


async def test_admin_recharge_negative(db_session):
    """管理员可以扣分(撤销)。"""
    admin = await ensure_admin_user(db_session)
    target, _ = await register_or_get_user(db_session, "13800010005")

    tx = await admin_recharge(db_session, target_user=target, amount=-30,
                                operator=admin, note="违规扣分")
    await db_session.refresh(target)
    assert target.points == 70
    assert tx.reason == TransactionReason.admin_revoke


async def test_recharge_non_admin_blocked(db_session):
    """非管理员调用充值应该被拒。"""
    user, _ = await register_or_get_user(db_session, "13800010006")
    target, _ = await register_or_get_user(db_session, "13800010007")
    with pytest.raises(PermissionError):
        await admin_recharge(db_session, target_user=target, amount=100, operator=user)


async def test_invite_three_levels(db_session):
    """三级邀请链: A → B → C, 验证物化路径 + 直属/间接区分。"""
    admin = await ensure_admin_user(db_session)
    a, _ = await register_or_get_user(db_session, "13800020001")
    b, _ = await register_or_get_user(db_session, "13800020002", invite_code=a.invite_code)
    c, _ = await register_or_get_user(db_session, "13800020003", invite_code=b.invite_code)

    await db_session.refresh(a)
    await db_session.refresh(b)
    await db_session.refresh(c)

    assert a.invite_path == ""
    assert b.invite_path == f"{a.id}/"
    assert c.invite_path == f"{a.id}/{b.id}/"

    # A 的直属 = [B], 间接 = [C]
    from sqlalchemy import select
    direct_b = (await db_session.execute(
        select(User).where(User.invited_by_user_id == a.id)
    )).scalars().all()
    assert len(direct_b) == 1
    assert direct_b[0].id == b.id

    desc = (await db_session.execute(
        select(User).where(User.invite_path.like(f"%/{a.id}/%"))
    )).scalars().all()
    # invite_path 格式: A 的下级 b 是 "A/" → 不匹配 "/A/"
    # 但 c 是 "A/B/" → 也不匹配 "/A/" 因为开头没 /
    # 实际: 用 like "A/%" 更准
    res = (await db_session.execute(
        select(User).where(User.invite_path.like(f"{a.id}/%"))
    )).scalars().all()
    assert len(res) == 2   # B 和 C 都在 A 下游
