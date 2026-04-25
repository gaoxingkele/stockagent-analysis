"""P1 验证: 11 张表 ORM 模型创建 + 关键关系正确。"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models import (
    AnalysisJob, AnalysisResult, AnalysisType, InviteMethod, InviteRelation,
    JobStatus, PointTransaction, ResultStatus, Subscription, TransactionReason,
    User, UserStatus,
)


pytestmark = pytest.mark.asyncio


async def test_create_user(db_session):
    user = User(
        phone="13800001234",
        nickname="测试",
        points=100,
        invite_code="A100001",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    assert user.id is not None
    assert user.points == 100
    assert user.is_admin is False
    assert user.status == UserStatus.active
    assert user.created_at is not None


async def test_user_invite_self_reference(db_session):
    """邀请关系: 自引用 + 物化路径。"""
    inviter = User(phone="13800001111", invite_code="A100001", invite_path="")
    db_session.add(inviter)
    await db_session.flush()

    invitee = User(
        phone="13800002222", invite_code="B200002",
        invited_by_user_id=inviter.id,
        invite_path=f"{inviter.id}/",
    )
    db_session.add(invitee)
    await db_session.commit()

    await db_session.refresh(invitee, ["inviter"])
    assert invitee.inviter is not None
    assert invitee.inviter.id == inviter.id

    # 物化路径查询: 找 inviter 的所有下级
    result = await db_session.execute(
        select(User).where(User.invite_path.like(f"%{inviter.id}/%"))
    )
    descendants = result.scalars().all()
    assert len(descendants) == 1
    assert descendants[0].phone == "13800002222"


async def test_point_transaction_strong_consistency(db_session):
    """积分流水强一致到 result_id。"""
    user = User(phone="13800003333", points=100, invite_code="C300003")
    db_session.add(user)
    await db_session.flush()

    # 先创建一条 quant 分析结果
    job = AnalysisJob(user_id=user.id, symbols_count=1, total_points_charged=1)
    db_session.add(job)
    await db_session.flush()

    result = AnalysisResult(
        job_id=job.id, user_id=user.id, symbol="600519",
        analysis_type=AnalysisType.quant_only,
        points_charged=1,
        status=ResultStatus.done,
        final_score=68.0, decision_level="hold", quant_score=68.0,
    )
    db_session.add(result)
    await db_session.flush()

    # 流水扣 1 积分,关联到 result
    tx = PointTransaction(
        user_id=user.id, delta=-1,
        reason=TransactionReason.analyze_quant,
        related_result_id=result.id,
        balance_before=100, balance_after=99,
    )
    db_session.add(tx)
    await db_session.commit()

    # 反查: 通过 result_id 找流水
    res = await db_session.execute(
        select(PointTransaction).where(PointTransaction.related_result_id == result.id)
    )
    tx_back = res.scalar_one()
    assert tx_back.delta == -1
    assert tx_back.reason == TransactionReason.analyze_quant


async def test_analysis_result_dual_mode(db_session):
    """双模式: full vs quant_only 字段差异。"""
    user = User(phone="13800004444", invite_code="D400004")
    db_session.add(user)
    await db_session.flush()

    job = AnalysisJob(user_id=user.id, symbols_count=2, total_points_charged=21)
    db_session.add(job)
    await db_session.flush()

    full = AnalysisResult(
        job_id=job.id, user_id=user.id, symbol="600519",
        analysis_type=AnalysisType.full,
        points_charged=20,
        status=ResultStatus.done,
        final_score=82.5, decision_level="strong_buy",
        quant_score=72.0, trader_decision="BUY",
        expert_scores_json={"k_structure": 85, "wave": 80, "capital": 78, "sentiment": 82},
        score_components_json={"expert_consensus": 81, "judge_adj": 80, "risk_mapped": 75},
        quant_components_json={"quant_score": 72, "adjustments": []},
    )
    db_session.add(full)
    await db_session.flush()

    quant = AnalysisResult(
        job_id=job.id, user_id=user.id, symbol="600519",
        analysis_type=AnalysisType.quant_only,
        parent_full_result_id=full.id,
        points_charged=1,
        status=ResultStatus.done,
        final_score=70.0, decision_level="hold", quant_score=70.0,
        quant_components_json={"quant_score": 70, "adjustments": []},
    )
    db_session.add(quant)
    await db_session.commit()

    # full 有专家分, quant 没有
    assert full.expert_scores_json is not None
    assert quant.expert_scores_json is None
    assert quant.parent_full_result_id == full.id


async def test_invite_relation_with_poster(db_session):
    """邀请关系: 海报来源关联 result。"""
    inviter = User(phone="13800005555", invite_code="E500005")
    invitee = User(phone="13800006666", invite_code="F600006")
    db_session.add_all([inviter, invitee])
    await db_session.flush()

    job = AnalysisJob(user_id=inviter.id, symbols_count=1, total_points_charged=20)
    db_session.add(job)
    await db_session.flush()

    poster_result = AnalysisResult(
        job_id=job.id, user_id=inviter.id, symbol="000858",
        analysis_type=AnalysisType.full,
        status=ResultStatus.done,
    )
    db_session.add(poster_result)
    await db_session.flush()

    relation = InviteRelation(
        inviter_user_id=inviter.id, invitee_user_id=invitee.id,
        invite_method=InviteMethod.poster,
        inviter_reward_points=100, invitee_reward_points=50,
        poster_result_id=poster_result.id,
    )
    db_session.add(relation)
    await db_session.commit()

    res = await db_session.execute(
        select(InviteRelation).where(InviteRelation.invitee_user_id == invitee.id)
    )
    r = res.scalar_one()
    assert r.invite_method == InviteMethod.poster
    assert r.poster_result_id == poster_result.id


async def test_subscription_unique_per_user_symbol(db_session):
    """订阅: (user_id, symbol) 唯一约束。"""
    from sqlalchemy.exc import IntegrityError

    user = User(phone="13800007777", invite_code="G700007")
    db_session.add(user)
    await db_session.flush()

    sub1 = Subscription(user_id=user.id, symbol="600519", name="贵州茅台",
                        notify_channels=["feishu"])
    db_session.add(sub1)
    await db_session.commit()

    sub2 = Subscription(user_id=user.id, symbol="600519", name="贵州茅台 dup")
    db_session.add(sub2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_seed_admin(db_session):
    """种子: ensure_admin_user 创建管理员。"""
    from app.services.seed import ensure_admin_user

    admin = await ensure_admin_user(db_session)
    assert admin.is_admin is True
    assert admin.invite_code is not None
    assert len(admin.invite_code) == 7   # 1 字母 + 6 数字
    assert admin.invite_code[0].isalpha()
    assert admin.invite_code[1:].isdigit()
    assert admin.points == 100   # register_bonus

    # 流水里应该有一条 register_bonus
    res = await db_session.execute(
        select(PointTransaction).where(PointTransaction.user_id == admin.id)
    )
    txs = res.scalars().all()
    assert len(txs) == 1
    assert txs[0].reason == TransactionReason.register_bonus
    assert txs[0].delta == 100

    # 再次调用应该幂等 (不创建第二个)
    admin2 = await ensure_admin_user(db_session)
    assert admin2.id == admin.id
