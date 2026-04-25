"""个人中心: 流水 / 团队 / 邀请信息。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..models import PointTransaction, User
from ..schemas.invite import (
    InviteInfoResponse, InviteLandingResponse, TeamMemberPublic, TeamResponse,
)
from ..schemas.points import TransactionItem, TransactionListResponse
from ..services.points_service import list_transactions

router = APIRouter(prefix="/api/me", tags=["me"])


def _mask_phone(p: str) -> str:
    return f"{p[:3]}****{p[-4:]}"


def _team_member(u: User) -> TeamMemberPublic:
    return TeamMemberPublic(
        id=u.id, phone=_mask_phone(u.phone), nickname=u.nickname,
        points=u.points, invite_count=u.invite_count,
        invite_earned_points=u.invite_earned_points,
        created_at=u.created_at.isoformat(),
    )


@router.get("/transactions", response_model=TransactionListResponse)
async def my_transactions(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    items, total = await list_transactions(db, user.id, limit=limit, offset=offset)
    return TransactionListResponse(
        items=[
            TransactionItem(
                id=tx.id, delta=tx.delta, reason=tx.reason.value,
                related_result_id=tx.related_result_id,
                related_invite_id=tx.related_invite_id,
                related_user_id=tx.related_user_id,
                note=tx.note,
                balance_before=tx.balance_before, balance_after=tx.balance_after,
                created_at=tx.created_at,
            ) for tx in items
        ],
        total=total, limit=limit, offset=offset,
    )


@router.get("/team", response_model=TeamResponse)
async def my_team(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """用户视角: 看自己的上级 + 直属下级 + 间接下级数量。"""
    # 上级
    inviter_obj: User | None = None
    if user.invited_by_user_id:
        res = await db.execute(select(User).where(User.id == user.invited_by_user_id))
        inviter_obj = res.scalar_one_or_none()

    # 直属下级
    res = await db.execute(
        select(User).where(User.invited_by_user_id == user.id).order_by(desc(User.created_at))
    )
    direct = list(res.scalars().all())

    # 间接 (物化路径包含 /user.id/, 减去直属)
    indirect_total = await db.scalar(
        select(func.count()).select_from(User).where(User.invite_path.like(f"%/{user.id}/%"))
    )
    # invite_path 格式 "1/5/12/" - 用户 5 的下级路径包含 /5/
    # 直属用户的 invite_path = "1/5/" 不含 /5/ 中间, 上面的 LIKE 不会匹配自己直属
    # 直属是 "1/5/" 末尾是 user.id/ , 间接是 "1/5/X/...", 都包含 /5/
    # 所以总数 = 直属 + 间接, 减去直属即可
    indirect_count = max(int(indirect_total or 0) - len(direct), 0)

    return TeamResponse(
        inviter=_team_member(inviter_obj) if inviter_obj else None,
        direct_invitees=[_team_member(u) for u in direct],
        direct_count=len(direct),
        indirect_count=indirect_count,
    )


@router.get("/invite-info", response_model=InviteInfoResponse)
async def my_invite_info(user: Annotated[User, Depends(get_current_user)]):
    return InviteInfoResponse(
        invite_code=user.invite_code or "",
        invite_count=user.invite_count,
        invite_earned_points=user.invite_earned_points,
        invite_url=f"{settings.base_url.rstrip('/')}/invite/{user.invite_code or ''}",
    )


@router.get("/invite-landing/{code}", response_model=InviteLandingResponse)
async def invite_landing(
    code: str, db: Annotated[AsyncSession, Depends(get_db)],
):
    """公开端点: 邀请落地页数据(无需登录)。"""
    res = await db.execute(select(User).where(User.invite_code == code))
    inviter = res.scalar_one_or_none()
    if inviter is None:
        return InviteLandingResponse(
            inviter_nickname=None, inviter_phone_masked="****",
            inviter_invite_count=0,
            new_user_bonus=settings.points_register_bonus + settings.points_invite_new_user,
            valid=False,
        )
    return InviteLandingResponse(
        inviter_nickname=inviter.nickname,
        inviter_phone_masked=_mask_phone(inviter.phone),
        inviter_invite_count=inviter.invite_count,
        new_user_bonus=settings.points_register_bonus + settings.points_invite_new_user,
        valid=True,
    )
