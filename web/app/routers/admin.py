"""管理员路由: 用户管理 / 充值 / 好友链 / 日志查看。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, or_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import get_db
from ..core.deps import get_admin_user
from ..models import PointTransaction, TransactionReason, User
from ..schemas.invite import TeamMemberPublic
from ..schemas.points import RechargeRequest, TransactionItem
from ..services.points_service import (
    InsufficientPointsError, admin_recharge,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users")
async def list_users(
    admin: Annotated[User, Depends(get_admin_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: str = Query("", max_length=64),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    q = select(User)
    if search:
        q = q.where(or_(
            User.phone.like(f"%{search}%"),
            User.nickname.like(f"%{search}%"),
            User.invite_code == search.upper(),
        ))
    total = await db.scalar(
        select(func.count()).select_from(User).where(
            or_(
                User.phone.like(f"%{search}%"),
                User.nickname.like(f"%{search}%"),
                User.invite_code == search.upper(),
            ) if search else True
        )
    )
    res = await db.execute(q.order_by(desc(User.created_at)).limit(limit).offset(offset))
    users = res.scalars().all()
    return {
        "items": [
            {
                "id": u.id, "phone": u.phone, "nickname": u.nickname,
                "points": u.points, "is_admin": u.is_admin,
                "invite_code": u.invite_code, "invite_count": u.invite_count,
                "invite_earned_points": u.invite_earned_points,
                "status": u.status.value,
                "language": u.language,
                "invited_by_user_id": u.invited_by_user_id,
                "created_at": u.created_at.isoformat(),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            }
            for u in users
        ],
        "total": int(total or 0),
        "limit": limit, "offset": offset,
    }


@router.post("/recharge")
async def recharge(
    body: RechargeRequest,
    admin: Annotated[User, Depends(get_admin_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    res = await db.execute(select(User).where(User.id == body.target_user_id))
    target = res.scalar_one_or_none()
    if target is None:
        raise HTTPException(404, "用户不存在")
    try:
        tx = await admin_recharge(db, target_user=target, amount=body.amount,
                                   operator=admin, note=body.note)
    except (InsufficientPointsError, ValueError, PermissionError) as e:
        raise HTTPException(400, str(e))
    return {
        "ok": True,
        "tx_id": tx.id,
        "balance_after": target.points,
    }


@router.get("/relations")
async def all_relations(
    admin: Annotated[User, Depends(get_admin_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """完整邀请树 (echarts tree 数据格式)。"""
    res = await db.execute(select(User).order_by(User.id))
    users = list(res.scalars().all())

    by_id = {u.id: u for u in users}
    children: dict[int, list[int]] = {u.id: [] for u in users}
    roots: list[int] = []
    for u in users:
        if u.invited_by_user_id and u.invited_by_user_id in by_id:
            children[u.invited_by_user_id].append(u.id)
        else:
            roots.append(u.id)

    def build(node_id: int) -> dict:
        u = by_id[node_id]
        return {
            "id": u.id,
            "name": u.nickname or u.phone[-4:],
            "phone": f"{u.phone[:3]}****{u.phone[-4:]}",
            "points": u.points,
            "invite_count": u.invite_count,
            "is_admin": u.is_admin,
            "children": [build(c) for c in children[node_id]],
        }

    return {"trees": [build(r) for r in roots], "total_users": len(users)}


@router.get("/transactions")
async def list_all_transactions(
    admin: Annotated[User, Depends(get_admin_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    user_id: int | None = None,
    reason: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    q = select(PointTransaction)
    if user_id:
        q = q.where(PointTransaction.user_id == user_id)
    if reason:
        try:
            r = TransactionReason(reason)
            q = q.where(PointTransaction.reason == r)
        except ValueError:
            raise HTTPException(400, f"未知 reason: {reason}")
    res = await db.execute(q.order_by(desc(PointTransaction.created_at)).limit(limit).offset(offset))
    items = res.scalars().all()
    return {
        "items": [
            {
                "id": tx.id, "user_id": tx.user_id, "delta": tx.delta,
                "reason": tx.reason.value,
                "related_result_id": tx.related_result_id,
                "related_invite_id": tx.related_invite_id,
                "note": tx.note,
                "balance_after": tx.balance_after,
                "created_at": tx.created_at.isoformat(),
            }
            for tx in items
        ],
    }
