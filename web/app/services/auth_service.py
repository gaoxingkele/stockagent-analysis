"""认证服务: 注册/登录(验证码 OR admin 密码) + JWT。"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.security import encode_token, verify_password
from ..models import (
    InviteMethod, PointTransaction, TransactionReason, User, UserStatus,
)
from .invite_service import (
    bind_invite_relation, find_inviter_by_code, generate_unique_invite_code,
)
from .sms_service import verify_sms_code

logger = logging.getLogger(__name__)


class AuthError(Exception):
    pass


async def register_or_get_user(
    db: AsyncSession, phone: str, *, invite_code: str | None = None,
) -> tuple[User, bool]:
    """根据手机号注册或返回已有用户。返回 (user, is_new)。"""
    res = await db.execute(select(User).where(User.phone == phone))
    user = res.scalar_one_or_none()

    if user is not None:
        return user, False

    # 新用户
    is_admin = (phone == settings.admin_phone)
    user = User(
        phone=phone,
        nickname=None,
        points=settings.points_register_bonus,
        is_admin=is_admin,
        language=settings.default_language,
        invite_code=await generate_unique_invite_code(db),
        invite_path="",   # 默认 root, 邀请绑定时再覆盖
        status=UserStatus.active,
    )
    db.add(user)
    await db.flush()

    db.add(PointTransaction(
        user_id=user.id, delta=settings.points_register_bonus,
        reason=TransactionReason.register_bonus,
        balance_before=0, balance_after=settings.points_register_bonus,
        note="注册赠送",
    ))
    await db.commit()
    await db.refresh(user)

    # 处理邀请关系
    if invite_code:
        inviter = await find_inviter_by_code(db, invite_code)
        if inviter and inviter.id != user.id:
            await bind_invite_relation(
                db, inviter=inviter, invitee=user,
                inviter_reward=settings.points_invite_referrer,
                invitee_reward=settings.points_invite_new_user,
                method=InviteMethod.code,
            )

    logger.info("[auth] 新用户注册: id=%d phone=%s admin=%s invite_code=%s",
                user.id, phone, is_admin, user.invite_code)
    return user, True


async def login_with_code(
    db: AsyncSession, phone: str, code: str, invite_code: str | None = None,
) -> tuple[User, str]:
    """验证码登录。新手机号自动注册。返回 (user, jwt_token)。"""
    if not await verify_sms_code(db, phone, code):
        raise AuthError("验证码错误或已过期")

    user, is_new = await register_or_get_user(db, phone, invite_code=invite_code)

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    token = encode_token(user.id, is_admin=user.is_admin)
    return user, token


async def login_with_password(
    db: AsyncSession, phone: str, password: str,
) -> tuple[User, str]:
    """管理员密码登录。仅 admin 可用。"""
    res = await db.execute(select(User).where(User.phone == phone))
    user = res.scalar_one_or_none()

    if user is None or not user.is_admin:
        raise AuthError("仅管理员账户支持密码登录")

    if not verify_password(password, user.password_hash):
        raise AuthError("密码错误")

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    token = encode_token(user.id, is_admin=user.is_admin)
    return user, token
