"""FastAPI 依赖注入: 当前用户 / DB / 语言。"""
from __future__ import annotations

from typing import AsyncIterator

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_db
from .i18n import normalize_lang
from .security import decode_token
from ..models import User, UserStatus


async def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """从 Bearer token / cookie 解析当前用户。"""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未登录")

    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已过期")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token 无效")

    user_id = int(payload.get("sub", 0))
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token 无效")

    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    if user.status != UserStatus.active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "账户已停用")
    return user


async def get_current_user_optional(
    request: Request,
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """可选当前用户(用于公开页面带身份显示)。"""
    try:
        return await get_current_user(request, authorization, db)
    except HTTPException:
        return None


async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
    return user


def get_lang(request: Request) -> str:
    """从 cookie / Accept-Language 头解析语言。"""
    cookie_lang = request.cookies.get("lang")
    if cookie_lang:
        return normalize_lang(cookie_lang)
    return normalize_lang(request.headers.get("accept-language"))
