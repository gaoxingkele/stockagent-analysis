"""认证 API: 验证码 / 密码登录 / 我的信息。"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..models import User
from ..schemas.auth import (
    CodeSentResponse, PasswordLoginRequest, SendCodeRequest, TokenResponse,
    UserPublic, VerifyCodeRequest,
)
from ..services.auth_service import (
    AuthError, login_with_code, login_with_password,
)
from ..services.sms_service import send_verification_code

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/send-code", response_model=CodeSentResponse)
async def send_code(
    body: SendCodeRequest, request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        ip = request.client.host if request.client else None
        code, expires_at = await send_verification_code(db, body.phone, ip=ip)
    except ValueError as e:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e))

    return CodeSentResponse(
        phone=body.phone,
        expires_in=settings.sms_code_ttl_seconds,
        dev_code=code if settings.sms_provider == "mock" else None,
    )


@router.post("/verify", response_model=TokenResponse)
async def verify_code(
    body: VerifyCodeRequest, response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        user, token, minutes = await login_with_code(
            db, body.phone, body.code, invite_code=body.invite_code,
            remember_me=body.remember_me,
        )
    except AuthError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    # 同时种 Cookie 方便 SSR 页面
    response.set_cookie(
        "access_token", token,
        max_age=minutes * 60,
        httponly=True, samesite="lax",
    )
    return TokenResponse(
        access_token=token,
        expires_in=minutes * 60,
        user=UserPublic.from_orm(user),
    )


@router.post("/password-login", response_model=TokenResponse)
async def password_login(
    body: PasswordLoginRequest, response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        user, token, minutes = await login_with_password(
            db, body.phone, body.password, remember_me=body.remember_me,
        )
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))

    response.set_cookie(
        "access_token", token,
        max_age=minutes * 60,
        httponly=True, samesite="lax",
    )
    return TokenResponse(
        access_token=token,
        expires_in=minutes * 60,
        user=UserPublic.from_orm(user),
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"ok": True}


@router.get("/me", response_model=UserPublic)
async def me(user: Annotated[User, Depends(get_current_user)]):
    return UserPublic.from_orm(user)
