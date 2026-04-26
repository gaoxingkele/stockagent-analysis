"""认证相关 Pydantic schemas。"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SendCodeRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)

    @field_validator("phone")
    @classmethod
    def _check_phone(cls, v: str) -> str:
        if not v.isdigit() or not v.startswith("1"):
            raise ValueError("手机号格式不正确")
        return v


class VerifyCodeRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    code: str = Field(..., min_length=4, max_length=8)
    invite_code: str | None = Field(default=None, max_length=8)
    remember_me: bool = False


class PasswordLoginRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=64)
    remember_me: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserPublic"


class UserPublic(BaseModel):
    id: int
    phone: str   # 已打码
    nickname: str | None
    avatar_url: str | None
    points: int
    is_admin: bool
    language: str
    invite_code: str | None
    invite_count: int
    invite_earned_points: int

    @classmethod
    def from_orm(cls, u) -> "UserPublic":
        return cls(
            id=u.id,
            phone=f"{u.phone[:3]}****{u.phone[-4:]}",
            nickname=u.nickname,
            avatar_url=u.avatar_url,
            points=u.points,
            is_admin=u.is_admin,
            language=u.language,
            invite_code=u.invite_code,
            invite_count=u.invite_count,
            invite_earned_points=u.invite_earned_points,
        )


TokenResponse.model_rebuild()


class CodeSentResponse(BaseModel):
    phone: str
    expires_in: int = 300
    dev_code: str | None = None   # mock 模式才返回
