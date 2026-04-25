"""邀请/团队 schemas。"""
from __future__ import annotations

from pydantic import BaseModel


class TeamMemberPublic(BaseModel):
    id: int
    phone: str   # 已打码
    nickname: str | None
    points: int
    invite_count: int
    invite_earned_points: int
    created_at: str


class TeamResponse(BaseModel):
    inviter: TeamMemberPublic | None
    direct_invitees: list[TeamMemberPublic]
    direct_count: int
    indirect_count: int   # 二级及以下


class InviteInfoResponse(BaseModel):
    invite_code: str
    invite_count: int
    invite_earned_points: int
    invite_url: str       # 完整分享 URL


class InviteLandingResponse(BaseModel):
    """邀请落地页给前端的数据。"""
    inviter_nickname: str | None
    inviter_phone_masked: str
    inviter_invite_count: int
    new_user_bonus: int    # 新人将得 = 100 + 50 = 150
    valid: bool
