"""用户表 - 多用户 SaaS 核心。"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.db import Base, TimestampMixin

if TYPE_CHECKING:
    from .invite import InviteRelation


class UserStatus(str, enum.Enum):
    active = "active"
    suspended = "suspended"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String(11), unique=True, index=True, nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(50))
    avatar_url: Mapped[str | None] = mapped_column(String(255))

    # 积分
    points: Mapped[int] = mapped_column(default=0, nullable=False)

    # 权限
    is_admin: Mapped[bool] = mapped_column(default=False, nullable=False)

    # 偏好
    language: Mapped[str] = mapped_column(String(8), default="zh-CN", nullable=False)

    # 密码 (仅 admin 必须设, 普通用户为 NULL 走验证码登录)
    password_hash: Mapped[str | None] = mapped_column(String(128))

    # 邀请相关
    invite_code: Mapped[str | None] = mapped_column(String(8), unique=True, index=True)
    invited_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    invite_path: Mapped[str | None] = mapped_column(String(500))   # 物化路径 "1/5/12/"
    invite_count: Mapped[int] = mapped_column(default=0, nullable=False)
    invite_earned_points: Mapped[int] = mapped_column(default=0, nullable=False)

    # 状态
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus, name="user_status_enum", values_callable=lambda x: [e.value for e in x]),
        default=UserStatus.active, nullable=False,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 自引用关系 (邀请人)
    inviter: Mapped["User | None"] = relationship(
        "User", remote_side=[id], foreign_keys=[invited_by_user_id], backref="invitees",
    )

    __table_args__ = (
        Index("ix_users_invite_path", "invite_path"),
        Index("ix_users_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} phone={self.phone[:3]}****{self.phone[-4:]} points={self.points}>"
