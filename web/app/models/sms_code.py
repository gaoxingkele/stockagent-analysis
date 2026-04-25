"""短信验证码表 - 5 分钟过期, 防滥用。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class SmsCode(Base, TimestampMixin):
    __tablename__ = "sms_codes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String(11), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(8), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45))   # IPv6 最长 45

    __table_args__ = (
        Index("ix_sms_phone_created", "phone", "created_at"),
    )
