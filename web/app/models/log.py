"""应用日志 - 关键事件持久化(同时也写文件)。"""
from __future__ import annotations

from sqlalchemy import ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class AppLog(Base, TimestampMixin):
    __tablename__ = "app_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    level: Mapped[str] = mapped_column(String(10), nullable=False)   # DEBUG/INFO/WARNING/ERROR
    module: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    request_id: Mapped[str | None] = mapped_column(String(40))

    context_json: Mapped[dict | None] = mapped_column(JSON)
    traceback: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_logs_level_created", "level", "created_at"),
        Index("ix_logs_user_created", "user_id", "created_at"),
        Index("ix_logs_module_created", "module", "created_at"),
    )
