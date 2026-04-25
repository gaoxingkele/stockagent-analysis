"""SSE 进度事件持久化 - 用户断线重连可回放。"""
from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class ProgressEvent(Base, TimestampMixin):
    __tablename__ = "progress_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    result_id: Mapped[int] = mapped_column(ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False)

    phase_id: Mapped[str] = mapped_column(String(50), nullable=False)
    percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    data_json: Mapped[dict | None] = mapped_column(JSON)   # 阶段附带的中间结果

    __table_args__ = (
        Index("ix_progress_result_created", "result_id", "created_at"),
    )
