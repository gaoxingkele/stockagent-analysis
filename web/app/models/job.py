"""分析任务 - 一次提交可含多只股票。"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    partial_done = "partial_done"
    done = "done"
    failed = "failed"


class AnalysisJob(Base, TimestampMixin):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    symbols_count: Mapped[int] = mapped_column(nullable=False)
    total_points_charged: Mapped[int] = mapped_column(default=0, nullable=False)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status_enum", values_callable=lambda x: [e.value for e in x]),
        default=JobStatus.pending, nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_jobs_user_created", "user_id", "created_at"),
        Index("ix_jobs_status_created", "status", "created_at"),
    )
