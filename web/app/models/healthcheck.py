"""健康检查记录 - 手动触发 + 工作日 9-16 点定时。"""
from __future__ import annotations

import enum

from sqlalchemy import Enum, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class HealthCheckTriggerType(str, enum.Enum):
    manual = "manual"
    cron = "cron"


class HealthCheck(Base, TimestampMixin):
    __tablename__ = "health_checks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    triggered_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    trigger_type: Mapped[HealthCheckTriggerType] = mapped_column(
        Enum(HealthCheckTriggerType, name="healthcheck_trigger_type_enum",
             values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    total_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    details_json: Mapped[list | None] = mapped_column(JSON)   # 每项 (api_name, status, latency, error)
    market_snapshot_json: Mapped[dict | None] = mapped_column(JSON)   # 大盘指数快照
