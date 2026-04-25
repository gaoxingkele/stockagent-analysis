"""自动跟踪订阅 - P10 每日 16:30 cron 自动量化 + 异常推送。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class Subscription(Base, TimestampMixin):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[str | None] = mapped_column(String(50))

    # 开关
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    auto_quant_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_on_change: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # 阈值
    notify_threshold_score_delta: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    notify_channels: Mapped[list | None] = mapped_column(JSON)   # ['feishu','dingtalk','email']

    # 上次跟踪
    last_quant_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_quant_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="SET NULL"),
    )

    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_subscription_user_symbol"),
    )
