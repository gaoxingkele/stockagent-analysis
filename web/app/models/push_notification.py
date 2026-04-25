"""推送通知日志 - P11 飞书/钉钉/邮件推送审计。"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class PushNotificationType(str, enum.Enum):
    decision_change = "decision_change"
    score_alert = "score_alert"
    quant_failed = "quant_failed"
    admin_message = "admin_message"


class PushChannel(str, enum.Enum):
    feishu = "feishu"
    dingtalk = "dingtalk"
    email = "email"
    site = "site"      # 站内通知


class PushStatus(str, enum.Enum):
    pending = "pending"
    sent = "sent"
    failed = "failed"


class PushNotification(Base, TimestampMixin):
    __tablename__ = "push_notifications"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    subscription_id: Mapped[int | None] = mapped_column(ForeignKey("subscriptions.id", ondelete="SET NULL"))

    type: Mapped[PushNotificationType] = mapped_column(
        Enum(PushNotificationType, name="push_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    channel: Mapped[PushChannel] = mapped_column(
        Enum(PushChannel, name="push_channel_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    status: Mapped[PushStatus] = mapped_column(
        Enum(PushStatus, name="push_status_enum", values_callable=lambda x: [e.value for e in x]),
        default=PushStatus.pending, nullable=False,
    )

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    related_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="SET NULL"),
    )
