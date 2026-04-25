"""邀请关系 - 完整审计 + 物化路径辅助查询。"""
from __future__ import annotations

import enum

from sqlalchemy import Enum, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class InviteMethod(str, enum.Enum):
    code = "code"        # 手动输入邀请码
    qr = "qr"            # 扫码
    poster = "poster"    # 海报
    link = "link"        # 分享链接


class InviteRelation(Base, TimestampMixin):
    __tablename__ = "invite_relations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    inviter_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    invitee_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    invite_method: Mapped[InviteMethod] = mapped_column(
        Enum(InviteMethod, name="invite_method_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    inviter_reward_points: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    invitee_reward_points: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 海报场景 (来源股票分析结果)
    poster_result_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_results.id", ondelete="SET NULL"))
