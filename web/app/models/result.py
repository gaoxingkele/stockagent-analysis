"""单股分析结果 - 双模式 (full LLM / quant_only) 共用此表。"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, JSON, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db import Base, TimestampMixin


class AnalysisType(str, enum.Enum):
    full = "full"               # LLM 全量评分(20pt 或 缓存 10pt)
    quant_only = "quant_only"   # 仅量化评分(1pt)


class ResultStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    refunded = "refunded"


class AnalysisResult(Base, TimestampMixin):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    job_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="SET NULL"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[str | None] = mapped_column(String(50))
    run_dir: Mapped[str | None] = mapped_column(String(255))   # quant_only 通常无

    # 双模式核心字段
    analysis_type: Mapped[AnalysisType] = mapped_column(
        Enum(AnalysisType, name="analysis_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    parent_full_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="SET NULL"),
    )   # quant_only 关联到最近一次 full

    # 缓存
    is_cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="SET NULL"),
    )

    # 积分
    points_charged: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 状态
    status: Mapped[ResultStatus] = mapped_column(
        Enum(ResultStatus, name="result_status_enum", values_callable=lambda x: [e.value for e in x]),
        default=ResultStatus.queued, nullable=False,
    )
    current_phase: Mapped[str | None] = mapped_column(String(50))
    progress_pct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 评分核心结果
    final_score: Mapped[float | None] = mapped_column(Float)
    decision_level: Mapped[str | None] = mapped_column(String(20))   # weak_buy/hold/weak_sell etc.
    quant_score: Mapped[float | None] = mapped_column(Float)
    trader_decision: Mapped[str | None] = mapped_column(String(10))   # full only: BUY/SELL/HOLD

    # 详情快照 (JSON)
    expert_scores_json: Mapped[dict | None] = mapped_column(JSON)        # full only
    score_components_json: Mapped[dict | None] = mapped_column(JSON)     # full only
    quant_components_json: Mapped[dict | None] = mapped_column(JSON)     # 两类都有(quant_only 必有)

    # 错误处理
    error_message: Mapped[str | None] = mapped_column(Text)

    duration_sec: Mapped[int | None] = mapped_column(Integer)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_results_symbol_created", "symbol", "created_at"),
        Index("ix_results_user_created", "user_id", "created_at"),
        Index("ix_results_status_created", "status", "created_at"),
        Index("ix_results_type_symbol_created", "analysis_type", "symbol", "created_at"),
    )

    def __repr__(self) -> str:
        return (f"<AnalysisResult id={self.id} {self.symbol} "
                f"type={self.analysis_type} status={self.status} score={self.final_score}>")
