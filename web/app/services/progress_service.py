"""进度推送服务: 同时落 DB + 推 Redis。"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.redis import publish_progress
from ..models import AnalysisResult, ProgressEvent

logger = logging.getLogger(__name__)


async def emit_progress(
    db: AsyncSession, result_id: int, *,
    phase_id: str, percent: int, message: str,
    data: dict[str, Any] | None = None,
) -> None:
    """记录进度事件 + 更新 result + 推送 SSE。"""
    # 落 DB
    evt = ProgressEvent(
        result_id=result_id, phase_id=phase_id,
        percent=percent, message=message, data_json=data,
    )
    db.add(evt)

    # 同时更新 result 进度字段
    res = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
    rec = res.scalar_one_or_none()
    if rec is not None:
        rec.current_phase = phase_id
        rec.progress_pct = percent
    await db.commit()

    # 推 SSE
    await publish_progress(result_id, {
        "type": "progress",
        "result_id": result_id,
        "phase": phase_id,
        "percent": percent,
        "message": message,
        "data": data,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


async def emit_done(
    db: AsyncSession, result_id: int, *,
    final_score: float | None, decision_level: str | None,
    extra: dict[str, Any] | None = None,
) -> None:
    await publish_progress(result_id, {
        "type": "done",
        "result_id": result_id,
        "final_score": final_score,
        "decision_level": decision_level,
        "extra": extra or {},
        "ts": datetime.now(timezone.utc).isoformat(),
    })


async def emit_failed(
    db: AsyncSession, result_id: int, error: str, refunded: int = 0,
) -> None:
    await publish_progress(result_id, {
        "type": "failed",
        "result_id": result_id,
        "error": error,
        "refunded": refunded,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


async def list_progress_events(
    db: AsyncSession, result_id: int,
) -> list[dict[str, Any]]:
    """断线重连用: 取所有历史事件回放。"""
    res = await db.execute(
        select(ProgressEvent).where(ProgressEvent.result_id == result_id)
        .order_by(ProgressEvent.created_at)
    )
    items = list(res.scalars().all())
    return [
        {
            "phase": e.phase_id, "percent": e.percent,
            "message": e.message, "data": e.data_json,
            "ts": e.created_at.isoformat(),
        } for e in items
    ]
