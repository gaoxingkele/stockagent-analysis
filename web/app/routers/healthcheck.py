"""健康检查 API: 手动触发 + SSE 实时推送 + 历史趋势。"""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import get_db, get_session_factory
from ..core.deps import get_current_user
from ..core.redis import subscribe_progress
from ..models import HealthCheck, HealthCheckTriggerType, User
from ..services.healthcheck_service import run_full_healthcheck

router = APIRouter(prefix="/api/healthcheck", tags=["healthcheck"])


@router.post("/run")
async def trigger_run(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """触发一次健康检查 (异步)。返回 check_id 用于订阅 SSE。"""
    # 先创建占位记录拿 id
    rec = HealthCheck(
        triggered_by_user_id=user.id,
        trigger_type=HealthCheckTriggerType.manual,
        total_items=0, passed=0, failed=0, duration_ms=0,
        details_json=[], market_snapshot_json={},
    )
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    check_id = rec.id

    # 异步跑
    factory = get_session_factory()
    async def _bg():
        async with factory() as db2:
            await run_full_healthcheck(
                db2, user_id=user.id, trigger_type=HealthCheckTriggerType.manual,
                stream=True, hc_record_id=check_id,
            )

    asyncio.create_task(_bg())
    return {"check_id": check_id, "status": "running"}


@router.get("/{check_id}/stream")
async def stream_check(
    check_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """SSE 实时输出每项结果。"""
    from sse_starlette.sse import EventSourceResponse
    import json

    res = await db.execute(select(HealthCheck).where(HealthCheck.id == check_id))
    rec = res.scalar_one_or_none()
    if rec is None:
        raise HTTPException(404, "记录不存在")

    async def event_gen():
        # 已完成 → 直接发完整结果
        if rec.passed + rec.failed > 0:
            yield {"event": "summary", "data": json.dumps({
                "total": rec.total_items, "passed": rec.passed, "failed": rec.failed,
                "duration_ms": rec.duration_ms,
                "items": rec.details_json,
                "market": rec.market_snapshot_json,
            }, ensure_ascii=False)}
            return

        # 实时订阅
        async for evt in subscribe_progress(check_id):
            yield {"event": evt.get("type", "message"),
                   "data": json.dumps(evt, ensure_ascii=False)}
            if evt.get("type") == "summary":
                break

    return EventSourceResponse(event_gen())


@router.get("/history")
async def history(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
):
    """历史健康检查记录 (24h 趋势用)。"""
    res = await db.execute(
        select(HealthCheck).order_by(desc(HealthCheck.created_at)).limit(limit)
    )
    items = list(res.scalars().all())
    return {
        "items": [
            {
                "id": h.id,
                "trigger_type": h.trigger_type.value,
                "total": h.total_items, "passed": h.passed, "failed": h.failed,
                "duration_ms": h.duration_ms,
                "created_at": h.created_at.isoformat(),
            }
            for h in items
        ],
    }


@router.get("/{check_id}")
async def detail(
    check_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    res = await db.execute(select(HealthCheck).where(HealthCheck.id == check_id))
    rec = res.scalar_one_or_none()
    if rec is None:
        raise HTTPException(404)
    return {
        "id": rec.id,
        "trigger_type": rec.trigger_type.value,
        "triggered_by_user_id": rec.triggered_by_user_id,
        "total": rec.total_items, "passed": rec.passed, "failed": rec.failed,
        "duration_ms": rec.duration_ms,
        "items": rec.details_json,
        "market": rec.market_snapshot_json,
        "created_at": rec.created_at.isoformat(),
    }
