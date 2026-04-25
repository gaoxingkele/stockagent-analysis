"""APScheduler 定时任务编排。"""
from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import settings
from ..services.healthcheck_service import cron_run_healthcheck

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def init_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone=settings.healthcheck_tz)

    if settings.healthcheck_cron_enabled:
        # 工作日 9-16 点每小时跑一次
        _scheduler.add_job(
            cron_run_healthcheck,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=settings.healthcheck_cron_hours,   # "9-16"
                minute=0,
                timezone=settings.healthcheck_tz,
            ),
            id="healthcheck_cron",
            name="工作日 9-16 点每小时健康检查",
            replace_existing=True,
        )
        logger.info("[scheduler] healthcheck_cron 已注册: 工作日 %s 点每小时 (%s)",
                    settings.healthcheck_cron_hours, settings.healthcheck_tz)

    _scheduler.start()
    return _scheduler


async def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("[scheduler] 已关闭")
