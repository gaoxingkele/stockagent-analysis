"""P4 双模式分析核心测试。"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.models import (
    AnalysisJob, AnalysisResult, AnalysisType, JobStatus, ResultStatus,
)
from app.services.analysis_runner import (
    determine_analysis_type, find_latest_full_score, submit_analysis,
    has_any_full_score,
)
from app.services.auth_service import register_or_get_user

pytestmark = pytest.mark.asyncio


async def test_no_history_returns_full(db_session):
    a_type, pts, cache = await determine_analysis_type(db_session, "600519")
    assert a_type == AnalysisType.full
    assert pts == 20
    assert cache is None


async def _seed_full_result(db, user_id, symbol, score=80.0):
    """辅助: 创建一条 full 已完成结果。"""
    job = AnalysisJob(user_id=user_id, symbols_count=1, total_points_charged=20,
                       status=JobStatus.done)
    db.add(job)
    await db.flush()
    r = AnalysisResult(
        job_id=job.id, user_id=user_id, symbol=symbol,
        analysis_type=AnalysisType.full, points_charged=20,
        status=ResultStatus.done,
        final_score=score, decision_level="strong_buy",
        quant_score=70.0,
    )
    db.add(r)
    await db.commit()
    return r


async def test_existing_full_returns_quant(db_session):
    """已有 LLM 历史 → 默认走 quant。"""
    user, _ = await register_or_get_user(db_session, "13800030001")
    await _seed_full_result(db_session, user.id, "600519")

    a_type, pts, _ = await determine_analysis_type(db_session, "600519")
    assert a_type == AnalysisType.quant_only
    assert pts == 1


async def test_force_full_with_today_cache(db_session):
    """force_full + 当日有缓存 → cache_hit (10pt)。"""
    user, _ = await register_or_get_user(db_session, "13800030002")
    await _seed_full_result(db_session, user.id, "600519")

    a_type, pts, cache = await determine_analysis_type(db_session, "600519", force_full=True)
    assert a_type == AnalysisType.full
    assert pts == 10
    assert cache is not None


async def test_submit_analysis_full(db_session):
    """新股票 → 创建 job + 扣 20 + queued。"""
    user, _ = await register_or_get_user(db_session, "13800030003")
    assert user.points == 100

    job = await submit_analysis(db_session, user, ["600519"])
    await db_session.refresh(user)
    assert user.points == 80   # 扣了 20
    assert job.symbols_count == 1
    assert job.total_points_charged == 20

    res = await db_session.execute(
        select(AnalysisResult).where(AnalysisResult.job_id == job.id)
    )
    rec = res.scalar_one()
    assert rec.symbol == "600519"
    assert rec.analysis_type == AnalysisType.full
    assert rec.points_charged == 20
    # 状态可能是 queued 或 running (异步任务可能已开始)
    assert rec.status in (ResultStatus.queued, ResultStatus.running)


async def test_submit_quant_only_after_full(db_session):
    """已有 full → 提交同股 → 走 quant_only 1pt。"""
    user, _ = await register_or_get_user(db_session, "13800030004")
    await _seed_full_result(db_session, user.id, "600519")

    job = await submit_analysis(db_session, user, ["600519"])
    await db_session.refresh(user)
    assert user.points == 99   # 100 - 1
    assert job.total_points_charged == 1

    res = await db_session.execute(
        select(AnalysisResult).where(AnalysisResult.job_id == job.id)
    )
    rec = res.scalar_one()
    assert rec.analysis_type == AnalysisType.quant_only
    assert rec.points_charged == 1


async def test_submit_cache_hit(db_session):
    """force_full + 当日已有 → 创建 cache_hit 直接 done。"""
    user_a, _ = await register_or_get_user(db_session, "13800030005")
    user_b, _ = await register_or_get_user(db_session, "13800030006")
    src = await _seed_full_result(db_session, user_a.id, "600519", score=85.0)

    job = await submit_analysis(db_session, user_b, ["600519"], force_full_set={"600519"})
    await db_session.refresh(user_b)
    assert user_b.points == 90   # 100 - 10

    res = await db_session.execute(
        select(AnalysisResult).where(AnalysisResult.job_id == job.id)
    )
    rec = res.scalar_one()
    assert rec.is_cache_hit
    assert rec.source_result_id == src.id
    assert rec.status == ResultStatus.done
    assert rec.final_score == 85.0


async def test_submit_insufficient(db_session):
    """积分不足应该抛错。"""
    from app.services.points_service import InsufficientPointsError
    user, _ = await register_or_get_user(db_session, "13800030007")
    user.points = 5
    await db_session.commit()

    with pytest.raises(InsufficientPointsError):
        await submit_analysis(db_session, user, ["600519", "600520", "600521"])
