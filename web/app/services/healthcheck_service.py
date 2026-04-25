"""健康检查服务: 14 项 API 联通性 + 大盘快照。

每项独立异步检查, 互不阻塞。失败项不影响其他项。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.db import get_session_factory
from ..core.redis import publish_progress
from ..models import HealthCheck, HealthCheckTriggerType

logger = logging.getLogger(__name__)


# 把主项目 src 加到路径
_PROJECT_SRC = settings.project_root / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))


# ─────────────── 单项检查 ───────────────

async def _check(name: str, coro_fn) -> dict[str, Any]:
    """统一封装: 计时 + 异常捕获。"""
    t0 = time.time()
    try:
        result = await asyncio.wait_for(coro_fn(), timeout=15)
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "name": name, "status": "ok", "latency_ms": latency_ms,
            "info": result if isinstance(result, str) else None,
            "data": result if isinstance(result, dict) else None,
        }
    except asyncio.TimeoutError:
        return {"name": name, "status": "timeout", "latency_ms": 15000, "error": "超时"}
    except Exception as e:
        return {
            "name": name, "status": "fail",
            "latency_ms": int((time.time() - t0) * 1000),
            "error": str(e)[:200],
        }


def _check_tushare_pro_connect() -> str:
    """同步函数 (放线程池)。"""
    from stockagent_analysis.tushare_enrich import _get_pro
    pro = _get_pro()
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN 未配置或 init 失败")
    return "pro_api 已初始化"


def _check_tushare_daily() -> dict:
    from stockagent_analysis.tushare_enrich import _get_pro
    pro = _get_pro()
    if pro is None:
        raise RuntimeError("Tushare not init")
    df = pro.daily(ts_code="600519.SH", limit=3)
    if df is None or df.empty:
        raise RuntimeError("daily 返回空")
    return {"rows": len(df), "latest_date": str(df.iloc[0]["trade_date"])}


def _check_tushare_factor_pro() -> dict:
    from stockagent_analysis.tushare_enrich import fetch_stk_factor_pro
    rows = fetch_stk_factor_pro("600519.SH", days=2)
    if not rows:
        raise RuntimeError("stk_factor_pro 返回空(可能权限不足)")
    return {"rows": len(rows), "factors_count": len(rows[0].keys()) if rows else 0}


def _check_tushare_cyq() -> dict:
    from stockagent_analysis.tushare_enrich import fetch_cyq_perf
    rows = fetch_cyq_perf("600519.SH", days=2)
    if not rows:
        raise RuntimeError("cyq_perf 返回空(可能权限不足)")
    return {"rows": len(rows), "latest_winner_rate": rows[-1].get("winner_rate")}


def _check_tushare_moneyflow() -> dict:
    from stockagent_analysis.tushare_enrich import fetch_moneyflow
    rows = fetch_moneyflow("600519.SH", days=2)
    if not rows:
        raise RuntimeError("moneyflow 返回空(可能权限不足)")
    return {"rows": len(rows)}


def _check_tushare_holdernumber() -> dict:
    from stockagent_analysis.tushare_enrich import fetch_holdernumber
    rows = fetch_holdernumber("600519.SH", periods=2)
    if not rows:
        raise RuntimeError("stk_holdernumber 返回空")
    return {"periods": len(rows)}


async def _check_akshare() -> dict:
    """AKShare 拉一只股票快照。"""
    def _pull():
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            raise RuntimeError("akshare 实时快照空")
        return {"total_stocks": len(df)}
    return await asyncio.to_thread(_pull)


async def _check_tdx() -> str:
    """TDX 测试连接 (简化: 跳过若不可用)。"""
    try:
        from pytdx.hq import TdxHq_API
    except ImportError:
        raise RuntimeError("pytdx 未安装")
    api = TdxHq_API()
    def _try():
        with api.connect("119.147.212.81", 7709):
            return "tdx connected"
    return await asyncio.to_thread(_try)


async def _check_redis() -> str:
    from ..core.redis import _redis, _use_inmemory
    if _use_inmemory:
        return "in-memory broker (Redis 未连接)"
    if _redis is None:
        raise RuntimeError("Redis 未初始化")
    pong = await _redis.ping()
    return f"PONG: {pong}"


async def _check_db(db: AsyncSession) -> dict:
    from sqlalchemy import func, select
    from ..models import User
    cnt = await db.scalar(select(func.count()).select_from(User))
    return {"users_count": int(cnt or 0)}


async def _check_llm_provider(name: str, env_key: str, base_url: str | None = None) -> dict:
    """轻量 LLM 探针: 只校验 token 存在, 不发请求(避免计费)。"""
    import os
    key = os.getenv(env_key, "").strip()
    if not key:
        raise RuntimeError(f"{env_key} 未配置")
    return {"key_length": len(key), "configured": True}


async def _check_market_index() -> dict:
    """大盘指数快照 (akshare)。"""
    def _pull():
        import akshare as ak
        df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
        if df is None or df.empty:
            raise RuntimeError("指数空")
        result = {}
        for _, row in df.iterrows():
            name = row.get("名称") or row.get("name", "")
            if name in ("上证指数", "深证成指", "创业板指", "沪深300", "中证500", "上证50"):
                result[name] = {
                    "current": float(row.get("最新价") or row.get("current", 0)),
                    "pct_chg": float(row.get("涨跌幅") or 0),
                }
        return result
    return await asyncio.to_thread(_pull)


# ─────────────── 编排 ───────────────

async def _to_thread(fn):
    """把同步函数包成 awaitable。"""
    return await asyncio.to_thread(fn)


CHECK_DEFINITIONS = [
    ("tushare.pro_init",       lambda: _to_thread(_check_tushare_pro_connect)),
    ("tushare.daily",          lambda: _to_thread(_check_tushare_daily)),
    ("tushare.stk_factor_pro", lambda: _to_thread(_check_tushare_factor_pro)),
    ("tushare.cyq_perf",       lambda: _to_thread(_check_tushare_cyq)),
    ("tushare.moneyflow",      lambda: _to_thread(_check_tushare_moneyflow)),
    ("tushare.holdernumber",   lambda: _to_thread(_check_tushare_holdernumber)),
    ("akshare.spot",           _check_akshare),
    ("tdx.connect",            _check_tdx),
    ("llm.kimi",       lambda: _check_llm_provider("kimi",       "KIMI_API_KEY")),
    ("llm.grok",       lambda: _check_llm_provider("grok",       "GROK_API_KEY")),
    ("llm.doubao",     lambda: _check_llm_provider("doubao",     "DOUBAO_API_KEY")),
    ("llm.deepseek",   lambda: _check_llm_provider("deepseek",   "DEEPSEEK_API_KEY")),
    ("redis",                  _check_redis),
    ("market.indices",         _check_market_index),
]


async def run_full_healthcheck(
    db: AsyncSession, user_id: int | None,
    trigger_type: HealthCheckTriggerType,
    *, stream: bool = True, hc_record_id: int | None = None,
) -> HealthCheck:
    """跑全套检查 + 落 DB。如果 stream=True, 同步把每项推 SSE channel。"""
    items: list[dict[str, Any]] = []
    market: dict[str, Any] = {}

    # DB 检查 (用同一 session)
    db_check = await _check("sqlite.db", lambda: _check_db(db))
    items.append(db_check)
    if hc_record_id is not None and stream:
        await publish_progress(hc_record_id, {"type": "check_item", "item": db_check})

    # 并发跑其他项
    tasks = []
    for name, fn in CHECK_DEFINITIONS:
        tasks.append(_check(name, fn))

    for coro in asyncio.as_completed(tasks):
        item = await coro
        items.append(item)
        if hc_record_id is not None and stream:
            await publish_progress(hc_record_id, {"type": "check_item", "item": item})
        # 提取大盘快照
        if item["name"] == "market.indices" and item["status"] == "ok":
            market = item.get("data") or {}

    passed = sum(1 for x in items if x["status"] == "ok")
    failed = len(items) - passed
    duration_ms = sum(x.get("latency_ms", 0) for x in items)

    rec = HealthCheck(
        triggered_by_user_id=user_id, trigger_type=trigger_type,
        total_items=len(items), passed=passed, failed=failed,
        duration_ms=duration_ms,
        details_json=items,
        market_snapshot_json=market,
    )
    if hc_record_id is not None:
        # 更新已有记录
        from sqlalchemy import select
        existing = await db.execute(select(HealthCheck).where(HealthCheck.id == hc_record_id))
        ex = existing.scalar_one_or_none()
        if ex is not None:
            ex.total_items = rec.total_items
            ex.passed = rec.passed
            ex.failed = rec.failed
            ex.duration_ms = rec.duration_ms
            ex.details_json = rec.details_json
            ex.market_snapshot_json = rec.market_snapshot_json
            await db.commit()
            if stream:
                await publish_progress(hc_record_id, {
                    "type": "summary",
                    "total": rec.total_items, "passed": passed, "failed": failed,
                    "duration_ms": duration_ms,
                })
            return ex

    db.add(rec)
    await db.commit()
    return rec


async def cron_run_healthcheck() -> None:
    """APScheduler 定时调用。"""
    factory = get_session_factory()
    async with factory() as db:
        try:
            rec = await run_full_healthcheck(
                db, user_id=None, trigger_type=HealthCheckTriggerType.cron,
                stream=False,
            )
            logger.info("[healthcheck cron] 完成: %d/%d 通过 耗时 %dms",
                        rec.passed, rec.total_items, rec.duration_ms)
        except Exception as e:
            logger.exception("[healthcheck cron] 失败: %s", e)
