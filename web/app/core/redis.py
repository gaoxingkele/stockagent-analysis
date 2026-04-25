"""Redis 连接 + 进度事件 pub/sub。

如果 Redis 连接失败 → 自动 fallback 到 in-memory broker (单进程开发)。
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


_redis = None
_use_inmemory = False
_inmemory_subs: dict[str, list[asyncio.Queue]] = {}


async def init_redis() -> None:
    """启动时初始化 Redis 连接, 失败则 fallback 到 in-memory。"""
    global _redis, _use_inmemory
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await _redis.ping()
        _use_inmemory = False
        logger.info("[redis] 已连接: %s", settings.redis_url)
    except Exception as e:
        logger.warning("[redis] 连接失败 (%s), 使用 in-memory broker", e)
        _redis = None
        _use_inmemory = True


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


def progress_channel(result_id: int) -> str:
    return f"progress:result:{result_id}"


async def publish_progress(result_id: int, payload: dict[str, Any]) -> None:
    """推送进度事件。同时落 SSE 订阅 + DB(由调用者负责落 DB)。"""
    msg = json.dumps(payload, ensure_ascii=False, default=str)
    ch = progress_channel(result_id)

    if _use_inmemory:
        for q in _inmemory_subs.get(ch, []):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
        return

    if _redis is not None:
        try:
            await _redis.publish(ch, msg)
        except Exception as e:
            logger.error("[redis publish] %s", e)


async def subscribe_progress(result_id: int) -> AsyncIterator[dict[str, Any]]:
    """订阅一个结果的进度流, 异步迭代器。"""
    ch = progress_channel(result_id)

    if _use_inmemory:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        _inmemory_subs.setdefault(ch, []).append(q)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=60)
                except asyncio.TimeoutError:
                    yield {"type": "ping"}   # 心跳
                    continue
                yield json.loads(msg)
        finally:
            try:
                _inmemory_subs.get(ch, []).remove(q)
            except ValueError:
                pass
        return

    if _redis is None:
        return

    pubsub = _redis.pubsub()
    await pubsub.subscribe(ch)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True, timeout=60), timeout=65)
            except asyncio.TimeoutError:
                yield {"type": "ping"}
                continue
            if msg and msg.get("type") == "message":
                yield json.loads(msg["data"])
    finally:
        await pubsub.unsubscribe(ch)
        await pubsub.close()
