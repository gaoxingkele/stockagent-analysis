"""pytest 全局配置 - 内存 SQLite + 异步 fixture + 屏蔽外部 IO。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

# 把 web/ 加到 sys.path
WEB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB_ROOT))

from app.core.db import Base, init_engine, get_session_factory   # noqa: E402
import app.models   # noqa: F401  触发模型注册


@pytest.fixture(autouse=True)
def _no_external_run(monkeypatch):
    """全局屏蔽真实 _run_one (避免测试时联网拉 Tushare/跑 LLM)。"""
    async def fake_run_one(*args, **kwargs):
        return None
    try:
        from app.services import analysis_runner
        monkeypatch.setattr(analysis_runner, "_run_one", fake_run_one)
    except ImportError:
        pass


@pytest_asyncio.fixture
async def db_session():
    """每个测试用一个全新的内存 SQLite 数据库。"""
    engine = init_engine(url="sqlite+aiosqlite:///:memory:", echo=False)

    # 每次重置全局单例 (防止多个测试互相干扰)
    import app.core.db as db_module
    db_module._engine = engine
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    db_module._SessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = get_session_factory()
    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
