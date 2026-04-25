"""SQLAlchemy 2.0 async 引擎 + Session 工厂 + Base。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import DateTime
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型基类。"""
    pass


class TimestampMixin:
    """带 created_at 字段的混入(给需要的表用)。"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# 模块级单例 (启动时初始化)
_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_engine(url: str | None = None, echo: bool = False) -> AsyncEngine:
    """初始化全局 engine 和 session 工厂。lifespan 启动时调用。"""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    db_url = url or settings.database_url
    _engine = create_async_engine(db_url, echo=echo, future=True)
    _SessionLocal = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False, autoflush=False,
    )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Engine 未初始化, 先调用 init_engine()")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _SessionLocal is None:
        raise RuntimeError("SessionLocal 未初始化, 先调用 init_engine()")
    return _SessionLocal


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖注入: yield 一个 AsyncSession, 自动关闭。"""
    factory = get_session_factory()
    async with factory() as session:
        yield session
