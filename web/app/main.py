"""FastAPI 应用入口。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .core.db import init_engine, get_session_factory
from .core.logging_config import setup_logging
from .core.redis import close_redis, init_redis
from .services.seed import ensure_admin_user

# 触发模型注册
from . import models  # noqa: F401

# P7: 配置日志
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子。"""
    logger.info("[startup] %s 启动 env=%s base_url=%s",
                settings.app_name, settings.app_env, settings.base_url)

    # P1: 初始化数据库 + 自动创建 admin
    init_engine(echo=False)
    factory = get_session_factory()
    async with factory() as session:
        admin = await ensure_admin_user(session)
        logger.info("[startup] admin ready: id=%d invite_code=%s", admin.id, admin.invite_code)

    # P5: Redis (失败自动 fallback in-memory)
    await init_redis()

    # P6: APScheduler 健康检查定时
    from .tasks.scheduler import init_scheduler, shutdown_scheduler
    init_scheduler()

    yield
    await shutdown_scheduler()
    await close_redis()
    logger.info("[shutdown] %s 关闭", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# 静态资源
app.mount("/static", StaticFiles(directory=str(settings.web_root / "static")), name="static")

# 路由注册
from .routers import admin, admin_logs, analysis, auth, healthcheck, i18n, users   # noqa: E402
app.include_router(auth.router)
app.include_router(i18n.router)
app.include_router(users.router)
app.include_router(analysis.router)
app.include_router(healthcheck.router)
app.include_router(admin.router)
app.include_router(admin_logs.router)


# === 临时路由 (P0 阶段, 后续 P2-P8 拆到 routers/) ===

@app.get("/")
async def root():
    return JSONResponse({
        "app": settings.app_name,
        "version": "0.1.0",
        "status": "ready",
        "phase": "P1 数据库 schema 完成",
        "tables_count": 11,
        "admin_phone": f"{settings.admin_phone[:3]}****{settings.admin_phone[-4:]}",
        "next_phases": [
            "P2 认证 + i18n",
            "P3 积分 + 邀请",
            "P4 双模式分析核心",
            "P5 SSE 进度推送",
            "P6 健康检查",
            "P7 日志",
            "P8 前端互动 HTML",
            "P9 部署",
            "P10 订阅自动跟踪",
            "P11 推送渠道",
        ],
    })


@app.get("/health")
async def health():
    """K8s/ngrok liveness 探针。"""
    return {"status": "ok"}


# === 主入口 (本地直接运行) ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
