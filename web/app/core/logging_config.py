"""结构化日志配置: 6 个分级文件 + JSON 格式 + RotatingFileHandler。

启动时调用 setup_logging() 替换默认 logging。
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings


class JsonFormatter(logging.Formatter):
    """简单 JSON 格式 (不引入 structlog 依赖也能用)。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        # 携带额外字段
        for k in ("user_id", "request_id", "result_id", "duration_ms"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _file_handler(filename: str, level: int) -> logging.handlers.RotatingFileHandler:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(
        log_dir / filename,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(JsonFormatter())
    return h


class _ModuleFilter(logging.Filter):
    """只放行特定 logger 命名空间的日志。"""

    def __init__(self, prefixes: list[str]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record):
        return any(record.name.startswith(p) for p in self.prefixes)


def setup_logging() -> None:
    """启动时调用一次。"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    # 清掉现有 handler 避免重复
    for h in list(root.handlers):
        root.removeHandler(h)

    # console handler (开发可读)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # app.log: 全局 INFO+
    app_h = _file_handler("app.log", logging.INFO)
    root.addHandler(app_h)

    # error.log: ERROR+ (含异常 traceback)
    err_h = _file_handler("error.log", logging.ERROR)
    root.addHandler(err_h)

    # analysis.log: 仅 app.services.analysis_runner / quant_runner / progress_service
    ana_h = _file_handler("analysis.log", logging.INFO)
    ana_h.addFilter(_ModuleFilter([
        "app.services.analysis_runner",
        "app.services.progress_service",
    ]))
    root.addHandler(ana_h)

    # healthcheck.log
    hc_h = _file_handler("healthcheck.log", logging.INFO)
    hc_h.addFilter(_ModuleFilter([
        "app.services.healthcheck_service",
        "app.routers.healthcheck",
        "app.tasks.scheduler",
    ]))
    root.addHandler(hc_h)

    # api.log: HTTP 路由
    api_h = _file_handler("api.log", logging.INFO)
    api_h.addFilter(_ModuleFilter([
        "app.routers", "uvicorn.access", "fastapi",
    ]))
    root.addHandler(api_h)

    # llm.log: LLM 调用 (主项目命名空间)
    llm_h = _file_handler("llm.log", logging.INFO)
    llm_h.addFilter(_ModuleFilter([
        "stockagent_analysis",
    ]))
    root.addHandler(llm_h)

    # 减少 SQLAlchemy 噪音
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
