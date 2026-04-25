"""管理员日志查看: 读取磁盘日志文件 (按 module/level 过滤)。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import settings
from ..core.deps import get_admin_user
from ..models import User

router = APIRouter(prefix="/api/admin/logs", tags=["admin-logs"])


LOG_FILES = {
    "app": "app.log",
    "error": "error.log",
    "analysis": "analysis.log",
    "healthcheck": "healthcheck.log",
    "api": "api.log",
    "llm": "llm.log",
}


@router.get("")
async def list_log_files(_: Annotated[User, Depends(get_admin_user)]):
    """列出可用日志文件 + 大小。"""
    log_dir = Path(settings.log_dir)
    items = []
    for key, fname in LOG_FILES.items():
        f = log_dir / fname
        if f.exists():
            items.append({
                "key": key, "filename": fname,
                "size_bytes": f.stat().st_size,
                "modified_at": f.stat().st_mtime,
            })
        else:
            items.append({"key": key, "filename": fname, "size_bytes": 0})
    return {"logs": items, "log_dir": str(log_dir.absolute())}


@router.get("/{key}")
async def tail_log(
    key: str,
    _: Annotated[User, Depends(get_admin_user)],
    lines: int = Query(200, ge=1, le=2000),
    level: str | None = Query(None),
):
    """tail 最近 N 行 + 按 level 过滤。"""
    if key not in LOG_FILES:
        raise HTTPException(404, f"未知日志: {key}")
    f = Path(settings.log_dir) / LOG_FILES[key]
    if not f.exists():
        return {"lines": [], "total": 0}

    # 读最后 N 行
    raw_lines = []
    try:
        with f.open("rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            chunk_size = min(size, 64 * 1024 * lines // 50)   # 估算
            fp.seek(max(0, size - chunk_size))
            data = fp.read().decode("utf-8", errors="replace")
            raw_lines = [l for l in data.splitlines() if l.strip()][-lines:]
    except Exception as e:
        return {"lines": [], "error": str(e)}

    parsed = []
    level_upper = (level or "").upper()
    for ln in raw_lines:
        try:
            obj = json.loads(ln)
            if level_upper and obj.get("level") != level_upper:
                continue
            parsed.append(obj)
        except Exception:
            parsed.append({"raw": ln})
    return {"lines": parsed, "total": len(parsed), "log_file": str(f)}
