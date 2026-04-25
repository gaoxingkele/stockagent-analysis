"""SSR 模板渲染路由 (前端页面)。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.db import get_db
from ..core.deps import get_current_user_optional, get_lang
from ..core.i18n import t as _t
from ..models import AnalysisResult, User

router = APIRouter(tags=["pages"])

templates = Jinja2Templates(directory=str(settings.web_root / "templates"))


def _ctx(request, user, lang, **extra):
    """统一的模板上下文。"""
    return {
        "request": request, "user": user, "lang": lang,
        "t": lambda key, **kw: _t(key, lang, **kw),
        "base_url": settings.base_url,
        **extra,
    }


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("dashboard.html", _ctx(request, user, lang))


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    lang: Annotated[str, Depends(get_lang)],
    invite_code: str | None = None,
):
    return templates.TemplateResponse("login.html",
        _ctx(request, None, lang, invite_code=invite_code))


@router.get("/invite/{code}", response_class=HTMLResponse)
async def invite_landing(
    code: str, request: Request,
    lang: Annotated[str, Depends(get_lang)],
):
    return RedirectResponse(f"/login?invite_code={code}")


@router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("analyze.html", _ctx(request, user, lang))


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(
    job_id: int, request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("job.html", _ctx(request, user, lang, job_id=job_id))


@router.get("/results/{result_id}", response_class=HTMLResponse)
async def result_page(
    result_id: int, request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if user is None:
        return RedirectResponse("/login")
    res = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
    rec = res.scalar_one_or_none()
    if rec is None:
        raise HTTPException(404, "结果不存在")
    if rec.user_id != user.id and not user.is_admin:
        raise HTTPException(403)
    return templates.TemplateResponse("result.html", _ctx(request, user, lang, result=rec))


@router.get("/stock/{symbol}/history", response_class=HTMLResponse)
async def stock_history_page(
    symbol: str, request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("stock_history.html",
        _ctx(request, user, lang, symbol=symbol.upper()))


@router.get("/share", response_class=HTMLResponse)
async def share_page(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("share.html", _ctx(request, user, lang))


@router.get("/me/jobs", response_class=HTMLResponse)
async def my_jobs_page(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("dashboard.html", _ctx(request, user, lang))


@router.get("/me/transactions", response_class=HTMLResponse)
async def my_transactions_page(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("dashboard.html", _ctx(request, user, lang))


@router.get("/system/health", response_class=HTMLResponse)
async def health_page(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse("health.html", _ctx(request, user, lang))


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    user: Annotated[User | None, Depends(get_current_user_optional)],
    lang: Annotated[str, Depends(get_lang)],
):
    if user is None or not user.is_admin:
        return RedirectResponse("/login")
    return templates.TemplateResponse("dashboard.html", _ctx(request, user, lang))
