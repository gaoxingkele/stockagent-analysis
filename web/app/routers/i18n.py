"""i18n API: 拉取翻译字典 + 切换语言。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from ..config import settings
from ..core.i18n import all_translations, normalize_lang

router = APIRouter(prefix="/api/i18n", tags=["i18n"])


@router.get("/{lang}")
async def get_translations(lang: str):
    if lang not in settings.supported_lang_list:
        raise HTTPException(404, f"Unsupported language: {lang}")
    return {
        "lang": lang,
        "supported": settings.supported_lang_list,
        "translations": all_translations(lang),
    }


@router.post("/set/{lang}")
async def set_language(lang: str, response: Response):
    """切换语言, 写 cookie。"""
    normalized = normalize_lang(lang)
    response.set_cookie("lang", normalized, max_age=365 * 24 * 3600, samesite="lax")
    return {"lang": normalized}
