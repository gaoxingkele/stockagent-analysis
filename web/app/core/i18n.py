"""轻量 i18n: JSON 字典 + 三语支持 (zh-CN / en-US / zh-TW)。

使用:
    from app.core.i18n import t
    t("login.title", "zh-CN")  → "登录"
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parent.parent.parent / "locales"


@lru_cache(maxsize=8)
def _load_lang(lang: str) -> dict[str, str]:
    f = LOCALES_DIR / f"{lang}.json"
    if not f.exists():
        logger.warning("[i18n] %s.json 缺失, fallback to %s", lang, settings.default_language)
        if lang != settings.default_language:
            return _load_lang(settings.default_language)
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("[i18n] load %s 失败: %s", lang, e)
        return {}


def t(key: str, lang: str | None = None, default: str | None = None, **kwargs) -> str:
    """翻译。支持 {var} 占位符替换。"""
    if not lang or lang not in settings.supported_lang_list:
        lang = settings.default_language
    bundle = _load_lang(lang)
    val = bundle.get(key, default if default is not None else key)
    if kwargs:
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val


def all_translations(lang: str) -> dict[str, str]:
    """前端用 GET /api/i18n/{lang} 一次拿全套。"""
    if lang not in settings.supported_lang_list:
        lang = settings.default_language
    return _load_lang(lang)


def normalize_lang(raw: str | None) -> str:
    """归一化语言标识。"""
    if not raw:
        return settings.default_language
    raw = raw.strip()
    # 处理 Accept-Language 头格式 zh-CN,zh;q=0.9
    if "," in raw:
        raw = raw.split(",")[0].split(";")[0].strip()
    # 处理 zh_CN → zh-CN
    raw = raw.replace("_", "-")
    for s in settings.supported_lang_list:
        if s.lower() == raw.lower():
            return s
        # zh → zh-CN, en → en-US
        if s.lower().startswith(raw.lower().split("-")[0]):
            return s
    return settings.default_language
