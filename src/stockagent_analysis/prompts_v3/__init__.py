"""v3 角色 Prompt 库 - 从文件加载 .txt prompts。"""
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent

_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """按名称加载 prompts_v3/<name>.txt。缓存到内存。"""
    if name in _CACHE:
        return _CACHE[name]
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    _CACHE[name] = text
    return text
