"""Phase 0 共享工具 - 字段提取+数值格式化。"""
from __future__ import annotations

from typing import Any


def fmt_num(v: Any, decimals: int = 2, default: str = "N/A", unit: str = "") -> str:
    """把数字格式化成字符串, None/无效时返回 default。"""
    try:
        if v is None:
            return default
        return f"{float(v):.{decimals}f}{unit}"
    except (ValueError, TypeError):
        return default


def fmt_pct(v: Any, decimals: int = 2, default: str = "N/A") -> str:
    return fmt_num(v, decimals, default, unit="%")


def g(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """嵌套取值: g(d, 'features', 'kline_indicators', 'day')。"""
    if not isinstance(d, dict):
        return default
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def tf_label(tf: str) -> str:
    return {"day": "日线", "week": "周线", "month": "月线", "hour_1": "1 小时线"}.get(tf, tf)
