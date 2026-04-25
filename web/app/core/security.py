"""安全工具: 密码哈希(PBKDF2-SHA256) + JWT 编解码。

用 hashlib 内置实现避开 passlib+bcrypt 在 Windows + Py3.14 的兼容问题。
PBKDF2-SHA256 + 200k iterations + 16 字节随机盐 = 行业标准强度。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from ..config import settings

_PBKDF2_ITERATIONS = 200_000
_SALT_BYTES = 16
_KEY_BYTES = 32
_ALG = "pbkdf2_sha256"


# ─────────────── 密码 ───────────────

def hash_password(plain: str) -> str:
    """格式: pbkdf2_sha256$<iter>$<salt_b64>$<key_b64>"""
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, _PBKDF2_ITERATIONS, _KEY_BYTES)
    return f"{_ALG}${_PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(key).decode()}"


def verify_password(plain: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        alg, iters_s, salt_b64, key_b64 = hashed.split("$", 3)
        if alg != _ALG:
            return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(key_b64)
        actual = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iters, len(expected))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# ─────────────── JWT ───────────────

def encode_token(user_id: int, *, is_admin: bool = False, expire_minutes: int | None = None) -> str:
    """颁发 access token。"""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expire_minutes or settings.jwt_access_token_expire_minutes
    )
    payload = {
        "sub": str(user_id),
        "is_admin": is_admin,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """解码 + 校验 token。失败抛 jwt.PyJWTError。"""
    return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
