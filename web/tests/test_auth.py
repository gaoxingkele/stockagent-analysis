"""P2 认证测试: 验证码登录 / admin 密码登录 / JWT / i18n。"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.core.security import (
    decode_token, encode_token, hash_password, verify_password,
)
from app.core.i18n import normalize_lang, t
from app.models import (
    PointTransaction, SmsCode, TransactionReason, User, UserStatus,
)
from app.services.auth_service import (
    AuthError, login_with_code, login_with_password, register_or_get_user,
)
from app.services.invite_service import (
    bind_invite_relation, find_inviter_by_code, generate_invite_code,
    generate_unique_invite_code,
)
from app.services.seed import DEFAULT_ADMIN_PASSWORD, ensure_admin_user
from app.services.sms_service import send_verification_code, verify_sms_code

pytestmark = pytest.mark.asyncio


# ─────────────── security ───────────────

def test_password_hash_verify():
    h = hash_password("MyP@ss123")
    assert verify_password("MyP@ss123", h)
    assert not verify_password("wrong", h)
    assert not verify_password("MyP@ss123", None)


def test_jwt_roundtrip():
    token = encode_token(user_id=42, is_admin=True)
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert payload["is_admin"] is True
    assert payload["type"] == "access"


# ─────────────── i18n ───────────────

def test_i18n_basic():
    assert t("login.title", "zh-CN") == "登录 / 注册"
    assert t("login.title", "en-US") == "Login / Register"
    assert t("login.title", "zh-TW") == "登入 / 註冊"
    # 未知 key fallback
    assert t("not.exists", "zh-CN") == "not.exists"
    assert t("not.exists", "zh-CN", default="X") == "X"


def test_i18n_format():
    msg = t("analyze.insufficient_points", "zh-CN", need=20, have=5)
    assert "20" in msg and "5" in msg


def test_i18n_normalize():
    assert normalize_lang("zh-CN") == "zh-CN"
    assert normalize_lang("zh_CN") == "zh-CN"
    assert normalize_lang("en") == "en-US"
    assert normalize_lang("zh-CN,zh;q=0.9") == "zh-CN"
    assert normalize_lang(None) == "zh-CN"
    assert normalize_lang("ja") == "zh-CN"   # fallback


# ─────────────── SMS ───────────────

async def test_send_and_verify_code(db_session):
    code, _ = await send_verification_code(db_session, "13800001234")
    assert len(code) == 6
    assert code.isdigit()
    assert await verify_sms_code(db_session, "13800001234", code)
    # 已用过不能再用
    assert not await verify_sms_code(db_session, "13800001234", code)


async def test_admin_master_code(db_session):
    """admin 万能码 8888 不需要先 send。"""
    assert await verify_sms_code(db_session, "18606099618", "8888")
    # 普通手机号无效
    assert not await verify_sms_code(db_session, "13800009999", "8888")


async def test_sms_rate_limit(db_session):
    from app.config import settings
    for _ in range(settings.sms_rate_limit_per_day):
        await send_verification_code(db_session, "13800002222")
    # 超出
    with pytest.raises(ValueError):
        await send_verification_code(db_session, "13800002222")


# ─────────────── 邀请 ───────────────

def test_invite_code_format():
    for _ in range(20):
        c = generate_invite_code()
        assert len(c) == 7
        assert c[0].isalpha() and c[0].isupper()
        assert c[1:].isdigit()


async def test_invite_relation_binding(db_session):
    inviter, _ = await register_or_get_user(db_session, "13800003001")
    # 模拟用户用邀请码注册
    invitee, _ = await register_or_get_user(
        db_session, "13800003002", invite_code=inviter.invite_code,
    )

    # 介绍人 +100
    await db_session.refresh(inviter)
    assert inviter.invite_count == 1
    assert inviter.invite_earned_points == 100
    # 介绍人原本 100, 现在 200
    assert inviter.points == 200

    # 新人 +50 (基础 100 + 邀请额外 50)
    await db_session.refresh(invitee)
    assert invitee.points == 150
    assert invitee.invited_by_user_id == inviter.id
    assert invitee.invite_path == f"{inviter.id}/"


# ─────────────── 注册 + 登录 ───────────────

async def test_admin_auto_marked(db_session):
    user, is_new = await register_or_get_user(db_session, "18606099618")
    assert is_new
    assert user.is_admin
    assert user.points == 100


async def test_login_with_code(db_session):
    user, token = await login_with_code(db_session, "18606099618", "8888")
    assert user.is_admin
    assert token
    payload = decode_token(token)
    assert payload["is_admin"] is True


async def test_login_with_code_wrong(db_session):
    with pytest.raises(AuthError):
        await login_with_code(db_session, "13800009999", "999999")


# ─────────────── admin 密码登录 ───────────────

async def test_admin_password_login(db_session):
    """seed 创建 admin 后, 用默认密码登录。"""
    admin = await ensure_admin_user(db_session)
    assert admin.password_hash is not None

    user, token = await login_with_password(db_session, "18606099618", DEFAULT_ADMIN_PASSWORD)
    assert user.id == admin.id
    payload = decode_token(token)
    assert payload["is_admin"] is True


async def test_password_login_wrong(db_session):
    await ensure_admin_user(db_session)
    with pytest.raises(AuthError):
        await login_with_password(db_session, "18606099618", "wrong")


async def test_password_login_non_admin(db_session):
    """普通用户不能用密码登录(没设密码 hash)。"""
    user, _ = await register_or_get_user(db_session, "13800004444")
    with pytest.raises(AuthError):
        await login_with_password(db_session, "13800004444", "anypass")
