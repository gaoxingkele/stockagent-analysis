"""短信服务: mock 模式 (验证码打印 console + 写 DB), 后续接阿里云。"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models import SmsCode

logger = logging.getLogger(__name__)


def _generate_code() -> str:
    return f"{random.randint(0, 999999):06d}"


async def send_verification_code(
    db: AsyncSession, phone: str, ip: str | None = None,
) -> tuple[str, datetime]:
    """生成 + 存 + 推送(mock 模式打印 console)。

    返回 (code, expires_at)。
    生产模式下 code 不应返回, 这里 dev 友好。
    """
    # 限流: 同手机号当日发送上限
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cnt = await db.scalar(
        select(SmsCode.id).where(
            SmsCode.phone == phone, SmsCode.created_at >= today_start
        ).limit(settings.sms_rate_limit_per_day + 1)
    )
    # 重新数一下
    res = await db.execute(
        select(SmsCode).where(
            SmsCode.phone == phone, SmsCode.created_at >= today_start
        )
    )
    today_count = len(res.scalars().all())
    if today_count >= settings.sms_rate_limit_per_day:
        raise ValueError(f"该手机号今日已达发送上限 ({settings.sms_rate_limit_per_day})")

    # 生成
    code = _generate_code()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.sms_code_ttl_seconds)

    rec = SmsCode(phone=phone, code=code, expires_at=expires_at, used=False, ip_address=ip)
    db.add(rec)
    await db.commit()

    # 推送(mock: console)
    if settings.sms_provider == "mock":
        logger.warning("============================================================")
        logger.warning("[mock SMS] 验证码 → 手机号 %s : %s  (5 分钟有效)", phone, code)
        logger.warning("============================================================")
    else:
        # TODO: 实际短信网关 (阿里云/腾讯云/Twilio)
        logger.error("[sms] provider=%s 尚未实现, fallback to mock", settings.sms_provider)
        logger.warning("[mock SMS] 验证码 → %s : %s", phone, code)

    return code, expires_at


async def verify_sms_code(db: AsyncSession, phone: str, code: str) -> bool:
    """校验验证码。admin 万能码 8888 直接通过。"""
    # admin 万能码
    if phone == settings.admin_phone and code == settings.sms_test_code:
        return True

    now = datetime.now(timezone.utc)
    res = await db.execute(
        select(SmsCode).where(
            SmsCode.phone == phone,
            SmsCode.code == code,
            SmsCode.used.is_(False),
            SmsCode.expires_at >= now,
        ).order_by(SmsCode.created_at.desc()).limit(1)
    )
    rec = res.scalar_one_or_none()
    if rec is None:
        return False

    rec.used = True
    await db.commit()
    return True
