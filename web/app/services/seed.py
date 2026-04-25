"""数据库种子: 启动时自动创建管理员账户。"""
from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models import PointTransaction, TransactionReason, User, UserStatus

logger = logging.getLogger(__name__)


def generate_invite_code() -> str:
    """生成 1 字母+6 数字邀请码 (A123456 格式)。"""
    letter = random.choice(string.ascii_uppercase)
    digits = "".join(random.choices(string.digits, k=6))
    return f"{letter}{digits}"


async def ensure_admin_user(db: AsyncSession) -> User:
    """检查/创建管理员账户 (config.admin_phone)。"""
    admin_phone = settings.admin_phone

    result = await db.execute(select(User).where(User.phone == admin_phone))
    admin = result.scalar_one_or_none()

    if admin is not None:
        # 确保 is_admin 标志正确(防止后台改库后状态不一致)
        if not admin.is_admin:
            admin.is_admin = True
            await db.commit()
            logger.warning("[seed] 已修正 admin 标志: phone=%s id=%d", admin_phone, admin.id)
        else:
            logger.info("[seed] admin 已存在: id=%d phone=%s points=%d invite_code=%s",
                        admin.id, admin_phone, admin.points, admin.invite_code)
        return admin

    # 创建新 admin
    admin = User(
        phone=admin_phone,
        nickname="Admin",
        points=settings.points_register_bonus,
        is_admin=True,
        language=settings.default_language,
        invite_code=generate_invite_code(),
        invite_path="",   # admin 是根, 没有上级
        status=UserStatus.active,
        last_login_at=None,
    )
    db.add(admin)
    await db.flush()    # 拿 admin.id 用于 transaction

    # 注册赠送积分流水
    tx = PointTransaction(
        user_id=admin.id,
        delta=settings.points_register_bonus,
        reason=TransactionReason.register_bonus,
        balance_before=0,
        balance_after=settings.points_register_bonus,
        note="管理员账户初始化",
    )
    db.add(tx)
    await db.commit()
    await db.refresh(admin)

    logger.info("[seed] admin 已创建: id=%d phone=%s invite_code=%s points=%d",
                admin.id, admin_phone, admin.invite_code, admin.points)
    return admin
