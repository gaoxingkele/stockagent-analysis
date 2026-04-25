"""数据库种子: 启动时自动创建管理员账户。"""
from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.security import hash_password
from ..models import PointTransaction, TransactionReason, User, UserStatus

logger = logging.getLogger(__name__)


def generate_invite_code() -> str:
    """生成 1 字母+6 数字邀请码 (A123456 格式)。"""
    letter = random.choice(string.ascii_uppercase)
    digits = "".join(random.choices(string.digits, k=6))
    return f"{letter}{digits}"


DEFAULT_ADMIN_PASSWORD = "Ab18606099618"


async def ensure_admin_user(db: AsyncSession) -> User:
    """检查/创建管理员账户 (config.admin_phone)。

    - 不存在 → 创建 (含 password_hash + invite_code + 100 积分)
    - 存在但 password_hash 缺失 → 补默认密码
    - 存在但 is_admin=False → 修正
    """
    admin_phone = settings.admin_phone

    result = await db.execute(select(User).where(User.phone == admin_phone))
    admin = result.scalar_one_or_none()

    if admin is not None:
        changed = False
        if not admin.is_admin:
            admin.is_admin = True
            changed = True
            logger.warning("[seed] 修正 admin 标志: phone=%s id=%d", admin_phone, admin.id)
        if not admin.password_hash:
            admin.password_hash = hash_password(DEFAULT_ADMIN_PASSWORD)
            changed = True
            logger.warning("[seed] admin 密码已设置为默认值")
        if changed:
            await db.commit()
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
        password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
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

    logger.info("[seed] admin 已创建: id=%d phone=%s invite_code=%s points=%d 密码=%s",
                admin.id, admin_phone, admin.invite_code, admin.points, DEFAULT_ADMIN_PASSWORD)
    return admin
