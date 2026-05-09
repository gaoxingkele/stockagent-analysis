"""extend analysis_type enum + add extra_data_json for V12

Revision ID: c0f3v12ext01
Revises: 9f7c7e8c093b
Create Date: 2026-05-09 19:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c0f3v12ext01"
down_revision: Union[str, None] = "9f7c7e8c093b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # 1. 扩 AnalysisType enum (Postgres 走 ALTER TYPE; SQLite 是字符串无需操作)
    if dialect == "postgresql":
        op.execute("ALTER TYPE analysis_type_enum ADD VALUE IF NOT EXISTS 'v12_market'")
        op.execute("ALTER TYPE analysis_type_enum ADD VALUE IF NOT EXISTS 'v12_llm_filter'")
    # SQLite: enum 作为 string 存储, 不需要 schema 变更 (SQLAlchemy enforces 在 ORM 层)

    # 2. 给 analysis_results 加 extra_data_json (V12 用来存推荐快照/llm 视觉结果)
    with op.batch_alter_table("analysis_results", schema=None) as batch_op:
        batch_op.add_column(sa.Column("extra_data_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("analysis_results", schema=None) as batch_op:
        batch_op.drop_column("extra_data_json")
    # enum 值无法回滚 (Postgres 不支持 DROP VALUE), SQLite 不需要
