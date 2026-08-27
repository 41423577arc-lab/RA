"""Add users, authentication sessions, conversations, and resource ownership.

Revision ID: 20260827_0005
Revises: 20260827_0004
"""

from datetime import datetime, timezone
from uuid import uuid4

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, select

from app.models.database import Base


revision = "20260827_0005"
down_revision = "20260827_0004"
branch_labels = None
depends_on = None

SYSTEM_TENANT_ID = "00000000-0000-0000-0000-000000000001"
SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000003"
NEW_TABLES = ("users", "auth_sessions", "conversations", "conversation_messages")
OWNERSHIP_COLUMNS = {
    "intake_sessions": ("tenant_id", "owner_id", "conversation_id"),
    "research_tasks": ("tenant_id", "owner_id", "conversation_id"),
    "agent_runs": ("owner_id", "started_by", "conversation_id"),
}


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(inspect(bind).get_table_names())
    for table_name in NEW_TABLES:
        if table_name not in existing_tables:
            Base.metadata.tables[table_name].create(bind=bind)
            existing_tables.add(table_name)

    users = Base.metadata.tables["users"]
    if bind.execute(select(users.c.id).where(users.c.id == SYSTEM_USER_ID)).first() is None:
        bind.execute(
            users.insert().values(
                id=SYSTEM_USER_ID,
                tenant_id=SYSTEM_TENANT_ID,
                email="system@local.invalid",
                display_name="Local User",
                password_hash="disabled",
                role="SYSTEM",
                status="ACTIVE",
            )
        )

    for table_name, column_names in OWNERSHIP_COLUMNS.items():
        existing_columns = {
            item["name"] for item in inspect(bind).get_columns(table_name)
        }
        for column_name in column_names:
            if column_name not in existing_columns:
                op.add_column(
                    table_name,
                    sa.Column(column_name, sa.String(length=36), nullable=True),
                )
                op.create_index(
                    f"ix_{table_name}_{column_name}", table_name, [column_name]
                )

    _backfill_legacy_resources(bind)


def _backfill_legacy_resources(bind) -> None:
    conversations = Base.metadata.tables["conversations"]
    intake_rows = bind.execute(
        sa.text("SELECT id, research_task_id FROM intake_sessions")
    ).mappings()
    for row in intake_rows:
        conversation_id = str(uuid4())
        now = datetime.now(timezone.utc)
        bind.execute(
            conversations.insert().values(
                id=conversation_id,
                tenant_id=SYSTEM_TENANT_ID,
                owner_id=SYSTEM_USER_ID,
                title="历史调查",
                status="ACTIVE",
                intake_session_id=row["id"],
                latest_task_id=row["research_task_id"],
                created_at=now,
                updated_at=now,
            )
        )
        bind.execute(
            sa.text(
                "UPDATE intake_sessions SET tenant_id=:tenant_id, owner_id=:owner_id, "
                "conversation_id=:conversation_id WHERE id=:id"
            ),
            {
                "tenant_id": SYSTEM_TENANT_ID,
                "owner_id": SYSTEM_USER_ID,
                "conversation_id": conversation_id,
                "id": row["id"],
            },
        )
        bind.execute(
            sa.text(
                "UPDATE research_tasks SET tenant_id=:tenant_id, owner_id=:owner_id, "
                "conversation_id=:conversation_id WHERE intake_session_id=:intake_id"
            ),
            {
                "tenant_id": SYSTEM_TENANT_ID,
                "owner_id": SYSTEM_USER_ID,
                "conversation_id": conversation_id,
                "intake_id": row["id"],
            },
        )
        bind.execute(
            sa.text(
                "UPDATE agent_runs SET owner_id=:owner_id, started_by=:owner_id, "
                "conversation_id=:conversation_id WHERE intake_session_id=:intake_id"
            ),
            {
                "owner_id": SYSTEM_USER_ID,
                "conversation_id": conversation_id,
                "intake_id": row["id"],
            },
        )

    task_rows = bind.execute(
        sa.text(
            "SELECT id FROM research_tasks WHERE conversation_id IS NULL"
        )
    ).mappings()
    for row in task_rows:
        conversation_id = str(uuid4())
        now = datetime.now(timezone.utc)
        bind.execute(
            conversations.insert().values(
                id=conversation_id,
                tenant_id=SYSTEM_TENANT_ID,
                owner_id=SYSTEM_USER_ID,
                title="历史分析任务",
                status="ACTIVE",
                latest_task_id=row["id"],
                created_at=now,
                updated_at=now,
            )
        )
        bind.execute(
            sa.text(
                "UPDATE research_tasks SET tenant_id=:tenant_id, owner_id=:owner_id, "
                "conversation_id=:conversation_id WHERE id=:id"
            ),
            {
                "tenant_id": SYSTEM_TENANT_ID,
                "owner_id": SYSTEM_USER_ID,
                "conversation_id": conversation_id,
                "id": row["id"],
            },
        )
        bind.execute(
            sa.text(
                "UPDATE agent_runs SET owner_id=:owner_id, started_by=:owner_id, "
                "conversation_id=:conversation_id WHERE research_task_id=:task_id"
            ),
            {
                "owner_id": SYSTEM_USER_ID,
                "conversation_id": conversation_id,
                "task_id": row["id"],
            },
        )


def downgrade() -> None:
    bind = op.get_bind()
    for table_name, column_names in reversed(list(OWNERSHIP_COLUMNS.items())):
        existing_columns = {
            item["name"] for item in inspect(bind).get_columns(table_name)
        }
        for column_name in reversed(column_names):
            if column_name not in existing_columns:
                continue
            index_name = f"ix_{table_name}_{column_name}"
            indexes = {item["name"] for item in inspect(bind).get_indexes(table_name)}
            if index_name in indexes:
                op.drop_index(index_name, table_name=table_name)
            if bind.dialect.name == "sqlite":
                with op.batch_alter_table(table_name) as batch_op:
                    batch_op.drop_column(column_name)
            else:
                op.drop_column(table_name, column_name)
    for table_name in reversed(NEW_TABLES):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
