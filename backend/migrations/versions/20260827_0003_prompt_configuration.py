"""Add versioned Prompt definitions and node bindings.

Revision ID: 20260827_0003
Revises: 20260826_0002
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.models.database import Base


revision = "20260827_0003"
down_revision = "20260826_0002"
branch_labels = None
depends_on = None


PROMPT_TABLES = (
    "prompt_definitions",
    "prompt_revisions",
)
PROMPT_BINDING_INDEX = "ix_agent_node_bindings_prompt_revision_id"


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(inspect(bind).get_table_names())
    for table_name in PROMPT_TABLES:
        if table_name not in existing_tables:
            Base.metadata.tables[table_name].create(bind=bind)
            existing_tables.add(table_name)

    columns = {item["name"] for item in inspect(bind).get_columns("agent_node_bindings")}
    if "prompt_revision_id" not in columns:
        op.add_column(
            "agent_node_bindings",
            sa.Column("prompt_revision_id", sa.String(length=36), nullable=True),
        )
        op.create_index(
            PROMPT_BINDING_INDEX,
            "agent_node_bindings",
            ["prompt_revision_id"],
        )
        if bind.dialect.name != "sqlite":
            op.create_foreign_key(
                "fk_agent_node_bindings_prompt_revision_id",
                "agent_node_bindings",
                "prompt_revisions",
                ["prompt_revision_id"],
                ["id"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("agent_node_bindings")}
    if "prompt_revision_id" in columns:
        indexes = {item["name"] for item in inspector.get_indexes("agent_node_bindings")}
        if PROMPT_BINDING_INDEX in indexes:
            op.drop_index(PROMPT_BINDING_INDEX, table_name="agent_node_bindings")
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("agent_node_bindings") as batch_op:
                batch_op.drop_column("prompt_revision_id")
        else:
            for foreign_key in inspector.get_foreign_keys("agent_node_bindings"):
                if foreign_key.get("constrained_columns") == [
                    "prompt_revision_id"
                ] and foreign_key.get("name"):
                    op.drop_constraint(
                        foreign_key["name"],
                        "agent_node_bindings",
                        type_="foreignkey",
                    )
            op.drop_column("agent_node_bindings", "prompt_revision_id")
    for table_name in reversed(PROMPT_TABLES):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
