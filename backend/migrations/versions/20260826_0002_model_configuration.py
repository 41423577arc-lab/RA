"""Add encrypted model configuration and profile revisions.

Revision ID: 20260826_0002
Revises: 20260826_0001
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.models.database import Base


revision = "20260826_0002"
down_revision = "20260826_0001"
branch_labels = None
depends_on = None


MODEL_TABLES = (
    "config_secrets",
    "model_connections",
    "model_connection_revisions",
    "model_profiles",
    "model_profile_revisions",
)


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(inspect(bind).get_table_names())
    for table_name in MODEL_TABLES:
        if table_name not in existing_tables:
            Base.metadata.tables[table_name].create(bind=bind)
            existing_tables.add(table_name)

    secret_columns = {
        item["name"] for item in inspect(bind).get_columns("config_secrets")
    }
    if "version" in secret_columns and "key_version" not in secret_columns:
        op.alter_column(
            "config_secrets",
            "version",
            new_column_name="key_version",
            existing_type=sa.Integer(),
            existing_nullable=False,
        )

    columns = {item["name"] for item in inspect(bind).get_columns("agent_node_bindings")}
    if "model_profile_revision_id" not in columns:
        op.add_column(
            "agent_node_bindings",
            sa.Column("model_profile_revision_id", sa.String(length=36), nullable=True),
        )
        op.create_index(
            "ix_agent_node_bindings_model_profile_revision_id",
            "agent_node_bindings",
            ["model_profile_revision_id"],
        )
        if bind.dialect.name != "sqlite":
            op.create_foreign_key(
                "fk_agent_node_bindings_model_profile_revision_id",
                "agent_node_bindings",
                "model_profile_revisions",
                ["model_profile_revision_id"],
                ["id"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("agent_node_bindings")}
    if "model_profile_revision_id" in columns:
        if bind.dialect.name == "sqlite":
            indexes = {
                item["name"] for item in inspector.get_indexes("agent_node_bindings")
            }
            if "ix_agent_node_bindings_model_profile_revision_id" in indexes:
                op.drop_index(
                    "ix_agent_node_bindings_model_profile_revision_id",
                    table_name="agent_node_bindings",
                )
            with op.batch_alter_table("agent_node_bindings") as batch_op:
                batch_op.drop_column("model_profile_revision_id")
        else:
            for foreign_key in inspector.get_foreign_keys("agent_node_bindings"):
                if foreign_key.get("constrained_columns") == [
                    "model_profile_revision_id"
                ] and foreign_key.get("name"):
                    op.drop_constraint(
                        foreign_key["name"],
                        "agent_node_bindings",
                        type_="foreignkey",
                    )
            indexes = {
                item["name"] for item in inspector.get_indexes("agent_node_bindings")
            }
            if "ix_agent_node_bindings_model_profile_revision_id" in indexes:
                op.drop_index(
                    "ix_agent_node_bindings_model_profile_revision_id",
                    table_name="agent_node_bindings",
                )
            op.drop_column("agent_node_bindings", "model_profile_revision_id")
    for table_name in reversed(MODEL_TABLES):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
