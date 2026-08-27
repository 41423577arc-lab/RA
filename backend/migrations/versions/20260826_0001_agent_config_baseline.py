"""Create the current application schema and Agent configuration baseline.

This one-time bridge is intentionally idempotent for both existing databases and
clean installations. The Agent tables are declared explicitly so later model,
prompt, and user fields cannot leak backward into this baseline revision.

Revision ID: 20260826_0001
Revises: None
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from app.models.database import Base, INTAKE_JSON


revision = "20260826_0001"
down_revision = None
branch_labels = None
depends_on = None


LEGACY_APPLICATION_TABLES = (
    "research_tasks",
    "intake_sessions",
    "intake_audio_jobs",
    "llm_call_logs",
    "execution_events",
)


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in LEGACY_APPLICATION_TABLES:
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)

    existing = set(inspect(bind).get_table_names())
    if "tenants" not in existing:
        op.create_table(
            "tenants",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("slug", sa.String(length=100), nullable=False, unique=True),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )
    if "agent_definitions" not in existing:
        op.create_table(
            "agent_definitions",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("tenant_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("slug", sa.String(length=100), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("published_version_id", sa.String(length=36)),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
            sa.UniqueConstraint("tenant_id", "slug"),
        )
        op.create_index("ix_agent_definitions_tenant_id", "agent_definitions", ["tenant_id"])
    if "agent_versions" not in existing:
        op.create_table(
            "agent_versions",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("agent_definition_id", sa.String(length=36), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("config_schema_version", sa.Integer(), nullable=False),
            sa.Column("config", INTAKE_JSON, nullable=False),
            sa.Column("config_hash", sa.String(length=64), nullable=False),
            sa.Column("published_at", sa.DateTime(timezone=True)),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["agent_definition_id"], ["agent_definitions.id"]),
            sa.UniqueConstraint("agent_definition_id", "version"),
        )
        op.create_index(
            "ix_agent_versions_agent_definition_id",
            "agent_versions",
            ["agent_definition_id"],
        )
    if "agent_node_bindings" not in existing:
        op.create_table(
            "agent_node_bindings",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("agent_version_id", sa.String(length=36), nullable=False),
            sa.Column("node_key", sa.String(length=100), nullable=False),
            sa.Column("model_config", INTAKE_JSON, nullable=False),
            sa.Column("prompt_config", INTAKE_JSON, nullable=False),
            sa.Column("allowed_tools", INTAKE_JSON, nullable=False),
            sa.ForeignKeyConstraint(["agent_version_id"], ["agent_versions.id"]),
            sa.UniqueConstraint("agent_version_id", "node_key"),
        )
        op.create_index(
            "ix_agent_node_bindings_agent_version_id",
            "agent_node_bindings",
            ["agent_version_id"],
        )
    if "agent_runs" not in existing:
        op.create_table(
            "agent_runs",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("tenant_id", sa.String(length=36), nullable=False),
            sa.Column("agent_definition_id", sa.String(length=36), nullable=False),
            sa.Column("agent_version_id", sa.String(length=36), nullable=False),
            sa.Column("config_schema_version", sa.Integer(), nullable=False),
            sa.Column("resolved_config_snapshot", INTAKE_JSON, nullable=False),
            sa.Column("config_hash", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("intake_session_id", sa.String(length=36)),
            sa.Column("research_task_id", sa.String(length=36)),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
            sa.ForeignKeyConstraint(["agent_definition_id"], ["agent_definitions.id"]),
            sa.ForeignKeyConstraint(["agent_version_id"], ["agent_versions.id"]),
        )
        op.create_index("ix_agent_runs_tenant_id", "agent_runs", ["tenant_id"])
        op.create_index(
            "ix_agent_runs_agent_definition_id",
            "agent_runs",
            ["agent_definition_id"],
        )
        op.create_index(
            "ix_agent_runs_agent_version_id",
            "agent_runs",
            ["agent_version_id"],
        )
        op.create_index(
            "ix_agent_runs_intake_session_id",
            "agent_runs",
            ["intake_session_id"],
            unique=True,
        )
        op.create_index(
            "ix_agent_runs_research_task_id",
            "agent_runs",
            ["research_task_id"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    for table_name in (
        "agent_runs",
        "agent_node_bindings",
        "agent_versions",
        "agent_definitions",
        "tenants",
    ):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
