"""Add versioned MCP servers, tool mappings, and Agent tool bindings.

Revision ID: 20260827_0004
Revises: 20260827_0003
"""

from alembic import op
from sqlalchemy import inspect

from app.models.database import Base


revision = "20260827_0004"
down_revision = "20260827_0003"
branch_labels = None
depends_on = None


MCP_TABLES = (
    "mcp_server_definitions",
    "mcp_server_revisions",
    "tool_mapping_definitions",
    "tool_mapping_revisions",
    "agent_tool_bindings",
)


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(inspect(bind).get_table_names())
    for table_name in MCP_TABLES:
        if table_name not in existing_tables:
            Base.metadata.tables[table_name].create(bind=bind)
            existing_tables.add(table_name)


def downgrade() -> None:
    bind = op.get_bind()
    for table_name in reversed(MCP_TABLES):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
