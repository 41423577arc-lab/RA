"""Add optional release notes to Agent versions.

Revision ID: 20260828_0006
Revises: 20260827_0005
"""

from alembic import op
import sqlalchemy as sa


revision = "20260828_0006"
down_revision = "20260827_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_versions", sa.Column("release_note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_versions", "release_note")
