"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-11-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op
from app.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    dims = get_settings().embedding_dimensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_code", sa.String(64), nullable=False, unique=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("access_level", sa.String(16), nullable=False),
        sa.Column("source_authority", sa.String(16), nullable=False),
        sa.Column(
            "related_document_codes",
            postgresql.ARRAY(sa.String(64)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source_updated_at", sa.Date(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("type IN ('SOP', 'Chat')", name="ck_documents_type"),
        sa.CheckConstraint("status IN ('active', 'inactive')", name="ck_documents_status"),
        sa.CheckConstraint(
            "access_level IN ('public', 'internal', 'confidential')",
            name="ck_documents_access_level",
        ),
        sa.CheckConstraint(
            "source_authority IN ('authoritative', 'example')",
            name="ck_documents_source_authority",
        ),
        sa.CheckConstraint("version > 0", name="ck_documents_version_positive"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_key", sa.String(200), nullable=False, unique=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dims), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("inserted_count", sa.Integer(), nullable=False),
        sa.Column("updated_count", sa.Integer(), nullable=False),
        sa.Column("skipped_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("error_summary", sa.Text()),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'partial', 'failed')",
            name="ck_ingestion_runs_status",
        ),
    )

    op.create_table(
        "query_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_id", sa.String(64), nullable=False, unique=True),
        sa.Column("actor_hash", sa.String(64), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False),
        sa.Column("redacted_query", sa.Text()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason_code", sa.String(64)),
        sa.Column("best_relevance", sa.Float()),
        sa.Column("model", sa.String(100)),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_query_logs_actor_hash", "query_logs", ["actor_hash"])
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])

    op.create_table(
        "retrieval_hits",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "query_log_id",
            sa.Uuid(),
            sa.ForeignKey("query_logs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.Uuid(), sa.ForeignKey("document_chunks.id", ondelete="SET NULL")),
        sa.Column("chunk_key", sa.String(200), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("cosine_similarity", sa.Float(), nullable=False),
    )
    op.create_index("ix_retrieval_hits_query_log_id", "retrieval_hits", ["query_log_id"])
    op.create_index("ix_retrieval_hits_chunk_id", "retrieval_hits", ["chunk_id"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"])


def downgrade() -> None:
    for table in (
        "evaluation_runs",
        "retrieval_hits",
        "query_logs",
        "ingestion_runs",
        "document_chunks",
        "documents",
    ):
        op.drop_table(table)
