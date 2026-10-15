"""Create the initial normalized backend schema.

Revision ID: 20261015_0001
Revises:
Create Date: 2026-10-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261015_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("algorithm", sa.String(length=100), nullable=False),
        sa.Column("artifact_path", sa.String(length=500), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
    )
    op.create_table(
        "patients",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_code", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("patient_code"),
    )
    op.create_index(op.f("ix_patients_patient_code"), "patients", ["patient_code"], unique=True)

    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_roles_name"), "roles", ["name"], unique=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("username"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_role_id"), "users", ["role_id"], unique=False)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=100), nullable=False),
        sa.Column("entity_id", sa.String(length=100), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audit_logs_user_id"), "audit_logs", ["user_id"], unique=False)

    op.create_table(
        "patient_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("pregnancies", sa.Integer(), nullable=True),
        sa.Column("glucose", sa.Float(), nullable=True),
        sa.Column("blood_pressure", sa.Float(), nullable=True),
        sa.Column("skin_thickness", sa.Float(), nullable=True),
        sa.Column("insulin", sa.Float(), nullable=True),
        sa.Column("bmi", sa.Float(), nullable=True),
        sa.Column("diabetes_pedigree_function", sa.Float(), nullable=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_patient_records_patient_id"), "patient_records", ["patient_id"], unique=False)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("patient_record_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("predicted_probability", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("predicted_probability >= 0 AND predicted_probability <= 1", name="ck_predictions_probability_range"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"]),
        sa.ForeignKeyConstraint(["patient_record_id"], ["patient_records.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_model_version_id"), "predictions", ["model_version_id"], unique=False)
    op.create_index(op.f("ix_predictions_patient_id"), "predictions", ["patient_id"], unique=False)
    op.create_index(op.f("ix_predictions_patient_record_id"), "predictions", ["patient_record_id"], unique=False)

    op.create_table(
        "explanations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(length=30), nullable=False),
        sa.Column("feature", sa.String(length=150), nullable=False),
        sa.Column("feature_value", sa.Float(), nullable=True),
        sa.Column("contribution", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(length=30), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_explanations_prediction_id"), "explanations", ["prediction_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_explanations_prediction_id"), table_name="explanations")
    op.drop_table("explanations")
    op.drop_index(op.f("ix_predictions_patient_record_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_patient_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_version_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_patient_records_patient_id"), table_name="patient_records")
    op.drop_table("patient_records")
    op.drop_index(op.f("ix_audit_logs_user_id"), table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_index(op.f("ix_users_role_id"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_roles_name"), table_name="roles")
    op.drop_table("roles")
    op.drop_index(op.f("ix_patients_patient_code"), table_name="patients")
    op.drop_table("patients")
    op.drop_table("model_versions")
