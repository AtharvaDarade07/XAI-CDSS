"""Import all ORM entities so Base.metadata is complete for Alembic."""

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.explanation import Explanation
from backend.app.db.models.model_version import ModelVersion
from backend.app.db.models.patient import Patient
from backend.app.db.models.patient_record import PatientRecord
from backend.app.db.models.prediction import Prediction
from backend.app.db.models.role import Role
from backend.app.db.models.user import User

__all__ = [
    "AuditLog",
    "Explanation",
    "ModelVersion",
    "Patient",
    "PatientRecord",
    "Prediction",
    "Role",
    "User",
]
