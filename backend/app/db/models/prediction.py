"""Persisted model prediction record; prediction calculation is elsewhere."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, Float, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("predicted_probability >= 0 AND predicted_probability <= 1", name="ck_predictions_probability_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True, nullable=False)
    patient_record_id: Mapped[int] = mapped_column(ForeignKey("patient_records.id"), index=True, nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="predictions")
    patient_record: Mapped["PatientRecord"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    explanations: Mapped[list["Explanation"]] = relationship(back_populates="prediction")
