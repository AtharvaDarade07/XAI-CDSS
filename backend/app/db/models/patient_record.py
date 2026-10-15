"""Raw clinical/dataset input fields; preprocessing stays in the ML pipeline."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class PatientRecord(Base):
    __tablename__ = "patient_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True, nullable=False)
    pregnancies: Mapped[int | None] = mapped_column(Integer, nullable=True)
    glucose: Mapped[float | None] = mapped_column(Float, nullable=True)
    blood_pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    skin_thickness: Mapped[float | None] = mapped_column(Float, nullable=True)
    insulin: Mapped[float | None] = mapped_column(Float, nullable=True)
    bmi: Mapped[float | None] = mapped_column(Float, nullable=True)
    diabetes_pedigree_function: Mapped[float | None] = mapped_column(Float, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="records")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="patient_record")
