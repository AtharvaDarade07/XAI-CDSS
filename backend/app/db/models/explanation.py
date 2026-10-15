"""Feature-level explanation detail associated with a prediction."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True, nullable=False)
    method: Mapped[str] = mapped_column(String(30), nullable=False)
    feature: Mapped[str] = mapped_column(String(150), nullable=False)
    feature_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    contribution: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(30), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prediction: Mapped["Prediction"] = relationship(back_populates="explanations")
