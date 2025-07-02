"""
Patient model.
"""

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ecg_analysis import ECGAnalysis


class Patient(Base):
    """Patient model."""

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    patient_id: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    mrn: Mapped[str | None] = mapped_column(
        String(50), index=True
    )  # Medical Record Number

    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    date_of_birth: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)

    phone: Mapped[str | None] = mapped_column(String(20))
    email: Mapped[str | None] = mapped_column(String(255))
    address: Mapped[str | None] = mapped_column(Text)

    height_cm: Mapped[int | None] = mapped_column(Integer)
    weight_kg: Mapped[float | None] = mapped_column()
    blood_type: Mapped[str | None] = mapped_column(String(5))

    emergency_contact_name: Mapped[str | None] = mapped_column(String(200))
    emergency_contact_phone: Mapped[str | None] = mapped_column(String(20))
    emergency_contact_relationship: Mapped[str | None] = mapped_column(String(50))

    allergies: Mapped[str | None] = mapped_column(Text)  # JSON
    medications: Mapped[str | None] = mapped_column(Text)  # JSON
    medical_history: Mapped[str | None] = mapped_column(Text)  # JSON
    family_history: Mapped[str | None] = mapped_column(Text)  # JSON

    insurance_provider: Mapped[str | None] = mapped_column(String(200))
    insurance_number: Mapped[str | None] = mapped_column(String(50))

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    consent_for_research: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    consent_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    data_retention_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    created_by: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    analyses: Mapped[list["ECGAnalysis"]] = relationship(
        "ECGAnalysis", back_populates="patient", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, patient_id='{self.patient_id}', name='{self.full_name}')>"

    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self) -> int:
        """Calculate age."""
        today = date.today()
        return (
            today.year
            - self.date_of_birth.year
            - (
                (today.month, today.day)
                < (self.date_of_birth.month, self.date_of_birth.day)
            )
        )

    @property
    def bmi(self) -> float | None:
        """Calculate BMI."""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m**2), 1)
        return None


class PatientNote(Base):
    """Patient note model."""

    __tablename__ = "patient_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    author_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    note_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # clinical, administrative, etc.

    is_confidential: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    def __repr__(self) -> str:
        return f"<PatientNote(id={self.id}, patient_id={self.patient_id}, type='{self.note_type}')>"
