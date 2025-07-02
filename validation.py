"""
Validation models.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import ValidationStatus
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ecg_analysis import ECGAnalysis
    from app.models.user import User


class Validation(Base):
    """Validation model for medical review of ECG analyses."""

    __tablename__ = "validations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ecg_analyses.id"), nullable=False, index=True
    )
    validator_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    status: Mapped[ValidationStatus] = mapped_column(
        String(20), nullable=False, default=ValidationStatus.PENDING
    )
    validation_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    agrees_with_ai: Mapped[bool | None] = mapped_column(Boolean)
    clinical_notes: Mapped[str | None] = mapped_column(Text)
    corrected_diagnosis: Mapped[str | None] = mapped_column(String(200))
    corrected_urgency: Mapped[str | None] = mapped_column(String(20))

    signal_quality_rating: Mapped[int | None] = mapped_column(Integer)  # 1-5 scale
    ai_confidence_rating: Mapped[int | None] = mapped_column(Integer)  # 1-5 scale
    overall_quality_score: Mapped[float | None] = mapped_column(Float)

    recommendations: Mapped[list[str] | None] = mapped_column(JSON)
    follow_up_required: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    follow_up_notes: Mapped[str | None] = mapped_column(Text)

    digital_signature: Mapped[str | None] = mapped_column(String(255))
    signature_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    validation_duration_minutes: Mapped[int | None] = mapped_column(Integer)
    requires_second_opinion: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    analysis: Mapped["ECGAnalysis"] = relationship(
        "ECGAnalysis", back_populates="validations"
    )
    validator: Mapped["User"] = relationship(
        "User", back_populates="validations", foreign_keys=[validator_id]
    )

    def __repr__(self) -> str:
        return f"<Validation(id={self.id}, analysis_id={self.analysis_id}, status='{self.status}')>"

    @property
    def is_completed(self) -> bool:
        """Check if validation is completed."""
        return self.status in [ValidationStatus.APPROVED, ValidationStatus.REJECTED]


class ValidationRule(Base):
    """Validation rule model for automated quality checks."""

    __tablename__ = "validation_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    rule_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # threshold, pattern, ml

    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # info, warning, error, critical

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    applies_to_conditions: Mapped[list[str] | None] = mapped_column(JSON)

    created_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")

    def __repr__(self) -> str:
        return f"<ValidationRule(id={self.id}, name='{self.name}', type='{self.rule_type}')>"


class ValidationResult(Base):
    """Validation result model for storing rule check results."""

    __tablename__ = "validation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ecg_analyses.id"), nullable=False, index=True
    )
    rule_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("validation_rules.id"), nullable=False, index=True
    )

    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    execution_time_ms: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<ValidationResult(id={self.id}, rule_id={self.rule_id}, passed={self.passed})>"


class QualityMetric(Base):
    """Quality metric model for tracking analysis quality over time."""

    __tablename__ = "quality_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    analysis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ecg_analyses.id"), nullable=False, index=True
    )

    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[str | None] = mapped_column(String(20))

    normal_min: Mapped[float | None] = mapped_column(Float)
    normal_max: Mapped[float | None] = mapped_column(Float)
    is_within_normal: Mapped[bool | None] = mapped_column(Boolean)

    calculation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<QualityMetric(id={self.id}, name='{self.metric_name}', value={self.metric_value})>"
