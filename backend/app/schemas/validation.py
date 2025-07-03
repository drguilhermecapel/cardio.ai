"""
Validation schemas.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.core.constants import ValidationStatus


class ValidationBase(BaseModel):
    """Base validation schema."""

    analysis_id: int
    validator_id: int


class ValidationCreate(ValidationBase):
    """Validation creation schema."""

    pass


class ValidationUpdate(BaseModel):
    """Validation update schema."""

    status: ValidationStatus | None = None
    agrees_with_ai: bool | None = None
    clinical_notes: str | None = None
    corrected_diagnosis: str | None = None
    corrected_urgency: str | None = None
    signal_quality_rating: int | None = Field(None, ge=1, le=5)
    ai_confidence_rating: int | None = Field(None, ge=1, le=5)
    overall_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    recommendations: list[str] | None = None
    follow_up_required: bool | None = None
    follow_up_notes: str | None = None
    validation_duration_minutes: int | None = None
    digital_signature: str | None = None


class ValidationSubmit(BaseModel):
    """Validation submission schema."""

    approved: bool = True
    agrees_with_ai: bool | None = None
    clinical_notes: str | None = None
    corrected_diagnosis: str | None = None
    corrected_urgency: str | None = None
    signal_quality_rating: int | None = Field(None, ge=1, le=5)
    ai_confidence_rating: int | None = Field(None, ge=1, le=5)
    overall_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    recommendations: list[str] | None = None
    follow_up_required: bool = False
    follow_up_notes: str | None = None
    validation_duration_minutes: int | None = None
    digital_signature: str | None = None


class ValidationInDB(ValidationBase):
    """Validation in database schema."""

    id: int
    status: ValidationStatus
    validation_date: datetime | None
    agrees_with_ai: bool | None
    clinical_notes: str | None
    corrected_diagnosis: str | None
    corrected_urgency: str | None
    signal_quality_rating: int | None
    ai_confidence_rating: int | None
    overall_quality_score: float | None
    recommendations: list[str] | None
    follow_up_required: bool
    follow_up_notes: str | None
    digital_signature: str | None
    signature_timestamp: datetime | None
    validation_duration_minutes: int | None
    requires_second_opinion: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Validation(ValidationInDB):
    """Validation response schema."""

    pass


class ValidationList(BaseModel):
    """Validation list response schema."""

    validations: list[Validation]
    total: int
    page: int
    size: int
