"""
Patient schemas.
"""

from datetime import date, datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


class PatientBase(BaseModel):
    """Base patient schema."""

    patient_id: str = Field(..., min_length=1, max_length=50)
    mrn: str | None = Field(None, max_length=50)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    gender: str = Field(..., pattern="^(male|female|other)$")
    phone: str | None = Field(None, max_length=20)
    email: EmailStr | None = None
    address: str | None = None
    height_cm: int | None = Field(None, ge=50, le=250)
    weight_kg: float | None = Field(None, ge=1.0, le=500.0)
    blood_type: str | None = Field(None, pattern="^(A|B|AB|O)[+-]?$")

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, v: date) -> date:
        """Validate date of birth."""
        today = date.today()
        if v > today:
            raise ValueError("Date of birth cannot be in the future")
        if (today - v).days > 365 * 150:  # 150 years
            raise ValueError("Date of birth cannot be more than 150 years ago")
        return v


class PatientCreate(PatientBase):
    """Patient creation schema."""

    emergency_contact_name: str | None = Field(None, max_length=200)
    emergency_contact_phone: str | None = Field(None, max_length=20)
    emergency_contact_relationship: str | None = Field(None, max_length=50)
    allergies: list[str] | None = None
    medications: list[str] | None = None
    medical_history: list[str] | None = None
    family_history: list[str] | None = None
    insurance_provider: str | None = Field(None, max_length=200)
    insurance_number: str | None = Field(None, max_length=50)
    consent_for_research: bool = False


class PatientUpdate(BaseModel):
    """Patient update schema."""

    mrn: str | None = Field(None, max_length=50)
    first_name: str | None = Field(None, min_length=1, max_length=100)
    last_name: str | None = Field(None, min_length=1, max_length=100)
    phone: str | None = Field(None, max_length=20)
    email: EmailStr | None = None
    address: str | None = None
    height_cm: int | None = Field(None, ge=50, le=250)
    weight_kg: float | None = Field(None, ge=1.0, le=500.0)
    blood_type: str | None = Field(None, pattern="^(A|B|AB|O)[+-]?$")
    emergency_contact_name: str | None = Field(None, max_length=200)
    emergency_contact_phone: str | None = Field(None, max_length=20)
    emergency_contact_relationship: str | None = Field(None, max_length=50)
    allergies: list[str] | None = None
    medications: list[str] | None = None
    medical_history: list[str] | None = None
    family_history: list[str] | None = None
    insurance_provider: str | None = Field(None, max_length=200)
    insurance_number: str | None = Field(None, max_length=50)


class PatientInDB(PatientBase):
    """Patient in database schema."""

    id: int
    is_active: bool
    consent_for_research: bool
    consent_date: datetime | None
    data_retention_until: datetime | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Patient(PatientInDB):
    """Patient response schema."""

    age: int
    bmi: float | None

    class Config:
        from_attributes = True


class PatientList(BaseModel):
    """Patient list response schema."""

    patients: list[Patient]
    total: int
    page: int
    size: int


class PatientSearch(BaseModel):
    """Patient search schema."""

    query: str = Field(..., min_length=1, max_length=100)
    search_fields: list[str] = Field(
        default=["patient_id", "mrn", "first_name", "last_name", "email"], min_length=1
    )


class PatientNoteBase(BaseModel):
    """Base patient note schema."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    note_type: str = Field(..., min_length=1, max_length=50)
    is_confidential: bool = False


class PatientNoteCreate(PatientNoteBase):
    """Patient note creation schema."""

    pass


class PatientNoteUpdate(BaseModel):
    """Patient note update schema."""

    title: str | None = Field(None, min_length=1, max_length=200)
    content: str | None = Field(None, min_length=1)
    note_type: str | None = Field(None, min_length=1, max_length=50)
    is_confidential: bool | None = None


class PatientNote(PatientNoteBase):
    """Patient note response schema."""

    id: int
    patient_id: int
    author_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatientNoteList(BaseModel):
    """Patient note list response schema."""

    notes: list[PatientNote]
    total: int
    page: int
    size: int
