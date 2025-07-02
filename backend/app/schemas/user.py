"""
User model.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import UserRoles
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ecg_analysis import ECGAnalysis
    from app.models.validation import Validation


class User(Base):
    """User model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20))

    role: Mapped[UserRoles] = mapped_column(
        String(20), nullable=False, default=UserRoles.VIEWER
    )
    license_number: Mapped[str | None] = mapped_column(String(50))
    specialty: Mapped[str | None] = mapped_column(String(100))
    institution: Mapped[str | None] = mapped_column(String(200))
    experience_years: Mapped[int | None] = mapped_column(Integer)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    digital_signature_key: Mapped[str | None] = mapped_column(Text)
    signature_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    notification_preferences: Mapped[str | None] = mapped_column(Text)  # JSON
    ui_preferences: Mapped[str | None] = mapped_column(Text)  # JSON

    analyses: Mapped[list["ECGAnalysis"]] = relationship(
        "ECGAnalysis",
        back_populates="created_by_user",
        foreign_keys="ECGAnalysis.created_by",
    )
    validations: Mapped[list["Validation"]] = relationship(
        "Validation", back_populates="validator", foreign_keys="Validation.validator_id"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}"

    @property
    def is_physician(self) -> bool:
        """Check if user is a physician."""
        return self.role in [UserRoles.PHYSICIAN, UserRoles.CARDIOLOGIST]

    @property
    def can_validate_critical(self) -> bool:
        """Check if user can validate critical findings."""
        return (
            self.role in [UserRoles.CARDIOLOGIST, UserRoles.PHYSICIAN]
            and self.experience_years is not None
            and self.experience_years >= 5
        )


class APIKey(Base):
    """API key model."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    scopes: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rate_limit_per_minute: Mapped[int] = mapped_column(
        Integer, default=60, nullable=False
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class UserSession(Base):
    """User session model for tracking active sessions."""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"
