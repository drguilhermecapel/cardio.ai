"""
Notification models.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import (
    NotificationPriority,
    NotificationType,
)
from app.models.base import Base


class Notification(Base):
    """Notification model."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    notification_type: Mapped[NotificationType] = mapped_column(
        String(50), nullable=False
    )
    priority: Mapped[NotificationPriority] = mapped_column(
        String(20), nullable=False, default=NotificationPriority.NORMAL
    )

    channels: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    related_resource_type: Mapped[str | None] = mapped_column(String(50))
    related_resource_id: Mapped[int | None] = mapped_column(Integer)

    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_delivery_attempt: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    delivery_status: Mapped[str | None] = mapped_column(String(50))

    notification_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<Notification(id={self.id}, type='{self.notification_type}', priority='{self.priority}')>"


class NotificationTemplate(Base):
    """Notification template model."""

    __tablename__ = "notification_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    notification_type: Mapped[NotificationType] = mapped_column(
        String(50), nullable=False
    )

    title_template: Mapped[str] = mapped_column(String(200), nullable=False)
    message_template: Mapped[str] = mapped_column(Text, nullable=False)

    email_subject_template: Mapped[str | None] = mapped_column(String(200))
    email_body_template: Mapped[str | None] = mapped_column(Text)
    sms_template: Mapped[str | None] = mapped_column(String(160))
    push_template: Mapped[str | None] = mapped_column(Text)

    default_channels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    default_priority: Mapped[NotificationPriority] = mapped_column(
        String(20), nullable=False, default=NotificationPriority.NORMAL
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return f"<NotificationTemplate(id={self.id}, name='{self.name}', type='{self.notification_type}')>"


class NotificationPreference(Base):
    """User notification preference model."""

    __tablename__ = "notification_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    notification_type: Mapped[NotificationType] = mapped_column(
        String(50), nullable=False
    )
    enabled_channels: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    quiet_hours_start: Mapped[str | None] = mapped_column(String(5))  # HH:MM
    quiet_hours_end: Mapped[str | None] = mapped_column(String(5))  # HH:MM
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="UTC")

    escalation_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    escalation_delay_minutes: Mapped[int] = mapped_column(
        Integer, default=30, nullable=False
    )
    escalation_channels: Mapped[list[str] | None] = mapped_column(JSON)

    def __repr__(self) -> str:
        return f"<NotificationPreference(id={self.id}, user_id={self.user_id}, type='{self.notification_type}')>"
