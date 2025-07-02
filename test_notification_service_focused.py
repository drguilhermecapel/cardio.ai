"""Focused Notification Service Tests."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from unittest.mock import AsyncMock
from app.services.notification_service import NotificationService
from app.models.notification import Notification
from app.core.constants import ClinicalUrgency


@pytest.fixture
def notification_service(test_db):
    """Create notification service instance."""
    return NotificationService(db=test_db)


@pytest.mark.asyncio
async def test_send_validation_assignment(notification_service):
    """Test sending validation assignment notification."""
    notification_service.repository.create_notification = AsyncMock()
    notification_service._send_notification = AsyncMock()

    await notification_service.send_validation_assignment(
        validator_id=1, analysis_id=1, urgency=ClinicalUrgency.HIGH
    )

    notification_service.repository.create_notification.assert_called_once()
    notification_service._send_notification.assert_called_once()


@pytest.mark.asyncio
async def test_send_urgent_validation_alert(notification_service):
    """Test sending urgent validation alert."""
    notification_service.repository.create_notification = AsyncMock()
    notification_service._send_notification = AsyncMock()

    await notification_service.send_urgent_validation_alert(
        validator_id=1, analysis_id=1
    )

    notification_service.repository.create_notification.assert_called_once()
    notification_service._send_notification.assert_called_once()


@pytest.mark.asyncio
async def test_send_validation_complete(notification_service):
    """Test sending validation complete notification."""
    notification_service.repository.create_notification = AsyncMock()
    notification_service._send_notification = AsyncMock()

    await notification_service.send_validation_complete(
        user_id=1, analysis_id=1, status="approved"
    )

    notification_service.repository.create_notification.assert_called_once()
    notification_service._send_notification.assert_called_once()


@pytest.mark.asyncio
async def test_send_analysis_complete(notification_service):
    """Test sending analysis complete notification."""
    notification_service.repository.create_notification = AsyncMock()
    notification_service._send_notification = AsyncMock()

    await notification_service.send_analysis_complete(
        user_id=1, analysis_id=1, has_critical_findings=True
    )

    notification_service.repository.create_notification.assert_called_once()
    notification_service._send_notification.assert_called_once()


@pytest.mark.asyncio
async def test_get_user_notifications(notification_service):
    """Test getting user notifications."""
    mock_notifications = [Notification(), Notification()]
    notification_service.repository.get_user_notifications = AsyncMock(
        return_value=mock_notifications
    )

    notifications = await notification_service.get_user_notifications(
        user_id=1, limit=10, offset=0
    )

    assert len(notifications) == 2


@pytest.mark.asyncio
async def test_mark_notification_read(notification_service):
    """Test marking notification as read."""
    notification_service.repository.mark_notification_read = AsyncMock(
        return_value=True
    )

    result = await notification_service.mark_notification_read(
        notification_id=1, user_id=1
    )

    assert result is True


@pytest.mark.asyncio
async def test_mark_all_read(notification_service):
    """Test marking all notifications as read."""
    notification_service.repository.mark_all_read = AsyncMock(return_value=5)

    count = await notification_service.mark_all_read(user_id=1)

    assert count == 5


@pytest.mark.asyncio
async def test_get_unread_count(notification_service):
    """Test getting unread notification count."""
    notification_service.repository.get_unread_count = AsyncMock(return_value=3)

    count = await notification_service.get_unread_count(user_id=1)

    assert count == 3
