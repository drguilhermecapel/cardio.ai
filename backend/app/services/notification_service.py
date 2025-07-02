"""
Notification Service - Real-time notifications and alerts.
"""

import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import (
    ClinicalUrgency,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
)
from app.models.notification import Notification, NotificationPreference
from app.repositories.notification_repository import NotificationRepository

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications and alerts."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repository = NotificationRepository(db)

    async def send_validation_assignment(
        self, validator_id: int, analysis_id: int, urgency: ClinicalUrgency
    ) -> None:
        """Send validation assignment notification."""
        try:
            priority = self._map_urgency_to_priority(urgency)

            notification = Notification()
            notification.user_id = validator_id
            notification.title = "New ECG Validation Assignment"
            notification.message = (
                f"You have been assigned to validate ECG analysis #{analysis_id}"
            )
            notification.notification_type = NotificationType.VALIDATION_REMINDER
            notification.priority = priority
            notification.channels = [
                NotificationChannel.IN_APP,
                NotificationChannel.EMAIL,
            ]
            notification.related_resource_type = "ecg_analysis"
            notification.related_resource_id = analysis_id

            await self.repository.create_notification(notification)
            await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send validation assignment: {str(e)}")

    async def send_urgent_validation_alert(
        self, validator_id: int, analysis_id: int
    ) -> None:
        """Send urgent validation alert."""
        try:
            notification = Notification()
            notification.user_id = validator_id
            notification.title = "URGENT: Critical ECG Requires Immediate Validation"
            notification.message = (
                f"Critical ECG analysis #{analysis_id} requires immediate validation"
            )
            notification.notification_type = NotificationType.CRITICAL_FINDING
            notification.priority = NotificationPriority.CRITICAL
            notification.channels = [
                NotificationChannel.IN_APP,
                NotificationChannel.EMAIL,
                NotificationChannel.SMS,
                NotificationChannel.PUSH,
            ]
            notification.related_resource_type = "ecg_analysis"
            notification.related_resource_id = analysis_id

            await self.repository.create_notification(notification)
            await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send urgent validation alert: {str(e)}")

    async def send_validation_complete(
        self, user_id: int, analysis_id: int, status: str
    ) -> None:
        """Send validation completion notification."""
        try:
            notification = Notification()
            notification.user_id = user_id
            notification.title = "ECG Validation Complete"
            notification.message = f"ECG analysis #{analysis_id} validation completed with status: {status}"
            notification.notification_type = NotificationType.ANALYSIS_COMPLETE
            notification.priority = NotificationPriority.NORMAL
            notification.channels = [
                NotificationChannel.IN_APP,
                NotificationChannel.EMAIL,
            ]
            notification.related_resource_type = "ecg_analysis"
            notification.related_resource_id = analysis_id

            await self.repository.create_notification(notification)
            await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send validation complete notification: {str(e)}")

    async def send_critical_rejection_alert(self, analysis_id: int) -> None:
        """Send alert when critical analysis is rejected."""
        try:
            recipients = await self.repository.get_critical_alert_recipients()

            for recipient in recipients:
                notification = Notification()
                notification.user_id = recipient.id
                notification.title = "ALERT: Critical ECG Analysis Rejected"
                notification.message = f"Critical ECG analysis #{analysis_id} was rejected during validation"
                notification.notification_type = NotificationType.CRITICAL_FINDING
                notification.priority = NotificationPriority.CRITICAL
                notification.channels = [
                    NotificationChannel.IN_APP,
                    NotificationChannel.EMAIL,
                    NotificationChannel.SMS,
                ]
                notification.related_resource_type = "ecg_analysis"
                notification.related_resource_id = analysis_id

                await self.repository.create_notification(notification)
                await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send critical rejection alert: {str(e)}")

    async def send_no_validator_alert(self, analysis_id: int) -> None:
        """Send alert when no validators are available."""
        try:
            admins = await self.repository.get_administrators()

            for admin in admins:
                notification = Notification()
                notification.user_id = admin.id
                notification.title = "ALERT: No Validators Available"
                notification.message = (
                    f"No validators available for critical ECG analysis #{analysis_id}"
                )
                notification.notification_type = NotificationType.SYSTEM_ALERT
                notification.priority = NotificationPriority.CRITICAL
                notification.channels = [
                    NotificationChannel.IN_APP,
                    NotificationChannel.EMAIL,
                    NotificationChannel.PHONE_CALL,
                ]
                notification.related_resource_type = "ecg_analysis"
                notification.related_resource_id = analysis_id

                await self.repository.create_notification(notification)
                await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send no validator alert: {str(e)}")

    async def send_analysis_complete(
        self, user_id: int, analysis_id: int, has_critical_findings: bool = False
    ) -> None:
        """Send analysis completion notification."""
        try:
            priority = (
                NotificationPriority.HIGH
                if has_critical_findings
                else NotificationPriority.NORMAL
            )
            channels = [NotificationChannel.IN_APP, NotificationChannel.EMAIL]

            if has_critical_findings:
                channels.extend([NotificationChannel.SMS, NotificationChannel.PUSH])

            notification = Notification()
            notification.user_id = user_id
            notification.title = "ECG Analysis Complete"
            notification.message = f"ECG analysis #{analysis_id} has been completed"
            notification.notification_type = NotificationType.ANALYSIS_COMPLETE
            notification.priority = priority
            notification.channels = [str(ch) for ch in channels]
            notification.related_resource_type = "ecg_analysis"
            notification.related_resource_id = analysis_id

            await self.repository.create_notification(notification)
            await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send analysis complete notification: {str(e)}")

    async def send_quality_alert(
        self, user_id: int, analysis_id: int, quality_issues: list[str]
    ) -> None:
        """Send quality alert notification."""
        try:
            issues_text = ", ".join(quality_issues)

            notification = Notification()
            notification.user_id = user_id
            notification.title = "ECG Quality Alert"
            notification.message = (
                f"Quality issues detected in ECG analysis #{analysis_id}: {issues_text}"
            )
            notification.notification_type = NotificationType.QUALITY_ALERT
            notification.priority = NotificationPriority.MEDIUM
            notification.channels = [
                NotificationChannel.IN_APP,
                NotificationChannel.EMAIL,
            ]
            notification.related_resource_type = "ecg_analysis"
            notification.related_resource_id = analysis_id
            notification.notification_metadata = {"quality_issues": quality_issues}

            await self.repository.create_notification(notification)
            await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send quality alert: {str(e)}")

    async def send_system_alert(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> None:
        """Send system-wide alert to administrators."""
        try:
            admins = await self.repository.get_administrators()

            for admin in admins:
                notification = Notification()
                notification.user_id = admin.id
                notification.title = title
                notification.message = message
                notification.notification_type = NotificationType.SYSTEM_ALERT
                notification.priority = priority
                notification.channels = [
                    NotificationChannel.IN_APP,
                    NotificationChannel.EMAIL,
                ]

                await self.repository.create_notification(notification)
                await self._send_notification(notification)

        except Exception as e:
            logger.error(f"Failed to send system alert: {str(e)}")

    async def _send_notification(self, notification: Notification) -> None:
        """Send notification through configured channels."""
        try:
            preferences = await self.repository.get_user_preferences(
                notification.user_id, notification.notification_type
            )

            enabled_channels = self._filter_channels(notification.channels, preferences)

            for channel in enabled_channels:
                try:
                    if channel == NotificationChannel.IN_APP:
                        await self._send_in_app(notification)
                    elif channel == NotificationChannel.EMAIL:
                        await self._send_email(notification)
                    elif channel == NotificationChannel.SMS:
                        await self._send_sms(notification)
                    elif channel == NotificationChannel.PUSH:
                        await self._send_push(notification)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook(notification)
                    elif channel == NotificationChannel.PHONE_CALL:
                        await self._send_phone_call(notification)

                except Exception as e:
                    logger.error(
                        f"Failed to send notification via {channel}: notification_id={notification.id}, error={str(e)}"
                    )

            await self.repository.mark_notification_sent(notification.id)

        except Exception as e:
            logger.error(f"Failed to send notification: {str(e)}")

    async def _send_in_app(self, notification: Notification) -> None:
        """Send in-app notification (already stored in database)."""
        pass

    async def _send_email(self, notification: Notification) -> None:
        """Send email notification."""
        logger.info(
            f"EMAIL: {notification.title} to user {notification.user_id}: {notification.message}"
        )

    async def _send_sms(self, notification: Notification) -> None:
        """Send SMS notification."""
        logger.info(
            f"SMS: {notification.title} to user {notification.user_id}: {notification.message}"
        )

    async def _send_push(self, notification: Notification) -> None:
        """Send push notification."""
        logger.info(
            f"PUSH: {notification.title} to user {notification.user_id}: {notification.message}"
        )

    async def _send_webhook(self, notification: Notification) -> None:
        """Send webhook notification."""
        logger.info(
            f"WEBHOOK: {notification.title} to user {notification.user_id}: {notification.message}"
        )

    async def _send_phone_call(self, notification: Notification) -> None:
        """Send phone call notification."""
        logger.info(
            f"PHONE: {notification.title} to user {notification.user_id}: {notification.message}"
        )

    def _map_urgency_to_priority(
        self, urgency: ClinicalUrgency
    ) -> NotificationPriority:
        """Map clinical urgency to notification priority."""
        mapping = {
            ClinicalUrgency.LOW: NotificationPriority.LOW,
            ClinicalUrgency.MEDIUM: NotificationPriority.NORMAL,
            ClinicalUrgency.HIGH: NotificationPriority.HIGH,
            ClinicalUrgency.CRITICAL: NotificationPriority.CRITICAL,
        }
        return mapping.get(urgency, NotificationPriority.NORMAL)

    def _filter_channels(
        self, channels: list[str], preferences: NotificationPreference | None
    ) -> list[str]:
        """Filter channels based on user preferences."""
        if not preferences:
            return channels

        if self._is_quiet_hours(preferences):
            if NotificationPriority.CRITICAL in channels:
                return [
                    ch
                    for ch in channels
                    if ch in [NotificationChannel.PHONE_CALL, NotificationChannel.SMS]
                ]
            else:
                return [NotificationChannel.IN_APP]  # Only in-app during quiet hours

        return [ch for ch in channels if ch in preferences.enabled_channels]

    def _is_quiet_hours(self, preferences: NotificationPreference) -> bool:
        """Check if current time is within quiet hours."""
        if not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            return False

        current_hour = datetime.now().hour
        start_hour = int(preferences.quiet_hours_start.split(":")[0])
        end_hour = int(preferences.quiet_hours_end.split(":")[0])

        if start_hour <= end_hour:
            return start_hour <= current_hour <= end_hour
        else:  # Overnight quiet hours
            return current_hour >= start_hour or current_hour <= end_hour

    async def get_user_notifications(
        self, user_id: int, limit: int = 50, offset: int = 0, unread_only: bool = False
    ) -> list[Notification]:
        """Get notifications for a user."""
        return await self.repository.get_user_notifications(
            user_id, limit, offset, unread_only
        )

    async def mark_notification_read(self, notification_id: int, user_id: int) -> bool:
        """Mark notification as read."""
        return await self.repository.mark_notification_read(notification_id, user_id)

    async def mark_all_read(self, user_id: int) -> int:
        """Mark all notifications as read for a user."""
        return await self.repository.mark_all_read(user_id)

    async def get_unread_count(self, user_id: int) -> int:
        """Get unread notification count for a user."""
        return await self.repository.get_unread_count(user_id)
