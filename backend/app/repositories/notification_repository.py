"""
Notification Repository - Data access layer for notifications.
"""

from sqlalchemy import and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.constants import UserRoles
from app.models.notification import Notification, NotificationPreference
from app.models.user import User


class NotificationRepository:
    """Repository for notification data access."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_notification(self, notification: Notification) -> Notification:
        """Create a new notification."""
        self.db.add(notification)
        await self.db.commit()
        await self.db.refresh(notification)
        return notification

    async def get_user_notifications(
        self, user_id: int, limit: int = 50, offset: int = 0, unread_only: bool = False
    ) -> list[Notification]:
        """Get notifications for a user."""
        stmt = (
            select(Notification)
            .where(Notification.user_id == user_id)
            .order_by(desc(Notification.created_at))
            .limit(limit)
            .offset(offset)
        )

        if unread_only:
            stmt = stmt.where(Notification.is_read.is_(False))

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def mark_notification_read(self, notification_id: int, user_id: int) -> bool:
        """Mark notification as read."""
        stmt = select(Notification).where(
            and_(
                Notification.id == notification_id,
                Notification.user_id == user_id,
            )
        )
        result = await self.db.execute(stmt)
        notification = result.scalar_one_or_none()

        if notification and not notification.is_read:
            notification.is_read = True
            notification.read_at = func.now()
            await self.db.commit()
            return True

        return False

    async def mark_all_read(self, user_id: int) -> int:
        """Mark all notifications as read for a user."""
        stmt = select(Notification).where(
            and_(
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
        )
        result = await self.db.execute(stmt)
        notifications = result.scalars().all()

        count = 0
        for notification in notifications:
            notification.is_read = True
            notification.read_at = func.now()
            count += 1

        if count > 0:
            await self.db.commit()

        return count

    async def get_unread_count(self, user_id: int) -> int:
        """Get unread notification count for a user."""
        stmt = select(func.count(Notification.id)).where(
            and_(
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
        )
        result = await self.db.execute(stmt)
        count = result.scalar()
        return count or 0

    async def mark_notification_sent(self, notification_id: int) -> bool:
        """Mark notification as sent."""
        stmt = select(Notification).where(Notification.id == notification_id)
        result = await self.db.execute(stmt)
        notification = result.scalar_one_or_none()

        if notification:
            notification.is_sent = True
            notification.sent_at = func.now()
            await self.db.commit()
            return True

        return False

    async def get_user_preferences(
        self, user_id: int, notification_type: str
    ) -> NotificationPreference | None:
        """Get user notification preferences."""
        stmt = select(NotificationPreference).where(
            and_(
                NotificationPreference.user_id == user_id,
                NotificationPreference.notification_type == notification_type,
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_administrators(self) -> list[User]:
        """Get all administrators."""
        stmt = select(User).where(
            and_(
                User.is_active.is_(True),
                User.role == UserRoles.ADMIN,
            )
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_critical_alert_recipients(self) -> list[User]:
        """Get users who should receive critical alerts."""
        stmt = select(User).where(
            and_(
                User.is_active.is_(True),
                User.role.in_([UserRoles.ADMIN, UserRoles.CARDIOLOGIST]),
            )
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
