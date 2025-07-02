"""
User Repository - Data access layer for users.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.user import User


class UserRepository:
    """Repository for user data access."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_user(self, user: User) -> User:
        """Create a new user."""
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_user_by_id(self, user_id: int) -> User | None:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        stmt = select(User).where(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_user(
        self, user_id: int, update_data: dict[str, Any]
    ) -> User | None:
        """Update user."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            for key, value in update_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            await self.db.commit()
            await self.db.refresh(user)

        return user

    async def get_users(
        self, limit: int = 50, offset: int = 0, is_active: bool | None = None
    ) -> list[User]:
        """Get users with pagination."""
        stmt = select(User).limit(limit).offset(offset)

        if is_active is not None:
            stmt = stmt.where(User.is_active == is_active)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())
