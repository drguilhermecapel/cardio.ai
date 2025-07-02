"""
User Service - User management and authentication.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.db.session import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


class UserService:
    """Service for user management and authentication."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repository = UserRepository(db)

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        hashed_password = get_password_hash(user_data.password)

        user = User()
        user.username = user_data.username
        user.email = user_data.email
        user.hashed_password = hashed_password
        user.first_name = user_data.first_name
        user.last_name = user_data.last_name
        user.phone = user_data.phone
        user.role = user_data.role
        user.license_number = user_data.license_number
        user.specialty = user_data.specialty
        user.institution = user_data.institution
        user.experience_years = user_data.experience_years

        return await self.repository.create_user(user)

    async def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user with username and password."""
        user = await self.repository.get_user_by_username(username)
        if not user:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        return user

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        return await self.repository.get_user_by_email(email)

    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        return await self.repository.get_user_by_username(username)

    async def update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp."""
        await self.repository.update_user(user_id, {"last_login": datetime.utcnow()})

    async def verify_refresh_token(self, refresh_token: str) -> User | None:
        """Verify refresh token and return user."""
        try:
            payload = jwt.decode(
                refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            username_raw = payload.get("sub")
            token_type_raw = payload.get("type")
            username: str = str(username_raw) if username_raw is not None else ""
            token_type: str = str(token_type_raw) if token_type_raw is not None else ""

            if not username or token_type != "refresh":
                return None

        except JWTError:
            return None

        user = await self.repository.get_user_by_username(username)
        return user

    @staticmethod
    async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        """Get current authenticated user."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            username_raw = payload.get("sub")
            username: str = str(username_raw) if username_raw is not None else ""
            if not username:
                raise credentials_exception

        except JWTError as e:
            raise credentials_exception from e

        user_service = UserService(db)
        user = await user_service.get_user_by_username(username)
        if user is None:
            raise credentials_exception

        return user

    async def update_user(
        self, user_id: int, update_data: dict[str, Any]
    ) -> User | None:
        """Update user with provided data."""
        return await self.repository.update_user(user_id, update_data)
