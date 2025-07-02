"""Focused User Service Tests."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from unittest.mock import AsyncMock, patch
from app.services.user_service import UserService
from app.schemas.user import UserCreate
from app.models.user import User
from app.core.constants import UserRoles


@pytest.fixture
def user_service(test_db):
    """Create user service instance."""
    return UserService(db=test_db)


@pytest.mark.asyncio
async def test_create_user_success(user_service):
    """Test successful user creation."""
    user_data = UserCreate(
        username="testuser",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        password="TestPassword123!",  # Valid password with uppercase
        role=UserRoles.PHYSICIAN,
        experience_years=5,
    )

    mock_user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        role=UserRoles.PHYSICIAN,
        experience_years=5,
    )

    user_service.repository.create_user = AsyncMock(return_value=mock_user)

    user = await user_service.create_user(user_data)

    assert user is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.role == UserRoles.PHYSICIAN


@pytest.mark.asyncio
async def test_authenticate_user_success(user_service):
    """Test successful user authentication."""
    mock_user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # Valid bcrypt hash
        is_active=True,
    )

    user_service.repository.get_user_by_username = AsyncMock(return_value=mock_user)

    with patch("app.services.user_service.verify_password", return_value=True):
        user = await user_service.authenticate_user("testuser", "password")

    assert user is not None
    assert user.username == "testuser"


@pytest.mark.asyncio
async def test_authenticate_user_invalid_password(user_service):
    """Test authentication with invalid password."""
    mock_user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        is_active=True,
    )

    user_service.repository.get_user_by_username = AsyncMock(return_value=mock_user)

    with patch("app.core.security.verify_password", return_value=False):
        user = await user_service.authenticate_user("testuser", "wrongpassword")

    assert user is None


@pytest.mark.asyncio
async def test_get_user_by_email(user_service):
    """Test getting user by email."""
    mock_user = User(id=1, username="testuser", email="test@example.com")

    user_service.repository.get_user_by_email = AsyncMock(return_value=mock_user)

    user = await user_service.get_user_by_email("test@example.com")

    assert user is not None
    assert user.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_user_by_username(user_service):
    """Test getting user by username."""
    mock_user = User(id=1, username="testuser", email="test@example.com")

    user_service.repository.get_user_by_username = AsyncMock(return_value=mock_user)

    user = await user_service.get_user_by_username("testuser")

    assert user is not None
    assert user.username == "testuser"


@pytest.mark.asyncio
async def test_update_last_login(user_service):
    """Test updating user's last login."""
    user_service.repository.update_user = AsyncMock()

    await user_service.update_last_login(1)

    user_service.repository.update_user.assert_called_once()
