import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip(
        "User service tests skipped in standalone mode", allow_module_level=True
    )

from app.services.user_service import UserService
from app.schemas.user import UserCreate


@pytest.mark.asyncio
async def test_user_service_initialization():
    """Test user service initialization."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    assert service.db == mock_session
    assert hasattr(service, "repository")


@pytest.mark.asyncio
async def test_create_user():
    """Test user creation."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    user_data = UserCreate(
        username="testuser",
        email="test@example.com",
        password="Password123!",
        first_name="Test",
        last_name="User",
    )

    with patch.object(service, "repository") as mock_repo:
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_repo.create_user = AsyncMock(return_value=mock_user)

        result = await service.create_user(user_data)

        mock_repo.create_user.assert_called_once()
        assert result.id == 1
        assert result.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_user_by_email():
    """Test getting user by email."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_repo.get_user_by_email = AsyncMock(return_value=mock_user)

        result = await service.get_user_by_email("test@example.com")

        mock_repo.get_user_by_email.assert_called_once_with("test@example.com")
        assert result.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_user_by_username():
    """Test getting user by username."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_repo.get_user_by_username = AsyncMock(return_value=mock_user)

        result = await service.get_user_by_username("testuser")

        mock_repo.get_user_by_username.assert_called_once_with("testuser")
        assert result.username == "testuser"


@pytest.mark.asyncio
async def test_update_user():
    """Test user update."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    update_data = {"first_name": "Updated", "last_name": "Name"}

    with patch.object(service, "repository") as mock_repo:
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.first_name = "Updated"
        mock_repo.update_user = AsyncMock(return_value=mock_user)

        result = await service.update_user(1, update_data)

        mock_repo.update_user.assert_called_once_with(1, update_data)
        assert result.first_name == "Updated"


@pytest.mark.asyncio
async def test_authenticate_user():
    """Test user authentication."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    with patch("app.services.user_service.verify_password") as mock_verify:
        with patch.object(service, "repository") as mock_repo:
            mock_user = MagicMock()
            mock_user.username = "testuser"
            mock_user.hashed_password = (
                "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Uvj/Ey"
            )
            mock_repo.get_user_by_username = AsyncMock(return_value=mock_user)
            mock_verify.return_value = True

            result = await service.authenticate_user("testuser", "password123")

            mock_repo.get_user_by_username.assert_called_once_with("testuser")
            mock_verify.assert_called_once_with(
                "password123",
                "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Uvj/Ey",
            )
            assert result == mock_user


@pytest.mark.asyncio
async def test_authenticate_user_invalid_password():
    """Test user authentication with invalid password."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    with patch("app.services.user_service.verify_password") as mock_verify:
        with patch.object(service, "repository") as mock_repo:
            mock_user = MagicMock()
            mock_user.username = "testuser"
            mock_user.hashed_password = (
                "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Uvj/Ey"
            )
            mock_repo.get_user_by_username = AsyncMock(return_value=mock_user)
            mock_verify.return_value = False

            result = await service.authenticate_user("testuser", "wrong_password")

            assert result is None


@pytest.mark.asyncio
async def test_authenticate_user_not_found():
    """Test user authentication with non-existent user."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_repo.get_user_by_username = AsyncMock(return_value=None)

        result = await service.authenticate_user("nonexistent", "password123")

        assert result is None


@pytest.mark.asyncio
async def test_update_last_login():
    """Test updating user's last login."""
    mock_session = AsyncMock()
    service = UserService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_repo.update_user = AsyncMock(return_value=None)

        await service.update_last_login(1)

        mock_repo.update_user.assert_called_once()
        args = mock_repo.update_user.call_args[0]
        assert args[0] == 1
        assert "last_login" in args[1]
