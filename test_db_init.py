import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip(
        "Database init tests skipped in standalone mode", allow_module_level=True
    )

from app.db.init_db import (
    init_db,
    create_admin_user,
    check_database_exists,
    ensure_database_ready,
)


@pytest.mark.asyncio
async def test_init_db():
    """Test database initialization."""
    mock_session = AsyncMock()

    with patch("app.db.init_db.get_session_factory") as mock_factory:
        mock_factory.return_value.return_value.__aenter__.return_value = mock_session

        with patch("app.db.init_db.create_admin_user") as mock_create_user:
            await init_db()
            mock_create_user.assert_called_once_with(mock_session)


@pytest.mark.asyncio
async def test_create_admin_user_new_user():
    """Test creating admin user when user doesn't exist."""
    mock_session = AsyncMock()

    with patch("sqlalchemy.future.select") as mock_select:
        with patch.object(mock_session, "execute") as mock_execute:
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None  # No existing user
            mock_execute.return_value = mock_result

            with patch("app.db.init_db.get_password_hash") as mock_hash:
                mock_hash.return_value = "hashed_password"

                with patch("app.db.init_db.User") as mock_user_class:
                    mock_user = MagicMock()
                    mock_user_class.return_value = mock_user

                    result = await create_admin_user(mock_session)

                    assert mock_session.add.called
                    assert mock_session.commit.called
                    assert result == mock_user


@pytest.mark.asyncio
async def test_create_admin_user_existing_user():
    """Test creating admin user when user already exists."""
    mock_session = AsyncMock()

    with patch("sqlalchemy.future.select") as mock_select:
        with patch.object(mock_session, "execute") as mock_execute:
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = MagicMock()  # Existing user
            mock_execute.return_value = mock_result

            result = await create_admin_user(mock_session)

            assert result is not None
            assert not mock_session.add.called


@pytest.mark.asyncio
async def test_init_db_with_tables():
    """Test database initialization with table creation."""
    mock_session = AsyncMock()
    mock_engine = MagicMock()
    mock_conn = AsyncMock()

    with patch("app.db.init_db.get_session_factory") as mock_factory:
        mock_factory.return_value.return_value.__aenter__.return_value = mock_session

        with patch("app.db.init_db.get_engine") as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            mock_engine.begin.return_value.__aenter__.return_value = mock_conn

            with patch("app.db.init_db.create_admin_user") as mock_create_user:
                await init_db()
                mock_conn.run_sync.assert_called_once()


@pytest.mark.asyncio
async def test_database_connection_error():
    """Test handling database connection errors."""
    with patch("app.db.init_db.get_session_factory") as mock_factory:
        mock_factory.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            await init_db()


@pytest.mark.asyncio
async def test_superuser_creation_error():
    """Test handling superuser creation errors."""
    mock_session = AsyncMock()

    with patch("app.db.init_db.get_session_factory") as mock_factory:
        mock_factory.return_value.return_value.__aenter__.return_value = mock_session

        with patch("app.db.init_db.create_admin_user") as mock_create_user:
            mock_create_user.side_effect = Exception("User creation failed")

            with pytest.raises(Exception, match="User creation failed"):
                await init_db()


def test_database_config_validation():
    """Test database configuration validation."""
    assert hasattr(settings, "DATABASE_URL")
    assert hasattr(settings, "STANDALONE_MODE")

    if settings.STANDALONE_MODE:
        assert "sqlite" in str(settings.DATABASE_URL)


@pytest.mark.asyncio
async def test_init_db_idempotent():
    """Test that init_db can be called multiple times safely."""
    mock_session = AsyncMock()

    with patch("app.db.init_db.get_session_factory") as mock_factory:
        mock_factory.return_value.return_value.__aenter__.return_value = mock_session

        with patch("app.db.init_db.create_admin_user") as mock_create_user:
            await init_db()
            await init_db()

            assert mock_create_user.call_count == 2


@pytest.mark.asyncio
async def test_database_migration_compatibility():
    """Test database initialization with migration compatibility."""
    mock_session = AsyncMock()

    with patch("app.db.init_db.get_session_factory") as mock_factory:
        mock_factory.return_value.return_value.__aenter__.return_value = mock_session

        with patch("app.models.base.Base.metadata") as mock_metadata:
            await init_db()

            assert mock_metadata.create_all.called
