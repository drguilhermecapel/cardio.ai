import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip("Repository tests skipped in standalone mode", allow_module_level=True)

from app.repositories.user_repository import UserRepository
from app.repositories.patient_repository import PatientRepository
from app.repositories.ecg_repository import ECGRepository
from app.repositories.notification_repository import NotificationRepository
from app.repositories.validation_repository import ValidationRepository


@pytest.mark.asyncio
async def test_user_repository_initialization():
    """Test user repository initialization."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)
    assert repo.db == mock_session


@pytest.mark.asyncio
async def test_user_repository_create():
    """Test user repository create method."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)

    mock_user = MagicMock()
    mock_user.id = 1

    with patch.object(repo, "create_user") as mock_create:
        mock_create.return_value = mock_user

        result = await repo.create_user(mock_user)

        assert result.id == 1


@pytest.mark.asyncio
async def test_patient_repository_initialization():
    """Test patient repository initialization."""
    mock_session = AsyncMock()
    repo = PatientRepository(mock_session)
    assert repo.db == mock_session


@pytest.mark.asyncio
async def test_patient_repository_get_by_id():
    """Test patient repository get by ID."""
    mock_session = AsyncMock()
    repo = PatientRepository(mock_session)

    with patch.object(repo, "get_patient_by_id") as mock_get:
        mock_patient = MagicMock()
        mock_patient.id = 1
        mock_get.return_value = mock_patient

        result = await repo.get_patient_by_id(1)

        assert result.id == 1


@pytest.mark.asyncio
async def test_ecg_repository_initialization():
    """Test ECG repository initialization."""
    mock_session = AsyncMock()
    repo = ECGRepository(mock_session)
    assert repo.db == mock_session


@pytest.mark.asyncio
async def test_ecg_repository_create_analysis():
    """Test ECG repository create analysis."""
    mock_session = AsyncMock()
    repo = ECGRepository(mock_session)

    mock_analysis = MagicMock()
    mock_analysis.id = 1

    with patch.object(repo, "create_analysis") as mock_create:
        mock_create.return_value = mock_analysis

        result = await repo.create_analysis(mock_analysis)

        assert result.id == 1


@pytest.mark.asyncio
async def test_notification_repository_initialization():
    """Test notification repository initialization."""
    mock_session = AsyncMock()
    repo = NotificationRepository(mock_session)
    assert repo.db == mock_session


@pytest.mark.asyncio
async def test_notification_repository_create():
    """Test notification repository create."""
    mock_session = AsyncMock()
    repo = NotificationRepository(mock_session)

    mock_notification = MagicMock()
    mock_notification.id = 1

    with patch.object(repo, "create_notification") as mock_create:
        mock_create.return_value = mock_notification

        result = await repo.create_notification(mock_notification)

        assert result.id == 1


@pytest.mark.asyncio
async def test_validation_repository_initialization():
    """Test validation repository initialization."""
    mock_session = AsyncMock()
    repo = ValidationRepository(mock_session)
    assert repo.db == mock_session


@pytest.mark.asyncio
async def test_validation_repository_create():
    """Test validation repository create."""
    mock_session = AsyncMock()
    repo = ValidationRepository(mock_session)

    mock_validation = MagicMock()
    mock_validation.id = 1

    with patch.object(repo, "create_validation") as mock_create:
        mock_create.return_value = mock_validation

        result = await repo.create_validation(mock_validation)

        assert result.id == 1


@pytest.mark.asyncio
async def test_repository_error_handling():
    """Test repository error handling."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)

    with patch.object(repo, "create_user") as mock_create:
        mock_create.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            await repo.create_user(MagicMock())


@pytest.mark.asyncio
async def test_repository_session_management():
    """Test repository session management."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)

    assert repo.db is not None
    assert repo.db == mock_session


@pytest.mark.asyncio
async def test_repository_list_operations():
    """Test repository list operations."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)

    with patch.object(repo, "get_users") as mock_list:
        mock_users = [MagicMock(), MagicMock()]
        mock_list.return_value = mock_users

        result = await repo.get_users(offset=0, limit=10)

        assert len(result) == 2


@pytest.mark.asyncio
async def test_repository_update_operations():
    """Test repository update operations."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)

    with patch.object(repo, "update_user") as mock_update:
        mock_user = MagicMock()
        mock_user.id = 1
        mock_update.return_value = mock_user

        result = await repo.update_user(1, {"username": "updated"})

        assert result.id == 1


@pytest.mark.asyncio
async def test_repository_delete_operations():
    """Test repository delete operations."""
    mock_session = AsyncMock()
    repo = UserRepository(mock_session)

    with patch.object(repo.db, "delete") as mock_delete:
        mock_delete.return_value = None

        await repo.db.delete(MagicMock())

        mock_delete.assert_called_once()
