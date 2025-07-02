import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip("Service tests skipped in standalone mode", allow_module_level=True)

from app.services.ecg_service import ECGAnalysisService
from app.services.notification_service import NotificationService
from app.services.validation_service import ValidationService
from app.services.ml_model_service import MLModelService


@pytest.mark.asyncio
async def test_ecg_service_initialization():
    """Test ECG service initialization."""
    mock_session = AsyncMock()
    mock_ml_service = MagicMock()
    mock_validation_service = MagicMock()

    service = ECGAnalysisService(mock_session, mock_ml_service, mock_validation_service)
    assert service.db == mock_session
    assert service.ml_service == mock_ml_service
    assert service.validation_service == mock_validation_service


@pytest.mark.asyncio
async def test_notification_service_initialization():
    """Test notification service initialization."""
    mock_session = AsyncMock()
    service = NotificationService(mock_session)
    assert service.db == mock_session


@pytest.mark.asyncio
async def test_validation_service_initialization():
    """Test validation service initialization."""
    mock_session = AsyncMock()
    mock_notification_service = MagicMock()

    service = ValidationService(mock_session, mock_notification_service)
    assert service.db == mock_session
    assert service.notification_service == mock_notification_service


@pytest.mark.asyncio
async def test_ml_model_service_initialization():
    """Test ML model service initialization."""
    service = MLModelService()
    assert service is not None
    assert hasattr(service, "models")
    assert hasattr(service, "model_metadata")


@pytest.mark.asyncio
async def test_service_basic_functionality():
    """Test basic service functionality."""
    mock_session = AsyncMock()
    mock_ml_service = MagicMock()
    mock_validation_service = MagicMock()

    ecg_service = ECGAnalysisService(
        mock_session, mock_ml_service, mock_validation_service
    )

    assert hasattr(ecg_service, "create_analysis")
    assert hasattr(ecg_service, "get_analysis_by_id")

    notification_service = NotificationService(mock_session)
    assert hasattr(notification_service, "send_validation_assignment")

    validation_service = ValidationService(mock_session, MagicMock())
    assert hasattr(validation_service, "create_validation")


@pytest.mark.asyncio
async def test_ml_model_service_methods():
    """Test ML model service methods."""
    service = MLModelService()

    assert hasattr(service, "analyze_ecg")
    assert hasattr(service, "get_model_info")
    assert hasattr(service, "unload_model")

    model_info = service.get_model_info()
    assert "loaded_models" in model_info
    assert "model_metadata" in model_info
