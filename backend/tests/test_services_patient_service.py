import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip(
        "Patient service tests skipped in standalone mode", allow_module_level=True
    )

from app.services.patient_service import PatientService
from app.schemas.patient import PatientCreate


@pytest.mark.asyncio
async def test_patient_service_initialization():
    """Test patient service initialization."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)

    assert service.db == mock_session
    assert hasattr(service, "repository")


@pytest.mark.asyncio
async def test_create_patient():
    """Test patient creation."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)

    patient_data = PatientCreate(
        patient_id="P123",
        mrn="MRN123",
        first_name="John",
        last_name="Doe",
        date_of_birth=date(1990, 1, 1),
        gender="male",
    )

    with patch.object(service, "repository") as mock_repo:
        mock_patient = MagicMock()
        mock_patient.id = 1
        mock_patient.first_name = "John"
        mock_patient.last_name = "Doe"
        mock_repo.create_patient = AsyncMock(return_value=mock_patient)

        result = await service.create_patient(patient_data, created_by=1)

        mock_repo.create_patient.assert_called_once()
        assert result.id == 1
        assert result.first_name == "John"


@pytest.mark.asyncio
async def test_get_patient_by_patient_id():
    """Test getting patient by patient ID."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_patient = MagicMock()
        mock_patient.patient_id = "P123"
        mock_patient.first_name = "John"
        mock_repo.get_patient_by_patient_id = AsyncMock(return_value=mock_patient)

        result = await service.get_patient_by_patient_id("P123")

        mock_repo.get_patient_by_patient_id.assert_called_once_with("P123")
        assert result.patient_id == "P123"


@pytest.mark.asyncio
async def test_update_patient():
    """Test patient update."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)

    update_data = {"first_name": "Jane", "last_name": "Smith"}

    with patch.object(service, "repository") as mock_repo:
        mock_patient = MagicMock()
        mock_patient.id = 1
        mock_patient.first_name = "Jane"
        mock_repo.update_patient = AsyncMock(return_value=mock_patient)

        result = await service.update_patient(1, update_data)

        mock_repo.update_patient.assert_called_once_with(1, update_data)
        assert result.first_name == "Jane"


@pytest.mark.asyncio
async def test_get_patients():
    """Test getting patients with pagination."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_patients = [MagicMock(), MagicMock()]
        mock_repo.get_patients = AsyncMock(return_value=(mock_patients, 2))

        result, count = await service.get_patients(limit=10, offset=0)

        mock_repo.get_patients.assert_called_once_with(10, 0)
        assert len(result) == 2
        assert count == 2


@pytest.mark.asyncio
async def test_search_patients():
    """Test searching patients."""
    mock_session = AsyncMock()
    service = PatientService(mock_session)

    with patch.object(service, "repository") as mock_repo:
        mock_patients = [MagicMock()]
        mock_repo.search_patients = AsyncMock(return_value=(mock_patients, 1))

        result, count = await service.search_patients(
            "John", ["first_name"], limit=10, offset=0
        )

        mock_repo.search_patients.assert_called_once_with("John", ["first_name"], 10, 0)
        assert len(result) == 1
        assert count == 1
