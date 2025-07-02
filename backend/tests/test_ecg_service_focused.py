"""Focused ECG service tests for actual methods."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from app.services.ecg_service import ECGAnalysisService
from app.schemas.ecg_analysis import ECGAnalysisCreate
from app.models.ecg_analysis import ECGAnalysis


@pytest.fixture
def ecg_service(test_db):
    """Create ECG service instance."""
    mock_ml_service = Mock()
    mock_validation_service = Mock()
    return ECGAnalysisService(
        db=test_db,
        ml_service=mock_ml_service,
        validation_service=mock_validation_service,
    )


@pytest.fixture
def sample_ecg_data():
    """Sample ECG analysis data."""
    return ECGAnalysisCreate(
        patient_id=1,
        original_filename="test_ecg.txt",
        acquisition_date=datetime.utcnow(),
        sample_rate=500,
        duration_seconds=10.0,
        leads_count=12,
        leads_names=[
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        device_manufacturer="Test Device",
        device_model="v1.0",
    )


@pytest.mark.asyncio
async def test_create_ecg_analysis(ecg_service, sample_ecg_data):
    """Test ECG analysis creation."""
    mock_analysis = ECGAnalysis()
    mock_analysis.id = 1
    mock_analysis.patient_id = 1
    mock_analysis.original_filename = "test_ecg.txt"
    mock_analysis.created_by = 1

    ecg_service_instance.repository.create_analysis = AsyncMock(return_value=mock_analysis)
    ecg_service_instance.processor.extract_metadata = AsyncMock(return_value={})
    ecg_service_instance._calculate_file_info = AsyncMock(return_value=("hash123", 1024))

    analysis = await ecg_service_instance.create_analysis(
        patient_id=sample_ecg_data.patient_id,
        file_path="/tmp/test_ecg.txt",
        original_filename=sample_ecg_data.original_filename,
        created_by=1,
    )

    assert analysis is not None
    assert analysis.patient_id == 1
    assert analysis.original_filename == "test_ecg.txt"
    assert analysis.created_by == 1


@pytest.mark.asyncio
async def test_get_analysis_by_id(ecg_service):
    """Test getting ECG analysis by ID."""
    mock_analysis = ECGAnalysis()
    mock_analysis.id = 1
    mock_analysis.patient_id = 1

    ecg_service_instance.repository.get_analysis_by_id = AsyncMock(return_value=mock_analysis)

    analysis = await ecg_service_instance.get_analysis_by_id(1)

    assert analysis is not None
    assert analysis.id == 1
    assert analysis.patient_id == 1


@pytest.mark.asyncio
async def test_get_analyses_for_patient(ecg_service):
    """Test getting ECG analyses for a patient."""
    mock_analyses = [ECGAnalysis(), ECGAnalysis()]
    ecg_service_instance.repository.get_analyses_by_patient = AsyncMock(
        return_value=mock_analyses
    )

    analyses = await ecg_service_instance.get_analyses_by_patient(
        patient_id=1, limit=10, offset=0
    )

    assert len(analyses) == 2


@pytest.mark.asyncio
async def test_delete_analysis(ecg_service):
    """Test deleting ECG analysis."""
    ecg_service_instance.repository.delete_analysis = AsyncMock(return_value=True)

    success = await ecg_service_instance.delete_analysis(1)

    assert success is True
    ecg_service_instance.repository.delete_analysis.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_search_analyses_with_correct_params(ecg_service):
    """Test searching analyses with correct parameters."""
    mock_analyses = [ECGAnalysis(), ECGAnalysis()]
    ecg_service_instance.repository.search_analyses = AsyncMock(return_value=(mock_analyses, 2))

    analyses, total = await ecg_service_instance.search_analyses(
        filters={"patient_id": 1, "status": "completed"}, limit=10, offset=0
    )

    assert len(analyses) == 2
    assert total == 2
