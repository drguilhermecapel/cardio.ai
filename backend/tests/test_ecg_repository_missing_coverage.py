"""ECG repository tests for missing coverage areas."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import os

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"

from app.repositories.ecg_repository import ECGRepository
from app.models.ecg_analysis import ECGAnalysis
from app.core.constants import AnalysisStatus


@pytest.fixture
def ecg_repository():
    """Create ECG repository instance."""
    mock_db = AsyncMock()
    return ECGRepository(mock_db)


@pytest.mark.asyncio
async def test_search_analyses_with_all_filters(ecg_repository):
    """Test search analyses with all possible filters - covers lines 92-116."""
    filters = {
        "patient_id": 1,
        "status": "completed",
        "clinical_urgency": "high",
        "diagnosis_category": "arrhythmia",
        "date_from": "2024-01-01",
        "date_to": "2024-12-31",
        "is_validated": True,
        "requires_validation": False,
        "created_by": 1,
    }

    mock_count_result = Mock()
    mock_count_result.scalar.return_value = 5
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []

    ecg_repository.db.execute = AsyncMock(side_effect=[mock_count_result, mock_result])

    analyses, total = await ecg_repository.search_analyses(filters, limit=10, offset=0)

    assert total == 5
    assert analyses == []


@pytest.mark.asyncio
async def test_search_analyses_empty_filters(ecg_repository):
    """Test search analyses with empty filters - covers lines 118-137."""
    filters = {}

    mock_count_result = Mock()
    mock_count_result.scalar.return_value = 10
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []

    ecg_repository.db.execute = AsyncMock(side_effect=[mock_count_result, mock_result])

    analyses, total = await ecg_repository.search_analyses(filters, limit=10, offset=0)

    assert total == 10
    assert analyses == []


@pytest.mark.asyncio
async def test_update_analysis_status_success(ecg_repository):
    """Test update analysis status success - covers lines 157-170."""
    mock_analysis = Mock(spec=ECGAnalysis)
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_analysis
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)
    ecg_repository.db.commit = AsyncMock()

    success = await ecg_repository.update_analysis_status(1, AnalysisStatus.COMPLETED)

    assert success is True
    assert mock_analysis.status == AnalysisStatus.COMPLETED


@pytest.mark.asyncio
async def test_update_analysis_status_not_found(ecg_repository):
    """Test update analysis status not found - covers lines 169-170."""
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)

    success = await ecg_repository.update_analysis_status(1, AnalysisStatus.COMPLETED)

    assert success is False


@pytest.mark.asyncio
async def test_delete_analysis(ecg_repository):
    """Test delete analysis - covers lines 172-176."""
    with patch.object(ecg_repository, "update_analysis") as mock_update:
        mock_update.return_value = Mock()

        result = await ecg_repository.delete_analysis(1)

        assert result is True


@pytest.mark.asyncio
async def test_create_measurement(ecg_repository):
    """Test create measurement - covers lines 178-183."""
    mock_measurement = Mock()
    ecg_repository.db.add = Mock()
    ecg_repository.db.commit = AsyncMock()
    ecg_repository.db.refresh = AsyncMock()

    result = await ecg_repository.create_measurement(mock_measurement)

    assert result == mock_measurement
    ecg_repository.db.add.assert_called_once_with(mock_measurement)


@pytest.mark.asyncio
async def test_get_measurements_by_analysis(ecg_repository):
    """Test get measurements by analysis - covers lines 185-195."""
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)

    measurements = await ecg_repository.get_measurements_by_analysis(1)

    assert measurements == []


@pytest.mark.asyncio
async def test_create_annotation(ecg_repository):
    """Test create annotation - covers lines 197-202."""
    mock_annotation = Mock()
    ecg_repository.db.add = Mock()
    ecg_repository.db.commit = AsyncMock()
    ecg_repository.db.refresh = AsyncMock()

    result = await ecg_repository.create_annotation(mock_annotation)

    assert result == mock_annotation
    ecg_repository.db.add.assert_called_once_with(mock_annotation)


@pytest.mark.asyncio
async def test_get_annotations_by_analysis(ecg_repository):
    """Test get annotations by analysis - covers lines 204-214."""
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)

    annotations = await ecg_repository.get_annotations_by_analysis(1)

    assert annotations == []


@pytest.mark.asyncio
async def test_get_critical_analyses(ecg_repository):
    """Test get critical analyses - covers lines 216-236."""
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)

    analyses = await ecg_repository.get_critical_analyses(limit=20)

    assert analyses == []


@pytest.mark.asyncio
async def test_get_pending_validations(ecg_repository):
    """Test get pending validations - covers lines 238-253."""
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)

    analyses = await ecg_repository.get_pending_validations(limit=50)

    assert analyses == []


@pytest.mark.asyncio
async def test_get_analysis_statistics_with_dates(ecg_repository):
    """Test get analysis statistics with date filters - covers lines 255-306."""
    mock_total_result = Mock()
    mock_total_result.scalar.return_value = 100
    mock_status_result = Mock()
    mock_status_result.all.return_value = [("completed", 80), ("pending", 20)]
    mock_critical_result = Mock()
    mock_critical_result.scalar.return_value = 5

    ecg_repository.db.execute = AsyncMock(
        side_effect=[mock_total_result, mock_status_result, mock_critical_result]
    )

    stats = await ecg_repository.get_analysis_statistics(
        date_from="2024-01-01", date_to="2024-12-31"
    )

    assert stats["total_analyses"] == 100
    assert stats["status_distribution"]["completed"] == 80
    assert stats["critical_analyses"] == 5
    assert "validation_rate" in stats


@pytest.mark.asyncio
async def test_get_analysis_statistics_no_dates(ecg_repository):
    """Test get analysis statistics without date filters - covers lines 259-306."""
    mock_total_result = Mock()
    mock_total_result.scalar.return_value = 50
    mock_status_result = Mock()
    mock_status_result.all.return_value = [("completed", 40), ("failed", 10)]
    mock_critical_result = Mock()
    mock_critical_result.scalar.return_value = 2

    ecg_repository.db.execute = AsyncMock(
        side_effect=[mock_total_result, mock_status_result, mock_critical_result]
    )

    stats = await ecg_repository.get_analysis_statistics()

    assert stats["total_analyses"] == 50
    assert stats["status_distribution"]["completed"] == 40
    assert stats["critical_analyses"] == 2


@pytest.mark.asyncio
async def test_update_analysis_with_attributes(ecg_repository):
    """Test update analysis with valid attributes - covers lines 147-155."""
    mock_analysis = Mock(spec=ECGAnalysis)
    mock_analysis.status = "pending"
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_analysis
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)
    ecg_repository.db.commit = AsyncMock()
    ecg_repository.db.refresh = AsyncMock()

    update_data = {"status": "completed", "confidence": 0.95}

    with patch(
        "builtins.hasattr",
        side_effect=lambda obj, attr: attr in ["status", "confidence"],
    ):
        result = await ecg_repository.update_analysis(1, update_data)

        assert result == mock_analysis
        ecg_repository.db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_update_analysis_not_found(ecg_repository):
    """Test update analysis when analysis not found - covers lines 155."""
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None
    ecg_repository.db.execute = AsyncMock(return_value=mock_result)

    result = await ecg_repository.update_analysis(1, {"status": "completed"})

    assert result is None
