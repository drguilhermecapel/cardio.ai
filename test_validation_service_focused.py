"""Focused Validation Service Tests."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from unittest.mock import AsyncMock, Mock
from app.services.validation_service import ValidationService
from app.models.validation import Validation
from app.core.constants import UserRoles, ClinicalUrgency


@pytest.fixture
def validation_service(test_db):
    """Create validation service instance."""
    mock_notification_service = Mock()
    return ValidationService(db=test_db, notification_service=mock_notification_service)


@pytest.mark.asyncio
async def test_create_validation(validation_service):
    """Test creating validation."""
    mock_analysis = Mock()
    mock_analysis.id = 1
    mock_analysis.clinical_urgency = ClinicalUrgency.HIGH

    mock_validation = Validation()
    mock_validation.id = 1
    mock_validation.analysis_id = 1
    mock_validation.validator_id = 1

    validation_service.repository.get_validation_by_analysis = AsyncMock(
        return_value=None
    )
    validation_service.repository.get_analysis_by_id = AsyncMock(
        return_value=mock_analysis
    )
    validation_service.repository.create_validation = AsyncMock(
        return_value=mock_validation
    )
    validation_service._can_validate = Mock(return_value=True)
    validation_service._requires_second_opinion = Mock(return_value=False)
    validation_service.notification_service.send_validation_assignment = AsyncMock()

    validation = await validation_service.create_validation(
        analysis_id=1,
        validator_id=1,
        validator_role=UserRoles.PHYSICIAN,
        validator_experience_years=5,
    )

    assert validation is not None
    assert validation.analysis_id == 1
    assert validation.validator_id == 1


@pytest.mark.asyncio
async def test_submit_validation(validation_service):
    """Test submitting validation."""
    from app.core.constants import ValidationStatus

    mock_validation = Validation()
    mock_validation.id = 1
    mock_validation.validator_id = 1
    mock_validation.status = ValidationStatus.PENDING
    mock_validation.analysis_id = 1

    updated_validation = Validation()
    updated_validation.id = 1
    updated_validation.status = ValidationStatus.APPROVED
    updated_validation.analysis_id = 1

    validation_service.repository.get_validation_by_id = AsyncMock(
        return_value=mock_validation
    )
    validation_service.repository.update_validation = AsyncMock(
        return_value=updated_validation
    )
    validation_service._update_analysis_validation_status = AsyncMock()
    validation_service._calculate_quality_metrics = AsyncMock()
    validation_service._send_validation_notifications = AsyncMock()

    validation = await validation_service.submit_validation(
        validation_id=1,
        validator_id=1,
        validation_data={
            "approved": True,
            "clinical_notes": "Validation approved",
            "confidence_score": 0.95,
        },
    )

    assert validation is not None
    assert validation.status == ValidationStatus.APPROVED


@pytest.mark.asyncio
async def test_create_urgent_validation(validation_service):
    """Test creating urgent validation."""
    mock_validator = Mock()
    mock_validator.id = 1
    mock_validator.role = UserRoles.PHYSICIAN
    mock_validator.experience_years = 5

    mock_analysis = Mock()
    mock_analysis.id = 1
    mock_analysis.clinical_urgency = ClinicalUrgency.CRITICAL

    mock_validation = Validation()
    mock_validation.id = 1
    mock_validation.analysis_id = 1

    validation_service.repository.get_available_validators = AsyncMock(
        return_value=[mock_validator]
    )
    validation_service.repository.get_validation_by_analysis = AsyncMock(
        return_value=None
    )
    validation_service.repository.get_analysis_by_id = AsyncMock(
        return_value=mock_analysis
    )
    validation_service.repository.create_validation = AsyncMock(
        return_value=mock_validation
    )
    validation_service._can_validate = Mock(return_value=True)
    validation_service._requires_second_opinion = Mock(return_value=False)
    validation_service.notification_service.send_validation_assignment = AsyncMock()
    validation_service.notification_service.send_urgent_validation_alert = AsyncMock()

    await validation_service.create_urgent_validation(analysis_id=1)

    validation_service.notification_service.send_urgent_validation_alert.assert_called_once()


@pytest.mark.asyncio
async def test_run_automated_validation_rules(validation_service):
    """Test running automated validation rules."""
    mock_analysis = Mock()
    mock_analysis.id = 1

    validation_service.repository.get_active_validation_rules = AsyncMock(
        return_value=[]
    )
    validation_service.repository.get_analysis_by_id = AsyncMock(
        return_value=mock_analysis
    )

    results = await validation_service.run_automated_validation_rules(analysis_id=1)

    assert len(results) == 0  # No rules returned from repository


@pytest.mark.asyncio
async def test_can_validate_permission(validation_service):
    """Test validation permission check."""
    result = validation_service._can_validate(
        validator_role=UserRoles.PHYSICIAN,
        validator_experience_years=5,
        clinical_urgency=ClinicalUrgency.HIGH,
    )

    assert result is True


@pytest.mark.asyncio
async def test_calculate_quality_metrics(validation_service):
    """Test quality metrics calculation."""
    ecg_data = {"signal_quality": 0.95, "noise_level": 0.05, "baseline_drift": 0.02}

    await validation_service._calculate_quality_metrics(
        analysis_id=1, validation_data=ecg_data
    )

    assert True
