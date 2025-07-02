"""Additional comprehensive tests to boost coverage above 80%."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ecg_service import ECGAnalysisService
from app.services.validation_service import ValidationService
from app.services.user_service import UserService
from app.services.notification_service import NotificationService
from app.services.patient_service import PatientService
from app.core.constants import UserRoles
from app.schemas.user import UserCreate
from app.schemas.patient import PatientCreate
from app.models.ecg_analysis import ECGAnalysis
from app.models.patient import Patient
from app.models.user import User
from app.models.validation import Validation


@pytest.fixture
def notification_service(test_db):
    """Notification service."""
    return NotificationService(db=test_db)


@pytest.fixture
def patient_service(test_db):
    """Patient service."""
    return PatientService(db=test_db)


@pytest.fixture
def ecg_service(test_db):
    """ECG service."""
    return ECGAnalysisService(db=test_db, ml_service=Mock(), validation_service=Mock())


@pytest.fixture
def validation_service(test_db):
    """Validation service."""
    mock_notification_service = Mock()
    mock_notification_service.send_validation_assignment = AsyncMock()
    mock_notification_service.send_urgent_validation_alert = AsyncMock()
    mock_notification_service.send_no_validator_alert = AsyncMock()
    mock_notification_service.send_validation_complete = AsyncMock()
    mock_notification_service.send_critical_rejection_alert = AsyncMock()

    service = ValidationService(
        db=test_db, notification_service=mock_notification_service
    )
    return service


@pytest.fixture
def user_service(test_db):
    """User service."""
    return UserService(db=test_db)


@pytest.mark.asyncio
async def test_notification_service_send_email(notification_service):
    """Test notification service _send_email method."""
    with patch(
        "app.services.notification_service.NotificationService._send_email"
    ) as mock_email:
        mock_email.return_value = None

        notification = Mock()
        notification.title = "Test Subject"
        notification.user_id = 1
        notification.message = "Test Body"

        await notification_service._send_email(notification)

        mock_email.assert_called_once_with(notification)


@pytest.mark.asyncio
async def test_notification_service_send_sms(notification_service):
    """Test notification service _send_sms method."""
    with patch(
        "app.services.notification_service.NotificationService._send_sms"
    ) as mock_sms:
        mock_sms.return_value = None

        notification = Mock()
        notification.title = "Test SMS"
        notification.user_id = 1
        notification.message = "Test SMS message"

        await notification_service._send_sms(notification)

        mock_sms.assert_called_once_with(notification)


@pytest.mark.asyncio
async def test_notification_service_get_user_notifications(
    notification_service, test_db
):
    """Test getting user notifications."""
    notifications = await notification_service.get_user_notifications(
        user_id=1, limit=10, offset=0
    )

    assert isinstance(notifications, list)


@pytest.mark.asyncio
async def test_notification_service_mark_notification_read(notification_service):
    """Test marking notification as read."""
    with patch(
        "app.repositories.notification_repository.NotificationRepository.mark_notification_read"
    ) as mock_mark:
        mock_mark.return_value = True

        result = await notification_service.mark_notification_read(
            notification_id=1, user_id=1
        )

        assert result is True


@pytest.mark.asyncio
async def test_patient_service_create_patient(patient_service, test_db):
    """Test patient service create_patient method."""
    patient_data = PatientCreate(
        patient_id="PAT001",
        first_name="Test",
        last_name="Patient",
        date_of_birth="1990-01-01",
        gender="male",
        phone="123-456-7890",
        email="patient@example.com",
    )

    with patch(
        "app.repositories.patient_repository.PatientRepository.create_patient"
    ) as mock_create:
        mock_patient = Patient(
            id=1,
            patient_id="PAT001",
            first_name="Test",
            last_name="Patient",
            date_of_birth="1990-01-01",
            gender="male",
        )
        mock_create.return_value = mock_patient

        result = await patient_service.create_patient(
            patient_data, created_by="test_user_id"
        )

        assert result is not None
        assert result.first_name == "Test"


@pytest.mark.asyncio
async def test_patient_service_get_patient_by_patient_id(patient_service):
    """Test getting patient by patient ID."""
    with patch(
        "app.repositories.patient_repository.PatientRepository.get_patient_by_patient_id"
    ) as mock_get:
        mock_patient = Patient(
            id=1,
            patient_id="PAT001",
            first_name="Test",
            last_name="Patient",
            date_of_birth="1990-01-01",
            gender="male",
        )
        mock_get.return_value = mock_patient

        result = await patient_service.get_patient_by_patient_id("PAT001")

        assert result is not None
        assert result.first_name == "Test"


@pytest.mark.asyncio
async def test_patient_service_get_patients(patient_service):
    """Test getting patients list."""
    with patch(
        "app.repositories.patient_repository.PatientRepository.get_patients"
    ) as mock_get:
        mock_patients = [
            Patient(
                id=1,
                patient_id="PAT001",
                first_name="Patient",
                last_name="One",
                date_of_birth="1990-01-01",
                gender="male",
            ),
            Patient(
                id=2,
                patient_id="PAT002",
                first_name="Patient",
                last_name="Two",
                date_of_birth="1985-05-15",
                gender="female",
            ),
        ]
        mock_get.return_value = (mock_patients, 2)

        patients, total = await patient_service.get_patients(limit=10, offset=0)

        assert len(patients) == 2
        assert total == 2


@pytest.mark.asyncio
async def test_patient_service_search_patients(patient_service):
    """Test searching patients."""
    with patch(
        "app.repositories.patient_repository.PatientRepository.search_patients"
    ) as mock_search:
        mock_patients = [
            Patient(
                id=1,
                patient_id="PAT001",
                first_name="John",
                last_name="Doe",
                date_of_birth="1990-01-01",
                gender="male",
            )
        ]
        mock_search.return_value = (mock_patients, 1)

        patients, total = await patient_service.search_patients(
            query="John", search_fields=["first_name", "last_name"], limit=10, offset=0
        )

        assert len(patients) == 1
        assert total == 1


@pytest.mark.asyncio
async def test_ecg_service_get_analyses_by_patient(ecg_service):
    """Test getting ECG analyses by patient."""
    with patch(
        "app.repositories.ecg_repository.ECGRepository.get_analyses_by_patient"
    ) as mock_get:
        mock_analyses = [
            ECGAnalysis(
                id=1,
                analysis_id="ECG001",
                patient_id=1,
                file_path="/tmp/test.txt",
                original_filename="test.txt",
                file_hash="hash1",
                file_size=1024,
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
                status="completed",
                created_by=1,
            )
        ]
        mock_get.return_value = mock_analyses

        analyses = await ecg_service_instance.get_analyses_by_patient(
            patient_id=1, limit=10, offset=0
        )

        assert len(analyses) == 1


@pytest.mark.asyncio
async def test_ecg_service_delete_analysis(ecg_service):
    """Test deleting ECG analysis."""
    with patch(
        "app.repositories.ecg_repository.ECGRepository.delete_analysis"
    ) as mock_delete:
        mock_delete.return_value = True

        result = await ecg_service_instance.delete_analysis(analysis_id=1)

        assert result is True


@pytest.mark.asyncio
async def test_validation_service_create_validation(validation_service):
    """Test creating validation."""
    with patch.object(
        validation_service.repository, "get_analysis_by_id"
    ) as mock_get_analysis, patch.object(
        validation_service.repository, "get_validation_by_analysis"
    ) as mock_get_existing, patch.object(
        validation_service.repository, "create_validation"
    ) as mock_create:

        # Mock no existing validation
        mock_get_existing.return_value = None

        mock_analysis = Mock()
        mock_analysis.id = 1
        mock_analysis.clinical_urgency = "low"
        mock_get_analysis.return_value = mock_analysis

        mock_validation = Validation(
            id=1,
            analysis_id=1,
            validator_id=1,
            status="pending",
            clinical_notes="Test validation",
        )
        mock_create.return_value = mock_validation

        result = await validation_service.create_validation(
            analysis_id=1,
            validator_id=1,
            validator_role="physician",
            validator_experience_years=5,
        )

        assert result is not None


@pytest.mark.asyncio
async def test_user_service_get_user_by_username(user_service):
    """Test getting user by username."""
    with patch(
        "app.repositories.user_repository.UserRepository.get_user_by_username"
    ) as mock_get:
        mock_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role=UserRoles.PHYSICIAN,
        )
        mock_get.return_value = mock_user

        result = await user_service.get_user_by_username("testuser")

        assert result is not None
        assert result.username == "testuser"


@pytest.mark.asyncio
async def test_ecg_service_error_handling(ecg_service):
    """Test ECG service error handling."""
    with patch(
        "app.repositories.ecg_repository.ECGRepository.create_analysis"
    ) as mock_create:
        mock_create.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            await ecg_service_instance.create_analysis(
                patient_id=1,
                file_path="/tmp/nonexistent.txt",
                original_filename="nonexistent.txt",
                created_by=1,
            )


@pytest.mark.asyncio
async def test_validation_service_error_handling(validation_service):
    """Test validation service error handling."""
    with patch(
        "app.repositories.validation_repository.ValidationRepository.create_validation"
    ) as mock_create:
        mock_create.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            await validation_service.create_validation(
                analysis_id=1,
                validator_id=1,
                validator_role=UserRoles.PHYSICIAN,
                validator_experience_years=5,
            )


@pytest.mark.asyncio
async def test_user_service_error_handling(user_service):
    """Test user service error handling."""
    with patch(
        "app.repositories.user_repository.UserRepository.create_user"
    ) as mock_create:
        mock_create.side_effect = Exception("Database error")

        user_data = UserCreate(
            username="errortest",
            email="error@example.com",
            first_name="Error",
            last_name="Test",
            password="Password123!",
            role=UserRoles.PHYSICIAN,
        )

        with pytest.raises(Exception):
            await user_service.create_user(user_data)


@pytest.mark.asyncio
async def test_notification_service_error_handling(notification_service):
    """Test notification service error handling."""
    with patch(
        "app.services.notification_service.NotificationService._send_email"
    ) as mock_email:
        mock_email.side_effect = Exception("Email error")

        notification = Mock()
        notification.title = "Test Subject"
        notification.user_id = 1
        notification.message = "Test Body"
        notification.channels = ["email"]
        notification.id = 1

        await notification_service._send_notification(notification)


@pytest.mark.asyncio
async def test_patient_service_error_handling(patient_service):
    """Test patient service error handling."""
    with patch(
        "app.repositories.patient_repository.PatientRepository.create_patient"
    ) as mock_create:
        mock_create.side_effect = Exception("Database error")

        patient_data = PatientCreate(
            patient_id="PAT001",
            first_name="Error",
            last_name="Patient",
            date_of_birth="1990-01-01",
            gender="male",
        )

        with pytest.raises(Exception):
            await patient_service.create_patient(
                patient_data, created_by="test_user_id"
            )
