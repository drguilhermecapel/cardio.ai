import pytest
from datetime import datetime
from app.models.user import User
from app.models.patient import Patient
from app.models.ecg_analysis import ECGAnalysis
from app.models.notification import Notification
from app.models.validation import Validation

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"


def test_user_model_creation():
    """Test User model creation."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        first_name="Test",
        last_name="User",
        is_active=True,
        is_superuser=False,
    )

    assert user.email == "test@example.com"
    assert user.first_name == "Test"
    assert user.last_name == "User"
    assert user.is_active is True
    assert user.is_superuser is False


def test_patient_model_creation():
    """Test Patient model creation."""
    from app.models.patient import Patient
    from datetime import date

    patient = Patient(
        patient_id="PAT123",
        first_name="John",
        last_name="Doe",
        date_of_birth=date(1990, 1, 1),
        gender="male",
    )

    assert patient.first_name == "John"
    assert patient.last_name == "Doe"
    assert patient.gender == "male"


def test_ecg_analysis_model_creation():
    """Test ECGAnalysis model creation."""
    from datetime import datetime

    analysis = ECGAnalysis(
        analysis_id="TEST123",
        patient_id=1,
        created_by=1,
        original_filename="test.csv",
        file_path="/path/to/ecg.csv",
        file_hash="abc123",
        file_size=1024,
        acquisition_date=datetime.now(),
        sample_rate=500,
        duration_seconds=10.0,
        leads_count=12,
        leads_names=["I", "II", "III"],
    )

    assert analysis.patient_id == 1
    assert analysis.analysis_id == "TEST123"
    assert analysis.original_filename == "test.csv"


def test_notification_model_creation():
    """Test Notification model creation."""
    notification = Notification(
        user_id=1,
        title="Test Notification",
        message="This is a test notification",
        notification_type="info",
        is_read=False,
    )

    assert notification.user_id == 1
    assert notification.title == "Test Notification"
    assert notification.notification_type == "info"
    assert notification.is_read is False


def test_validation_model_creation():
    """Test Validation model creation."""
    from app.core.constants import ValidationStatus

    validation = Validation(
        analysis_id=1,
        validator_id=1,
        status=ValidationStatus.PENDING,
        clinical_notes="Needs review",
    )

    assert validation.analysis_id == 1
    assert validation.validator_id == 1
    assert validation.status == ValidationStatus.PENDING
    assert validation.clinical_notes == "Needs review"


def test_user_model_str_representation():
    """Test User model string representation."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        first_name="Test",
        last_name="User",
    )

    assert "testuser" in str(user)


def test_patient_model_str_representation():
    """Test Patient model string representation."""
    from app.models.patient import Patient
    from datetime import date

    patient = Patient(
        patient_id="PAT123",
        first_name="John",
        last_name="Doe",
        date_of_birth=date(1990, 1, 1),
        gender="male",
    )

    assert "John" in str(patient) or "PAT123" in str(patient)


def test_model_timestamps():
    """Test model timestamp fields."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        first_name="Test",
        last_name="User",
    )

    assert hasattr(user, "created_at")
    assert hasattr(user, "updated_at")


def test_model_relationships():
    """Test model relationship definitions."""
    from app.models.patient import Patient

    assert hasattr(User, "analyses")
    assert hasattr(User, "validations")
    assert hasattr(ECGAnalysis, "validations")


def test_model_table_names():
    """Test model table names."""
    from app.models.patient import Patient

    assert User.__tablename__ == "users"
    assert Patient.__tablename__ == "patients"
    assert ECGAnalysis.__tablename__ == "ecg_analyses"
    assert Notification.__tablename__ == "notifications"
    assert Validation.__tablename__ == "validations"
