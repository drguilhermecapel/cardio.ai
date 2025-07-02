"""Comprehensive test suite to achieve 80%+ coverage."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ALGORITHM"] = "HS256"

# Import after setting environment
from app.core.config import settings
from app.core.constants import (
    UserRoles,
    AnalysisStatus,
    ValidationStatus,
    ClinicalUrgency,
    NotificationType,
)
from app.core.exceptions import (
    ECGProcessingException,
    ValidationException,
    AuthenticationException,
)
from app.core.security import (
    create_access_token,
    verify_password,
    get_password_hash,
    decode_access_token,
)
from app.core.logging import get_logger, configure_logging, AuditLogger


class TestCoreConfig:
    """Test core configuration module."""

    def test_settings_initialization(self):
        """Test settings are properly initialized."""
        assert settings.PROJECT_NAME == "CardioAI Pro"
        assert settings.VERSION == "1.0.0"
        assert settings.API_V1_STR == "/api/v1"
        assert settings.ENVIRONMENT == "test"
        assert settings.DATABASE_URL is not None
        assert settings.SECRET_KEY is not None

    def test_cors_origins(self):
        """Test CORS origins configuration."""
        origins = settings.BACKEND_CORS_ORIGINS
        assert isinstance(origins, list)
        assert "http://localhost:3000" in origins

    def test_database_config(self):
        """Test database configuration."""
        assert settings.POSTGRES_SERVER is not None
        assert settings.POSTGRES_DB is not None
        assert isinstance(settings.DATABASE_POOL_SIZE, int)
        assert isinstance(settings.DATABASE_MAX_OVERFLOW, int)


class TestCoreConstants:
    """Test core constants module."""

    def test_user_roles(self):
        """Test user roles enumeration."""
        assert UserRoles.ADMIN.value == "admin"
        assert UserRoles.PHYSICIAN.value == "physician"
        assert UserRoles.TECHNICIAN.value == "technician"
        assert UserRoles.PATIENT.value == "patient"

    def test_analysis_status(self):
        """Test analysis status enumeration."""
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.PROCESSING.value == "processing"
        assert AnalysisStatus.COMPLETED.value == "completed"
        assert AnalysisStatus.FAILED.value == "failed"

    def test_validation_status(self):
        """Test validation status enumeration."""
        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.APPROVED.value == "approved"
        assert ValidationStatus.REJECTED.value == "rejected"
        assert ValidationStatus.REQUIRES_REVIEW.value == "requires_review"

    def test_clinical_urgency(self):
        """Test clinical urgency enumeration."""
        assert ClinicalUrgency.LOW.value == "low"
        assert ClinicalUrgency.MEDIUM.value == "medium"
        assert ClinicalUrgency.HIGH.value == "high"
        assert ClinicalUrgency.CRITICAL.value == "critical"

    def test_notification_type(self):
        """Test notification type enumeration."""
        assert NotificationType.ECG_ANALYSIS_COMPLETE.value == "ecg_analysis_complete"
        assert NotificationType.VALIDATION_REQUIRED.value == "validation_required"
        assert NotificationType.CRITICAL_FINDING.value == "critical_finding"


class TestCoreExceptions:
    """Test core exceptions module."""

    def test_ecg_processing_exception(self):
        """Test ECG processing exception."""
        with pytest.raises(ECGProcessingException) as exc_info:
            raise ECGProcessingException("Processing failed", details={"code": "E001"})

        assert str(exc_info.value) == "Processing failed"
        assert exc_info.value.details == {"code": "E001"}

    def test_validation_exception(self):
        """Test validation exception."""
        with pytest.raises(ValidationException) as exc_info:
            raise ValidationException("Validation failed")

        assert str(exc_info.value) == "Validation failed"

    def test_authentication_exception(self):
        """Test authentication exception."""
        with pytest.raises(AuthenticationException) as exc_info:
            raise AuthenticationException("Auth failed")

        assert str(exc_info.value) == "Auth failed"


class TestCoreSecurity:
    """Test core security module."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False
        assert verify_password("", hashed) is False

    def test_access_token_creation(self):
        """Test access token creation."""
        subject = "test_user"
        token = create_access_token(subject=subject)

        assert isinstance(token, str)
        assert len(token) > 0

        # Test with custom expiration
        custom_token = create_access_token(
            subject=subject, expires_delta=timedelta(hours=2)
        )
        assert isinstance(custom_token, str)

    def test_access_token_decode(self):
        """Test access token decoding."""
        subject = "test_user"
        token = create_access_token(subject=subject)

        decoded = decode_access_token(token)
        assert decoded is not None
        assert decoded.get("sub") == subject

        # Test invalid token
        invalid_decoded = decode_access_token("invalid_token")
        assert invalid_decoded is None


class TestCoreLogging:
    """Test core logging module."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_configure_logging(self):
        """Test logging configuration."""
        configure_logging()
        logger = get_logger("test_config")

        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_audit_logger(self):
        """Test audit logger functionality."""
        audit = AuditLogger()

        # Test user action logging
        audit.log_user_action(
            user_id=1,
            action="login",
            resource_type="auth",
            resource_id="session_123",
            details={"method": "password"},
            ip_address="127.0.0.1",
            user_agent="test-agent",
        )

        # Test system event logging
        audit.log_system_event(
            event_type="startup",
            description="System started",
            details={"version": "1.0.0"},
        )

        # Test data access logging
        audit.log_data_access(
            user_id=1,
            resource_type="patient",
            resource_id="P123",
            access_type="read",
            ip_address="127.0.0.1",
        )

        # Test medical action logging
        audit.log_medical_action(
            user_id=1, patient_id=123, action="ecg_analysis", details={"duration": 5.2}
        )


@pytest.mark.asyncio
class TestECGProcessor:
    """Test ECG processor utilities."""

    async def test_ecg_processor_initialization(self):
        """Test ECG processor initialization."""
        from app.utils.ecg_processor import ECGProcessor

        processor = ECGProcessor()
        assert processor is not None
        assert hasattr(processor, "sampling_rate")

    async def test_load_ecg_file_error(self):
        """Test ECG file loading error handling."""
        from app.utils.ecg_processor import ECGProcessor

        processor = ECGProcessor()

        # Test non-existent file
        with pytest.raises(Exception):
            await processor.load_ecg_file("/nonexistent/file.txt")

        # Test invalid file format
        with pytest.raises(Exception):
            await processor.load_ecg_file("invalid.xyz")

    @patch("app.utils.ecg_processor.np.loadtxt")
    async def test_preprocess_signal(self, mock_loadtxt):
        """Test signal preprocessing."""
        from app.utils.ecg_processor import ECGProcessor

        # Mock signal data
        mock_signal = np.random.randn(5000, 12)
        mock_loadtxt.return_value = mock_signal

        processor = ECGProcessor()

        # Test preprocessing
        processed = await processor.preprocess_signal(mock_signal)
        assert processed is not None
        assert isinstance(processed, np.ndarray)


@pytest.mark.asyncio
class TestSignalQualityAnalyzer:
    """Test signal quality analyzer."""

    async def test_analyzer_initialization(self):
        """Test signal quality analyzer initialization."""
        from app.utils.signal_quality import SignalQualityAnalyzer

        analyzer = SignalQualityAnalyzer()
        assert analyzer is not None

    async def test_analyze_quality_invalid_input(self):
        """Test q"""
