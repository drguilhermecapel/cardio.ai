"""Testes para configuração e constantes."""

import os
os.environ["ENVIRONMENT"] = "test"

from app.core.config import settings
from app.core.constants import *


class TestConfiguration:
    """Test configuration coverage."""
    
    def test_all_settings(self):
        """Test all settings attributes."""
        # Basic settings
        assert settings.PROJECT_NAME == "CardioAI Pro"
        assert settings.VERSION == "1.0.0"
        assert settings.API_V1_STR == "/api/v1"
        
        # Database settings
        assert settings.POSTGRES_SERVER
        assert settings.POSTGRES_USER
        assert settings.POSTGRES_PASSWORD
        assert settings.POSTGRES_DB
        
        # Security settings
        assert settings.SECRET_KEY
        assert settings.ALGORITHM
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
        # CORS settings
        assert isinstance(settings.BACKEND_CORS_ORIGINS, list)
        
        # ML settings
        assert settings.ML_MODEL_PATH
        assert settings.ML_BATCH_SIZE
        assert settings.ML_MAX_QUEUE_SIZE
        
    def test_all_constants(self):
        """Test all constant enumerations."""
        # User roles
        assert UserRoles.ADMIN.value == "admin"
        assert UserRoles.PHYSICIAN.value == "physician"
        assert UserRoles.TECHNICIAN.value == "technician"
        assert UserRoles.PATIENT.value == "patient"
        
        # Analysis status
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.PROCESSING.value == "processing"
        assert AnalysisStatus.COMPLETED.value == "completed"
        assert AnalysisStatus.FAILED.value == "failed"
        
        # Validation status
        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.APPROVED.value == "approved"
        assert ValidationStatus.REJECTED.value == "rejected"
        assert ValidationStatus.REQUIRES_REVIEW.value == "requires_review"
        
        # Clinical urgency
        assert ClinicalUrgency.ROUTINE.value == "routine"
        assert ClinicalUrgency.URGENT.value == "urgent"
        assert ClinicalUrgency.EMERGENT.value == "emergent"
        
        # Notification types
        assert NotificationType.ECG_ANALYSIS_COMPLETE.value == "ecg_analysis_complete"
        assert NotificationType.VALIDATION_REQUIRED.value == "validation_required"
        assert NotificationType.CRITICAL_FINDING.value == "critical_finding"
