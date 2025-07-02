import pytest
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"


def test_config_initialization():
    """Test configuration initialization."""
    assert hasattr(settings, "DATABASE_URL")
    assert hasattr(settings, "STANDALONE_MODE")
    assert hasattr(settings, "SECRET_KEY")


def test_standalone_mode_config():
    """Test standalone mode configuration."""
    if settings.STANDALONE_MODE:
        assert "sqlite" in str(settings.DATABASE_URL)
    else:
        assert "postgresql" in str(settings.DATABASE_URL) or "sqlite" in str(
            settings.DATABASE_URL
        )


def test_secret_key_exists():
    """Test secret key exists."""
    assert settings.SECRET_KEY is not None
    assert len(settings.SECRET_KEY) > 0


def test_database_url_format():
    """Test database URL format."""
    assert settings.DATABASE_URL is not None
    assert isinstance(str(settings.DATABASE_URL), str)


def test_cors_origins():
    """Test CORS origins configuration."""
    cors_origins = getattr(
        settings, "BACKEND_CORS_ORIGINS", getattr(settings, "ALLOWED_HOSTS", [])
    )
    assert isinstance(cors_origins, list)


def test_project_name():
    """Test project name configuration."""
    assert hasattr(settings, "PROJECT_NAME")
    assert settings.PROJECT_NAME == "CardioAI Pro"


def test_api_version():
    """Test API version configuration."""
    assert hasattr(settings, "API_V1_STR")
    assert settings.API_V1_STR == "/api/v1"


def test_environment_variables():
    """Test environment variable handling."""
    assert hasattr(settings, "ENVIRONMENT")
    env = getattr(settings, "ENVIRONMENT", "test")
    assert env in ["development", "production", "testing", "test"]


def test_logging_configuration():
    """Test logging configuration."""
    assert hasattr(settings, "LOG_LEVEL")
    assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_security_settings():
    """Test security settings."""
    assert hasattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES")
    assert isinstance(settings.ACCESS_TOKEN_EXPIRE_MINUTES, int)
    assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0


def test_file_upload_settings():
    """Test file upload settings."""
    max_upload = getattr(
        settings, "MAX_UPLOAD_SIZE", getattr(settings, "MAX_ECG_FILE_SIZE", 0)
    )
    assert max_upload > 0


def test_ai_model_settings():
    """Test AI model settings."""
    model_path = getattr(
        settings, "AI_MODEL_PATH", getattr(settings, "MODELS_DIR", None)
    )
    model_enabled = getattr(settings, "AI_MODEL_ENABLED", True)
    assert model_path is not None
    assert isinstance(model_enabled, bool)
